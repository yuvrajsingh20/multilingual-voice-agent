"""Composition root.

Everything is constructed here and passed in explicitly. There is no module-level
singleton that a request path reaches for: a test builds its own
:class:`Runtime` with a fixed clock and an empty session store, and gets exactly
the behaviour production gets.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.core.checks import PolicyConfig
from app.core.policy import PolicyEngine
from app.core.rules import RuleSet, load_rule_set
from app.core.session import SessionStore
from app.services.llm import LlmService, NotConfiguredLlmService
from app.services.stt import TranscriptNormalizer, WhitespaceTranscriptNormalizer
from app.services.tts import SpanDetectingTtsNormalizer, TtsNormalizer
from app.services.turn import (
    BargeInClassifier,
    HeuristicBargeInClassifier,
    SilenceTurnDetector,
)
from app.services.validation import ResponseValidator
from app.tools.banking import BankingBackend, build_registry
from app.tools.base import ToolRegistry
from app.core.clock import Clock, SystemClock


@dataclass(frozen=True)
class Runtime:
    settings: Settings
    clock: Clock
    rule_set: RuleSet
    policy: PolicyEngine
    sessions: SessionStore
    tools: ToolRegistry
    validator: ResponseValidator
    llm: LlmService
    transcript_normalizer: TranscriptNormalizer
    tts_normalizer: TtsNormalizer
    barge_in: BargeInClassifier
    turn_detector: SilenceTurnDetector


def build_runtime(
    settings: Settings | None = None,
    *,
    clock: Clock | None = None,
    backend: BankingBackend | None = None,
    llm: LlmService | None = None,
) -> Runtime:
    resolved = settings or get_settings()
    rule_set = load_rule_set(resolved.regulatory_rules_path)
    policy = PolicyEngine(
        rule_set,
        PolicyConfig(
            policy_version=resolved.policy_version,
            timezone=resolved.default_timezone,
            max_contacts_per_day=resolved.max_recovery_calls_per_day,
        ),
    )
    tools = build_registry(backend)
    return Runtime(
        settings=resolved,
        clock=clock or SystemClock(resolved.default_timezone),
        rule_set=rule_set,
        policy=policy,
        sessions=SessionStore(max_sessions=resolved.max_active_sessions),
        tools=tools,
        validator=ResponseValidator(rule_set, tools.names),
        llm=llm or NotConfiguredLlmService(),
        transcript_normalizer=WhitespaceTranscriptNormalizer(),
        tts_normalizer=SpanDetectingTtsNormalizer(),
        barge_in=HeuristicBargeInClassifier(),
        turn_detector=SilenceTurnDetector(),
    )
