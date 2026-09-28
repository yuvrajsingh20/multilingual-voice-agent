"""Composition root.

Everything is constructed here and passed in explicitly. There is no module-level
singleton that a request path reaches for: a test builds its own
:class:`Runtime` with a fixed clock and an empty session store, and gets exactly
the behaviour production gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.core.checks import PolicyConfig
from app.core.policy import PolicyEngine
from app.core.rules import RuleSet, load_rule_set
from app.core.session import SessionStore
from app.services.decision import (
    DecisionConfigurationError,
    DecisionService,
    DisabledDecisionService,
)
from app.services.llm import (
    LlmConfigurationError,
    LlmService,
    NotConfiguredLlmService,
)
from app.services.llm_openai import OpenAiCompatibleLlmService
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
    # Optional semantic decision provider. Defaulted so that everything built
    # before the decision layer existed builds unchanged.
    decisions: DecisionService = field(default_factory=DisabledDecisionService)


#: Top-level modules the Jev adapter needs that a deployment might lack.
_JEV_REQUIREMENTS = frozenset({"typesafe_sdk", "tenacity", "httpx2", "typing_extensions"})


def build_decision_service(settings: Settings) -> DecisionService:
    """Choose the decision provider from configuration. The only place that choice is made.

    * ``DECISION_PROVIDER=disabled`` and ``JEV_ENABLED=false`` - the default.
      :class:`DisabledDecisionService`; the Jev adapter module is never
      imported, so neither is the SDK, and no client exists.
    * The two switches disagree - a deployment mistake. It raises, so the
      process fails to start rather than guessing which switch was meant.
    * Both on - the Jev adapter, which additionally requires ``JEV_API_KEY``
      and ``JEV_MODEL``. Neither is guessed.
    """
    provider = settings.decision_provider
    wants_jev = provider == "jev"
    if not wants_jev and not settings.jev_enabled:
        return DisabledDecisionService()
    if wants_jev != settings.jev_enabled:
        raise DecisionConfigurationError(
            f"DECISION_PROVIDER={provider} but JEV_ENABLED={str(settings.jev_enabled).lower()}. "
            "Jev is used only when DECISION_PROVIDER=jev and JEV_ENABLED=true; set both or neither."
        )

    missing = [
        name
        for name, value in (("JEV_API_KEY", settings.jev_api_key), ("JEV_MODEL", settings.jev_model))
        if value is None
    ]
    if missing:
        raise DecisionConfigurationError(
            f"Jev is enabled but {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
            "not set; the application will not guess them."
        )

    try:
        # Deferred on purpose: the SDK is loaded only by a process that asked for it.
        from app.services.decision_jev import JevDecisionService
    except ModuleNotFoundError as exc:
        # Only the SDK and its own requirements are reported as "not installed".
        # Anything else missing is a defect here and keeps its own traceback.
        if (exc.name or "").split(".")[0] not in _JEV_REQUIREMENTS:
            raise
        raise DecisionConfigurationError(
            f"Jev is enabled but a module it needs is not installed ({exc.name}). "
            "Install the pinned requirement typesafe-sdk==0.7.1."
        ) from exc

    return JevDecisionService(
        api_key=settings.jev_api_key.get_secret_value(),  # type: ignore[union-attr]
        model=settings.jev_model,  # type: ignore[arg-type]
        base_url=settings.jev_base_url,
        timeout_seconds=settings.jev_timeout_seconds,
        max_retries=settings.jev_max_retries,
    )


def build_llm_service(settings: Settings) -> LlmService:
    """Choose the model service from configuration. The only place that choice is made.

    Three outcomes, and no fourth:

    * Neither a base URL nor a model name - the model is not configured, and the
      application keeps :class:`NotConfiguredLlmService`. No HTTP client is
      constructed, so an unconfigured process cannot reach a network at all. This
      is the default, and it is what keeps the test suite offline.
    * Exactly one of the two - a deployment mistake. It raises, so the process
      fails to start. Guessing a model name or a URL would silently point a
      compliance-sensitive call at the wrong model.
    * Both - the OpenAI-compatible adapter, pointed wherever configuration says.

    There is no fallback path. A configured model that is unreachable fails the
    turn; it never degrades to a different model.
    """
    has_url = settings.model_base_url is not None
    has_model = settings.model_name is not None

    if not has_url and not has_model:
        return NotConfiguredLlmService()
    if has_url != has_model:
        missing = "MODEL_NAME" if has_url else "MODEL_BASE_URL"
        raise LlmConfigurationError(
            f"{missing} is not set. A model endpoint needs both MODEL_BASE_URL and "
            "MODEL_NAME; the application will not guess one of them."
        )

    return OpenAiCompatibleLlmService(
        base_url=settings.model_base_url,  # type: ignore[arg-type]
        model=settings.model_name,  # type: ignore[arg-type]
        api_key=(
            settings.model_api_key.get_secret_value() if settings.model_api_key else None
        ),
        timeout_seconds=settings.model_timeout_seconds,
        connect_timeout_seconds=settings.model_connect_timeout_seconds,
        max_output_tokens=settings.model_max_output_tokens,
        temperature=settings.model_temperature,
        max_retries=settings.model_max_retries,
        provider=settings.model_provider,
    )


def build_runtime(
    settings: Settings | None = None,
    *,
    clock: Clock | None = None,
    backend: BankingBackend | None = None,
    llm: LlmService | None = None,
    decisions: DecisionService | None = None,
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
        llm=llm if llm is not None else build_llm_service(resolved),
        transcript_normalizer=WhitespaceTranscriptNormalizer(),
        tts_normalizer=SpanDetectingTtsNormalizer(),
        barge_in=HeuristicBargeInClassifier(),
        turn_detector=SilenceTurnDetector(),
        decisions=decisions if decisions is not None else build_decision_service(resolved),
    )
