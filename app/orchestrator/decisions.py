"""The decision coordinator: the application's async entry point to the
optional decision layer.

It owns, for each registered decision:

* **routing** - whether a decision is needed at all. Most audio and most turns
  never reach a provider: noise, missing transcripts, sustained speech, an
  intent upstream NLU already supplied, and an escalation policy already
  requires are all settled without one;
* **the deadline** - enforced here for every provider, on top of whatever the
  provider enforces itself;
* **fallback** - every failure, and every answer below its threshold, becomes
  an UNCERTAIN result, and fusion then returns exactly the pre-existing
  behaviour;
* **measurement** - one structured record per decision attempted: name,
  provider, model, outcome, latency, confidence bucket and whether the fallback
  was used. Never the customer's words;
* **fusion** - :mod:`app.services.decisions.fusion`, which is where hard
  signals and policy bound what a decision can change.

It depends on :class:`~app.services.decision.DecisionService` only. It never
imports a provider SDK, never calls the LLM, never touches the tool registry,
and never writes conversation state: every resolution it returns is advice the
caller applies through the paths that already exist.

Why :class:`~app.orchestrator.pipeline.ConversationOrchestrator` does not call it
-------------------------------------------------------------------------------
The turn pipeline is synchronous, and in this codebase it has no response path
that does not go through the LLM. Calling a decision provider inside
``process_turn`` would add its latency to every turn without removing a single
model call, and the only intents safe to apply without confirmation change
nothing policy reads. The coordinator is therefore built for the caller that
does not exist yet - the asynchronous audio loop, which decides barge-in in
real time and can hand an applied intent to ``process_turn`` as an ordinary
upstream signal. See REPORT.md, "Stage 3A — Optional Decision Layer".
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from pydantic import ValidationError

from app.config import Settings
from app.models.conversation import ConversationState
from app.models.enums import ConversationStage, Intent, Language
from app.models.policy import PolicyDecision
from app.observability import get_logger, log_event
from app.services.decision import (
    MAX_DECISION_TEXT_CHARS,
    MAX_RECENT_UTTERANCES,
    DecisionAnswer,
    DecisionContext,
    DecisionError,
    DecisionMalformedResponse,
    DecisionName,
    DecisionOutcome,
    DecisionRequest,
    DecisionResult,
    DecisionService,
    DecisionTimeout,
    DisabledDecisionService,
    failed_result,
    result_from_answer,
)
from app.services.decisions.fusion import (
    DEFAULT_HARD_INTERRUPTION_MS,
    BargeInResolution,
    EscalationResolution,
    IntentResolution,
    barge_in_ineligibility,
    escalation_already_required,
    fuse_barge_in,
    fuse_escalation,
    fuse_intent,
)
from app.services.decisions.registry import DECISION_REGISTRY, get_spec
from app.services.turn import BargeInClassifier, BargeInSignal, HeuristicBargeInClassifier

if TYPE_CHECKING:  # the coordinator never holds a Runtime; see from_runtime
    from app.runtime import Runtime

_logger = get_logger(__name__)

#: Below this much remaining acknowledgement window, a barge-in decision is not
#: attempted: no provider answers in less, and the call would only be a cost.
MIN_PROVIDER_BUDGET_MS = 50.0


class DecisionCoordinator:
    """Routes, bounds, measures and fuses semantic decisions.

    Stateless between calls and safe to share across concurrent calls: the
    only shared resource is the decision service, which owns one connection
    pool for the process.
    """

    def __init__(
        self,
        *,
        service: DecisionService,
        settings: Settings,
        barge_in: BargeInClassifier,
    ) -> None:
        # Handed exactly what it uses. It never holds the tool registry, the
        # LLM, the policy engine or the session store, so it has nothing to
        # execute, generate, decide or write with even by mistake.
        self._service = service
        self._heuristic = barge_in
        self._deadline = settings.jev_timeout_seconds
        self._thresholds: dict[DecisionName, float] = {
            name: float(getattr(settings, spec.threshold_setting))
            for name, spec in DECISION_REGISTRY.items()
        }
        self._hard_interruption_ms = (
            barge_in.backchannel_max_ms
            if isinstance(barge_in, HeuristicBargeInClassifier)
            else DEFAULT_HARD_INTERRUPTION_MS
        )

    @classmethod
    def from_runtime(cls, runtime: Runtime) -> "DecisionCoordinator":
        """Build from the composition root, taking only the three things it needs."""
        return cls(service=runtime.decisions, settings=runtime.settings, barge_in=runtime.barge_in)

    @property
    def enabled(self) -> bool:
        """False when no provider is configured. Nothing is then called or logged."""
        return not isinstance(self._service, DisabledDecisionService)

    def threshold(self, name: DecisionName) -> float:
        return self._thresholds[get_spec(name).name]

    # -- decisions ----------------------------------------------------------

    async def resolve_barge_in(
        self,
        signal: BargeInSignal,
        *,
        agent_speaking: bool,
        recent_customer_utterances: tuple[str, ...] = (),
        session_id: str | None = None,
    ) -> BargeInResolution:
        """Continue, stop or wait, for customer audio heard while the agent speaks.

        The existing heuristic classifier always runs first and is the
        fallback. A provider is consulted only for audio that passes the
        acoustic gate, has a transcript, and carries no hard interruption
        signal; VAD and raw audio are never its business.

        The agent keeps speaking while a provider is consulted, so the wait is
        charged to the acknowledgement window: the provider gets only what is
        left of the hard-interruption bound after the speech already heard
        (never more than ``JEV_TIMEOUT_SECONDS``). Speech so far plus the wait
        therefore never exceeds that bound, and an answer that would arrive
        later is discarded as a timeout.

        Caller contract: keep feeding partial transcripts while a call is
        pending, and act on the newest resolution. Any resolution that says
        ``STOP_TTS`` stops the agent, whichever call produced it.
        """
        heuristic = self._heuristic.classify(signal)
        semantic: DecisionResult | None = None
        eligible = barge_in_ineligibility(
            signal,
            heuristic,
            agent_speaking=agent_speaking,
            hard_interruption_ms=self._hard_interruption_ms,
        ) is None
        budget_seconds = min(
            self._deadline,
            (self._hard_interruption_ms - signal.vad.duration_ms) / 1000.0,
        )
        if eligible and budget_seconds * 1000.0 >= MIN_PROVIDER_BUDGET_MS:
            context = _context(
                utterance=signal.partial_transcript or "",
                language=signal.language,
                agent_speaking=True,
                recent_customer_utterances=recent_customer_utterances,
            )
            if context is not None:
                semantic = await self._decide(
                    DecisionName.BARGE_IN, context, session_id, deadline=budget_seconds
                )
        return fuse_barge_in(
            heuristic=heuristic,
            signal=signal,
            agent_speaking=agent_speaking,
            semantic=semantic,
            hard_interruption_ms=self._hard_interruption_ms,
        )

    async def resolve_intent(
        self,
        utterance: str,
        *,
        caller_intent: Intent | None = None,
        language: Language | None = None,
        stage: ConversationStage | None = None,
        session_id: str | None = None,
    ) -> IntentResolution:
        """Classify a customer utterance's intent, unless upstream already has.

        The result is advice. Only a low-risk intent may be applied, and only by
        the caller passing it to ``process_turn`` as an upstream signal; a
        high-risk one is returned for confirmation and never applied here.
        """
        semantic: DecisionResult | None = None
        if caller_intent is None:
            context = _context(utterance=utterance, language=language, stage=stage)
            if context is not None:
                semantic = await self._decide(DecisionName.CUSTOMER_INTENT, context, session_id)
        return fuse_intent(caller_intent=caller_intent, semantic=semantic)

    async def resolve_escalation(
        self,
        utterance: str,
        *,
        state: ConversationState,
        policy: PolicyDecision | None,
        language: Language | None = None,
        recent_customer_utterances: tuple[str, ...] = (),
        session_id: str | None = None,
    ) -> EscalationResolution:
        """Whether a human should take over, as a recommendation.

        A provider is not asked when policy or session state already requires
        escalation - that is authoritative, and no answer could change it.
        """
        semantic: DecisionResult | None = None
        if not escalation_already_required(policy, state):
            context = _context(
                utterance=utterance,
                language=language,
                recent_customer_utterances=recent_customer_utterances,
            )
            if context is not None:
                semantic = await self._decide(DecisionName.HUMAN_ESCALATION, context, session_id)
        return fuse_escalation(policy=policy, state=state, semantic=semantic)

    # -- internals ----------------------------------------------------------

    async def _decide(
        self,
        name: DecisionName,
        context: DecisionContext,
        session_id: str | None,
        *,
        deadline: float | None = None,
    ) -> DecisionResult | None:
        """One bounded provider call. Never raises for a provider failure.

        ``deadline`` narrows the configured one for this call (barge-in passes
        what is left of its acknowledgement window). An answer measured after
        the deadline is discarded as a timeout: it is stale. That happens when
        the event loop was stalled while the answer was pending, whether by
        the provider or by other work on the loop, so a timeout record whose
        ``deadline_ms`` is small does not by itself implicate the provider.
        """
        if not self.enabled:
            return None
        spec = get_spec(name)
        threshold = self._thresholds[name]
        budget = self._deadline if deadline is None else min(deadline, self._deadline)
        provider = str(getattr(self._service, "provider", "unknown"))

        def failed(category: str) -> DecisionResult:
            return failed_result(
                name, threshold=threshold, provider=provider,
                category=category, latency_ms=_elapsed(started),
            )

        started = time.perf_counter()
        try:
            answer = await asyncio.wait_for(
                self._service.decide(DecisionRequest(name=name, context=context)),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            result = failed(DecisionTimeout.category)
        except DecisionError as exc:
            result = failed(exc.category)
        except Exception:  # noqa: BLE001 - a provider must never take the audio path down with it
            result = failed(DecisionError.category)
        else:
            result = self._accept(answer, name, spec.label_values, threshold, budget, started, failed)
        self._record(result, session_id, budget)
        return result

    @staticmethod
    def _accept(answer, name, labels, threshold, budget, started, failed) -> DecisionResult:
        """Turn whatever the provider returned into a result, or into a fallback.

        Checked here whatever the provider claims to have checked. Something
        that is not an answer, an answer that does not survive re-validation
        (``model_construct`` skips validation), an answer to another question,
        or a label nobody offered, is not a decision. Neither is an answer that
        arrived after the deadline: the event loop was stalled - by the provider
        or by anything else running on it - and the moment has passed.
        """
        latency_ms = _elapsed(started)
        if latency_ms > budget * 1000.0:
            return failed(DecisionTimeout.category)
        try:
            if not isinstance(answer, DecisionAnswer):
                return failed(DecisionMalformedResponse.category)
            checked = DecisionAnswer.model_validate(dict(answer.__dict__), strict=True)
            if checked.name is not name or checked.label not in labels:
                return failed(DecisionMalformedResponse.category)
            return result_from_answer(checked, threshold=threshold, latency_ms=latency_ms)
        except Exception:  # noqa: BLE001 - whatever a provider returned, it must not raise here
            return failed(DecisionMalformedResponse.category)

    def _record(self, result: DecisionResult, session_id: str | None, budget: float) -> None:
        """One record per decision. Categories and measurements only."""
        log_event(
            _logger,
            "decision",
            level=logging.WARNING if result.outcome is DecisionOutcome.FAILED else logging.INFO,
            session_id=session_id,
            decision_name=result.name.value,
            provider=result.provider,
            model=result.model,
            outcome=result.outcome.value,
            label=result.label,
            confidence_bucket=result.confidence_bucket,
            threshold=result.threshold,
            latency_ms=round(result.latency_ms, 3),
            deadline_ms=round(budget * 1000.0, 3),
            fallback_used=result.fallback_used,
            fallback_reason=result.failure,
        )


def _context(
    *,
    utterance: str,
    language: Language | None = None,
    agent_speaking: bool | None = None,
    stage: ConversationStage | None = None,
    recent_customer_utterances: tuple[str, ...] = (),
) -> DecisionContext | None:
    """Build a decision context, or ``None`` when the input is not eligible.

    Over-long earlier utterances are dropped and only the most recent are kept;
    an over-long or empty current utterance makes the decision ineligible
    rather than being truncated into something the customer did not say.
    """
    recent = tuple(
        item for item in recent_customer_utterances
        if item and item.strip() and len(item.strip()) <= MAX_DECISION_TEXT_CHARS
    )[-MAX_RECENT_UTTERANCES:]
    try:
        return DecisionContext(
            utterance=utterance,
            language=language,
            agent_speaking=agent_speaking,
            stage=stage,
            recent_customer_utterances=recent,
        )
    except ValidationError:
        return None


def _elapsed(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0
