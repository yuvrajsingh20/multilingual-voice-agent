"""The conversation orchestrator.

One customer turn in, one :class:`~app.orchestrator.result.TurnResult` out. This
module owns the *order* in which things happen and the decision to stop; it owns
no policy, no tool execution, no validation and no language.

Why the orchestrator, and not the model, drives
-----------------------------------------------
The model is asked for two things: words, and structured tool requests. It is
never asked whether collection is lawful, whether a tool may run, whether a
violation is acceptable, or whether a blocked sentence may be spoken. Those are
decided here, from :class:`~app.core.policy.PolicyEngine` output, before and
after the model is involved. A model that returns a perfectly-worded demand for
payment on an account under a pending grievance gets that demand discarded.

Policy is evaluated three times, and none of the three is redundant
------------------------------------------------------------------
``PRE_LLM``    before the model is called at all. If collection is prohibited the
               model is not invoked: there is nothing for it to usefully say, and
               generating a compliance-sensitive sentence would be a liability.
``POST_TOOL``  after every round of tool execution. A tool reads the *system of
               record*, and what it returns can change the decision - an account
               the session believed was a retail loan may come back as a digital
               lending product with its pre-contact obligations unmet. The
               pre-tool decision is not assumed to survive.
``PRE_TTS``    immediately before validation and speech, against the state the
               turn actually ended in.

Grounding
---------
:class:`~app.services.validation.GroundingFacts` is built from backend context
and from tool results, never from the model. If a fact was never loaded it is
absent, the model is told so, and any amount or date it states anyway is blocked
by :class:`~app.services.validation.ResponseValidator`. Nothing here invents a
fallback amount, date, DPD or account status.

Logging
-------
One structured record per turn, containing categories and measurements only. The
transcript, the draft, the response text, account and customer references,
amounts and backend error messages are all deliberately absent; they travel on
the :class:`~app.orchestrator.result.TurnResult` to the caller instead.
"""

from __future__ import annotations

import time
import uuid
from contextlib import nullcontext
from datetime import date
from typing import Any, Iterable

from pydantic import ValidationError

from app.core.policy import PolicyConfigurationError
from app.core.session import EventSignals, Session
from app.models.conversation import ConversationState
from app.models.customer import AccountContext
from app.models.enums import (
    EventKind,
    Intent,
    Language,
    RequiredAction,
    ToolStatus,
)
from app.models.policy import PolicyContext, PolicyDecision
from app.models.tools import ToolRequest, ToolResult
from app.observability import get_logger, log_event
from app.orchestrator.prompt import build_llm_request, replayed_turns
from app.orchestrator.result import (
    GroundingSource,
    PolicyCheckpoint,
    PolicyEvaluation,
    ToolAttempt,
    TurnError,
    TurnErrorCategory,
    TurnLatency,
    TurnOutcome,
    TurnResult,
    TurnStage,
)
from app.runtime import Runtime
from app.services.llm import (
    LlmError,
    LlmMessage,
    LlmNotConfigured,
    LlmToolCall,
    incomplete_reason,
    is_usable_call_id,
)
from app.services.stt import TranscriptSegment
from app.services.validation import DraftResponse, GroundingFacts

_logger = get_logger(__name__)


def _llm_error_category(exc: LlmError) -> TurnErrorCategory:
    """Map a model-boundary failure onto a turn error category.

    The two enumerations share their string values deliberately, so this is a
    lookup rather than a table that could drift. An unrecognised category - a
    failure class added at the boundary and not yet given a category here - falls
    back to the generic one rather than raising inside an error path.
    """
    try:
        return TurnErrorCategory(exc.category)
    except ValueError:
        return TurnErrorCategory.LLM_FAILED


class UnknownSession(KeyError):
    """No such session. A caller error, not a turn outcome."""


class IncompleteTurn(ValueError):
    """The customer has not finished speaking.

    Deciding that a turn is over belongs to
    :mod:`app.services.turn`, upstream. Running the pipeline on a partial
    hypothesis would advance conversation state and could commit a backend write
    on half a sentence.
    """


#: Tool payload keys that may update an :class:`AccountContext`, per tool.
#:
#: A tool can only refresh the fields listed for it. Anything else a backend
#: returns is ignored rather than merged, so no tool can introduce a field the
#: orchestrator did not expect it to own.
_ACCOUNT_PROJECTION: dict[str, frozenset[str]] = {
    "get_account_status": frozenset({"product_type", "currency", "dpd", "due_date"}),
    "get_outstanding_amount": frozenset({"currency", "outstanding_minor", "minimum_due_minor"}),
    "get_dpd": frozenset({"dpd"}),
}

#: Arguments that identify *who* a tool call is about.
#:
#: These are never taken from the model. The registry validates that arguments
#: are well-formed but has no way to know which borrower is on the call, so the
#: orchestrator binds these itself from the session's loaded context and
#: overwrites whatever the model supplied. The model is never told an account or
#: customer reference and cannot address any account but this session's - not
#: because it is asked not to, but because it has no way to.
_SCOPED_ARGUMENTS: tuple[str, ...] = ("account_ref", "customer_ref")

#: Every argument :meth:`ConversationOrchestrator._bind` sets itself. Left out
#: of the tool calls echoed back to the model on the next round: whatever the
#: model wrote there was overwritten, repeating it would tell the model its
#: choice was honoured, and repeating the bound value instead would tell it an
#: account or customer reference.
_BOUND_ARGUMENTS: frozenset[str] = frozenset((*_SCOPED_ARGUMENTS, "promise_date"))

#: Tools that change the bank's records, and the conversation state that must
#: already support them.
#:
#: These are the three tools whose effects are not reversible by a later turn: a
#: promise, a dispute and an escalation are all written to the case. The model
#: does not get to decide that the customer promised to pay, disputed the dues or
#: asked for a human - those are facts about what was *said*, they arrive as
#: upstream NLU signals, and the reducer records them. Without the corresponding
#: state the write is refused, so a model cannot manufacture the event it is
#: reporting.
_WRITE_TOOL_PRECONDITIONS: dict[str, tuple[str, str]] = {
    "record_payment_promise": ("payment_promise", "the customer has not made a payment promise"),
    "create_dispute": ("dispute", "the customer has not disputed the dues"),
    "escalate_case": ("escalation_required", "no escalation has been requested or required"),
}

#: The signal that authorises each write. A state flag, once raised, stays
#: raised, so the flag alone would let one promise be written again and again -
#: in a later round of the same turn, or on any later turn. Each time the
#: customer says it authorises one write of that kind: a write is refused if
#: one already succeeded since the most recent event carrying this intent.
_WRITE_TOOL_INTENTS: dict[str, Intent] = {
    "record_payment_promise": Intent.PAYMENT_PROMISE,
    "create_dispute": Intent.DISPUTE,
    "escalate_case": Intent.ESCALATION_REQUEST,
}


class _Working:
    """Mutable bookkeeping for one turn. Never returned; projected into a result."""

    def __init__(self) -> None:
        self.account: AccountContext | None = None
        self.errors: list[TurnError] = []
        self.tools: list[ToolAttempt] = []
        self.evaluations: list[PolicyEvaluation] = []
        self.event_ids: list[str] = []
        self.extra_amounts: set[int] = set()
        self.extra_dates: set[date] = set()
        self.sources: set[GroundingSource] = set()
        self.llm_calls: int = 0
        self.tool_ms: float = 0.0
        self.llm_ms: float = 0.0

    def fail(
        self,
        category: TurnErrorCategory,
        stage: TurnStage,
        detail: str,
        *,
        safety_critical: bool = True,
    ) -> None:
        self.errors.append(
            TurnError(
                category=category, stage=stage, detail=detail, safety_critical=safety_critical
            )
        )


class ConversationOrchestrator:
    """Deterministic turn pipeline.

    Stateless between turns: everything that persists lives in the
    :class:`~app.core.session.SessionStore`. Construct it once per process or
    per request; both behave identically.
    """

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        # Which arguments each registered tool accepts, read from the specs the
        # registry publishes. Used to bind session-scoped identity arguments.
        self._tool_properties: dict[str, frozenset[str]] = {
            spec.name: frozenset(spec.parameters.get("properties", {}))
            for spec in runtime.tools.specs()
        }

    # -- public API ---------------------------------------------------------

    def process_turn(
        self,
        session_id: str,
        segment: TranscriptSegment,
        *,
        signals: EventSignals | None = None,
        claimed_actions: tuple[RequiredAction, ...] = (),
    ) -> TurnResult:
        """Run one customer turn end to end.

        ``signals`` are upstream NLU outputs (intent, emotion, a promise date).
        They are *inputs to state*, never compliance decisions.

        ``claimed_actions`` is the caller's assertion about what the agent's
        utterance will perform - typically that this turn discloses recording or
        identifies the bank. The orchestrator cannot verify an assertion about
        the meaning of a Hindi or Marathi sentence, so it does not invent one:
        with no assertion, a required in-turn action is simply reported as unmet
        and the response is blocked. See REPORT.md, Stage 2 known limitations.
        """
        started = time.perf_counter()
        work = _Working()

        if not segment.is_final:
            raise IncompleteTurn(
                "process_turn expects a final transcript; end-of-turn is decided upstream"
            )

        session = self._runtime.sessions.get(session_id)
        if session is None:
            raise UnknownSession(session_id)

        # 1. Normalise the transcript. ------------------------------------
        normalization_started = time.perf_counter()
        try:
            transcript = self._runtime.transcript_normalizer.normalize(segment)
        except Exception as exc:  # noqa: BLE001 - boundary: a normaliser is replaceable
            work.fail(
                TurnErrorCategory.TRANSCRIPT_NORMALIZATION_FAILED,
                TurnStage.NORMALIZATION,
                f"Transcript normaliser raised {type(exc).__name__}.",
            )
            return self._abort(session, work, started, turn_id=session.state.turn_count)
        normalization_ms = _elapsed(normalization_started)

        language = transcript.language or segment.language or session.state.language

        # 2-4. Turn the utterance into an event and apply it to state. -----
        state_started = time.perf_counter()
        turn_id = session.state.turn_count + 1  # USER_UTTERANCE advances it by exactly one
        try:
            session = self._append(
                session,
                kind=EventKind.USER_UTTERANCE,
                work=work,
                language=language,
                text=transcript.normalized_text,
                data=(signals or EventSignals()).model_dump(mode="json", exclude_none=True),
            )
        except ValidationError as exc:
            work.fail(
                TurnErrorCategory.INVALID_CONVERSATION_EVENT,
                TurnStage.STATE_UPDATE,
                f"Event rejected by the state reducer: {exc.error_count()} invalid signal(s).",
            )
            return self._abort(session, work, started, turn_id=turn_id)
        state_ms = _elapsed(state_started)

        # 5. Load backend context. The session is where it lives; the -------
        #    orchestrator never reaches past the registry for it.
        context_started = time.perf_counter()
        account = session.account
        if account is None:
            work.fail(
                TurnErrorCategory.MISSING_BACKEND_CONTEXT,
                TurnStage.CONTEXT_LOAD,
                "No account context is loaded for this session; no account fact may be stated.",
                safety_critical=False,
            )
        else:
            work.sources.add(GroundingSource.ACCOUNT_CONTEXT)
        work.account = account
        context_ms = _elapsed(context_started)

        # 6. Policy, before the model is involved. --------------------------
        policy_started = time.perf_counter()
        decision = self._evaluate(session, account, work, PolicyCheckpoint.PRE_LLM)
        if decision is None:
            return self._abort(
                session, work, started, turn_id=turn_id,
                transcript_text=transcript.normalized_text,
                transcript_applied=transcript.applied,
                latency_parts=(normalization_ms, state_ms, context_ms, _elapsed(policy_started)),
            )
        policy_ms = _elapsed(policy_started)

        # 7. Collection prohibited: stop here. No model call. ---------------
        if not decision.allowed:
            self._record_policy_event(session, work, decision)
            return self._finish(
                session,
                work,
                started,
                turn_id=turn_id,
                outcome=TurnOutcome.POLICY_BLOCKED,
                transcript_text=transcript.normalized_text,
                transcript_applied=transcript.applied,
                language=language,
                decision=decision,
                latency=TurnLatency(
                    normalization_ms=normalization_ms,
                    state_update_ms=state_ms,
                    context_load_ms=context_ms,
                    policy_ms=policy_ms,
                    total_ms=_elapsed(started),
                ),
            )

        # 8-11. Model generation, with a bounded tool loop. -----------------
        budget = self._runtime.settings.max_tool_calls_per_turn
        history: list[LlmMessage] = []
        call_ids: set[str] = set()
        draft_text: str | None = None
        outcome: TurnOutcome | None = None
        last_good_decision = decision

        while True:
            grounding = self._grounding(account, work)
            generation = self._generate(
                session, work, decision, grounding, account, language,
                transcript.normalized_text, tuple(history),
            )
            if generation is None:
                return self._abort(
                    session, work, started, turn_id=turn_id, decision=decision,
                    transcript_text=transcript.normalized_text,
                    transcript_applied=transcript.applied,
                    latency_parts=(normalization_ms, state_ms, context_ms, policy_ms),
                )
            draft_text = generation.text

            if not generation.tool_calls:
                break

            if len(work.tools) + len(generation.tool_calls) > budget:
                work.fail(
                    TurnErrorCategory.TOOL_LIMIT_EXCEEDED,
                    TurnStage.TOOLS,
                    f"Model requested more than {budget} tool execution(s) in one turn; "
                    "tool execution stopped and no answer was produced.",
                )
                outcome = TurnOutcome.FAILED
                break

            # 10. Execute through the registry, then reload and re-decide.
            calls = _identified(generation.tool_calls, call_ids)
            refreshed, tool_messages, tool_failed = self._run_tools(
                session, work, calls, account
            )
            if refreshed is not None and refreshed != account:
                # A tool read the system of record, so the session's copy is now
                # known to be stale. Persist the correction and record it, or the
                # next turn would re-decide on data this turn already disproved.
                self._runtime.sessions.set_account(session.session_id, refreshed)
                session = self._append(
                    session, kind=EventKind.ACCOUNT_CONTEXT_LOADED, work=work
                )
            account = refreshed
            work.account = account
            # The model's own turn first, tool calls included, then one result
            # per call - the order the chat-completions contract requires. The
            # loop continues only when every call in the round was dispatched
            # and answered; any failure ends the turn below, before another
            # model call could send an unanswered call id.
            history.append(_assistant_turn(generation.text, calls))
            history.extend(tool_messages)
            if tool_failed:
                outcome = TurnOutcome.FAILED
                break

            policy_started = time.perf_counter()
            decision = self._evaluate(session, account, work, PolicyCheckpoint.POST_TOOL)
            policy_ms += _elapsed(policy_started)
            if decision is None:
                return self._abort(
                    session, work, started, turn_id=turn_id, decision=last_good_decision,
                    transcript_text=transcript.normalized_text,
                    transcript_applied=transcript.applied,
                    latency_parts=(normalization_ms, state_ms, context_ms, policy_ms),
                )
            last_good_decision = decision
            if not decision.allowed:
                self._record_policy_event(session, work, decision)
                outcome = TurnOutcome.POLICY_BLOCKED
                break

        if outcome is not None:
            return self._finish(
                session, work, started, turn_id=turn_id, outcome=outcome,
                transcript_text=transcript.normalized_text,
                transcript_applied=transcript.applied, language=language, decision=decision,
                draft_text=draft_text,
                latency=TurnLatency(
                    normalization_ms=normalization_ms, state_update_ms=state_ms,
                    context_load_ms=context_ms, policy_ms=policy_ms, llm_ms=work.llm_ms,
                    tool_ms=work.tool_ms, total_ms=_elapsed(started),
                ),
            )

        # 12a. Policy once more, against the state the turn ended in. -------
        policy_started = time.perf_counter()
        final_decision = self._evaluate(session, account, work, PolicyCheckpoint.PRE_TTS)
        policy_ms += _elapsed(policy_started)
        if final_decision is None:
            return self._abort(
                session, work, started, turn_id=turn_id, decision=decision,
                transcript_text=transcript.normalized_text,
                transcript_applied=transcript.applied,
                latency_parts=(normalization_ms, state_ms, context_ms, policy_ms),
            )
        decision = final_decision
        self._record_policy_event(session, work, decision)

        latency_so_far = TurnLatency(
            normalization_ms=normalization_ms, state_update_ms=state_ms,
            context_load_ms=context_ms, policy_ms=policy_ms, llm_ms=work.llm_ms,
            tool_ms=work.tool_ms, total_ms=_elapsed(started),
        )

        if not decision.allowed:
            return self._finish(
                session, work, started, turn_id=turn_id, outcome=TurnOutcome.POLICY_BLOCKED,
                transcript_text=transcript.normalized_text,
                transcript_applied=transcript.applied, language=language, decision=decision,
                draft_text=draft_text, latency=latency_so_far,
            )

        # 12b. Validate the draft. -----------------------------------------
        grounding = self._grounding(account, work)
        draft = DraftResponse(
            text=draft_text or "",
            language=language,
            tool_calls=(),
            claimed_actions=claimed_actions,
        )
        validation_started = time.perf_counter()
        try:
            validation = self._runtime.validator.validate(draft, decision, grounding)
        except Exception as exc:  # noqa: BLE001 - the gate itself failing is safety-critical
            work.fail(
                TurnErrorCategory.RESPONSE_VALIDATION_FAILED,
                TurnStage.VALIDATION,
                f"Response validator raised {type(exc).__name__}; nothing may be spoken.",
            )
            return self._finish(
                session, work, started, turn_id=turn_id, outcome=TurnOutcome.FAILED,
                transcript_text=transcript.normalized_text,
                transcript_applied=transcript.applied, language=language, decision=decision,
                draft_text=draft_text, latency=latency_so_far,
            )
        validation_ms = _elapsed(validation_started)

        # 13. Blocked: nothing reaches TTS. ---------------------------------
        if validation.blocked:
            work.fail(
                TurnErrorCategory.RESPONSE_VALIDATION_FAILED,
                TurnStage.VALIDATION,
                "Draft response blocked by validation: "
                + ", ".join(sorted({issue.code.value for issue in validation.issues if issue.blocking})),
            )
            return self._finish(
                session, work, started, turn_id=turn_id, outcome=TurnOutcome.RESPONSE_BLOCKED,
                transcript_text=transcript.normalized_text,
                transcript_applied=transcript.applied, language=language, decision=decision,
                draft_text=draft_text, validation=validation,
                latency=latency_so_far.model_copy(
                    update={"validation_ms": validation_ms, "total_ms": _elapsed(started)}
                ),
            )

        # 14. TTS normalisation. --------------------------------------------
        tts_started = time.perf_counter()
        try:
            speech = self._runtime.tts_normalizer.normalize(draft.text, language)
        except Exception as exc:  # noqa: BLE001 - boundary: a normaliser is replaceable
            work.fail(
                TurnErrorCategory.TTS_NORMALIZATION_FAILED,
                TurnStage.TTS_NORMALIZATION,
                f"TTS normaliser raised {type(exc).__name__}.",
            )
            return self._finish(
                session, work, started, turn_id=turn_id, outcome=TurnOutcome.FAILED,
                transcript_text=transcript.normalized_text,
                transcript_applied=transcript.applied, language=language, decision=decision,
                draft_text=draft_text, validation=validation,
                latency=latency_so_far.model_copy(
                    update={"validation_ms": validation_ms, "total_ms": _elapsed(started)}
                ),
            )
        tts_ms = _elapsed(tts_started)

        # A span this implementation cannot render is not a cosmetic gap: an
        # unrendered amount or date is read out wrongly by a synthesiser, which
        # is a false representation of an account fact. The turn stops rather
        # than speaking it. Text with no such spans is speakable in every
        # language, so this does not silently disable non-English calls.
        if speech.unrendered_kinds:
            work.fail(
                TurnErrorCategory.TTS_NORMALIZATION_FAILED,
                TurnStage.TTS_NORMALIZATION,
                "Response contains spans this implementation cannot render for "
                f"{language.value if language else 'an unknown language'}: "
                + ", ".join(kind.value for kind in speech.unrendered_kinds),
            )
            return self._finish(
                session, work, started, turn_id=turn_id, outcome=TurnOutcome.FAILED,
                transcript_text=transcript.normalized_text,
                transcript_applied=transcript.applied, language=language, decision=decision,
                draft_text=draft_text, validation=validation,
                tts_fully_normalized=speech.fully_normalized,
                tts_unrendered_kinds=speech.unrendered_kinds,
                latency=latency_so_far.model_copy(
                    update={
                        "validation_ms": validation_ms,
                        "tts_normalization_ms": tts_ms,
                        "total_ms": _elapsed(started),
                    }
                ),
            )

        # 15. Record the agent turn and return a speakable result. ----------
        session = self._append(
            session,
            kind=EventKind.AGENT_UTTERANCE,
            work=work,
            language=language,
            # The validated written form, not the spoken one: it is replayed to
            # the model on later turns, and grounding can only check amounts and
            # dates written as figures. The spoken form is derived from it.
            text=draft.text,
            data={
                "disclosed_recording": RequiredAction.DISCLOSE_CALL_RECORDING in claimed_actions,
                "identified_agent": RequiredAction.IDENTIFY_BANK_AND_AGENT in claimed_actions,
            },
        )

        return self._finish(
            session, work, started, turn_id=turn_id, outcome=TurnOutcome.COMPLETED,
            transcript_text=transcript.normalized_text,
            transcript_applied=transcript.applied, language=language, decision=decision,
            draft_text=draft_text, validation=validation, response_text=speech.text,
            tts_fully_normalized=speech.fully_normalized,
            tts_unrendered_kinds=speech.unrendered_kinds,
            latency=latency_so_far.model_copy(
                update={
                    "validation_ms": validation_ms,
                    "tts_normalization_ms": tts_ms,
                    "total_ms": _elapsed(started),
                }
            ),
        )

    # -- stages -------------------------------------------------------------

    def _evaluate(
        self,
        session: Session,
        account: AccountContext | None,
        work: _Working,
        checkpoint: PolicyCheckpoint,
    ) -> PolicyDecision | None:
        """Evaluate policy at ``checkpoint``. ``None`` means the engine failed."""
        try:
            decision = self._runtime.policy.evaluate(
                PolicyContext(
                    state=session.state,
                    now=self._runtime.clock.now(),
                    customer=session.customer,
                    account=account,
                    compliance=session.compliance,
                )
            )
        except PolicyConfigurationError as exc:
            work.fail(
                TurnErrorCategory.POLICY_CONFIGURATION_ERROR,
                TurnStage.POLICY,
                f"Policy engine is misconfigured: {exc}",
            )
            return None
        except Exception as exc:  # noqa: BLE001 - an undecidable policy must stop the turn
            work.fail(
                TurnErrorCategory.POLICY_EVALUATION_FAILED,
                TurnStage.POLICY,
                f"Policy evaluation raised {type(exc).__name__}; the turn cannot proceed.",
            )
            return None
        work.evaluations.append(PolicyEvaluation(checkpoint=checkpoint, decision=decision))
        return decision

    def _generate(
        self,
        session: Session,
        work: _Working,
        decision: PolicyDecision,
        grounding: GroundingFacts,
        account: AccountContext | None,
        language: Language | None,
        transcript: str,
        history: tuple[LlmMessage, ...],
    ):
        """One model call. ``None`` means the model is unavailable or failed."""
        request = build_llm_request(
            decision=decision,
            grounding=grounding,
            language=language,
            currency=account.currency if account else "INR",
            context_available=account is not None,
            account=account,
            transcript=transcript,
            prior_turns=replayed_turns(session.events),
            history=history,
            tools=self._runtime.tools.specs(),
        )
        started = time.perf_counter()
        try:
            generation = self._runtime.llm.generate(request)
        except LlmNotConfigured:
            work.llm_ms += _elapsed(started)
            work.fail(
                TurnErrorCategory.LLM_NOT_CONFIGURED,
                TurnStage.LLM,
                "No model endpoint is configured; this turn produces no response.",
            )
            return None
        except LlmError as exc:
            # A classified failure from the model boundary. `exc.detail` is
            # authored by app.services.llm and is guaranteed to carry no upstream
            # body, URL, header or key; `str(exc)` is not used for that reason.
            work.llm_ms += _elapsed(started)
            work.fail(
                _llm_error_category(exc),
                TurnStage.LLM,
                f"Model call failed: {exc.detail}.",
            )
            return None
        except Exception as exc:  # noqa: BLE001 - boundary: never leak a provider trace
            work.llm_ms += _elapsed(started)
            work.fail(
                TurnErrorCategory.LLM_FAILED,
                TurnStage.LLM,
                f"Model call raised {type(exc).__name__}.",
            )
            return None
        work.llm_ms += _elapsed(started)
        work.llm_calls += 1
        # The HTTP adapter refuses a cut-off generation itself; this holds the
        # same line for any other LlmService, so that no implementation can
        # hand the pipeline a fragment marked as one and have it spoken.
        cut_off = incomplete_reason(generation.finish_reason)
        if cut_off is not None:
            work.fail(
                TurnErrorCategory.LLM_INCOMPLETE_RESPONSE,
                TurnStage.LLM,
                f"Model call failed: llm_incomplete_response ({cut_off}).",
            )
            return None
        return generation

    def _run_tools(
        self,
        session: Session,
        work: _Working,
        calls: Iterable[LlmToolCall],
        account: AccountContext | None,
    ) -> tuple[AccountContext | None, list[LlmMessage], bool]:
        """Execute one round of tool requests through the registry.

        Returns the (possibly refreshed) account context, the messages to feed
        back to the model, and whether a failure ended the turn.
        """
        messages: list[LlmMessage] = []
        failed = False

        calls = tuple(calls)
        for call in calls:
            request_id = call.call_id or uuid.uuid4().hex

            # A write is not authorised until this whole check-execute-record
            # sequence has completed: two concurrent turns on this session must
            # not both pass the check before either's write is on record. Reads
            # need no such lock, so only a write tool call takes it.
            write_lock = (
                self._runtime.sessions.write_lock(session.session_id)
                if call.tool_name in _WRITE_TOOL_PRECONDITIONS
                else nullcontext()
            )
            with write_lock:
                arguments, refusal = self._bind(call, session, account)
                declared = self._tool_properties.get(call.tool_name, frozenset())
                if refusal is None and call.tool_name in _WRITE_TOOL_PRECONDITIONS and len(calls) > 1:
                    # A write commits before the round's policy re-evaluation can see
                    # what the round's reads returned. Keeping a write alone means the
                    # decision immediately before it is the freshest one available.
                    refusal = "a write must be the only tool call in its round"
                # Only keys the tool's own schema declares. The rest were invented by
                # the model, and a model-authored key name is not something to copy
                # into an audit record.
                argument_keys = tuple(
                    sorted(k for k in (arguments if refusal is None else call.arguments) if k in declared)
                )

                if refusal is not None:
                    work.tools.append(
                        ToolAttempt(
                            request_id=request_id,
                            tool_name=call.tool_name,
                            dispatched=False,
                            argument_keys=argument_keys,
                            refusal_reason=refusal,
                        )
                    )
                    work.fail(
                        TurnErrorCategory.INVALID_TOOL_REQUEST,
                        TurnStage.TOOLS,
                        f"Refused tool request {call.tool_name!r}: {refusal}",
                    )
                    self._record_tool_event(session, work, EventKind.TOOL_CALL, {
                        "tool_name": call.tool_name,
                        "request_id": request_id,
                        "dispatched": False,
                        "argument_keys": list(argument_keys),
                    })
                    failed = True
                    break

                try:
                    request = ToolRequest(
                        request_id=request_id,
                        session_id=session.session_id,
                        turn_id=session.state.turn_count,
                        tool_name=call.tool_name,
                        arguments=arguments,
                        requested_by="llm",
                    )
                except ValidationError:
                    work.tools.append(
                        ToolAttempt(
                            request_id=request_id,
                            tool_name=call.tool_name,
                            dispatched=False,
                            argument_keys=argument_keys,
                            refusal_reason="tool request envelope is malformed",
                        )
                    )
                    work.fail(
                        TurnErrorCategory.INVALID_TOOL_REQUEST,
                        TurnStage.TOOLS,
                        f"Tool request envelope for {call.tool_name!r} is malformed.",
                    )
                    failed = True
                    break

                self._record_tool_event(session, work, EventKind.TOOL_CALL, {
                    "tool_name": call.tool_name,
                    "request_id": request_id,
                    "dispatched": True,
                    "argument_keys": list(argument_keys),
                })

                started = time.perf_counter()
                result = self._runtime.tools.execute(request)
                work.tool_ms += _elapsed(started)

                work.tools.append(
                    ToolAttempt(
                        request_id=request_id,
                        tool_name=call.tool_name,
                        dispatched=True,
                        status=result.status,
                        argument_keys=argument_keys,
                        latency_ms=result.latency_ms,
                    )
                )
                self._record_tool_event(session, work, EventKind.TOOL_RESULT, {
                    "tool_name": call.tool_name,
                    "request_id": request_id,
                    "status": result.status.value,
                    "tool_latency_ms": round(result.latency_ms, 3),
                })

                usable = True
                if result.status is ToolStatus.OK:
                    account, usable = self._absorb(
                        call.tool_name, request.arguments, result, account, work
                    )

                messages.append(self._tool_message(request_id, result, usable=usable))

                if result.status is ToolStatus.NOT_IMPLEMENTED:
                    # Explicit, structured unavailability. The turn continues; the
                    # model is told the fact could not be read and must not state it.
                    work.fail(
                        TurnErrorCategory.TOOL_BACKEND_NOT_IMPLEMENTED,
                        TurnStage.TOOLS,
                        f"Tool {call.tool_name!r} has no backend behind it; the fact is unavailable.",
                        safety_critical=False,
                    )
                    continue
                if result.status in (ToolStatus.NOT_FOUND, ToolStatus.INVALID_REQUEST):
                    work.fail(
                        TurnErrorCategory.INVALID_TOOL_REQUEST,
                        TurnStage.TOOLS,
                        f"Registry rejected tool request {call.tool_name!r} as {result.status.value}.",
                    )
                    failed = True
                    break
                if result.status is not ToolStatus.OK:
                    work.fail(
                        TurnErrorCategory.TOOL_EXECUTION_FAILED,
                        TurnStage.TOOLS,
                        f"Tool {call.tool_name!r} failed with status {result.status.value}.",
                    )
                    failed = True
                    break

        return account, messages, failed

    # -- tool helpers -------------------------------------------------------

    def _bind(
        self, call: LlmToolCall, session: Session, account: AccountContext | None
    ) -> tuple[dict[str, Any], str | None]:
        """Bind session-scoped identity arguments, overwriting the model's.

        Returns the arguments to dispatch and, if the call cannot proceed, the
        reason. Argument *validity* remains the registry's job; what is settled
        here is only *whose* record may be read, which the registry cannot know.

        An unknown tool name is not refused here: it is dispatched so that the
        registry - the security boundary - is the thing that rejects it.
        """
        properties = self._tool_properties.get(call.tool_name)
        if properties is None:
            if call.tool_name in self._runtime.tools.names:
                # Registered after this orchestrator read the specs. Its identity
                # arguments cannot be bound, so it must not run: binding failing
                # open would hand the model back the choice of whose record to
                # read.
                return dict(call.arguments), (
                    "tool was registered after the orchestrator was built; "
                    "its identity arguments cannot be bound"
                )
            # Genuinely unknown to the registry too. Dispatch it so the registry -
            # the security boundary - is what rejects it.
            return dict(call.arguments), None

        arguments = dict(call.arguments)

        precondition = _WRITE_TOOL_PRECONDITIONS.get(call.tool_name)
        if precondition is not None:
            flag, reason = precondition
            if not getattr(session.state, flag, False):
                return arguments, f"{reason}, so this write must not be made"
            if self._already_written(session.session_id, call.tool_name):
                return arguments, (
                    "this was already recorded for what the customer last said; "
                    "a second write needs the customer to say it again"
                )

        for name in _SCOPED_ARGUMENTS:
            if name not in properties:
                continue
            if name == "account_ref":
                bound = account.account_ref if account is not None else None
            else:
                bound = session.customer.customer_ref if session.customer is not None else None
            if bound is None:
                return arguments, (
                    f"{name} is required but this session has no such context loaded"
                )
            arguments[name] = bound

        # The promised date is a fact about what the customer said, so it comes
        # from state, not from the model. Without it the promise cannot be
        # recorded accurately, and a wrong date on a case file is worse than no
        # promise at all.
        if "promise_date" in properties:
            if session.state.promise_date is None:
                return arguments, "no promise date was captured from the customer"
            arguments["promise_date"] = session.state.promise_date

        return arguments, None

    def _already_written(self, session_id: str, tool_name: str) -> bool:
        """True if this write succeeded after the customer last authorised it.

        Read from the session's own event log - the audit trail - newest first:
        a successful TOOL_RESULT for this tool met before any event carrying the
        authorising intent means the authorisation has been used. A failed or
        unavailable write does not use it.
        """
        current = self._runtime.sessions.get(session_id)
        if current is None:  # pragma: no cover - the session was deleted mid-turn
            return True
        intent = _WRITE_TOOL_INTENTS[tool_name].value
        for event in reversed(current.events):
            data = event.data or {}
            if (
                event.kind is EventKind.TOOL_RESULT
                and data.get("tool_name") == tool_name
                and data.get("status") == ToolStatus.OK.value
            ):
                return True
            if data.get("intent") == intent:
                return False
        return False

    def _tool_message(
        self, request_id: str, result: ToolResult, *, usable: bool = True
    ) -> LlmMessage:
        """Render a tool result for the model: an outcome, never a payload.

        No backend value enters the prompt. Not the amounts - they are in minor
        units, and a model handed ``outstanding_minor=1234500`` will read out one
        and a quarter million rupees for a twelve thousand rupee debt. Not the
        identifiers or the customer's name either, which is what makes the claim
        in ``_SCOPED_ARGUMENTS`` true rather than aspirational.

        The facts themselves reach the model the only way they should: absorbed
        into the account context, formatted by :mod:`app.orchestrator.prompt`,
        and rebuilt into the FACTS block of the next round's system message.

        A failure is likewise reported as a status, never as the backend's
        message, which routinely carries identifiers. So is a payload that
        arrived but could not be used: the FACTS block did not change, and the
        model is not told that it did.
        """
        if result.status is ToolStatus.OK and not usable:
            content = (
                f"{result.tool_name} returned data that could not be used. "
                "Do not state this fact; say you will check."
            )
        elif result.status is ToolStatus.OK:
            content = (
                f"{result.tool_name} ok. The FACTS block below is now up to date; "
                "state figures only from there."
            )
        else:
            content = (
                f"{result.tool_name} unavailable ({result.status.value}). "
                "Do not state this fact; say you will check."
            )
        return LlmMessage(role="tool", content=content, tool_call_id=request_id)

    def _absorb(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        account: AccountContext | None,
        work: _Working,
    ) -> tuple[AccountContext | None, bool]:
        """Fold a successful tool result into backend context and grounding.

        Nothing is constructed from nothing: a partial payload cannot build an
        :class:`AccountContext` that was never loaded, and this returns ``None``
        unchanged rather than inventing the fields the payload lacks.

        Also returns whether the payload was usable - ``False`` only when it
        does not fit the account context and was discarded.
        """
        data = result.data or {}
        usable = True

        allowed = _ACCOUNT_PROJECTION.get(tool_name)
        if allowed and account is not None:
            update = {key: value for key, value in data.items() if key in allowed}
            if update:
                try:
                    account = AccountContext.model_validate({**account.model_dump(), **update})
                except ValidationError:
                    usable = False
                    work.fail(
                        TurnErrorCategory.TOOL_EXECUTION_FAILED,
                        TurnStage.TOOLS,
                        f"Backend payload from {tool_name!r} does not fit the account context.",
                        safety_critical=False,
                    )
                else:
                    work.sources.add(GroundingSource.TOOL_RESULT)

        if tool_name == "record_payment_promise" and result.status is ToolStatus.OK:
            # Only the date is grounded, and only because the orchestrator - not
            # the model - supplied it: _bind takes it from ConversationState,
            # where it was put by an upstream NLU signal from the customer's own
            # words. The *amount* is deliberately NOT grounded. There is no
            # signal carrying a promised amount, so it is whatever the model
            # wrote; the registry checks only that it is a positive integer, and
            # the backend contract does not require it to be validated against
            # the account. Admitting it here would let the model mint a figure
            # and then state it as the outstanding balance, because GroundingFacts
            # records no role for an amount. See REPORT.md, Stage 2 limitations.
            promised = arguments.get("promise_date")
            if isinstance(promised, date):
                work.extra_dates.add(promised)
                work.sources.add(GroundingSource.TOOL_RESULT)
            elif isinstance(promised, str):
                try:
                    work.extra_dates.add(date.fromisoformat(promised))
                except ValueError:
                    pass
                else:
                    work.sources.add(GroundingSource.TOOL_RESULT)

        return account, usable

    def _grounding(self, account: AccountContext | None, work: _Working) -> GroundingFacts:
        """The only figures the agent may state this turn.

        ``allow_concession_offer`` is always false: no tool in this foundation
        authorises a waiver, settlement or discount, so none may be offered.
        """
        base = GroundingFacts.from_account(account) if account is not None else GroundingFacts()
        return GroundingFacts(
            amounts_minor=tuple(sorted(set(base.amounts_minor) | work.extra_amounts)),
            dates=tuple(sorted(set(base.dates) | work.extra_dates)),
            numbers=base.numbers,
            allow_concession_offer=False,
        )

    # -- events -------------------------------------------------------------

    def _append(
        self,
        session: Session,
        *,
        kind: EventKind,
        work: _Working,
        language: Language | None = None,
        text: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> Session:
        """Append one event and advance state through the reducer."""
        event = self._runtime.sessions.next_event(
            session, kind=kind, now=self._runtime.clock.now(),
            language=language, text=text, data=data,
        )
        updated = self._runtime.sessions.append_event(session.session_id, event)
        if updated is None:  # pragma: no cover - the session was deleted mid-turn
            raise UnknownSession(session.session_id)
        work.event_ids.append(event.event_id)
        return updated

    def _record_tool_event(
        self, session: Session, work: _Working, kind: EventKind, data: dict[str, Any]
    ) -> None:
        """Write a tool call or result to the audit trail. Never state-bearing."""
        self._append(session, kind=kind, work=work, data=data)

    def _record_policy_event(
        self, session: Session, work: _Working, decision: PolicyDecision
    ) -> None:
        self._append(
            session,
            kind=EventKind.POLICY_DECISION,
            work=work,
            data={
                "allowed": decision.allowed,
                "escalate": decision.escalate,
                "rule_ids": list(decision.rule_ids),
                "violation_rule_ids": [v.rule_id for v in decision.violations],
                "policy_version": decision.policy_version,
                "rules_version": decision.rules_version,
            },
        )

    # -- results ------------------------------------------------------------

    def _abort(
        self,
        session: Session,
        work: _Working,
        started: float,
        *,
        turn_id: int,
        decision: PolicyDecision | None = None,
        transcript_text: str = "",
        transcript_applied: tuple[str, ...] = (),
        latency_parts: tuple[float, ...] = (),
    ) -> TurnResult:
        """End a turn that failed before it could produce anything.

        ``transcript_text`` is carried through whenever normalisation had already
        succeeded: the turn produced no answer, but the audit record must still
        say what was heard, and the event log already holds it.
        """
        normalization, state_update, context, policy = (list(latency_parts) + [0.0] * 4)[:4]
        return self._finish(
            session, work, started, turn_id=turn_id, outcome=TurnOutcome.FAILED,
            transcript_text=transcript_text, transcript_applied=transcript_applied,
            language=session.state.language,
            decision=decision,
            latency=TurnLatency(
                normalization_ms=normalization, state_update_ms=state_update,
                context_load_ms=context, policy_ms=policy, llm_ms=work.llm_ms,
                tool_ms=work.tool_ms, total_ms=_elapsed(started),
            ),
        )

    def _finish(
        self,
        session: Session,
        work: _Working,
        started: float,
        *,
        turn_id: int,
        outcome: TurnOutcome,
        transcript_text: str,
        transcript_applied: tuple[str, ...],
        language: Language | None,
        decision: PolicyDecision | None,
        draft_text: str | None = None,
        validation: Any = None,
        response_text: str | None = None,
        tts_fully_normalized: bool = False,
        tts_unrendered_kinds: tuple[Any, ...] = (),
        latency: TurnLatency | None = None,
    ) -> TurnResult:
        grounding = self._grounding(work.account, work)
        result = TurnResult(
            session_id=session.session_id,
            turn_id=turn_id,
            outcome=outcome,
            speakable=outcome is TurnOutcome.COMPLETED and response_text is not None,
            normalized_transcript=transcript_text,
            transcript_applied=transcript_applied,
            language=language,
            state=session.state.model_copy(deep=True),
            event_ids=tuple(work.event_ids),
            policy=decision,
            policy_evaluations=tuple(work.evaluations),
            required_actions=decision.required_actions if decision else (),
            tools=tuple(work.tools),
            grounding_sources=tuple(sorted(work.sources, key=lambda s: s.value)),
            grounded_fact_count=len(grounding.amounts_minor) + len(grounding.dates),
            llm_calls=work.llm_calls,
            draft_text=draft_text,
            validation=validation,
            response_text=response_text if outcome is TurnOutcome.COMPLETED else None,
            tts_fully_normalized=tts_fully_normalized,
            tts_unrendered_kinds=tts_unrendered_kinds,
            errors=tuple(work.errors),
            latency=latency or TurnLatency(total_ms=_elapsed(started)),
        )
        self._log(result, session.state)
        return result

    def _log(self, result: TurnResult, state: ConversationState) -> None:
        """One record per turn. Categories and measurements only.

        Deliberately absent: the transcript, the draft, the spoken text, the
        account reference, the customer reference, every amount, and every
        backend message. What is here is what an operator needs to see a call
        going wrong without being able to read the customer's file.
        """
        decision = result.policy
        log_event(
            _logger,
            "turn",
            session_id=result.session_id,
            turn_id=result.turn_id,
            language=result.language.value if result.language else None,
            intent=state.intent.value if state.intent else None,
            dpd_stage=decision.dpd_stage.value if decision and decision.dpd_stage else None,
            outcome=result.outcome.value,
            speakable=result.speakable,
            policy_allowed=decision.allowed if decision else None,
            policy_escalate=decision.escalate if decision else None,
            policy_rule_ids=list(decision.rule_ids) if decision else [],
            policy_violation_rule_ids=(
                [v.rule_id for v in decision.violations] if decision else []
            ),
            policy_checkpoints=[e.checkpoint.value for e in result.policy_evaluations],
            tool_count=len(result.tools),
            # A name the registry does not know is model text; it is not logged.
            tool_name=(
                (
                    result.tools[-1].tool_name
                    if result.tools[-1].tool_name in self._runtime.tools.names
                    else "<unregistered>"
                )
                if result.tools
                else None
            ),
            tool_statuses=[t.status.value for t in result.tools if t.status is not None],
            tool_latency_ms=round(result.latency.tool_ms, 3),
            model_latency_ms=round(result.latency.llm_ms, 3),
            llm_calls=result.llm_calls,
            stt_latency_ms=round(result.latency.normalization_ms, 3),
            tts_latency_ms=round(result.latency.tts_normalization_ms, 3),
            validation_valid=result.validation.valid if result.validation else None,
            validation_blocked=result.validation.blocked if result.validation else None,
            validation_codes=(
                sorted({i.code.value for i in result.validation.issues})
                if result.validation
                else []
            ),
            tts_fully_normalized=result.tts_fully_normalized,
            tts_unrendered_kinds=[k.value for k in result.tts_unrendered_kinds],
            latency_ms=round(result.latency.total_ms, 3),
            barge_in=None,
            error_type=result.errors[0].category.value if result.errors else None,
            error_categories=[e.category.value for e in result.errors],
        )


def _identified(
    calls: Iterable[LlmToolCall], used: set[str]
) -> tuple[LlmToolCall, ...]:
    """Give each tool call an id no other call in this turn has, and record it in ``used``.

    The id is what a tool result answers, on the wire and in the audit trail. A
    model that sent none, or reused one - including an index-style id that a
    server restarts at ``call_0`` every round - has not identified the call, so
    one is minted, as a missing id always was. So is an id that is not a short
    plain token (:func:`~app.services.llm.is_usable_call_id`): it is copied into
    the audit trail and the logs. A usable model id is kept.
    """
    identified: list[LlmToolCall] = []
    for call in calls:
        call_id = call.call_id
        if not call_id or call_id in used or not is_usable_call_id(call_id):
            call_id = uuid.uuid4().hex
        used.add(call_id)
        identified.append(
            call if call_id == call.call_id else call.model_copy(update={"call_id": call_id})
        )
    return tuple(identified)


def _assistant_turn(text: str | None, calls: tuple[LlmToolCall, ...]) -> LlmMessage:
    """The model's tool-requesting turn, as it goes back to the model.

    Each call keeps its id and tool name, so the tool results that follow
    answer something on record. Its arguments are the model's own, minus
    :data:`_BOUND_ARGUMENTS`.
    """
    return LlmMessage(
        role="assistant",
        content=text or "",
        tool_calls=tuple(
            LlmToolCall(
                call_id=call.call_id,
                tool_name=call.tool_name,
                arguments={
                    key: value
                    for key, value in call.arguments.items()
                    if key not in _BOUND_ARGUMENTS
                },
            )
            for call in calls
        ),
    )


def _elapsed(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0
