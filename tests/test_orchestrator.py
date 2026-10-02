"""Stage 2: the conversation orchestrator.

Every test here is deterministic: a fixed clock, a scripted model, an in-memory
backend and the repository's own rule file. No test depends on the wall clock,
the host timezone, a network call or a real model.

The amounts used are the sample account's real ones, in the exact written form
the validator parses back: ``INR 12,345.00`` is 1 234 500 paise. A draft that
writes an amount any other way is - correctly - not grounded.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from app.config import Settings
from app.core.clock import FixedClock
from app.core.session import EventSignals, replay
from app.models.customer import AccountContext, ComplianceContext
from app.models.enums import (
    ConversationStage,
    EventKind,
    Intent,
    Language,
    ProductType,
    RequiredAction,
    ToolStatus,
)
from app.orchestrator import (
    ConversationOrchestrator,
    GroundingSource,
    IncompleteTurn,
    PolicyCheckpoint,
    TurnErrorCategory,
    TurnOutcome,
    UnknownSession,
)
from app.runtime import build_runtime
from app.services.llm import (
    LlmGeneration,
    LlmToolCall,
    NotConfiguredLlmService,
    ScriptedLlmService,
)
from app.services.stt import TranscriptSegment
from tests.conftest import DEFAULT_NOW, RULES_PATH
from tests.fakes import (
    BrokenPolicyEngine,
    BrokenTranscriptNormalizer,
    BrokenTtsNormalizer,
    BrokenValidator,
    FailingLlmService,
    InMemoryBankingBackend,
    RejectingSessionStore,
    sample_account,
    sample_customer,
    text_generation,
    tool_generation,
)

IST = ZoneInfo("Asia/Kolkata")

#: The sample account's outstanding and minimum due, written the way the
#: validator parses them back. 1_234_500 paise and 250_000 paise.
OUTSTANDING = "INR 12,345.00"
MINIMUM_DUE = "INR 2,500.00"

GROUNDED_REPLY = f"Your outstanding balance is {OUTSTANDING} and the minimum due is {MINIMUM_DUE}."


# --- harness ---------------------------------------------------------------


def make_settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        default_timezone="Asia/Kolkata",
        policy_version="test-0.1.0",
        regulatory_rules_path=RULES_PATH,
    )
    base.update(overrides)
    return Settings(**base)


def make_runtime(*, llm=None, backend=None, settings=None, **replacements):
    runtime = build_runtime(
        settings or make_settings(),
        clock=FixedClock(DEFAULT_NOW),
        backend=backend if backend is not None else InMemoryBankingBackend(
            customers={"CUST-1": sample_customer()},
            accounts={"ACC-1": sample_account()},
            compliance={"ACC-1": ComplianceContext(grievance_pending=False)},
        ),
        llm=llm or ScriptedLlmService([text_generation(GROUNDED_REPLY)]),
    )
    return dataclasses.replace(runtime, **replacements) if replacements else runtime


#: Distinguishes "use the sample account" from an explicit "no account loaded".
_SAMPLE = object()


def open_session(
    runtime,
    *,
    account: AccountContext | None = _SAMPLE,
    compliance: ComplianceContext | None = None,
    language: Language = Language.ENGLISH,
    stage: ConversationStage = ConversationStage.ACCOUNT_DISCUSSION,
    disclosed: bool = True,
):
    """A session mid-call: identity verified, duties discharged, account loaded.

    Those flags are set through events, not by writing to state, so the session
    is one a replay would reproduce.
    """
    session = runtime.sessions.create(
        now=runtime.clock.now(),
        customer=sample_customer(),
        account=sample_account() if account is _SAMPLE else account,
        compliance=compliance or ComplianceContext(grievance_pending=False),
        language=language,
    )
    event = runtime.sessions.next_event(
        session,
        kind=EventKind.SESSION_STARTED,
        now=runtime.clock.now(),
        language=language,
        data=EventSignals(
            identity_verified=disclosed,
            disclosed_recording=disclosed,
            identified_agent=disclosed,
            stage=stage,
        ).model_dump(mode="json", exclude_none=True),
    )
    return runtime.sessions.append_event(session.session_id, event)


def say(text: str = "How much do I have to pay?") -> TranscriptSegment:
    return TranscriptSegment(text=text, is_final=True, language=Language.ENGLISH)


# --- 1. a normal turn ------------------------------------------------------


def test_normal_transcript_produces_a_speakable_turn() -> None:
    runtime = make_runtime()
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True
    assert result.response_text is not None
    # TTS rendered the amount into a spoken form, so the synthesiser cannot
    # misread it.
    assert "twelve thousand three hundred forty-five rupees" in result.response_text
    assert result.tts_fully_normalized is True
    assert result.tts_unrendered_kinds == ()
    assert result.errors == ()
    assert result.llm_calls == 1
    assert result.turn_id == 1
    assert result.validation is not None and result.validation.valid is True


def test_transcript_is_normalized_before_anything_else_sees_it() -> None:
    runtime = make_runtime()
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("  how   much   do I owe?  ")
    )

    assert result.normalized_transcript == "how much do I owe?"
    assert "whitespace_collapse" in result.transcript_applied
    # The model saw the normalised form, not the raw one.
    assert runtime.llm.requests[0].messages[1].content == "how much do I owe?"


def test_every_stage_is_measured_separately() -> None:
    runtime = make_runtime()
    session = open_session(runtime)

    latency = ConversationOrchestrator(runtime).process_turn(session.session_id, say()).latency

    assert latency.policy_ms > 0
    assert latency.llm_ms > 0
    assert latency.validation_ms > 0
    assert latency.tts_normalization_ms > 0
    assert latency.total_ms > 0
    # Stages that did not run stay at zero rather than being guessed at.
    assert latency.tool_ms == 0


def test_the_turn_is_replayable_from_its_event_log() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-1"}),
            text_generation(GROUNDED_REPLY),
        ])
    )
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    stored = runtime.sessions.get(session.session_id)
    # Tool and policy events are in the log, and replaying the whole log - audit
    # events included - still reproduces the live state exactly.
    kinds = [event.kind for event in stored.events]
    assert EventKind.TOOL_CALL in kinds and EventKind.TOOL_RESULT in kinds
    assert EventKind.POLICY_DECISION in kinds
    assert replay(session.session_id, stored.events) == stored.state


# --- 2. policy blocks before the model is called ---------------------------


def test_policy_blocked_turn_never_calls_the_model() -> None:
    llm = ScriptedLlmService([text_generation("should never be produced")])
    runtime = make_runtime(llm=llm)
    session = open_session(
        runtime, compliance=ComplianceContext(grievance_pending=True)
    )

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.POLICY_BLOCKED
    assert result.speakable is False
    assert result.response_text is None
    assert result.draft_text is None
    assert result.llm_calls == 0
    assert llm.requests == []  # the model was not consulted at all


def test_policy_blocked_turn_returns_the_required_actions_not_a_sentence() -> None:
    runtime = make_runtime()
    session = open_session(runtime, compliance=ComplianceContext(grievance_pending=True))

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert RequiredAction.TERMINATE_COLLECTION_DISCUSSION in result.required_actions
    assert RequiredAction.ESCALATE_TO_HUMAN in result.required_actions
    # The orchestrator stays language-neutral: it reports categories, not copy.
    assert result.response_text is None
    assert result.policy is not None and result.policy.allowed is False


def test_a_call_outside_permitted_hours_is_blocked_before_generation() -> None:
    runtime = build_runtime(
        make_settings(),
        clock=FixedClock(datetime(2026, 9, 24, 20, 30, tzinfo=IST)),
        backend=InMemoryBankingBackend(accounts={"ACC-1": sample_account()}),
        llm=ScriptedLlmService([]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.POLICY_BLOCKED
    assert result.llm_calls == 0
    assert RequiredAction.END_CALL in result.required_actions


# --- 3-5. tools ------------------------------------------------------------


def test_llm_with_no_tool_request_goes_straight_to_validation() -> None:
    runtime = make_runtime()
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.tools == ()
    assert result.validation is not None
    assert [e.checkpoint for e in result.policy_evaluations] == [
        PolicyCheckpoint.PRE_LLM,
        PolicyCheckpoint.PRE_TTS,
    ]


def test_valid_tool_request_is_executed_through_the_registry() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-1"}),
            text_generation(GROUNDED_REPLY),
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    assert len(result.tools) == 1
    attempt = result.tools[0]
    assert attempt.tool_name == "get_outstanding_amount"
    assert attempt.dispatched is True
    assert attempt.status is ToolStatus.OK
    assert attempt.argument_keys == ("account_ref",)
    assert result.llm_calls == 2
    assert GroundingSource.TOOL_RESULT in result.grounding_sources
    assert result.latency.tool_ms > 0


def test_the_model_cannot_choose_which_account_a_tool_reads() -> None:
    """The orchestrator binds the identity arguments; the model's are discarded."""
    # A second borrower, whose balance is unmistakably different.
    other = sample_account("ACC-OTHER", outstanding_minor=7_777_700)
    backend = InMemoryBankingBackend(
        customers={"CUST-1": sample_customer()},
        accounts={"ACC-1": sample_account(), "ACC-OTHER": other},
    )
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-OTHER"}),
            text_generation(GROUNDED_REPLY),
        ]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.tools[0].status is ToolStatus.OK
    assert result.outcome is TurnOutcome.COMPLETED
    # The figure the tool returned is this session's, not the account the model
    # named: the other borrower's balance never enters the facts or the reply.
    facts = runtime.llm.requests[1].messages[0].content
    assert "- outstanding: INR 12,345.00" in facts
    assert "77,777" not in facts
    assert "INR 12,345.00" in (result.draft_text or "")


def test_a_tool_request_the_registry_does_not_know_is_blocked() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([tool_generation("wire_transfer", {"amount": 1})])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert result.response_text is None
    assert result.tools[0].status is ToolStatus.NOT_FOUND
    assert TurnErrorCategory.INVALID_TOOL_REQUEST in result.error_categories


def test_a_tool_request_with_bad_arguments_is_blocked_by_the_registry() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1", "sql": "drop table"})
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.tools[0].status is ToolStatus.INVALID_REQUEST
    assert result.response_text is None


def test_a_tool_needing_an_account_is_refused_when_none_is_loaded() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([tool_generation("get_dpd", {"account_ref": "ACC-1"})])
    )
    session = runtime.sessions.create(now=runtime.clock.now(), language=Language.ENGLISH)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    attempt = result.tools[0]
    assert attempt.dispatched is False  # never reached the registry
    assert attempt.status is None
    assert "no such context loaded" in (attempt.refusal_reason or "")
    assert result.outcome is TurnOutcome.FAILED


# --- 6. a tool with nothing behind it --------------------------------------


def test_not_implemented_backend_does_not_fabricate_a_fact() -> None:
    runtime = make_runtime(
        backend=InMemoryBankingBackend(),  # knows no accounts at all
        llm=ScriptedLlmService([
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-1"}),
            text_generation("Let me check that and call you back."),
        ]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.tools[0].status is ToolStatus.NOT_IMPLEMENTED
    assert TurnErrorCategory.TOOL_BACKEND_NOT_IMPLEMENTED in result.error_categories
    # Unavailability is explicit, not fatal: the agent may still say it will check.
    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True
    # No account fact was invented from the failure.
    assert result.grounded_fact_count == len(
        set(
            [sample_account().outstanding_minor, sample_account().minimum_due_minor,
             sample_account().last_payment_minor]
        )
    ) + 2


def test_a_failed_tool_is_reported_to_the_model_as_a_status_not_a_backend_message() -> None:
    runtime = make_runtime(
        backend=InMemoryBankingBackend(),
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1"}),
            text_generation("Let me check that and call you back."),
        ]),
    )
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    follow_up = runtime.llm.requests[1]
    tool_messages = [m for m in follow_up.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert "not_implemented" in tool_messages[0].content
    # The backend's own words - which carry the account reference - never reach
    # the prompt.
    assert "unknown account_ref" not in tool_messages[0].content
    assert "ACC-1" not in tool_messages[0].content


# --- 7. a tool result can change the decision ------------------------------


def test_policy_is_re_evaluated_after_a_tool_changes_the_account_context() -> None:
    """The session's cached product type is stale; the system of record differs.

    Before the tool the account looks like a retail loan and collection is
    allowed. The tool reads the system of record, which says digital lending -
    and the pre-contact particulars for digital lending were never sent, so
    collection is prohibited. A decision taken before the tool would have been
    wrong.
    """
    backend = InMemoryBankingBackend(
        customers={"CUST-1": sample_customer()},
        accounts={"ACC-1": sample_account(product_type=ProductType.DIGITAL_LENDING)},
    )
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation("get_account_status", {"account_ref": "ACC-1"}),
            text_generation(GROUNDED_REPLY),
        ]),
    )
    session = open_session(
        runtime,
        account=sample_account(product_type=ProductType.RETAIL_LOAN),
        compliance=ComplianceContext(
            grievance_pending=False, digital_lending_particulars_sent=False
        ),
    )

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    checkpoints = [e.checkpoint for e in result.policy_evaluations]
    assert checkpoints == [PolicyCheckpoint.PRE_LLM, PolicyCheckpoint.POST_TOOL]
    assert result.policy_evaluations[0].decision.allowed is True
    assert result.policy_evaluations[1].decision.allowed is False
    assert result.outcome is TurnOutcome.POLICY_BLOCKED
    assert result.speakable is False
    blocking = {
        v.rule_id
        for v in result.policy_evaluations[1].decision.violations
        if v.blocks_collection
    }
    assert "RBI-DL-2025-8-V-AGENT-PARTICULARS-BEFORE-CONTACT" in blocking


def test_a_tool_that_changes_dpd_changes_the_tone_the_model_is_given() -> None:
    backend = InMemoryBankingBackend(accounts={"ACC-1": sample_account(dpd=120)})
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1"}),
            text_generation(GROUNDED_REPLY),
        ]),
    )
    session = open_session(runtime, account=sample_account(dpd=2))

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.policy_evaluations[0].decision.dpd_stage.value == "dpd_5"
    assert result.policy_evaluations[1].decision.dpd_stage.value == "dpd_90"
    assert result.policy_evaluations[1].decision.tone.value == "formal_firm"
    # And the model was re-prompted with the escalated tone.
    assert "tone: formal_firm" in runtime.llm.requests[1].messages[0].content


# --- 8. the final response is gated ----------------------------------------


def test_a_response_that_violates_policy_never_reaches_tts() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            text_generation("If you do not pay we will have you arrested tomorrow.")
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert result.speakable is False
    assert result.response_text is None
    assert result.latency.tts_normalization_ms == 0  # TTS was never run
    assert result.validation is not None and result.validation.blocked is True
    codes = {issue.code.value for issue in result.validation.issues}
    assert "prohibited_content" in codes
    assert TurnErrorCategory.RESPONSE_VALIDATION_FAILED in result.error_categories


def test_an_unauthorised_concession_is_blocked() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation("We can waive the late charges for you.")])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert "unsupported_promise" in {i.code.value for i in result.validation.issues}


def test_a_required_in_turn_action_the_caller_does_not_claim_blocks_the_response() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation("Good morning, am I speaking with the borrower?")])
    )
    session = open_session(runtime, stage=ConversationStage.GREETING, disclosed=False)

    blocked = ConversationOrchestrator(runtime).process_turn(session.session_id, say("Hello?"))
    assert blocked.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert "missing_required_action" in {i.code.value for i in blocked.validation.issues}


def test_claiming_the_required_actions_lets_the_greeting_through() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            text_generation("Good morning. This call is recorded. I am calling from the bank.")
        ])
    )
    session = open_session(runtime, stage=ConversationStage.GREETING, disclosed=False)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        say("Hello?"),
        claimed_actions=(
            RequiredAction.DISCLOSE_CALL_RECORDING,
            RequiredAction.IDENTIFY_BANK_AND_AGENT,
        ),
    )

    assert result.outcome is TurnOutcome.COMPLETED
    # The claim is recorded as state, so the duty is not demanded again next turn.
    assert result.state.recording_disclosed is True
    assert result.state.agent_identified is True


# --- 9. grounding ----------------------------------------------------------


def test_an_amount_written_without_a_currency_prefix_is_still_checked() -> None:
    """Writing a figure as a bare number must not get it past grounding."""
    runtime = make_runtime(
        llm=ScriptedLlmService([
            text_generation("Your outstanding balance is 99,999 rupees. Please pay 50,000 rupees.")
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert result.speakable is False
    assert "ungrounded_number" in {i.code.value for i in result.validation.issues}


def test_a_fabricated_days_past_due_is_blocked() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation("You are 180 days past due.")])
    )
    session = open_session(runtime)  # the sample account is 35 days past due

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert "ungrounded_number" in {i.code.value for i in result.validation.issues}


def test_the_real_days_past_due_may_be_stated() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation("Your account is 35 days past due.")])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True


def test_an_amount_in_paise_cannot_be_read_out_as_rupees() -> None:
    """1_234_500 paise is INR 12,345.00, not 'twelve lakh thirty-four thousand'."""
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation("Your outstanding is 1234500.")])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert "ungrounded_number" in {i.code.value for i in result.validation.issues}


def test_no_backend_value_is_ever_placed_in_the_prompt() -> None:
    """Tool results reach the model as facts in FACTS, never as raw payloads."""
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_customer_context", {"customer_ref": "CUST-1"}),
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-1"}, call_id="c2"),
            text_generation(GROUNDED_REPLY),
        ]),
        settings=make_settings(max_tool_calls_per_turn=2),
    )
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    every_message = "\n".join(
        message.content for request in runtime.llm.requests for message in request.messages
    )
    # Identifiers and the customer's name never reach the model endpoint.
    assert "ACC-1" not in every_message
    assert "CUST-1" not in every_message
    assert "Test Borrower" not in every_message
    # Nor do raw minor units, which a model would read out a hundredfold wrong.
    assert "1234500" not in every_message
    assert "250000" not in every_message
    # The fact itself is present, correctly formatted.
    assert "INR 12,345.00" in every_message


def test_an_amount_no_backend_fact_supports_is_blocked() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation("Your outstanding balance is INR 99,999.00.")])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert "ungrounded_amount" in {i.code.value for i in result.validation.issues}


def test_with_no_account_context_no_amount_or_date_may_be_stated() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation(f"You owe {OUTSTANDING} since 2026-08-20.")])
    )
    session = open_session(runtime, account=None)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert TurnErrorCategory.MISSING_BACKEND_CONTEXT in result.error_categories
    assert result.grounding_sources == ()
    assert result.grounded_fact_count == 0
    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    codes = {i.code.value for i in result.validation.issues}
    assert "ungrounded_amount" in codes and "ungrounded_date" in codes


def test_missing_backend_context_alone_does_not_stop_a_turn_that_states_no_facts() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation("I can help with that. When would you like to pay?")])
    )
    session = open_session(runtime, account=None)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    assert TurnErrorCategory.MISSING_BACKEND_CONTEXT in result.error_categories
    assert result.errors[0].safety_critical is False


def test_the_model_cannot_manufacture_the_promise_it_reports() -> None:
    """A write the conversation does not support is refused before the registry."""
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation(
                "record_payment_promise",
                {"account_ref": "ACC-1", "promise_date": "2026-10-05", "amount_minor": 250000},
            )
        ])
    )
    session = open_session(runtime)

    # The customer said nothing about paying; no PAYMENT_PROMISE signal exists.
    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("I am not discussing this today.")
    )

    attempt = result.tools[0]
    assert attempt.dispatched is False
    assert attempt.status is None
    assert "has not made a payment promise" in (attempt.refusal_reason or "")
    assert result.outcome is TurnOutcome.FAILED
    assert runtime.tools  # nothing was written
    assert TurnErrorCategory.INVALID_TOOL_REQUEST in result.error_categories


def test_the_promise_date_is_bound_from_state_not_taken_from_the_model() -> None:
    backend = InMemoryBankingBackend(
        customers={"CUST-1": sample_customer()}, accounts={"ACC-1": sample_account()}
    )
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation(
                "record_payment_promise",
                # The model lies about the date and invents an amount.
                {"account_ref": "ACC-1", "promise_date": "2099-01-01", "amount_minor": 9_999_999},
            ),
            text_generation("Noted for 2026-10-05. Thank you."),
        ]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        say("I can pay on the fifth of October."),
        signals=EventSignals(intent=Intent.PAYMENT_PROMISE, promise_date=date(2026, 10, 5)),
    )

    assert result.tools[0].status is ToolStatus.OK
    # The date written to the case file is the customer's, not the model's.
    assert backend.promises == [("ACC-1", date(2026, 10, 5), 9_999_999)]
    # And that date - and only that date - became a fact the agent may state.
    assert result.outcome is TurnOutcome.COMPLETED
    assert result.validation is not None and result.validation.valid is True


def test_the_amount_the_model_invented_for_a_promise_never_becomes_a_fact() -> None:
    """A write tool must not be a laundry for numbers the model made up."""
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation(
                "record_payment_promise",
                {"account_ref": "ACC-1", "promise_date": "2026-10-05", "amount_minor": 9_876_543},
            ),
            # The model now states its own invented figure as the balance.
            text_generation("Your total payable is INR 98,765.43."),
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        say("I can pay on the fifth."),
        signals=EventSignals(intent=Intent.PAYMENT_PROMISE, promise_date=date(2026, 10, 5)),
    )

    assert result.tools[0].status is ToolStatus.OK
    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert result.speakable is False
    assert "ungrounded_amount" in {i.code.value for i in result.validation.issues}


def test_a_write_must_be_the_only_tool_call_in_its_round() -> None:
    """A write commits before the round's policy re-check can see the reads."""
    runtime = make_runtime(
        settings=make_settings(max_tool_calls_per_turn=4),
        llm=ScriptedLlmService([
            LlmGeneration(
                tool_calls=(
                    LlmToolCall(call_id="a", tool_name="get_account_status", arguments={}),
                    LlmToolCall(
                        call_id="b",
                        tool_name="record_payment_promise",
                        arguments={"amount_minor": 250000},
                    ),
                )
            )
        ]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        say("I'll pay on the fifth."),
        signals=EventSignals(intent=Intent.PAYMENT_PROMISE, promise_date=date(2026, 10, 5)),
    )

    refused = [t for t in result.tools if not t.dispatched]
    assert refused and "only tool call in its round" in (refused[0].refusal_reason or "")
    assert result.outcome is TurnOutcome.FAILED


def test_a_tool_registered_after_construction_is_refused_not_dispatched() -> None:
    """Identity binding must fail closed, never fall back to the model's argument."""
    runtime = make_runtime(
        llm=ScriptedLlmService([tool_generation("get_dpd", {"account_ref": "ACC-OTHER"})])
    )
    orchestrator = ConversationOrchestrator(runtime)
    # Simulate a registry that grew after the orchestrator read its specs.
    orchestrator._tool_properties.pop("get_dpd")
    session = open_session(runtime)

    result = orchestrator.process_turn(session.session_id, say())

    attempt = result.tools[0]
    assert attempt.dispatched is False
    assert "registered after the orchestrator was built" in (attempt.refusal_reason or "")


def test_a_tool_refresh_of_the_account_survives_the_turn() -> None:
    """A tool read the system of record; the session must not revert to stale data."""
    backend = InMemoryBankingBackend(
        customers={"CUST-1": sample_customer()},
        accounts={"ACC-1": sample_account(dpd=120)},
    )
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1"}),
            text_generation("Understood."),
        ]),
    )
    session = open_session(runtime, account=sample_account(dpd=2))

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    refreshed = runtime.sessions.get(session.session_id)
    assert refreshed.account is not None and refreshed.account.dpd == 120
    # And the correction is in the audit trail.
    assert any(e.kind is EventKind.ACCOUNT_CONTEXT_LOADED for e in refreshed.events)


def test_a_partial_transcript_never_drives_a_turn() -> None:
    runtime = make_runtime()
    session = open_session(runtime)

    with pytest.raises(IncompleteTurn):
        ConversationOrchestrator(runtime).process_turn(
            session.session_id,
            TranscriptSegment(text="I can pay on", is_final=False, language=Language.ENGLISH),
        )
    # Nothing was appended and no turn was counted.
    assert runtime.sessions.get(session.session_id).state.turn_count == 0


def test_argument_keys_the_model_invents_are_not_copied_into_the_audit_record() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1", "note_from_customer": "my PAN"})
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.tools[0].argument_keys == ("account_ref",)
    assert "note_from_customer" not in json.dumps(result.model_dump(mode="json"))


def test_the_facts_given_to_the_model_are_named() -> None:
    """An unlabelled figure invites the last payment to be spoken as the balance."""
    runtime = make_runtime()
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    facts = runtime.llm.requests[0].messages[0].content
    assert "- outstanding: INR 12,345.00" in facts
    assert "- minimum_due: INR 2,500.00" in facts
    assert "- last_payment_amount: INR 5,000.00" in facts
    assert "- days_past_due: 35" in facts


def test_a_policy_failure_after_a_tool_keeps_the_decision_already_taken() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1"}),
            text_generation("Understood."),
        ])
    )
    orchestrator = ConversationOrchestrator(runtime)
    session = open_session(runtime)
    real_policy = runtime.policy
    calls = {"n": 0}

    class _FailsAfterFirst:
        def evaluate(self, ctx):
            calls["n"] += 1
            if calls["n"] > 1:
                raise ValueError("engine fell over")
            return real_policy.evaluate(ctx)

    orchestrator._runtime = dataclasses.replace(runtime, policy=_FailsAfterFirst())
    result = orchestrator.process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.POLICY_EVALUATION_FAILED in result.error_categories
    # The PRE_LLM decision was valid and is still reported, so the turn is auditable.
    assert result.policy is not None and result.policy.allowed is True


def test_the_pre_tts_checkpoint_can_stop_a_turn_on_its_own() -> None:
    """The third checkpoint must be enforced, not merely evaluated."""
    runtime = make_runtime(llm=ScriptedLlmService([text_generation(GROUNDED_REPLY)]))
    orchestrator = ConversationOrchestrator(runtime)
    session = open_session(runtime)
    real_policy = runtime.policy
    seen = {"n": 0}

    class _BlocksAtTheLastCheckpoint:
        def evaluate(self, ctx):
            seen["n"] += 1
            decision = real_policy.evaluate(ctx)
            if seen["n"] == 1:
                return decision
            # Something changed between generation and speech.
            return decision.model_copy(update={"allowed": False})

    orchestrator._runtime = dataclasses.replace(runtime, policy=_BlocksAtTheLastCheckpoint())
    result = orchestrator.process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.POLICY_BLOCKED
    assert result.speakable is False
    assert result.response_text is None
    assert result.latency.validation_ms == 0  # validation never ran either
    assert [e.checkpoint for e in result.policy_evaluations] == [
        PolicyCheckpoint.PRE_LLM,
        PolicyCheckpoint.PRE_TTS,
    ]


def test_a_backend_error_fails_the_turn() -> None:
    class _Exploding(InMemoryBankingBackend):
        def get_account(self, account_ref: str):
            raise RuntimeError("postgres://user:hunter2@db is unreachable")

    runtime = make_runtime(
        backend=_Exploding(accounts={"ACC-1": sample_account()}),
        llm=ScriptedLlmService([tool_generation("get_dpd", {"account_ref": "ACC-1"})]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.tools[0].status is ToolStatus.BACKEND_ERROR
    assert TurnErrorCategory.TOOL_EXECUTION_FAILED in result.error_categories
    assert result.outcome is TurnOutcome.FAILED
    # The connection string never escapes into the audit record.
    assert "hunter2" not in json.dumps(result.model_dump(mode="json"))


# --- 10. multi-turn --------------------------------------------------------


def test_a_session_carries_state_across_turns() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            text_generation("Thank you for confirming."),
            text_generation("Noted, Friday works."),
            text_generation(f"The amount due is {OUTSTANDING}."),
        ])
    )
    session = open_session(runtime)
    orchestrator = ConversationOrchestrator(runtime)

    first = orchestrator.process_turn(
        session.session_id, say("Yes, speaking."),
        signals=EventSignals(intent=Intent.IDENTITY_CONFIRMED),
    )
    second = orchestrator.process_turn(
        session.session_id, say("I can pay on Friday."),
        signals=EventSignals(intent=Intent.PAYMENT_PROMISE, promise_date=date(2026, 9, 25)),
    )
    third = orchestrator.process_turn(session.session_id, say("How much do I have to pay?"))

    assert [first.turn_id, second.turn_id, third.turn_id] == [1, 2, 3]
    assert third.state.turn_count == 3
    # Turn 3 answers from state plus backend facts, not from a replayed transcript.
    assert third.state.payment_promise is True
    assert third.state.promise_date == date(2026, 9, 25)
    assert third.state.intent is Intent.PAYMENT_PROMISE
    assert third.outcome is TurnOutcome.COMPLETED


def test_the_transcript_is_not_accumulated_in_conversation_state() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation("Understood."), text_generation("Understood.")])
    )
    session = open_session(runtime)
    orchestrator = ConversationOrchestrator(runtime)

    orchestrator.process_turn(session.session_id, say("First thing I said."))
    result = orchestrator.process_turn(session.session_id, say("Second thing I said."))

    dumped = json.dumps(result.state.model_dump(mode="json"))
    assert "First thing I said" not in dumped


# --- 11. tool-loop safety --------------------------------------------------


def test_the_tool_loop_stops_at_the_configured_limit() -> None:
    runtime = make_runtime(
        settings=make_settings(max_tool_calls_per_turn=1),
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1"}, call_id="a"),
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-1"}, call_id="b"),
            text_generation("never reached"),
        ]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert len(result.tools) == 1  # the second round was never dispatched
    assert result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.TOOL_LIMIT_EXCEEDED in result.error_categories
    assert result.response_text is None  # no answer was fabricated
    assert result.speakable is False


def test_a_zero_tool_budget_permits_no_tool_at_all() -> None:
    runtime = make_runtime(
        settings=make_settings(max_tool_calls_per_turn=0),
        llm=ScriptedLlmService([tool_generation("get_dpd", {"account_ref": "ACC-1"})]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.tools == ()
    assert TurnErrorCategory.TOOL_LIMIT_EXCEEDED in result.error_categories


def test_the_default_tool_budget_is_conservative() -> None:
    assert make_settings().max_tool_calls_per_turn == 2


# --- 12-13. boundary failures ----------------------------------------------


def test_an_unconfigured_model_fails_the_turn_loudly() -> None:
    runtime = make_runtime(llm=NotConfiguredLlmService())
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.error_categories == (TurnErrorCategory.LLM_NOT_CONFIGURED,)
    assert result.llm_calls == 0
    assert result.speakable is False


def test_a_failed_turn_still_records_what_was_heard() -> None:
    """A turn that produces no answer is still an audit record of the turn."""
    runtime = make_runtime(llm=NotConfiguredLlmService())
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("  I  can  pay  on  Friday  ")
    )

    assert result.outcome is TurnOutcome.FAILED
    assert result.normalized_transcript == "I can pay on Friday"
    assert result.transcript_applied == ("whitespace_collapse",)
    # And it matches what the event log holds.
    stored = runtime.sessions.get(session.session_id)
    heard = [e.text for e in stored.events if e.kind is EventKind.USER_UTTERANCE]
    assert heard == [result.normalized_transcript]


def test_a_model_that_errors_is_distinguished_from_one_that_is_absent() -> None:
    runtime = make_runtime(llm=FailingLlmService())
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.error_categories == (TurnErrorCategory.LLM_FAILED,)
    # The provider's message is not carried through.
    assert "timed out" not in result.errors[0].detail
    assert "TimeoutError" in result.errors[0].detail


def test_tts_normalization_failure_stops_the_turn() -> None:
    runtime = make_runtime(tts_normalizer=BrokenTtsNormalizer())
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.TTS_NORMALIZATION_FAILED in result.error_categories
    assert result.speakable is False
    assert result.response_text is None


def test_an_amount_that_cannot_be_rendered_for_the_language_is_not_spoken() -> None:
    """Pure Hindi rendering is not implemented, so an amount would be misread."""
    runtime = make_runtime(llm=ScriptedLlmService([text_generation(GROUNDED_REPLY)]))
    session = open_session(runtime, language=Language.HINDI)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        TranscriptSegment(text="कितना देना है?", is_final=True, language=Language.HINDI),
    )

    assert result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.TTS_NORMALIZATION_FAILED in result.error_categories
    assert result.tts_unrendered_kinds != ()
    assert result.speakable is False


@pytest.mark.parametrize("language", [Language.HINGLISH, Language.MARATHI_ENGLISH])
def test_a_code_mixed_turn_speaks_its_amount_in_english_words(language: Language) -> None:
    runtime = make_runtime(llm=ScriptedLlmService([text_generation(GROUNDED_REPLY)]))
    session = open_session(runtime, language=language)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        TranscriptSegment(text="Kitna dena hai?", is_final=True, language=language),
    )

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True
    assert "rupees" in result.response_text
    assert OUTSTANDING not in result.response_text


def test_quoting_the_balance_after_a_dispute_is_never_spoken() -> None:
    runtime = make_runtime(llm=ScriptedLlmService([text_generation(GROUNDED_REPLY)]))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("I already paid."), signals=EventSignals(intent=Intent.DISPUTE)
    )

    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert result.speakable is False
    assert "recovery_after_dispute" in {issue.code.value for issue in result.validation.issues}


def test_a_dispute_is_recorded_and_acknowledged_without_pressure() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService(
            [
                tool_generation("create_dispute", {"account_ref": "ACC-1", "reason_code": "already_paid"}),
                text_generation("Understood, I have noted your dispute for review."),
            ]
        )
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("I already paid."), signals=EventSignals(intent=Intent.DISPUTE)
    )

    assert result.outcome is TurnOutcome.COMPLETED
    assert [(t.tool_name, t.status) for t in result.tools] == [("create_dispute", ToolStatus.OK)]
    system = runtime.llm.requests[0].messages[0].content
    assert "THIS TURN" in system
    assert "Call create_dispute now" in system


def test_the_next_turn_sees_the_earlier_exchange_in_its_written_form() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService(
            [text_generation(GROUNDED_REPLY), text_generation("Theek hai, main note kar leta hoon.")]
        )
    )
    session = open_session(runtime, language=Language.HINGLISH)
    orchestrator = ConversationOrchestrator(runtime)

    orchestrator.process_turn(
        session.session_id,
        TranscriptSegment(text="Kitna dena hai?", is_final=True, language=Language.HINGLISH),
    )
    orchestrator.process_turn(
        session.session_id,
        TranscriptSegment(text="Main kal dunga.", is_final=True, language=Language.HINGLISH),
    )

    second = runtime.llm.requests[1].messages
    assert [(m.role, m.content) for m in second[1:]] == [
        ("user", "Kitna dena hai?"),
        ("assistant", GROUNDED_REPLY),
        ("user", "Main kal dunga."),
    ]
    assert "Latin (Roman) script" in second[0].content


def test_a_hinglish_turn_with_no_numeric_spans_is_still_speakable() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([text_generation("Theek hai, main note kar leta hoon.")])
    )
    session = open_session(runtime, language=Language.HINGLISH)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        TranscriptSegment(text="Main kal karunga.", is_final=True, language=Language.HINGLISH),
    )

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True


def test_transcript_normalization_failure_stops_the_turn_before_state_changes() -> None:
    runtime = make_runtime(transcript_normalizer=BrokenTranscriptNormalizer())
    session = open_session(runtime)
    events_before = len(runtime.sessions.get(session.session_id).events)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.error_categories == (TurnErrorCategory.TRANSCRIPT_NORMALIZATION_FAILED,)
    assert result.errors[0].safety_critical is True
    assert len(runtime.sessions.get(session.session_id).events) == events_before
    assert result.llm_calls == 0


def test_an_event_the_reducer_rejects_stops_the_turn() -> None:
    settings = make_settings()
    runtime = make_runtime(
        settings=settings,
        sessions=RejectingSessionStore(settings.max_active_sessions, EventKind.USER_UTTERANCE),
    )
    session = runtime.sessions.create(now=runtime.clock.now(), language=Language.ENGLISH)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.error_categories == (TurnErrorCategory.INVALID_CONVERSATION_EVENT,)
    assert result.llm_calls == 0


def test_a_policy_configuration_error_stops_the_turn() -> None:
    from app.core.policy import PolicyConfigurationError

    runtime = make_runtime(
        policy=BrokenPolicyEngine(
            PolicyConfigurationError("rule file references unimplemented checks: ['made_up']")
        )
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.error_categories == (TurnErrorCategory.POLICY_CONFIGURATION_ERROR,)
    assert result.policy is None
    assert result.llm_calls == 0


def test_an_undecidable_policy_stops_the_turn() -> None:
    runtime = make_runtime(policy=BrokenPolicyEngine(ValueError("naive instant")))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.error_categories == (TurnErrorCategory.POLICY_EVALUATION_FAILED,)
    assert result.speakable is False


def test_a_validator_that_cannot_decide_blocks_speech() -> None:
    runtime = make_runtime(validator=BrokenValidator())
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.RESPONSE_VALIDATION_FAILED in result.error_categories
    assert result.speakable is False


def test_an_unknown_session_is_a_caller_error() -> None:
    runtime = make_runtime()
    with pytest.raises(UnknownSession):
        ConversationOrchestrator(runtime).process_turn("no-such-session", say())


# --- 14-15. nothing sensitive escapes --------------------------------------

SENSITIVE_VALUES = ("ACC-1", "CUST-1", "Test Borrower", "1234500", "12,345.00")


def test_no_sensitive_value_reaches_the_log(log_stream) -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-1"}),
            text_generation(GROUNDED_REPLY),
        ])
    )
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("Mera balance kitna hai, my PAN is ABCDE1234F")
    )

    captured = log_stream.getvalue()
    assert captured  # the turn really did log something
    for value in SENSITIVE_VALUES:
        assert value not in captured, f"{value!r} leaked into the log"
    # Nor the transcript, nor the draft, nor what was spoken.
    assert "ABCDE1234F" not in captured
    assert "outstanding balance is" not in captured
    assert "twelve thousand" not in captured


def test_the_turn_log_carries_the_standard_observability_fields(log_stream) -> None:
    runtime = make_runtime()
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    records = [json.loads(line) for line in log_stream.getvalue().splitlines() if line.strip()]
    turn_records = [r for r in records if r.get("event") == "turn"]
    assert len(turn_records) == 1
    record = turn_records[0]

    from app.observability import missing_standard_fields

    assert missing_standard_fields(record) == ()
    assert record["outcome"] == "completed"
    assert record["policy_allowed"] is True
    assert record["dpd_stage"] == "dpd_30"
    assert record["validation_blocked"] is False
    assert record["error_type"] is None
    assert record["latency_ms"] > 0


def test_the_turn_result_does_not_carry_account_or_customer_references() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-1"}),
            text_generation(GROUNDED_REPLY),
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())
    dumped = json.dumps(result.model_dump(mode="json"))

    assert "ACC-1" not in dumped
    assert "CUST-1" not in dumped
    assert "Test Borrower" not in dumped
    # Argument shape is recorded; argument values are not.
    assert result.tools[0].argument_keys == ("account_ref",)
    # Grounding is reported as provenance and a count, never as values.
    assert result.grounded_fact_count > 0
    assert all(isinstance(s.value, str) for s in result.grounding_sources)


def test_the_turn_result_keeps_no_copy_of_the_account_record() -> None:
    runtime = make_runtime()
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())
    fields = set(result.model_dump().keys())

    assert "account" not in fields
    assert "customer" not in fields
    assert "compliance" not in fields
