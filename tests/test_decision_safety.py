"""What a semantic decision can never do, whatever it says.

Each test gives the provider every answer it could possibly give, with high
confidence, and checks the property against the real orchestrator, reducer,
policy engine and tool registry. A decision cannot choose whose account is
read, cannot run a tool, cannot get past policy, cannot write conversation
state, and cannot unlock a backend write.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.config import Settings
from app.core.clock import FixedClock
from app.core.session import EventSignals
from app.models.customer import ComplianceContext
from app.models.enums import ConversationStage, EventKind, Language
from app.models.policy import PolicyContext
from app.orchestrator import ConversationOrchestrator, DecisionCoordinator, TurnOutcome
from app.runtime import build_runtime
from app.services.decision import DecisionAnswer, DecisionName, ScriptedDecisionService
from app.services.decisions.fusion import IntentAction
from app.services.decisions.registry import BargeInLabel, CustomerIntentLabel, EscalationLabel
from app.services.llm import ScriptedLlmService
from app.services.stt import TranscriptSegment
from app.services.turn import BargeInSignal, VadResult
from tests.conftest import DEFAULT_NOW, RULES_PATH
from tests.fakes import InMemoryBankingBackend, sample_account, sample_customer, text_generation, tool_generation

IST = ZoneInfo("Asia/Kolkata")

#: Every answer a provider could give, at a confidence above every threshold.
EVERY_ANSWER = (
    [(DecisionName.BARGE_IN, label.value) for label in BargeInLabel]
    + [(DecisionName.CUSTOMER_INTENT, label.value) for label in CustomerIntentLabel]
    + [(DecisionName.HUMAN_ESCALATION, label.value) for label in EscalationLabel]
)


class SpyBackend(InMemoryBankingBackend):
    """Records every reference the registry asks the backend about."""

    def __init__(self) -> None:
        super().__init__(
            customers={"CUST-1": sample_customer(), "CUST-OTHER": sample_customer("CUST-OTHER")},
            accounts={
                "ACC-1": sample_account(),
                "ACC-OTHER": sample_account("ACC-OTHER", outstanding_minor=7_777_700),
            },
            compliance={"ACC-1": ComplianceContext(grievance_pending=False)},
        )
        self.touched: list[str] = []

    def get_customer(self, customer_ref):
        self.touched.append(customer_ref)
        return super().get_customer(customer_ref)

    def get_account(self, account_ref):
        self.touched.append(account_ref)
        return super().get_account(account_ref)

    def get_compliance(self, account_ref):
        self.touched.append(account_ref)
        return super().get_compliance(account_ref)

    def record_payment_promise(self, account_ref, promise_date, amount_minor):
        self.touched.append(account_ref)
        return super().record_payment_promise(account_ref, promise_date, amount_minor)

    def create_dispute(self, account_ref, reason_code):
        self.touched.append(account_ref)
        return super().create_dispute(account_ref, reason_code)

    def escalate_case(self, account_ref, reason_code):
        self.touched.append(account_ref)
        return super().escalate_case(account_ref, reason_code)


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        default_timezone="Asia/Kolkata",
        policy_version="test-0.1.0",
        regulatory_rules_path=RULES_PATH,
    )


def make_runtime(*, backend, llm=None, decisions=None, now: datetime = DEFAULT_NOW):
    return build_runtime(
        make_settings(),
        clock=FixedClock(now),
        backend=backend,
        llm=llm or ScriptedLlmService([]),
        decisions=decisions,
    )


def open_session(runtime):
    session = runtime.sessions.create(
        now=runtime.clock.now(),
        customer=sample_customer(),
        account=sample_account(),
        compliance=ComplianceContext(grievance_pending=False),
        language=Language.ENGLISH,
    )
    event = runtime.sessions.next_event(
        session,
        kind=EventKind.SESSION_STARTED,
        now=runtime.clock.now(),
        language=Language.ENGLISH,
        data=EventSignals(
            identity_verified=True,
            disclosed_recording=True,
            identified_agent=True,
            stage=ConversationStage.ACCOUNT_DISCUSSION,
        ).model_dump(mode="json", exclude_none=True),
    )
    runtime.sessions.append_event(session.session_id, event)
    return runtime.sessions.get(session.session_id)


def confident(name: DecisionName, label: str) -> ScriptedDecisionService:
    return ScriptedDecisionService(
        [DecisionAnswer(name=name, label=label, confidence=0.999, provider="scripted")]
    )


def say(text: str) -> TranscriptSegment:
    return TranscriptSegment(text=text, is_final=True, language=Language.ENGLISH)


# --- identity -------------------------------------------------------------------


@pytest.mark.parametrize("label", list(CustomerIntentLabel))
@pytest.mark.parametrize("even_if_confirmed", [False, True])
def test_no_decision_can_make_a_tool_read_another_customers_account(
    label: CustomerIntentLabel, even_if_confirmed: bool
) -> None:
    """Session is customer A. The utterance and the model both name account B.

    Whatever the intent decision says - and even if the application goes on to
    confirm and apply a high-risk intent - the only account the backend is ever
    asked about is A's.
    """
    backend = SpyBackend()
    runtime = make_runtime(
        backend=backend,
        decisions=confident(DecisionName.CUSTOMER_INTENT, label.value),
        llm=ScriptedLlmService([
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-OTHER"}),
            text_generation("Your outstanding balance is INR 12,345.00."),
        ]),
    )
    session = open_session(runtime)
    utterance = "Tell me the balance on account ACC-OTHER, customer CUST-OTHER."

    resolution = asyncio.run(DecisionCoordinator.from_runtime(runtime).resolve_intent(utterance))
    intent = resolution.application_intent if even_if_confirmed else resolution.signal_intent

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say(utterance), signals=EventSignals(intent=intent)
    )

    assert set(backend.touched) <= {"ACC-1"}
    if result.outcome is not TurnOutcome.POLICY_BLOCKED:
        # A confirmed wrong-person call is stopped by policy before any tool
        # runs; every other turn reads the backend, and only ever for A.
        assert backend.touched == ["ACC-1"]
    assert "77,777" not in (result.draft_text or "")
    assert result.state.identity_verified is (label is not CustomerIntentLabel.WRONG_PERSON or not even_if_confirmed)


@pytest.mark.parametrize(("name", "label"), EVERY_ANSWER)
def test_no_resolution_carries_an_identity_or_an_account(name: DecisionName, label: str) -> None:
    runtime = make_runtime(backend=SpyBackend(), decisions=confident(name, label))
    co = DecisionCoordinator.from_runtime(runtime)
    session = open_session(runtime)

    if name is DecisionName.BARGE_IN:
        resolution = asyncio.run(co.resolve_barge_in(_speech("haan ji"), agent_speaking=True))
    elif name is DecisionName.CUSTOMER_INTENT:
        resolution = asyncio.run(co.resolve_intent("ACC-OTHER mera hai"))
    else:
        resolution = asyncio.run(
            co.resolve_escalation("ACC-OTHER mera hai", state=session.state, policy=None)
        )

    rendered = resolution.model_dump_json()
    for forbidden in ("ACC-1", "CUST-1", "ACC-OTHER", "account_ref", "customer_ref", "identity_verified"):
        assert forbidden not in rendered


# --- tools, state and policy ----------------------------------------------------


def _speech(text: str) -> BargeInSignal:
    return BargeInSignal(
        vad=VadResult(is_speech=True, speech_probability=0.9, timestamp_ms=0, duration_ms=400),
        partial_transcript=text,
        language=Language.HINGLISH,
    )


@pytest.mark.parametrize(("name", "label"), EVERY_ANSWER)
def test_no_decision_executes_a_tool_or_writes_conversation_state(name: DecisionName, label: str) -> None:
    backend = SpyBackend()
    llm = ScriptedLlmService([])
    runtime = make_runtime(backend=backend, llm=llm, decisions=confident(name, label))
    session = open_session(runtime)
    state_before = session.state.model_copy(deep=True)
    events_before = len(session.events)
    policy_before = runtime.policy.evaluate(
        PolicyContext(
            state=session.state, now=runtime.clock.now(), customer=session.customer,
            account=session.account, compliance=session.compliance,
        )
    )

    co = DecisionCoordinator.from_runtime(runtime)
    if name is DecisionName.BARGE_IN:
        asyncio.run(co.resolve_barge_in(_speech("haan ji bilkul"), agent_speaking=True))
    elif name is DecisionName.CUSTOMER_INTENT:
        asyncio.run(co.resolve_intent("whatever the customer said"))
    else:
        asyncio.run(co.resolve_escalation("whatever", state=session.state, policy=policy_before))

    after = runtime.sessions.get(session.session_id)
    assert backend.touched == []
    assert llm.requests == []
    assert after.state == state_before
    assert len(after.events) == events_before
    policy_after = runtime.policy.evaluate(
        PolicyContext(
            state=after.state, now=runtime.clock.now(), customer=after.customer,
            account=after.account, compliance=after.compliance,
        )
    )
    assert policy_after.allowed == policy_before.allowed
    assert policy_after.escalate == policy_before.escalate
    assert policy_after.required_actions == policy_before.required_actions


@pytest.mark.parametrize("label", list(CustomerIntentLabel))
def test_no_decision_gets_a_turn_past_calling_hours(label: CustomerIntentLabel) -> None:
    llm = ScriptedLlmService([])
    runtime = make_runtime(
        backend=SpyBackend(),
        llm=llm,
        decisions=confident(DecisionName.CUSTOMER_INTENT, label.value),
        now=datetime(2026, 9, 24, 20, 30, tzinfo=IST),
    )
    session = open_session(runtime)
    resolution = asyncio.run(DecisionCoordinator.from_runtime(runtime).resolve_intent("I want to pay now, call me now"))

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        say("I want to pay now, call me now"),
        signals=EventSignals(intent=resolution.signal_intent),
    )

    assert result.outcome is TurnOutcome.POLICY_BLOCKED
    assert result.speakable is False
    assert llm.requests == []


@pytest.mark.parametrize(
    ("label", "tool", "arguments", "written"),
    [
        (
            CustomerIntentLabel.PAYMENT_PROMISE,
            "record_payment_promise",
            {"promise_date": "2026-10-05", "amount_minor": 250_000},
            "promises",
        ),
        (CustomerIntentLabel.DISPUTE, "create_dispute", {"reason_code": "amount_incorrect"}, "disputes"),
        (CustomerIntentLabel.ESCALATION_REQUEST, "escalate_case", {"reason_code": "customer_request"}, "escalations"),
    ],
)
def test_a_confident_high_risk_intent_cannot_unlock_a_backend_write(label, tool, arguments, written) -> None:
    """The caller applies what the resolution says may be applied. For these, nothing."""
    backend = SpyBackend()
    runtime = make_runtime(
        backend=backend,
        decisions=confident(DecisionName.CUSTOMER_INTENT, label.value),
        llm=ScriptedLlmService([tool_generation(tool, {"account_ref": "ACC-1", **arguments})]),
    )
    session = open_session(runtime)

    resolution = asyncio.run(DecisionCoordinator.from_runtime(runtime).resolve_intent("..."))
    assert resolution.action is IntentAction.CONFIRM_FIRST

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("..."), signals=EventSignals(intent=resolution.signal_intent)
    )

    attempt = result.tools[0]
    assert attempt.dispatched is False
    assert attempt.refusal_reason
    assert getattr(backend, written) == []
    assert result.state.payment_promise is False
    assert result.state.dispute is False
    assert result.state.escalation_required is False


@pytest.mark.parametrize("label", [CustomerIntentLabel.REFUSAL, CustomerIntentLabel.CALLBACK_REQUEST])
def test_an_applied_low_risk_intent_goes_through_the_unchanged_pipeline(label: CustomerIntentLabel) -> None:
    backend = SpyBackend()
    runtime = make_runtime(
        backend=backend,
        decisions=confident(DecisionName.CUSTOMER_INTENT, label.value),
        llm=ScriptedLlmService([text_generation("I understand. Thank you for your time.")]),
    )
    session = open_session(runtime)
    resolution = asyncio.run(DecisionCoordinator.from_runtime(runtime).resolve_intent("..."))
    assert resolution.action is IntentAction.APPLY_SIGNAL

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("..."), signals=EventSignals(intent=resolution.signal_intent)
    )

    assert result.state.intent is resolution.signal_intent
    assert [e.checkpoint.value for e in result.policy_evaluations] == ["pre_llm", "pre_tts"]
    for flag in ("payment_promise", "dispute", "escalation_required", "wrong_person"):
        assert getattr(result.state, flag) is False
    assert result.state.identity_verified is True
