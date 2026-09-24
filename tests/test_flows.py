"""End-to-end conversation flows through the deterministic layers.

Each test drives state with the reducer and asserts what the policy engine then
decides, which is how the orchestrator will use both.
"""

from __future__ import annotations

from datetime import datetime

from zoneinfo import ZoneInfo

from app.core.policy import PolicyEngine
from app.core.session import apply_event
from app.models.conversation import ConversationEvent, ConversationState
from app.models.customer import ComplianceContext
from app.models.enums import (
    ConversationStage,
    EventKind,
    Intent,
    ProhibitedConduct,
    RequiredAction,
)
from tests.fakes import sample_account

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 24, 10, 30, tzinfo=IST)


def _event(turn: int, kind: EventKind = EventKind.USER_UTTERANCE, **data) -> ConversationEvent:
    return ConversationEvent(
        event_id=f"e{turn}", session_id="s1", turn_id=turn, occurred_at=NOW, kind=kind, data=data
    )


def _verified_state(**overrides) -> ConversationState:
    base = dict(
        session_id="s1",
        identity_verified=True,
        recording_disclosed=True,
        agent_identified=True,
        current_stage=ConversationStage.ACCOUNT_DISCUSSION,
    )
    base.update(overrides)
    return ConversationState(**base)


# --- wrong person -----------------------------------------------------------


def test_wrong_person_blocks_collection_and_ends_the_call(engine: PolicyEngine, make_context) -> None:
    state = apply_event(_verified_state(), _event(0, intent=Intent.WRONG_PERSON.value))
    decision = engine.evaluate(make_context(state=state, account=sample_account()))

    assert decision.allowed is False
    assert RequiredAction.TERMINATE_COLLECTION_DISCUSSION in decision.required_actions
    assert RequiredAction.END_CALL in decision.required_actions
    assert ProhibitedConduct.UNAUTHORISED_DISCLOSURE in decision.prohibited_conduct


def test_wrong_person_violation_cites_the_confidentiality_rule(engine: PolicyEngine, make_context) -> None:
    state = apply_event(_verified_state(), _event(0, intent=Intent.WRONG_PERSON.value))
    decision = engine.evaluate(make_context(state=state, account=sample_account()))
    cited = {v.rule_id for v in decision.violations}
    assert "RBI-CB-RBC-2025-446-CUSTOMER-CONFIDENTIALITY" in cited


def test_wrong_person_after_2027_also_cites_the_discuss_only_rule(
    engine: PolicyEngine, make_context
) -> None:
    state = apply_event(_verified_state(), _event(0, intent=Intent.WRONG_PERSON.value))
    decision = engine.evaluate(
        make_context(
            state=state,
            now=datetime(2027, 3, 1, 10, 30, tzinfo=IST),
            account=sample_account(),
        )
    )
    assert "RBI-CB-RBC-AMD4-2026-454Y-1-DISCUSS-ONLY-WITH-BORROWER" in {
        v.rule_id for v in decision.violations
    }


def test_unverified_identity_blocks_account_discussion(engine: PolicyEngine, make_context) -> None:
    state = _verified_state(identity_verified=False)
    decision = engine.evaluate(make_context(state=state, account=sample_account()))
    assert decision.allowed is False
    assert RequiredAction.VERIFY_IDENTITY_BEFORE_DISCLOSURE in decision.required_actions


# --- dispute ----------------------------------------------------------------


def test_dispute_requires_recording_it_and_offering_the_grievance_route(
    engine: PolicyEngine, make_context
) -> None:
    state = apply_event(_verified_state(), _event(0, intent=Intent.DISPUTE.value))
    decision = engine.evaluate(make_context(state=state, account=sample_account()))

    assert state.dispute is True
    assert RequiredAction.RECORD_DISPUTE in decision.required_actions
    assert RequiredAction.PROVIDE_GRIEVANCE_MECHANISM_DETAILS in decision.required_actions


def test_an_in_call_dispute_does_not_by_itself_stop_collection(engine: PolicyEngine, make_context) -> None:
    """Paragraph 442(6) is about a *lodged* grievance, not about any objection."""
    state = apply_event(_verified_state(), _event(0, intent=Intent.DISPUTE.value))
    decision = engine.evaluate(
        make_context(
            state=state,
            account=sample_account(),
            compliance=ComplianceContext(grievance_pending=False),
        )
    )
    assert decision.allowed is True


def test_a_lodged_undisposed_grievance_stops_collection(engine: PolicyEngine, make_context) -> None:
    decision = engine.evaluate(
        make_context(
            state=_verified_state(),
            account=sample_account(),
            compliance=ComplianceContext(grievance_pending=True),
        )
    )
    assert decision.allowed is False
    assert "RBI-CB-RBC-2025-442-6-GRIEVANCE-PENDING-HOLD" in {v.rule_id for v in decision.violations}
    assert decision.escalate is True


def test_documented_frivolous_complaints_allow_recovery_to_continue(
    engine: PolicyEngine, make_context
) -> None:
    """The exception is stated in paragraph 442(6) and needs the bank's proof."""
    decision = engine.evaluate(
        make_context(
            state=_verified_state(),
            account=sample_account(),
            compliance=ComplianceContext(grievance_pending=True, grievance_held_frivolous=True),
        )
    )
    assert decision.allowed is True


def test_sub_judice_escalates_rather_than_blocking(engine: PolicyEngine, make_context) -> None:
    """Paragraph 442(6) says "utmost caution" for sub judice matters, not a ban."""
    decision = engine.evaluate(
        make_context(
            state=_verified_state(),
            account=sample_account(),
            compliance=ComplianceContext(grievance_pending=False, sub_judice=True),
        )
    )
    assert decision.allowed is True
    assert decision.escalate is True
    assert RequiredAction.ESCALATE_TO_HUMAN in decision.required_actions


# --- escalation -------------------------------------------------------------


def test_customer_escalation_request_sets_escalation(engine: PolicyEngine, make_context) -> None:
    state = apply_event(_verified_state(), _event(0, intent=Intent.ESCALATION_REQUEST.value))
    decision = engine.evaluate(make_context(state=state, account=sample_account()))
    assert decision.escalate is True
    assert RequiredAction.ESCALATE_TO_HUMAN in decision.required_actions


def test_escalation_does_not_by_itself_block_collection(engine: PolicyEngine, make_context) -> None:
    state = _verified_state(escalation_required=True)
    decision = engine.evaluate(make_context(state=state, account=sample_account()))
    assert decision.escalate is True
    assert decision.allowed is True


def test_missing_agency_details_escalates_without_blocking(engine: PolicyEngine, make_context) -> None:
    """The source text does not say contact is prohibited, so the check must not claim it is."""
    decision = engine.evaluate(
        make_context(
            state=_verified_state(),
            account=sample_account(),
            compliance=ComplianceContext(recovery_agency_details_shared=False),
        )
    )
    assert decision.allowed is True
    assert decision.escalate is True
    assert "RBI-CB-RBC-2025-442-3-AGENCY-DETAILS-TO-BORROWER" in {v.rule_id for v in decision.violations}


# --- a clean call -----------------------------------------------------------


def test_a_compliant_call_produces_no_violations(engine: PolicyEngine, make_context) -> None:
    decision = engine.evaluate(
        make_context(
            state=_verified_state(),
            account=sample_account(),
            compliance=ComplianceContext(
                grievance_pending=False,
                recovery_agency_details_shared=True,
            ),
        )
    )
    assert decision.allowed is True
    assert decision.escalate is False
    assert decision.violations == ()
