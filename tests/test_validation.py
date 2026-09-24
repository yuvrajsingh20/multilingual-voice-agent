"""Response validation: the gate between the model and TTS."""

from __future__ import annotations

from datetime import date, datetime

from zoneinfo import ZoneInfo

from app.core.policy import PolicyEngine
from app.core.rules import RuleSet
from app.models.conversation import ConversationState
from app.models.enums import ConversationStage, RequiredAction
from app.services.llm import LlmToolCall
from app.services.validation import (
    DraftResponse,
    GroundingFacts,
    ResponseValidator,
    ValidationCode,
)
from app.tools.banking import build_registry
from tests.fakes import sample_account

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 24, 10, 30, tzinfo=IST)

ALL_DISCLOSED = dict(
    identity_verified=True,
    recording_disclosed=True,
    agent_identified=True,
    current_stage=ConversationStage.ACCOUNT_DISCUSSION,
)


def _validator(rule_set: RuleSet) -> ResponseValidator:
    return ResponseValidator(rule_set, build_registry().names)


def _decision(engine: PolicyEngine, make_context, **state_overrides):
    state = ConversationState(session_id="s1", **{**ALL_DISCLOSED, **state_overrides})
    return engine.evaluate(make_context(state=state, account=sample_account()))


def _grounding() -> GroundingFacts:
    return GroundingFacts.from_account(sample_account())


def _codes(result) -> set[ValidationCode]:
    return {issue.code for issue in result.issues}


def test_a_grounded_neutral_response_passes(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="Your outstanding balance is Rs. 12,345.00. When can you pay?"),
        _decision(engine, make_context),
        _grounding(),
    )
    assert result.valid is True
    assert result.blocked is False


def test_an_empty_response_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="   "), _decision(engine, make_context), _grounding()
    )
    assert ValidationCode.EMPTY_RESPONSE in _codes(result)
    assert result.blocked is True


# --- prohibited conduct -----------------------------------------------------


def test_a_threat_of_arrest_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="If you do not pay we will have you arrested tomorrow."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.PROHIBITED_CONTENT in _codes(result)
    assert result.blocked is True


def test_a_blocked_threat_cites_the_rule_it_breaks(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="Aap jail jaoge if this is not cleared."),
        _decision(engine, make_context),
        _grounding(),
    )
    issue = next(i for i in result.issues if i.code is ValidationCode.PROHIBITED_CONTENT)
    assert "RBI-CB-RBC-2025-445-NO-THREATENING-CALLS" in issue.rule_ids


def test_abusive_language_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="Shut up and pay, you are a chor."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.PROHIBITED_CONTENT in _codes(result)


def test_public_humiliation_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="We will tell your employer about this default."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.PROHIBITED_CONTENT in _codes(result)


def test_contacting_relatives_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="Then we will call your family and your friends."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.PROHIBITED_CONTENT in _codes(result)


def test_intimidation_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="This is your final warning."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.PROHIBITED_CONTENT in _codes(result)


def test_a_firm_but_permitted_reminder_passes(rule_set, engine, make_context) -> None:
    """Firmness itself is allowed; only the enumerated conduct is not."""
    result = _validator(rule_set).validate(
        DraftResponse(text="This account is seriously overdue and needs to be regularised."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert result.blocked is False


# --- grounding --------------------------------------------------------------


def test_an_invented_amount_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="You owe Rs. 99,999.00 today."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.UNGROUNDED_AMOUNT in _codes(result)
    assert result.blocked is True


def test_a_backend_supplied_amount_passes(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="The minimum due is Rs. 2,500.00."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.UNGROUNDED_AMOUNT not in _codes(result)


def test_an_invented_date_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="Your payment was due on 2026-01-01."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.UNGROUNDED_DATE in _codes(result)


def test_a_backend_supplied_date_passes(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="Your payment was due on 2026-08-20."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.UNGROUNDED_DATE not in _codes(result)


def test_grounding_from_account_collects_every_stated_figure() -> None:
    grounding = GroundingFacts.from_account(sample_account())
    assert 1_234_500 in grounding.amounts_minor
    assert 250_000 in grounding.amounts_minor
    assert date(2026, 8, 20) in grounding.dates


# --- unsupported promises ---------------------------------------------------


def test_offering_a_waiver_without_authorisation_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(text="We can waive the late charges for you."),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.UNSUPPORTED_PROMISE in _codes(result)
    assert result.blocked is True


def test_an_authorised_concession_passes(rule_set, engine, make_context) -> None:
    grounding = GroundingFacts.from_account(sample_account(), allow_concession_offer=True)
    result = _validator(rule_set).validate(
        DraftResponse(text="A settlement has been approved on this account."),
        _decision(engine, make_context),
        grounding,
    )
    assert ValidationCode.UNSUPPORTED_PROMISE not in _codes(result)


# --- tool requests ----------------------------------------------------------


def test_an_unknown_tool_request_is_blocked(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(
            text="One moment.",
            tool_calls=(LlmToolCall(call_id="c1", tool_name="transfer_funds", arguments={}),),
        ),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.INVALID_TOOL_REQUEST in _codes(result)
    assert result.blocked is True


def test_a_registered_tool_request_passes(rule_set, engine, make_context) -> None:
    result = _validator(rule_set).validate(
        DraftResponse(
            text="One moment.",
            tool_calls=(LlmToolCall(call_id="c1", tool_name="get_dpd", arguments={"account_ref": "ACC-1"}),),
        ),
        _decision(engine, make_context),
        _grounding(),
    )
    assert ValidationCode.INVALID_TOOL_REQUEST not in _codes(result)


# --- mandatory confirmations ------------------------------------------------


def test_a_missing_recording_disclosure_blocks_the_response(rule_set, engine, make_context) -> None:
    decision = _decision(engine, make_context, recording_disclosed=False, current_stage=ConversationStage.GREETING)
    result = _validator(rule_set).validate(
        DraftResponse(text="Good morning, am I speaking with the account holder?"),
        decision,
        _grounding(),
    )
    issue = next(i for i in result.issues if i.code is ValidationCode.MISSING_REQUIRED_ACTION)
    assert issue.blocking is True
    assert result.blocked is True


def test_claiming_the_disclosure_clears_the_block(rule_set, engine, make_context) -> None:
    decision = _decision(engine, make_context, recording_disclosed=False, current_stage=ConversationStage.GREETING)
    result = _validator(rule_set).validate(
        DraftResponse(
            text="Good morning. This call is recorded. This is a call from the bank.",
            claimed_actions=(
                RequiredAction.DISCLOSE_CALL_RECORDING,
                RequiredAction.IDENTIFY_BANK_AND_AGENT,
            ),
        ),
        decision,
        _grounding(),
    )
    assert result.blocked is False


def test_a_soft_required_action_is_reported_but_does_not_block(rule_set, engine, make_context) -> None:
    decision = _decision(
        engine,
        make_context,
        identity_verified=False,
        current_stage=ConversationStage.IDENTITY_VERIFICATION,
    )
    result = _validator(rule_set).validate(
        DraftResponse(
            text="Could you confirm your date of birth?",
            claimed_actions=(RequiredAction.DISCLOSE_CALL_RECORDING, RequiredAction.IDENTIFY_BANK_AND_AGENT),
        ),
        decision,
        _grounding(),
    )
    issue = next(
        i
        for i in result.issues
        if i.code is ValidationCode.MISSING_REQUIRED_ACTION
        and "verify_identity" in i.detail
    )
    assert issue.blocking is False


# --- policy block -----------------------------------------------------------


def test_speaking_at_all_is_blocked_when_policy_stopped_collection(rule_set, engine, make_context) -> None:
    decision = _decision(engine, make_context, wrong_person=True)
    assert decision.allowed is False
    result = _validator(rule_set).validate(
        DraftResponse(text="You still owe Rs. 12,345.00.", claimed_actions=()),
        decision,
        _grounding(),
    )
    assert ValidationCode.COLLECTION_NOT_ALLOWED in _codes(result)
    assert result.blocked is True


def test_winding_the_call_down_is_permitted_when_collection_is_blocked(rule_set, engine, make_context) -> None:
    decision = _decision(engine, make_context, wrong_person=True)
    result = _validator(rule_set).validate(
        DraftResponse(
            text="Sorry for the disturbance. Ending the call now.",
            claimed_actions=(RequiredAction.TERMINATE_COLLECTION_DISCUSSION, RequiredAction.END_CALL),
        ),
        decision,
        _grounding(),
    )
    assert ValidationCode.COLLECTION_NOT_ALLOWED not in _codes(result)
