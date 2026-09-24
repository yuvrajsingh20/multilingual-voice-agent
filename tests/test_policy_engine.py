"""Policy engine behaviour that is not specific to one rule."""

from __future__ import annotations

from datetime import datetime

import pytest
from zoneinfo import ZoneInfo

from app.core.checks import PolicyConfig
from app.core.policy import (
    PolicyConfigurationError,
    PolicyEngine,
    dpd_stage_for,
    tone_for,
)
from app.core.rules import RuleSet
from app.models.conversation import ConversationState
from app.models.customer import ComplianceContext
from app.models.enums import (
    ConversationStage,
    DpdStage,
    ProductType,
    ProhibitedConduct,
    RequiredAction,
    ToneLevel,
)
from tests.conftest import DEFAULT_NOW
from tests.fakes import sample_account

IST = ZoneInfo("Asia/Kolkata")


def test_decision_is_deterministic(engine: PolicyEngine, make_context) -> None:
    ctx = make_context(account=sample_account())
    assert engine.evaluate(ctx).model_dump() == engine.evaluate(ctx).model_dump()


def test_decision_carries_versions_and_instant(engine: PolicyEngine, make_context, rule_set: RuleSet) -> None:
    decision = engine.evaluate(make_context(account=sample_account()))
    assert decision.policy_version == "test-0.1.0"
    assert decision.rules_version == rule_set.rules_version
    assert decision.evaluated_at == DEFAULT_NOW


def test_decision_reports_rules_it_could_not_evaluate(engine: PolicyEngine, make_context) -> None:
    """A rule needing backend data it did not get must be visible, not silently dropped."""
    decision = engine.evaluate(make_context(account=sample_account()))
    assert "RBI-CB-RBC-2025-442-6-GRIEVANCE-PENDING-HOLD" in decision.not_evaluable_rule_ids


def test_decision_reports_rules_with_no_automated_check(engine: PolicyEngine, make_context) -> None:
    decision = engine.evaluate(make_context(account=sample_account()))
    assert "RBI-CB-RBC-2025-452-BANK-RESPONSIBLE-FOR-AGENTS" in decision.unenforced_rule_ids


def test_evaluated_not_evaluable_and_unenforced_are_disjoint(engine: PolicyEngine, make_context) -> None:
    decision = engine.evaluate(make_context(account=sample_account()))
    groups = [
        set(decision.rule_ids),
        set(decision.not_evaluable_rule_ids),
        set(decision.unenforced_rule_ids),
    ]
    for i, first in enumerate(groups):
        for second in groups[i + 1 :]:
            assert not (first & second)


def test_every_applicable_rule_appears_in_exactly_one_group(
    engine: PolicyEngine, make_context, rule_set: RuleSet
) -> None:
    account = sample_account()
    decision = engine.evaluate(make_context(account=account))
    applicable = {r.rule_id for r in rule_set.applicable(DEFAULT_NOW.date(), account.product_type)}
    reported = set(decision.rule_ids) | set(decision.not_evaluable_rule_ids) | set(decision.unenforced_rule_ids)
    assert reported == applicable


def test_engine_rejects_a_rule_file_naming_an_unknown_check(rule_set: RuleSet) -> None:
    broken = rule_set.model_copy(
        update={
            "rules": tuple(
                r.model_copy(update={"check_id": "no_such_check"}) if i == 0 else r
                for i, r in enumerate(rule_set.rules)
            )
        }
    )
    with pytest.raises(PolicyConfigurationError, match="no_such_check"):
        PolicyEngine(broken, PolicyConfig(policy_version="t"))


def test_no_account_means_no_product_specific_rules(engine: PolicyEngine, make_context) -> None:
    """Without account context the engine must not activate a product-scoped rule."""
    decision = engine.evaluate(make_context(account=None))
    assert decision.dpd_stage is None
    assert "RBI-CB-RBC-2025-410-2-MFI-CALLING-HOURS" not in decision.rule_ids
    assert "RBI-DL-2025-8-V-AGENT-PARTICULARS-BEFORE-CONTACT" not in decision.rule_ids


# --- DPD banding and tone (business rules, not regulatory) -----------------


@pytest.mark.parametrize(
    ("dpd", "stage"),
    [
        (0, DpdStage.CURRENT),
        (1, DpdStage.DPD_5),
        (5, DpdStage.DPD_5),
        (29, DpdStage.DPD_5),
        (30, DpdStage.DPD_30),
        (89, DpdStage.DPD_30),
        (90, DpdStage.DPD_90),
        (365, DpdStage.DPD_90),
    ],
)
def test_dpd_banding_boundaries(dpd: int, stage: DpdStage) -> None:
    assert dpd_stage_for(dpd) is stage


@pytest.mark.parametrize(
    ("stage", "tone"),
    [
        (DpdStage.CURRENT, ToneLevel.NEUTRAL),
        (DpdStage.DPD_5, ToneLevel.NEUTRAL),
        (DpdStage.DPD_30, ToneLevel.FIRM),
        (DpdStage.DPD_90, ToneLevel.FORMAL_FIRM),
        (None, ToneLevel.NEUTRAL),
    ],
)
def test_tone_escalates_with_dpd(stage, tone) -> None:
    assert tone_for(stage) is tone


def test_firmer_tone_never_relaxes_a_prohibition(engine: PolicyEngine, make_context) -> None:
    """The 90 DPD stage must carry the same prohibitions as the 5 DPD stage."""
    early = engine.evaluate(make_context(account=sample_account(dpd=5)))
    late = engine.evaluate(make_context(account=sample_account(dpd=200)))
    assert late.tone is ToneLevel.FORMAL_FIRM
    assert set(early.prohibited_conduct) <= set(late.prohibited_conduct)
    assert ProhibitedConduct.INTIMIDATION in late.prohibited_conduct
    assert ProhibitedConduct.THREATENING_CALL in late.prohibited_conduct


# --- persistent calling: no invented threshold ------------------------------


def test_persistent_calling_is_not_evaluable_without_a_bank_threshold(
    engine: PolicyEngine, make_context
) -> None:
    decision = engine.evaluate(
        make_context(
            account=sample_account(),
            compliance=ComplianceContext(contact_attempts_today=50),
        )
    )
    assert "RBI-CB-RBC-2025-445-NO-PERSISTENT-CALLING" in decision.not_evaluable_rule_ids
    assert decision.allowed is True


def test_persistent_calling_blocks_once_the_bank_sets_a_threshold(rule_set: RuleSet, make_context) -> None:
    engine = PolicyEngine(rule_set, PolicyConfig(policy_version="t", max_contacts_per_day=3))
    under = engine.evaluate(
        make_context(account=sample_account(), compliance=ComplianceContext(contact_attempts_today=2))
    )
    assert under.allowed is True

    at_threshold = engine.evaluate(
        make_context(account=sample_account(), compliance=ComplianceContext(contact_attempts_today=3))
    )
    assert at_threshold.allowed is False
    assert ProhibitedConduct.PERSISTENT_CALLING in at_threshold.prohibited_conduct


def test_persistent_calling_needs_the_contact_count_too(rule_set: RuleSet, make_context) -> None:
    engine = PolicyEngine(rule_set, PolicyConfig(policy_version="t", max_contacts_per_day=3))
    decision = engine.evaluate(make_context(account=sample_account(), compliance=ComplianceContext()))
    assert "RBI-CB-RBC-2025-445-NO-PERSISTENT-CALLING" in decision.not_evaluable_rule_ids


# --- disclosure duties ------------------------------------------------------


def test_disclosures_are_required_but_do_not_block_at_the_greeting(
    engine: PolicyEngine, make_context
) -> None:
    decision = engine.evaluate(make_context(account=sample_account()))
    assert decision.allowed is True
    assert RequiredAction.DISCLOSE_CALL_RECORDING in decision.required_actions
    assert RequiredAction.IDENTIFY_BANK_AND_AGENT in decision.required_actions


def test_undisclosed_recording_blocks_once_the_account_is_discussed(
    engine: PolicyEngine, make_context
) -> None:
    state = ConversationState(
        session_id="s",
        current_stage=ConversationStage.ACCOUNT_DISCUSSION,
        identity_verified=True,
        agent_identified=True,
    )
    decision = engine.evaluate(make_context(state=state, account=sample_account()))
    assert decision.allowed is False
    assert "RBI-CB-RBC-2025-442-4-CALL-RECORDING-INTIMATION" in {
        v.rule_id for v in decision.violations
    }


def test_unidentified_agent_blocks_once_the_account_is_discussed(
    engine: PolicyEngine, make_context
) -> None:
    state = ConversationState(
        session_id="s",
        current_stage=ConversationStage.ACCOUNT_DISCUSSION,
        identity_verified=True,
        recording_disclosed=True,
    )
    decision = engine.evaluate(make_context(state=state, account=sample_account()))
    assert decision.allowed is False
    assert ProhibitedConduct.ANONYMOUS_CALL in decision.prohibited_conduct


def test_all_disclosures_done_allows_account_discussion(engine: PolicyEngine, make_context) -> None:
    state = ConversationState(
        session_id="s",
        current_stage=ConversationStage.ACCOUNT_DISCUSSION,
        identity_verified=True,
        recording_disclosed=True,
        agent_identified=True,
    )
    decision = engine.evaluate(make_context(state=state, account=sample_account()))
    assert decision.allowed is True
    assert decision.violations == ()


# --- digital lending, conditional ------------------------------------------


def test_digital_lending_particulars_block_contact_when_not_sent(
    engine: PolicyEngine, make_context
) -> None:
    account = sample_account(product_type=ProductType.DIGITAL_LENDING)
    decision = engine.evaluate(
        make_context(
            account=account,
            compliance=ComplianceContext(digital_lending_particulars_sent=False),
        )
    )
    assert decision.allowed is False
    assert "RBI-DL-2025-8-V-AGENT-PARTICULARS-BEFORE-CONTACT" in {
        v.rule_id for v in decision.violations
    }


def test_digital_lending_rule_does_not_touch_a_retail_loan(engine: PolicyEngine, make_context) -> None:
    decision = engine.evaluate(
        make_context(
            account=sample_account(product_type=ProductType.RETAIL_LOAN),
            compliance=ComplianceContext(digital_lending_particulars_sent=False),
        )
    )
    assert decision.allowed is True
