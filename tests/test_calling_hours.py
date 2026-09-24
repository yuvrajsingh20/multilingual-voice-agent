"""Calling-hour rules, including the exact boundaries.

Timezone is stated explicitly in every test. The engine converts whatever instant
it is given into the configured zone, so none of these depend on the machine's
own timezone.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from zoneinfo import ZoneInfo

from app.core.checks import PolicyConfig
from app.core.policy import PolicyEngine
from app.models.enums import ProductType, ProhibitedConduct, RequiredAction
from tests.fakes import sample_account

IST = ZoneInfo("Asia/Kolkata")

GENERAL_RULE = "RBI-CB-RBC-2025-445-CALLING-HOURS"
MICROFINANCE_RULE = "RBI-CB-RBC-2025-410-2-MFI-CALLING-HOURS"
AMENDED_RULE = "RBI-CB-RBC-AMD4-2026-454Y-4-CONTACT-HOURS"


def _ist(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 9, 24, hour, minute, second, tzinfo=IST)


def _blocking_rule_ids(decision) -> set[str]:
    return {v.rule_id for v in decision.violations if v.blocks_collection}


# --- general window: 08:00 to 19:00 inclusive ------------------------------


@pytest.mark.parametrize(
    ("instant", "allowed"),
    [
        (_ist(7, 59, 59), False),  # one second before the window opens
        (_ist(8, 0, 0), True),     # boundary: 08:00:00 is permitted
        (_ist(8, 0, 1), True),
        (_ist(12, 0, 0), True),
        (_ist(19, 0, 0), True),    # boundary: 19:00:00 is permitted
        (_ist(19, 0, 1), False),   # one second after the window closes
        (_ist(23, 30, 0), False),
        (_ist(3, 0, 0), False),
    ],
)
def test_general_calling_window_boundaries(engine: PolicyEngine, make_context, instant, allowed) -> None:
    decision = engine.evaluate(make_context(now=instant, account=sample_account()))
    assert decision.allowed is allowed
    if not allowed:
        assert GENERAL_RULE in _blocking_rule_ids(decision)


def test_out_of_hours_call_must_end(engine: PolicyEngine, make_context) -> None:
    decision = engine.evaluate(make_context(now=_ist(20, 0), account=sample_account()))
    assert RequiredAction.END_CALL in decision.required_actions
    assert ProhibitedConduct.CONTACT_OUTSIDE_PERMITTED_HOURS in decision.prohibited_conduct


def test_violation_cites_its_paragraph(engine: PolicyEngine, make_context) -> None:
    decision = engine.evaluate(make_context(now=_ist(7, 0), account=sample_account()))
    violation = next(v for v in decision.violations if v.rule_id == GENERAL_RULE)
    assert violation.citation.document_id == "rbi_cb_rbc_2025"
    assert violation.citation.paragraph == "445"


# --- microfinance window: 09:00 to 18:00 inclusive -------------------------


@pytest.mark.parametrize(
    ("instant", "allowed"),
    [
        (_ist(8, 30, 0), False),  # inside the general window, outside the microfinance one
        (_ist(8, 59, 59), False),
        (_ist(9, 0, 0), True),
        (_ist(18, 0, 0), True),
        (_ist(18, 0, 1), False),
        (_ist(18, 30, 0), False),
    ],
)
def test_microfinance_calling_window_boundaries(engine: PolicyEngine, make_context, instant, allowed) -> None:
    account = sample_account(product_type=ProductType.MICROFINANCE)
    decision = engine.evaluate(make_context(now=instant, account=account))
    assert decision.allowed is allowed
    if not allowed:
        assert MICROFINANCE_RULE in _blocking_rule_ids(decision)


def test_microfinance_is_excluded_from_paragraph_445(engine: PolicyEngine, make_context) -> None:
    account = sample_account(product_type=ProductType.MICROFINANCE)
    decision = engine.evaluate(make_context(now=_ist(10, 0), account=account))
    evaluated = set(decision.rule_ids)
    assert MICROFINANCE_RULE in evaluated
    assert GENERAL_RULE not in evaluated


# --- timezone is configuration, never the host clock -----------------------


def test_instant_is_converted_into_the_configured_zone(engine: PolicyEngine, make_context) -> None:
    """02:00 UTC is 07:30 in Asia/Kolkata, which is outside the window."""
    utc_instant = datetime(2026, 9, 24, 2, 0, tzinfo=timezone.utc)
    decision = engine.evaluate(make_context(now=utc_instant, account=sample_account()))
    assert decision.allowed is False
    assert GENERAL_RULE in _blocking_rule_ids(decision)


def test_the_same_instant_decides_differently_in_different_zones(rule_set, engine, make_context) -> None:
    """14:00 UTC is 19:30 in Asia/Kolkata: allowed in one zone, blocked in the other.

    The point is that the zone is configuration. Nothing may be inferred from the
    host clock.
    """
    instant = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)
    in_kolkata = engine.evaluate(make_context(now=instant, account=sample_account()))
    assert in_kolkata.allowed is False
    assert GENERAL_RULE in _blocking_rule_ids(in_kolkata)

    utc_engine = PolicyEngine(rule_set, PolicyConfig(policy_version="t", timezone="UTC"))
    in_utc = utc_engine.evaluate(make_context(now=instant, account=sample_account()))
    assert in_utc.allowed is True


def test_naive_instant_is_rejected(engine: PolicyEngine, make_context) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        engine.evaluate(make_context(now=datetime(2026, 9, 24, 10, 0), account=sample_account()))


# --- the 2027 replacement --------------------------------------------------


def test_amended_window_takes_over_on_2027_01_01(engine: PolicyEngine, make_context) -> None:
    account = sample_account()
    before = engine.evaluate(make_context(now=datetime(2026, 12, 31, 10, 0, tzinfo=IST), account=account))
    after = engine.evaluate(make_context(now=datetime(2027, 1, 1, 10, 0, tzinfo=IST), account=account))
    assert GENERAL_RULE in before.rule_ids and AMENDED_RULE not in before.rule_ids
    assert AMENDED_RULE in after.rule_ids and GENERAL_RULE not in after.rule_ids


def test_amended_window_applies_to_microfinance_too(engine: PolicyEngine, make_context) -> None:
    """From 2027 there is one window; the narrower microfinance one is deleted."""
    account = sample_account(product_type=ProductType.MICROFINANCE)
    decision = engine.evaluate(make_context(now=datetime(2027, 1, 2, 8, 30, tzinfo=IST), account=account))
    assert decision.allowed is True
    assert AMENDED_RULE in decision.rule_ids
    assert MICROFINANCE_RULE not in decision.rule_ids


def test_express_authorisation_permits_out_of_hours_contact_from_2027(engine: PolicyEngine, make_context) -> None:
    from app.models.customer import ComplianceContext

    account = sample_account()
    instant = datetime(2027, 1, 2, 20, 30, tzinfo=IST)
    without = engine.evaluate(make_context(now=instant, account=account))
    assert without.allowed is False

    with_authorisation = engine.evaluate(
        make_context(
            now=instant,
            account=account,
            compliance=ComplianceContext(borrower_authorised_out_of_hours=True),
        )
    )
    assert with_authorisation.allowed is True


def test_express_authorisation_does_not_apply_before_2027(engine: PolicyEngine, make_context) -> None:
    """Paragraph 445 states no such exception, so the 2025 rule must ignore the flag."""
    from app.models.customer import ComplianceContext

    decision = engine.evaluate(
        make_context(
            now=_ist(21, 0),
            account=sample_account(),
            compliance=ComplianceContext(borrower_authorised_out_of_hours=True),
        )
    )
    assert decision.allowed is False
    assert GENERAL_RULE in _blocking_rule_ids(decision)
