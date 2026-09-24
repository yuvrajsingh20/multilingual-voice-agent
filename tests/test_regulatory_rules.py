"""The rule file must stay tied to the corpus.

The load test is not a formality: if a quote drifts from its source, or a rule
claims a check that does not exist, these tests fail and the rule set cannot be
trusted for an audit.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.core.checks import known_check_ids
from app.core.rules import RuleSet, RuleSourceError, load_rule_set
from app.models.enums import Enforcement, ProductType, RuleStatus
from tests.conftest import REPO_ROOT


def test_rule_file_loads_and_is_not_empty(rule_set: RuleSet) -> None:
    assert rule_set.schema_version == "1.0"
    assert len(rule_set.rules) > 0


def test_every_quote_is_verbatim_in_its_source_file(rule_set: RuleSet) -> None:
    """A rule is only traceable if its quote is literally in the cited document."""
    missing = []
    for rule in rule_set.rules:
        source_path = REPO_ROOT / rule.source.text_file
        text = source_path.read_text(encoding="utf-8")
        if rule.source.quote not in text:
            missing.append((rule.rule_id, rule.source.text_file))
    assert missing == [], f"quotes not found verbatim in their source: {missing}"


def test_every_cited_document_is_in_the_corpus_catalog(rule_set: RuleSet) -> None:
    import json

    catalog = json.loads((REPO_ROOT / "data" / "regulatory" / "catalog.json").read_text(encoding="utf-8"))
    known = {doc["id"] for doc in catalog["documents"]}
    cited = {rule.source.document_id for rule in rule_set.rules}
    assert cited <= known, f"rules cite documents absent from the catalog: {sorted(cited - known)}"


def test_every_source_file_exists(rule_set: RuleSet) -> None:
    for rule in rule_set.rules:
        assert (REPO_ROOT / rule.source.text_file).is_file(), rule.rule_id


def test_every_check_id_is_implemented(rule_set: RuleSet) -> None:
    declared = {rule.check_id for rule in rule_set.rules if rule.check_id}
    assert declared <= known_check_ids(), f"unimplemented checks: {sorted(declared - known_check_ids())}"


def test_every_implemented_check_is_used(rule_set: RuleSet) -> None:
    """A check with no rule behind it would enforce something untraceable."""
    declared = {rule.check_id for rule in rule_set.rules if rule.check_id}
    assert known_check_ids() <= declared, f"checks with no rule: {sorted(known_check_ids() - declared)}"


def test_rule_ids_are_unique(rule_set: RuleSet) -> None:
    ids = [rule.rule_id for rule in rule_set.rules]
    assert len(ids) == len(set(ids))


def test_organisational_obligations_have_no_check(rule_set: RuleSet) -> None:
    for rule in rule_set.rules:
        if rule.enforcement is Enforcement.ORGANISATIONAL_OBLIGATION:
            assert rule.check_id is None, rule.rule_id


def test_conversation_deterministic_rules_all_have_a_check(rule_set: RuleSet) -> None:
    for rule in rule_set.rules:
        if rule.enforcement is Enforcement.CONVERSATION_DETERMINISTIC:
            assert rule.check_id is not None, rule.rule_id


def test_historical_rules_are_never_active(rule_set: RuleSet) -> None:
    historical = [r for r in rule_set.rules if r.status is RuleStatus.HISTORICAL_SUPPORTING]
    assert historical, "the corpus retains historical provenance; it should be represented"
    for rule in historical:
        assert rule.check_id is None
        assert not rule.is_active_on(date(2026, 9, 24))
        assert not rule.is_active_on(date(2027, 6, 1))


def test_future_effective_rules_are_dormant_then_active(rule_set: RuleSet) -> None:
    future = [r for r in rule_set.rules if r.status is RuleStatus.FUTURE_EFFECTIVE]
    assert future, "the 2026 Fourth Amendment Directions should be encoded"
    for rule in future:
        assert rule.effective_from == date(2027, 1, 1), rule.rule_id
        assert not rule.is_active_on(date(2026, 12, 31)), rule.rule_id
        assert rule.is_active_on(date(2027, 1, 1)), rule.rule_id


def test_superseded_rules_stop_when_the_amendment_starts(rule_set: RuleSet) -> None:
    """RBC 2025 paragraphs 442-454 and 408-416 are deleted from 2027-01-01."""
    superseded = [r for r in rule_set.rules if r.effective_until is not None]
    assert superseded
    for rule in superseded:
        assert rule.effective_until == date(2026, 12, 31), rule.rule_id
        assert rule.is_active_on(date(2026, 12, 31))
        assert not rule.is_active_on(date(2027, 1, 1))


def test_superseded_by_points_at_real_rules(rule_set: RuleSet) -> None:
    ids = {rule.rule_id for rule in rule_set.rules}
    for rule in rule_set.rules:
        for successor in rule.superseded_by:
            assert successor in ids, f"{rule.rule_id} -> {successor}"


def test_fair_practices_harassment_rule_survives_the_amendment(rule_set: RuleSet) -> None:
    """Paragraph 343 sits outside the deleted range, so it must have no end date."""
    rule = rule_set.by_id("RBI-CB-RBC-2025-343-NO-UNDUE-HARASSMENT")
    assert rule.effective_until is None
    assert rule.is_active_on(date(2027, 6, 1))


def test_microfinance_calling_window_is_the_narrower_one(rule_set: RuleSet) -> None:
    general = rule_set.by_id("RBI-CB-RBC-2025-445-CALLING-HOURS")
    microfinance = rule_set.by_id("RBI-CB-RBC-2025-410-2-MFI-CALLING-HOURS")
    assert general.parameters["earliest_local_time"] == "08:00:00"
    assert general.parameters["latest_local_time"] == "19:00:00"
    assert microfinance.parameters["earliest_local_time"] == "09:00:00"
    assert microfinance.parameters["latest_local_time"] == "18:00:00"
    assert ProductType.MICROFINANCE in general.applies_when.excluded_product_types
    assert microfinance.applies_when.product_types == (ProductType.MICROFINANCE,)


def test_paragraph_445_microfinance_exception_is_recorded(rule_set: RuleSet) -> None:
    rule = rule_set.by_id("RBI-CB-RBC-2025-445-CALLING-HOURS")
    assert any("microfinance" in exception.lower() for exception in rule.exceptions)


def test_digital_lending_rule_is_conditional_on_the_product(rule_set: RuleSet) -> None:
    rule = rule_set.by_id("RBI-DL-2025-8-V-AGENT-PARTICULARS-BEFORE-CONTACT")
    assert rule.status is RuleStatus.CONDITIONAL
    assert rule.applies_when.matches(ProductType.DIGITAL_LENDING)
    assert not rule.applies_when.matches(ProductType.RETAIL_LOAN)


def test_applicable_filters_by_product_and_date(rule_set: RuleSet) -> None:
    today = date(2026, 9, 24)
    retail = {r.rule_id for r in rule_set.applicable(today, ProductType.RETAIL_LOAN)}
    micro = {r.rule_id for r in rule_set.applicable(today, ProductType.MICROFINANCE)}
    assert "RBI-CB-RBC-2025-445-CALLING-HOURS" in retail
    assert "RBI-CB-RBC-2025-445-CALLING-HOURS" not in micro
    assert "RBI-CB-RBC-2025-410-2-MFI-CALLING-HOURS" in micro
    assert "RBI-CB-RBC-2025-410-2-MFI-CALLING-HOURS" not in retail


def test_loader_rejects_a_bad_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "rules.json"
    bad.write_text(
        '{"schema_version": "9.9", "rules_version": "0", "generated_on": "2026-01-01",'
        ' "corpus_catalog": "x", "corpus_as_of": "x", "scope": "x", "notes": "x", "rules": []}',
        encoding="utf-8",
    )
    with pytest.raises(RuleSourceError):
        load_rule_set(bad)


def test_loader_rejects_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "rules.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuleSourceError):
        load_rule_set(bad)


def test_loader_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuleSourceError):
        load_rule_set(tmp_path / "absent.json")
