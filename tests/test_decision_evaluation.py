"""The decision evaluation harness and the measurement script.

The harness is how thresholds are meant to be chosen and latency measured, so
its arithmetic is pinned here. The shipped cases are synthetic; these tests
check their coverage and shape, not their realism.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.evaluation.decisions import DecisionCase, load_cases, percentile, run_cases, summarise
from app.main import create_app
from app.runtime import build_runtime
from app.services.decision import (
    DecisionAnswer,
    DecisionContext,
    DecisionDisabled,
    DecisionName,
    DecisionTimeout,
    ScriptedDecisionService,
)
from app.services.decisions.registry import DECISION_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES = REPO_ROOT / "data" / "eval" / "decision_cases.json"


def test_the_shipped_cases_cover_every_label_of_every_decision() -> None:
    cases = load_cases(CASES)
    for name, spec in DECISION_REGISTRY.items():
        covered = {case.expected for case in cases if case.decision is name}
        assert covered == spec.label_values, name


def test_the_shipped_cases_are_marked_synthetic() -> None:
    assert "SYNTHETIC" in json.loads(CASES.read_text(encoding="utf-8"))["description"]


def test_a_case_must_expect_a_registered_label() -> None:
    with pytest.raises(ValidationError):
        DecisionCase(
            id="x", decision=DecisionName.HUMAN_ESCALATION, expected="maybe",
            context=DecisionContext(utterance="hi"),
        )


def _case(case_id: str, expected: str) -> DecisionCase:
    return DecisionCase(
        id=case_id, decision=DecisionName.CUSTOMER_INTENT, expected=expected,
        context=DecisionContext(utterance=f"utterance {case_id}"),
    )


def _answer(label: str, confidence: float) -> DecisionAnswer:
    return DecisionAnswer(
        name=DecisionName.CUSTOMER_INTENT, label=label, confidence=confidence, provider="scripted"
    )


def test_the_threshold_sweep_reports_coverage_and_accuracy() -> None:
    cases = [_case("a", "dispute"), _case("b", "refusal"), _case("c", "unclear"), _case("d", "dispute")]
    service = ScriptedDecisionService([
        _answer("dispute", 0.95),  # right, confident
        _answer("dispute", 0.88),  # wrong, fairly confident
        _answer("unclear", 0.55),  # right, unsure
        DecisionTimeout(),         # no answer
    ])
    report = summarise(asyncio.run(run_cases(service, cases, timeout_seconds=1.0)))["customer_intent"]

    assert report["cases"] == 4
    assert report["answered"] == 3
    assert report["failures"] == {"decision_timeout": 1}
    assert report["top_label_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    sweep = report["threshold_sweep"]
    assert sweep["0.50"] == {"coverage": 0.75, "accuracy": pytest.approx(0.667, abs=1e-3), "decided": 3}
    assert sweep["0.85"] == {"coverage": 0.5, "accuracy": 0.5, "decided": 2}
    assert sweep["0.90"] == {"coverage": 0.25, "accuracy": 1.0, "decided": 1}
    assert sweep["0.95"]["decided"] == 1


def test_the_harness_records_failures_rather_than_stopping() -> None:
    service = ScriptedDecisionService([DecisionDisabled(), _answer("dispute", 0.9)])
    outcomes = asyncio.run(run_cases(service, [_case("a", "dispute"), _case("b", "dispute")], timeout_seconds=1.0))
    assert [o.failure for o in outcomes] == ["decision_disabled", None]


def test_the_harness_enforces_its_deadline() -> None:
    service = ScriptedDecisionService([_answer("dispute", 0.9)], delay_seconds=5.0)
    (outcome,) = asyncio.run(run_cases(service, [_case("a", "dispute")], timeout_seconds=0.05))
    assert outcome.failure == "decision_timeout"


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [([], 50, None), ([5.0], 95, 5.0), ([1, 2, 3, 4], 50, 2), ([1, 2, 3, 4], 95, 4), (list(range(1, 101)), 95, 95)],
)
def test_percentiles_are_nearest_rank(values, q, expected) -> None:
    assert percentile([float(v) for v in values], q) == expected


def test_the_measurement_script_runs_offline_through_the_real_adapter() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/eval_decisions.py", "--mock"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["provider"] == "jev" and report["mode"].startswith("mock")
    for decision in ("barge_in", "customer_intent", "needs_human_escalation"):
        assert report["results"][decision]["failures"] == {}


def test_the_measurement_script_refuses_live_mode_when_disabled() -> None:
    env = {"PATH": "/usr/bin:/bin", "DECISION_PROVIDER": "disabled", "JEV_ENABLED": "false"}
    result = subprocess.run(
        [sys.executable, "scripts/eval_decisions.py", "--live"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode != 0
    assert "disabled" in result.stderr


def test_shutdown_closes_the_decision_client(settings) -> None:
    closed: list[bool] = []

    class ClosableService:
        provider = "closable"

        async def decide(self, request):
            raise DecisionDisabled()

        async def aclose(self) -> None:
            closed.append(True)

    runtime = build_runtime(settings, decisions=ClosableService())
    with TestClient(create_app(settings, runtime)):
        assert closed == []
    assert closed == [True]


def test_every_decision_has_cases_in_every_language_in_scope() -> None:
    """The set is still mostly English; this pins the minimum, not a balance."""
    from app.models.enums import Language

    cases = load_cases(CASES)
    for name in DECISION_REGISTRY:
        languages = {case.context.language for case in cases if case.decision is name}
        assert languages >= set(Language), name


def test_mock_mode_handles_cases_whose_words_get_masked(tmp_path) -> None:
    """Real labelled calls will contain identifiers; masking must not break the mock run."""
    cases = {
        "description": "test",
        "cases": [
            {"id": "a", "decision": "customer_intent", "expected": "payment_already_made",
             "context": {"utterance": "I already paid, reference 12345678"}},
            {"id": "b", "decision": "needs_human_escalation", "expected": "yes",
             "context": {"utterance": "I already paid, reference 12345678"}},
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/eval_decisions.py", "--mock", "--cases", str(path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)["results"]
    assert report["customer_intent"]["failures"] == {} and report["customer_intent"]["top_label_accuracy"] == 1.0
    assert report["needs_human_escalation"]["failures"] == {}
