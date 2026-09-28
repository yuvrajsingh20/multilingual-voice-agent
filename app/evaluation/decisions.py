"""Offline evaluation of a decision provider: accuracy by threshold, and latency.

Provider-agnostic: it drives any :class:`~app.services.decision.DecisionService`
with labelled cases and reports, per decision,

* how often the provider failed outright, by category;
* latency percentiles over the calls that answered;
* for each candidate threshold, the share of cases that would be decided
  (coverage) and how many of those match the expected label (accuracy).

That last table is what a threshold should be chosen from. The defaults in
:mod:`app.config` were not - no labelled data existed when they were set.

The case file shipped in ``data/eval/decision_cases.json`` is SYNTHETIC: written
by hand to cover every label in the languages in scope. It is a smoke set for
wiring and a template for real data, not a benchmark. A threshold is only as
good as the labelled calls it was measured on.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.decision import (
    DecisionContext,
    DecisionError,
    DecisionName,
    DecisionRequest,
    DecisionService,
    DecisionTimeout,
)
from app.services.decisions.registry import get_spec

#: Thresholds reported in the sweep.
SWEEP: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


class DecisionCase(BaseModel):
    """One labelled example. ``expected`` must be a registered label for ``decision``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    decision: DecisionName
    context: DecisionContext
    expected: str

    @model_validator(mode="after")
    def _expected_is_a_registered_label(self) -> "DecisionCase":
        if self.expected not in get_spec(self.decision).label_values:
            raise ValueError(f"case {self.id}: {self.expected!r} is not a {self.decision.value} label")
        return self


class CaseOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    decision: DecisionName
    expected: str
    label: str | None = None
    confidence: float | None = None
    latency_ms: float
    failure: str | None = None


def load_cases(path: Path) -> list[DecisionCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [DecisionCase.model_validate(item) for item in raw["cases"]]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return cases


async def run_cases(
    service: DecisionService,
    cases: list[DecisionCase],
    *,
    timeout_seconds: float,
    concurrency: int = 1,
    repeat: int = 1,
) -> list[CaseOutcome]:
    """Ask ``service`` every case, ``repeat`` times, at most ``concurrency`` at once.

    ``concurrency=1`` measures latency without queueing the provider behind
    itself; raise it to see how latency behaves under load.
    """
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(case: DecisionCase) -> CaseOutcome:
        async with gate:
            started = time.perf_counter()
            try:
                answer = await asyncio.wait_for(
                    service.decide(DecisionRequest(name=case.decision, context=case.context)),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                failure = DecisionTimeout.category
            except DecisionError as exc:
                failure = exc.category
            except Exception:  # noqa: BLE001 - an evaluation reports failures, it does not stop on them
                failure = DecisionError.category
            else:
                return CaseOutcome(
                    case_id=case.id, decision=case.decision, expected=case.expected,
                    label=answer.label, confidence=answer.confidence,
                    latency_ms=_elapsed(started),
                )
            return CaseOutcome(
                case_id=case.id, decision=case.decision, expected=case.expected,
                latency_ms=_elapsed(started), failure=failure,
            )

    return list(await asyncio.gather(*(one(case) for _ in range(repeat) for case in cases)))


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile. ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def summarise(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    """Per-decision failure counts, latency and the threshold sweep.

    Latency percentiles are over answered calls, so they are truncated at the
    deadline: anything slower is counted under ``failures`` as
    ``decision_timeout`` instead. To see the tail, run with a generous
    ``JEV_TIMEOUT_SECONDS`` (up to 10) and choose the production value from it.
    """
    report: dict[str, Any] = {}
    for name in DecisionName:
        mine = [o for o in outcomes if o.decision is name]
        if not mine:
            continue
        answered = [o for o in mine if o.failure is None]
        latencies = [o.latency_ms for o in answered]
        sweep = {}
        for threshold in SWEEP:
            decided = [o for o in answered if (o.confidence or 0.0) >= threshold]
            correct = sum(1 for o in decided if o.label == o.expected)
            sweep[f"{threshold:.2f}"] = {
                "coverage": round(len(decided) / len(mine), 3),
                "accuracy": round(correct / len(decided), 3) if decided else None,
                "decided": len(decided),
            }
        report[name.value] = {
            "cases": len(mine),
            "answered": len(answered),
            "failures": dict(Counter(o.failure for o in mine if o.failure)),
            "top_label_accuracy": (
                round(sum(1 for o in answered if o.label == o.expected) / len(answered), 3)
                if answered else None
            ),
            "latency_ms": {
                "p50": _round(percentile(latencies, 50)),
                "p95": _round(percentile(latencies, 95)),
                "max": _round(max(latencies) if latencies else None),
            },
            "threshold_sweep": sweep,
        }
    return report


def _round(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _elapsed(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0
