"""PS-3 · Tool calls under code-mixing: runner, scorer and metrics.

Model output, scoring and metrics are separate on purpose and live in separate
files: ``raw.jsonl`` (what the model returned, unparsed), ``scores.jsonl`` (one
verdict per case, from this module's rules) and ``metrics.json`` (aggregates).
Re-scoring never re-runs a model.

Metric definitions
------------------
Fixed before any result was seen. "Emitted call" means an entry in the
response's ``tool_calls``; a tool call written into the text instead is not a
call - it is counted as ``tool_call_in_text`` and treated as malformed.

A call is **malformed** when any of these hold: its name is not one of the five
schemas; its arguments are not a JSON object; a required argument is missing;
an argument has the wrong JSON type (a number sent as a string counts); an enum
argument is outside its enum; ``promised_date`` is not a valid ``YYYY-MM-DD``
date. Extra, undeclared keys are recorded but are not malformed, because the
fixed schemas do not forbid them.

For a case whose expectation includes at least one call ("tool-expected"):

* **correct tool** - every tool in some required alternative was emitted (by
  name, whatever its arguments).
* **missed call** - no call at all was emitted.
* **wrong tool** - calls were emitted, but no required alternative is covered.
* **argument-correct** - some alternative is covered by well-formed calls whose
  scored arguments all equal an accepted value (numbers within 0.5, dates and
  enums exactly). A capture_ptp with the wrong date or amount is wrong.

For every case:

* **spurious call** - at least one emitted call is neither required nor
  permitted, or is a permitted tool with arguments outside what is permitted.
* **strict correct** - tool-expected: argument-correct, no spurious call, no
  malformed call; no-call-expected: nothing emitted that is not permitted.

Rates are reported with 95% Wilson intervals. The correct-tool, missed, wrong-
tool and argument-accuracy rates are over tool-expected, non-ambiguous cases;
the spurious rate is over all non-ambiguous cases; the malformed rate is over
all emitted calls. Ambiguous cases are reported separately: over-fire rate
(a forbidden tool fired on a ``should_not_fire`` case), under-fire rate (the
required tool did not fire on a ``should_fire`` case), and the firing rate on
``either`` cases.

The language delta is computed on matched scenarios: for every scenario present
in both languages, the English verdict is paired with the other language's
verdict. The delta is (English rate - other rate); discordant pairs and an exact
McNemar p-value accompany it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from app.evaluation.challenge import CALL_CONTEXT, SCHEMAS_BY_NAME, openai_tools
from app.evaluation.client import ChatClient, CallRecord
from app.evaluation.metrics import latency_summary, mcnemar_exact, rate
from app.evaluation.ps3_suite import persona_for

MAX_TOKENS = 256
NUMBER_TOLERANCE = 0.5

_TEXT_TOOL_MARKERS = ("<tool_call>", "</tool_call>", "<function=", '"name": "', '"name":"',
                      "capture_ptp(", "send_payment_link(", "mark_dispute(",
                      "escalate_human(", "log_disposition(")


# --- running ------------------------------------------------------------------


def messages_for(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": persona_for(case).system_prompt()},
        {"role": "system", "content": CALL_CONTEXT},
        *case["history"],
        {"role": "user", "content": case["utterance"]},
    ]


def run(
    cases: list[dict[str, Any]],
    client: ChatClient,
    raw_path: Path,
    *,
    progress: bool = True,
) -> None:
    """Run every case not already in ``raw_path``. Appends; safe to resume."""
    done = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["case_id"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    tools = openai_tools()
    with raw_path.open("a", encoding="utf-8") as fh:
        for index, case in enumerate(cases, 1):
            if case["id"] in done:
                continue
            record = client.complete(messages_for(case), tools=tools, max_tokens=MAX_TOKENS)
            fh.write(json.dumps({"case_id": case["id"], "record": record.to_json()},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if progress:
                names = [c.name for c in record.tool_calls]
                print(f"[{index}/{len(cases)}] {case['id']} ttft={record.ttft_ms} "
                      f"total={record.total_ms} calls={names} err={record.error}", flush=True)


# --- parsing and validation ---------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 10:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def parse_call(name: str | None, raw_arguments: str) -> dict[str, Any]:
    """Validate one emitted call against the fixed schemas. Never repairs it."""
    problems: list[str] = []
    args: dict[str, Any] | None = None
    schema = SCHEMAS_BY_NAME.get(name or "")
    if schema is None:
        problems.append("unknown_tool")
    text = (raw_arguments or "").strip()
    try:
        parsed = json.loads(text) if text else {}
    except json.JSONDecodeError:
        problems.append("arguments_not_json")
        parsed = None
    if parsed is not None and not isinstance(parsed, dict):
        problems.append("arguments_not_object")
        parsed = None
    args = parsed
    extra: list[str] = []
    if schema is not None and args is not None:
        props = schema["parameters"]["properties"]
        for required in schema["parameters"]["required"]:
            if required not in args:
                problems.append(f"missing_required:{required}")
        for key, value in args.items():
            spec = props.get(key)
            if spec is None:
                extra.append(key)
                continue
            if spec["type"] == "number" and not _is_number(value):
                problems.append(f"wrong_type:{key}")
            elif spec["type"] == "string" and not isinstance(value, str):
                problems.append(f"wrong_type:{key}")
            elif "enum" in spec and value not in spec["enum"]:
                problems.append(f"invalid_enum:{key}")
            elif spec.get("format") == "date" and not _valid_iso_date(value):
                problems.append(f"bad_date_format:{key}")
    return {"name": name, "args": args, "problems": problems, "extra_keys": extra,
            "malformed": bool(problems)}


def text_holds_tool_call(content: str) -> bool:
    lowered = (content or "").lower()
    return any(marker.lower() in lowered for marker in _TEXT_TOOL_MARKERS)


# --- matching ------------------------------------------------------------------------


def _value_matches(actual: Any, accepted: list[Any]) -> bool:
    for value in accepted:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if _is_number(actual) and abs(float(actual) - float(value)) <= NUMBER_TOLERANCE:
                return True
        elif actual == value:
            return True
    return False


def args_match(call: dict[str, Any], spec: dict[str, Any]) -> tuple[bool, list[str]]:
    """Whether a well-formed call's scored arguments equal the spec's. Returns mismatches."""
    if call["malformed"] or call["args"] is None:
        return False, ["malformed"]
    wrong = [key for key, accepted in spec.get("args", {}).items()
             if not _value_matches(call["args"].get(key), accepted)]
    return not wrong, wrong


def _covered(calls: list[dict[str, Any]], alternative: list[dict[str, Any]]) -> bool:
    names = {c["name"] for c in calls}
    return all(spec["name"] in names for spec in alternative)


def _alternative_args_ok(calls: list[dict[str, Any]], alternative: list[dict[str, Any]]) -> bool:
    return all(any(c["name"] == spec["name"] and args_match(c, spec)[0] for c in calls)
               for spec in alternative)


def _allowed(call: dict[str, Any], alternative: list[dict[str, Any]],
             permitted: list[dict[str, Any]]) -> bool:
    if any(call["name"] == spec["name"] for spec in alternative):
        return True
    for spec in permitted:
        if call["name"] == spec["name"] and (call["malformed"] or args_match(call, spec)[0]):
            return True
    return False


# --- scoring one case ------------------------------------------------------------------


def score_case(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    calls = [parse_call(tc.get("name"), tc.get("arguments", "")) for tc in record["tool_calls"]]
    in_text = text_holds_tool_call(record.get("content", ""))
    required: list[list[dict[str, Any]]] = case["expected"]["required"]
    permitted: list[dict[str, Any]] = case["expected"]["permitted"]
    ambiguity = case.get("ambiguity")

    best: list[dict[str, Any]] = required[0] if required else []
    for alternative in required:
        if _covered(calls, alternative) and _alternative_args_ok(calls, alternative):
            best = alternative
            break
    else:
        for alternative in required:
            if _covered(calls, alternative):
                best = alternative
                break

    tool_expected = bool(required)
    correct_tool = tool_expected and any(_covered(calls, alt) for alt in required)
    args_correct = tool_expected and any(
        _covered(calls, alt) and _alternative_args_ok(calls, alt) for alt in required
    )
    spurious_calls = [c["name"] for c in calls if not _allowed(c, best, permitted)]
    malformed = [c for c in calls if c["malformed"]]
    missed = tool_expected and not calls
    wrong_tool = tool_expected and bool(calls) and not correct_tool
    any_malformed = bool(malformed) or in_text
    if tool_expected:
        strict = args_correct and not spurious_calls and not any_malformed
    else:
        strict = not spurious_calls and not any_malformed

    confidence = None
    for spec in best:
        if spec["name"] == "capture_ptp" and spec.get("confidence"):
            fired = next((c for c in calls if c["name"] == "capture_ptp" and c["args"]), None)
            confidence = {
                "expected": spec["confidence"],
                "actual": fired["args"].get("confidence") if fired else None,
            }

    ambiguous_outcome = None
    if ambiguity:
        names = {c["name"] for c in calls}
        direction = ambiguity["direction"]
        if direction == "should_not_fire":
            ambiguous_outcome = {"over_fired": bool(names & set(ambiguity["forbidden"]))}
        elif direction == "should_fire":
            ambiguous_outcome = {"under_fired": not (names & set(ambiguity["required"]))}
        else:
            ambiguous_outcome = {"fired": ambiguity["tool"] in names}

    return {
        "case_id": case["id"],
        "scenario": case["scenario"],
        "language": case["language"],
        "category": case["category"],
        "ambiguous": case["ambiguous"],
        "tool_expected": tool_expected,
        "emitted": [{"name": c["name"], "args": c["args"], "problems": c["problems"],
                     "extra_keys": c["extra_keys"]} for c in calls],
        "tool_call_in_text": in_text,
        "correct_tool": correct_tool,
        "args_correct": args_correct,
        "missed": missed,
        "wrong_tool": wrong_tool,
        "spurious_calls": spurious_calls,
        "malformed_calls": len(malformed),
        "strict_correct": strict,
        "confidence": confidence,
        "ambiguous_outcome": ambiguous_outcome,
        "errors": error_shapes(case, calls, best, spurious_calls, in_text, record),
        "model_error": record.get("error"),
        "thinking_leaked": record.get("thinking_leaked", False),
        "ttft_ms": record.get("ttft_ms"),
        "total_ms": record.get("total_ms"),
        "completion_tokens": record.get("completion_tokens"),
        "finish_reason": record.get("finish_reason"),
    }


def error_shapes(case, calls, best, spurious, in_text, record) -> list[str]:
    """The recurring shape of a failure, not just that it failed. Feeds the error taxonomy."""
    shapes: list[str] = []
    if record.get("error"):
        shapes.append(f"infra:{record['error']}")
    if record.get("finish_reason") == "length":
        shapes.append("truncated_at_max_tokens")
    if in_text:
        shapes.append("tool_call_written_as_text")
    for c in calls:
        for problem in c["problems"]:
            shapes.append(f"malformed:{problem.split(':')[0]}:{c['name']}")
    if case["expected"]["required"] and not calls:
        shapes.append(f"no_call:{case['category']}")
    elif case["expected"]["required"]:
        names = {c["name"] for c in calls}
        for spec in best:
            if spec["name"] not in names:
                got = ",".join(sorted(names)) or "none"
                shapes.append(f"wrong_tool:{spec['name']}->{got}")
                continue
            same = [c for c in calls if c["name"] == spec["name"] and not c["malformed"]]
            if same and not any(args_match(c, spec)[0] for c in same):
                for key in args_match(same[0], spec)[1]:
                    shapes.append(f"wrong_arg:{spec['name']}.{key}")
    for name in spurious:
        shapes.append(f"spurious:{name}")
    return shapes


# --- aggregation ------------------------------------------------------------------------


def _core(scores: list[dict[str, Any]]) -> dict[str, Any]:
    plain = [s for s in scores if not s["ambiguous"]]
    expected = [s for s in plain if s["tool_expected"]]
    emitted = sum(len(s["emitted"]) for s in scores)
    malformed = sum(s["malformed_calls"] for s in scores)
    in_text = sum(1 for s in scores if s["tool_call_in_text"])
    with_tool = [s for s in expected if s["correct_tool"]]
    return {
        "cases": len(scores),
        "correct_tool_rate": rate([s["correct_tool"] for s in expected]),
        "argument_accuracy": rate([s["args_correct"] for s in expected]),
        "argument_accuracy_given_correct_tool": rate([s["args_correct"] for s in with_tool]),
        "missed_call_rate": rate([s["missed"] for s in expected]),
        "wrong_tool_rate": rate([s["wrong_tool"] for s in expected]),
        "spurious_call_rate": rate([bool(s["spurious_calls"]) for s in plain]),
        "spurious_rate_on_no_call_cases": rate(
            [bool(s["spurious_calls"]) for s in plain if not s["tool_expected"]]
        ),
        "malformed_argument_rate": {
            "malformed_calls": malformed,
            "emitted_calls": emitted,
            "rate": round(malformed / emitted, 4) if emitted else None,
        },
        "tool_call_written_as_text_cases": in_text,
        "strict_accuracy": rate([s["strict_correct"] for s in plain]),
        "confidence_agreement": rate([
            s["confidence"]["actual"] == s["confidence"]["expected"]
            for s in plain if s["confidence"] and s["correct_tool"]
        ]),
    }


def _ambiguous(scores: list[dict[str, Any]]) -> dict[str, Any]:
    amb = [s for s in scores if s["ambiguous"] and s["ambiguous_outcome"]]
    return {
        "cases": len(amb),
        "over_fire_rate": rate([s["ambiguous_outcome"]["over_fired"] for s in amb
                                if "over_fired" in s["ambiguous_outcome"]]),
        "under_fire_rate": rate([s["ambiguous_outcome"]["under_fired"] for s in amb
                                 if "under_fired" in s["ambiguous_outcome"]]),
        "either_fire_rate": rate([s["ambiguous_outcome"]["fired"] for s in amb
                                  if "fired" in s["ambiguous_outcome"]]),
    }


def paired_delta(scores: list[dict[str, Any]], other: str, metric: str) -> dict[str, Any]:
    """English minus ``other`` on scenarios present in both, over non-ambiguous tool-expected
    cases for tool metrics and over all non-ambiguous cases for strict/spurious."""
    by = defaultdict(dict)
    for s in scores:
        if s["ambiguous"]:
            continue
        if metric in ("correct_tool", "args_correct", "missed") and not s["tool_expected"]:
            continue
        by[s["scenario"]][s["language"]] = s
    pairs = [(v["en"], v[other]) for v in by.values() if "en" in v and other in v]

    def value(s: dict[str, Any]) -> bool:
        return bool(s["spurious_calls"]) if metric == "spurious" else bool(s[metric])

    en = [value(a) for a, _ in pairs]
    ot = [value(b) for _, b in pairs]
    b = sum(1 for x, y in zip(en, ot) if x and not y)
    c = sum(1 for x, y in zip(en, ot) if y and not x)
    n = len(pairs)
    return {
        "pairs": n,
        "en_rate": round(sum(en) / n, 4) if n else None,
        f"{other}_rate": round(sum(ot) / n, 4) if n else None,
        "delta_en_minus_other": round((sum(en) - sum(ot)) / n, 4) if n else None,
        "discordant_en_only": b,
        "discordant_other_only": c,
        "mcnemar_p": mcnemar_exact(b, c),
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    by_language = defaultdict(list)
    by_category = defaultdict(list)
    for s in scores:
        by_language[s["language"]].append(s)
        by_category[s["category"]].append(s)
    deltas = {}
    for other in ("hi-en", "mr", "mr-en"):
        deltas[f"en_vs_{other}"] = {
            metric: paired_delta(scores, other, metric)
            for metric in ("correct_tool", "args_correct", "strict_correct", "spurious", "missed")
        }
    errors = Counter(e for s in scores for e in s["errors"])
    errors_by_language = {
        lang: dict(Counter(e for s in group for e in s["errors"]).most_common())
        for lang, group in sorted(by_language.items())
    }
    return {
        "overall": _core(scores),
        "by_language": {k: _core(v) for k, v in sorted(by_language.items())},
        "by_category": {k: _core(v) for k, v in sorted(by_category.items())},
        "ambiguous": _ambiguous(scores),
        "ambiguous_by_language": {k: _ambiguous(v) for k, v in sorted(by_language.items())},
        "language_deltas": deltas,
        "error_taxonomy": dict(errors.most_common()),
        "error_taxonomy_by_language": errors_by_language,
        "thinking_leaked_calls": sum(1 for s in scores if s["thinking_leaked"]),
        "model_errors": sum(1 for s in scores if s["model_error"]),
        "latency_ms": {
            "ttft": latency_summary([s["ttft_ms"] for s in scores]),
            "total": latency_summary([s["total_ms"] for s in scores]),
        },
        "completion_tokens": latency_summary([s["completion_tokens"] for s in scores]),
    }


def score_run(cases: list[dict[str, Any]], raw_path: Path) -> list[dict[str, Any]]:
    by_id = {c["id"]: c for c in cases}
    scores = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case = by_id.get(row["case_id"])
        if case is not None:
            scores.append(score_case(case, row["record"]))
    return scores


def suite_digest(cases: list[dict[str, Any]]) -> str:
    blob = json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
