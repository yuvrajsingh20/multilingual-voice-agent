"""Render the Track 1 result tables as Markdown from the saved metrics files.

    python scripts/track1_tables.py > data/eval/track1/tables.md

Reads only ``data/eval/track1/ps*/runs/<model>/metrics.json``; every number in the
report's result tables comes from here, so it can be regenerated and checked.
Missing runs print as "not run".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "data" / "eval" / "track1"
MODELS = sys.argv[1:] or ["qwen-voice-4b", "qwen-voice-9b"]


def _load(ps: str, model: str) -> dict | None:
    path = ROOT / ps / "runs" / model / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _r(entry) -> str:
    if not entry or entry.get("rate") is None:
        return "n/a"
    ci = entry.get("ci95")
    ci_text = f" [{ci[0]:.2f}-{ci[1]:.2f}]" if ci and ci[0] is not None else ""
    return f"{entry['rate']:.3f} ({entry['k']}/{entry['n']}){ci_text}"


def _v(x) -> str:
    return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _p(x) -> str:
    if x is None:
        return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _ms(summary) -> str:
    if not summary or not summary.get("n"):
        return "n/a"
    return (f"n={summary['n']} p50={summary['p50'] / 1000:.1f}s p95={summary['p95'] / 1000:.1f}s "
            f"max={summary['max'] / 1000:.1f}s")


def ps3_tables() -> None:
    print("## PS-3 results\n")
    print("| Model | Correct tool | Argument accuracy | Strict | Missed | Wrong tool | Spurious (non-ambiguous) | Malformed args |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        d = _load("ps3", m)
        if not d:
            print(f"| {m} | not run | | | | | | |")
            continue
        o = d["overall"]
        mal = o["malformed_argument_rate"]
        print(f"| {m} | {_r(o['correct_tool_rate'])} | {_r(o['argument_accuracy'])} | {_r(o['strict_accuracy'])} | "
              f"{_r(o['missed_call_rate'])} | {_r(o['wrong_tool_rate'])} | {_r(o['spurious_call_rate'])} | "
              f"{_v(mal['rate'])} ({mal['malformed_calls']}/{mal['emitted_calls']}) |")
    print("\n### By language (correct tool / argument accuracy / strict)\n")
    print("| Model | en | hi-en | mr | mr-en |")
    print("|---|---|---|---|---|")
    for m in MODELS:
        d = _load("ps3", m)
        if not d:
            continue
        cells = []
        for lang in ("en", "hi-en", "mr", "mr-en"):
            o = d["by_language"].get(lang)
            cells.append("n/a" if not o else
                         f"{_v(o['correct_tool_rate']['rate'])} / {_v(o['argument_accuracy']['rate'])} / {_v(o['strict_accuracy']['rate'])}")
        print(f"| {m} | " + " | ".join(cells) + " |")
    print("\n### Paired language delta (English minus other, same scenarios; McNemar exact p)\n")
    print("| Model | Comparison | Metric | Pairs | en | other | Delta | p |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        d = _load("ps3", m)
        if not d:
            continue
        for comp, metrics in d["language_deltas"].items():
            other = comp.split("_vs_")[1]
            for metric in ("correct_tool", "args_correct", "strict_correct"):
                x = metrics[metric]
                print(f"| {m} | {comp} | {metric} | {x['pairs']} | {_v(x['en_rate'])} | {_v(x.get(other + '_rate'))} | "
                      f"{_v(x['delta_en_minus_other'])} | {_p(x['mcnemar_p'])} |")
    print("\n### Ambiguous cases\n")
    print("| Model | Over-fire (should not fire) | Under-fire (should fire) | Fired (either) |")
    print("|---|---|---|---|")
    for m in MODELS:
        d = _load("ps3", m)
        if d:
            a = d["ambiguous"]
            print(f"| {m} | {_r(a['over_fire_rate'])} | {_r(a['under_fire_rate'])} | {_r(a['either_fire_rate'])} |")
    print("\n### By category (correct tool / argument accuracy)\n")
    cats = sorted({c for m in MODELS if _load("ps3", m) for c in _load("ps3", m)["by_category"]})
    print("| Category | " + " | ".join(MODELS) + " |")
    print("|---|" + "---|" * len(MODELS))
    for c in cats:
        cells = []
        for m in MODELS:
            d = _load("ps3", m)
            o = d["by_category"].get(c) if d else None
            cells.append("n/a" if not o else f"{_v(o['correct_tool_rate']['rate'])} / {_v(o['argument_accuracy']['rate'])}")
        print(f"| {c} | " + " | ".join(cells) + " |")
    print("\n### Error taxonomy (top 12 shapes)\n")
    for m in MODELS:
        d = _load("ps3", m)
        if not d:
            continue
        print(f"**{m}**: " + "; ".join(f"`{k}` {v}" for k, v in list(d["error_taxonomy"].items())[:12]) + "\n")
    print("### Integrity and latency\n")
    for m in MODELS:
        d = _load("ps3", m)
        if d:
            print(f"- {m}: thinking leaked in {d['thinking_leaked_calls']} calls; model/transport errors {d['model_errors']}; "
                  f"TTFT {_ms(d['latency_ms']['ttft'])}; total {_ms(d['latency_ms']['total'])}")
    print()


def ps1_tables() -> None:
    print("## PS-1 results (judge verdicts)\n")
    print("| Model | Judge | Judged / turns | Violation rate | Attack success | mark_dispute compliance | escalate_human compliance |")
    print("|---|---|---|---|---|---|---|")
    for m in MODELS:
        d = _load("ps1", m)
        if not d:
            print(f"| {m} | not run | | | | | |")
            continue
        o = d["overall"]
        print(f"| {m} | {d.get('judge_model')} | {o['judged']}/{o['turns']} | {_r(o['violation_rate'])} | "
              f"{_r(o['attack_success_rate'])} | {_r(o['mark_dispute_compliance'])} | {_r(o['escalate_human_compliance'])} |")
    print("\n### By language (violation rate)\n")
    print("| Model | en | hi | hi-en | mr | Indic minus en |")
    print("|---|---|---|---|---|---|")
    for m in MODELS:
        d = _load("ps1", m)
        if not d:
            continue
        cells = [_r(d["by_language"].get(l, {}).get("violation_rate")) for l in ("en", "hi", "hi-en", "mr")]
        print(f"| {m} | " + " | ".join(cells) + f" | {_v(d['english_vs_indic']['delta_indic_minus_en'])} |")
    print("\n### Paired English-vs-Indic (same attack; McNemar exact p)\n")
    print("| Model | Comparison | Pairs | en | other | Violating en only | other only | p |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        d = _load("ps1", m)
        if not d:
            continue
        for comp, x in d["english_vs_indic"]["paired"].items():
            other = comp.split("_vs_")[1]
            print(f"| {m} | {comp} | {x['pairs']} | {_v(x['en_violation_rate'])} | {_v(x.get(other + '_violation_rate'))} | "
                  f"{x['violating_in_en_only']} | {x.get('violating_in_' + other + '_only')} | {_p(x['mcnemar_p'])} |")
    print("\n### By violation category (rate of the judge assigning the code, all turns)\n")
    codes = [f"V{i}" for i in range(1, 9)]
    print("| Model | " + " | ".join(codes) + " |")
    print("|---|" + "---|" * len(codes))
    for m in MODELS:
        d = _load("ps1", m)
        if d:
            bv = d["overall"]["by_violation"]
            print(f"| {m} | " + " | ".join(f"{bv[c]['k']}" for c in codes) + " |")
    print("\n### By attack surface (violation rate)\n")
    surfaces = sorted({s for m in MODELS if _load("ps1", m) for s in _load("ps1", m)["by_surface"]})
    print("| Surface | " + " | ".join(MODELS) + " |")
    print("|---|" + "---|" * len(MODELS))
    for s in surfaces:
        cells = []
        for m in MODELS:
            d = _load("ps1", m)
            cells.append(_r(d["by_surface"][s]["violation_rate"]) if d and s in d["by_surface"] else "n/a")
        print(f"| {s} | " + " | ".join(cells) + " |")
    print("\n### Rules vs judge (cross-check, not validation)\n")
    for m in MODELS:
        d = _load("ps1", m)
        if d:
            print(f"**{m}**: " + "; ".join(
                f"{c} rule {v['rule_positive']} / judge {v['judge_positive']} / both {v['both']} (kappa {_v(v['kappa'])}, {v['rule_kind']})"
                for c, v in d["rules_vs_judge"].items()) + "\n")
    print("### Integrity and latency\n")
    for m in MODELS:
        d = _load("ps1", m)
        if d:
            print(f"- {m}: judge verdicts contradicting their own rationale {len(d.get('judge_contradictions', []))}; "
                  f"empty replies {d['empty_replies']}; truncated {d['truncated_replies']}; errors {d['model_errors']}; "
                  f"thinking leaked {d['thinking_leaked']}; TTFT {_ms(d['latency_ms']['ttft'])}")
    print()


def ps2_tables() -> None:
    print("## PS-2 results (text-side signals; not TTS survival)\n")
    print("| Model | Calls/turns | Script by language | Language mismatch turns | Tone collapse calls | Numeral/currency turns | Non-Latin-script turns | Long turns |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        d = _load("ps2", m)
        if not d:
            print(f"| {m} | not run | | | | | | |")
            continue
        f = d["failure_modes"]
        scripts = "; ".join(f"{k}: " + ", ".join(f"{s} {n}" for s, n in v.items()) for k, v in d["script_by_language"].items())
        lm = f.get("language_mismatch", {})
        print(f"| {m} | {d['calls']}/{d['turns']} | {scripts} | {lm.get('turns', 'n/a')}/{lm.get('of_turns', 'n/a')} | {f['tone_collapse_under_refusal']['calls']}/{f['tone_collapse_under_refusal']['of_calls']} | "
              f"{f['numeral_currency_handling']['turns']}/{f['numeral_currency_handling']['of_turns']} | "
              f"{f['script_inconsistency']['turns']}/{f['script_inconsistency']['of_turns']} | "
              f"{f['long_turns']['turns']}/{f['long_turns']['of_turns']} |")
    print("\n### Per call\n")
    print("| Model | Call | Scripts | Matrix language | Switches | Honorific dropped | Repeated turns | Pressure after refusal / other | Courtesy after refusal / other | Mean words | Introduced amounts |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        d = _load("ps2", m)
        if not d:
            continue
        for cid, c in d["per_call"].items():
            print(f"| {m} | {cid} | {','.join(c['scripts_used'])} | {','.join(c.get('matrix_languages', []))} | {c['script_switches']} | {c['honorific_dropped']} | "
                  f"{c['repeated_turns']} | {_v(c['pressure_per_turn_after_refusal'])} / {_v(c['pressure_per_turn_other'])} | "
                  f"{_v(c['courtesy_per_turn_after_refusal'])} / {_v(c['courtesy_per_turn_other'])} | {_v(c['mean_words'])} | "
                  f"{c['introduced_amounts'] or '-'} |")
    print("\n### First agent turn by bucket (pressure markers / courtesy markers / words)\n")
    for m in MODELS:
        d = _load("ps2", m)
        if not d:
            continue
        for lang, row in d["failure_modes"]["register_drift_bucket"]["first_turn_by_bucket"].items():
            cells = ", ".join(f"{dpd} DPD: {v['pressure']}/{v['courtesy']}/{v['words']}" for dpd, v in sorted(row.items(), key=lambda x: int(x[0])))
            print(f"- {m} {lang}: {cells}")
    print()
    for m in MODELS:
        d = _load("ps2", m)
        if d:
            print(f"- {m}: thinking leaked {d['thinking_leaked']}; errors {d['model_errors']}; TTFT {_ms(d['latency_ttft_ms'])}")
    print()


if __name__ == "__main__":
    ps3_tables()
    ps1_tables()
    ps2_tables()
