"""PS-1 command line.

    python scripts/track1_ps1.py build
    python scripts/track1_ps1.py run   --model qwen-voice-4b          # agent replies
    python scripts/track1_ps1.py judge --model qwen-voice-4b --judge qwen-voice-9b
    python scripts/track1_ps1.py score --model qwen-voice-4b          # no model calls
    python scripts/track1_ps1.py sample --models qwen-voice-4b qwen-voice-9b
    python scripts/track1_ps1.py label --rater <your-name>             # a human labels
    python scripts/track1_ps1.py agree                                 # judge vs humans

A hosted baseline runs with --base-url/--api-key-env/--no-reasoning-effort, as
in track1_ps3.py. Challenge section 8 restricts which hosted API may be used.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.evaluation import ps1  # noqa: E402
from app.evaluation.challenge import CHALLENGE_VERSION, VIOLATION_TAXONOMY  # noqa: E402
from app.evaluation.client import (  # noqa: E402
    ChatClient,
    ModelEndpoint,
    ollama_model_digest,
    ollama_version,
)
from app.evaluation.ps1_suite import build_cases  # noqa: E402
from app.evaluation.ps3 import suite_digest  # noqa: E402

OUT = REPO / "data" / "eval" / "track1" / "ps1"
SUITE = OUT / "suite.jsonl"
HV = OUT / "human_validation"


def _load() -> list[dict]:
    return [json.loads(l) for l in SUITE.read_text(encoding="utf-8").splitlines() if l]


def _endpoint(args, model: str) -> ModelEndpoint:
    return ModelEndpoint(model=model, base_url=args.base_url, api_key_env=args.api_key_env,
                         reasoning_effort=None if args.no_reasoning_effort else "none")


def _manifest(run_dir: Path, update: dict) -> None:
    path = run_dir / "manifest.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(update)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_build(_):
    cases = build_cases()
    OUT.mkdir(parents=True, exist_ok=True)
    with SUITE.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    (OUT / "taxonomy.json").write_text(json.dumps(VIOLATION_TAXONOMY, indent=2, ensure_ascii=False))
    (OUT / "judge_prompt.txt").write_text(ps1.JUDGE_SYSTEM_PROMPT + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} turns to {SUITE.relative_to(REPO)} sha256={suite_digest(cases)}")


def cmd_run(args):
    cases = _load()[: args.limit or None]
    run_dir = OUT / "runs" / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    endpoint = _endpoint(args, args.model)
    local = "11434" in args.base_url
    _manifest(run_dir, {
        "challenge_version": CHALLENGE_VERSION, "problem": "PS-1", "model": args.model,
        "base_url": args.base_url,
        "ollama_version": ollama_version() if local else None,
        "ollama_model": ollama_model_digest(args.model) if local else None,
        "request_settings": {"reasoning_effort": endpoint.reasoning_effort,
                             "temperature": endpoint.temperature, "seed": endpoint.seed,
                             "max_tokens": ps1.MAX_TOKENS, "tools": "challenge 6.3 schemas"},
        "suite_sha256": suite_digest(_load()),
        "run_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    client = ChatClient(endpoint)
    try:
        ps1.run(cases, client, run_dir / "raw.jsonl")
    finally:
        client.close()
    _manifest(run_dir, {"run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def cmd_judge(args):
    cases = _load()
    run_dir = OUT / "runs" / args.model
    raw = ps1.load_raw(run_dir / "raw.jsonl")
    local = "11434" in args.base_url
    _manifest(run_dir, {
        "judge_model": args.judge,
        "judge_ollama_model": ollama_model_digest(args.judge) if local else None,
        "judge_prompt_version": ps1.JUDGE_PROMPT_VERSION,
        "judge_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    client = ChatClient(_endpoint(args, args.judge))
    try:
        ps1.judge(cases, raw, client, run_dir / f"judged_by_{args.judge}.jsonl")
    finally:
        client.close()
    _manifest(run_dir, {"judge_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def _judged(run_dir: Path, judge: str) -> dict:
    path = run_dir / f"judged_by_{judge}.jsonl"
    if not path.exists():
        return {}
    return {r["case_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l)}


def cmd_score(args):
    cases = _load()
    run_dir = OUT / "runs" / args.model
    rows = ps1.score(cases, ps1.load_raw(run_dir / "raw.jsonl"), _judged(run_dir, args.judge))
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = ps1.aggregate(rows)
    metrics.update({"model": args.model, "judge_model": args.judge,
                    "judge_prompt_version": ps1.JUDGE_PROMPT_VERSION})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    o = metrics["overall"]
    print(json.dumps({"model": args.model, "turns": o["turns"], "judged": o["judged"],
                      "violation_rate": o["violation_rate"]["rate"],
                      "attack_success": o["attack_success_rate"]["rate"],
                      "mark_dispute_compliance": o["mark_dispute_compliance"]["rate"],
                      "escalate_compliance": o["escalate_human_compliance"]["rate"],
                      "en_vs_indic": {k: v for k, v in metrics["english_vs_indic"].items() if k != "paired"}},
                     indent=2))


def cmd_sample(args):
    cases = {c["id"]: c for c in _load()}
    rows = {}
    for model in args.models:
        path = OUT / "runs" / model / "scores.jsonl"
        rows[model] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    items, key = ps1.human_sample(rows, cases, per_model=args.per_model)
    HV.mkdir(parents=True, exist_ok=True)
    with (HV / "sample_blind.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    (HV / "key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False))
    print(f"wrote {len(items)} blinded items to {(HV / 'sample_blind.jsonl').relative_to(REPO)}")


def cmd_label(args):
    """Interactive labelling for a human rater. Resumable; one line per item."""
    items = [json.loads(l) for l in (HV / "sample_blind.jsonl").read_text(encoding="utf-8").splitlines() if l]
    out = HV / f"labels_{args.rater}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["sample_id"] for l in out.read_text(encoding="utf-8").splitlines() if l}
    print("Taxonomy:")
    for v in VIOLATION_TAXONOMY:
        print(f"  {v['code']} {v['category']}: {v['boundary']}\n")
    with out.open("a", encoding="utf-8") as fh:
        for item in items:
            if item["sample_id"] in done:
                continue
            print("=" * 78)
            print(f"{item['sample_id']}  language={item['language']}  borrower={item['borrower_on_file']}")
            for turn in item["conversation"]:
                print(f"  [{'AGENT' if turn['role'] == 'assistant' else 'CALLER'}] {turn['content']}")
            print(f"  [AGENT REPLY] {item['agent_reply']}")
            print(f"  [TOOL CALLS] {item['agent_tool_calls']}")
            answer = input("Violations (e.g. V1,V4; blank = none; q = quit): ").strip()
            if answer.lower() == "q":
                break
            codes = sorted({c.strip().upper() for c in answer.split(",") if c.strip()})
            fh.write(json.dumps({"sample_id": item["sample_id"], "violations": codes,
                                 "rater": args.rater}) + "\n")
            fh.flush()


def cmd_agree(_):
    key = json.loads((HV / "key.json").read_text())
    files = {p.stem.removeprefix("labels_"): p for p in sorted(HV.glob("labels_*.jsonl"))}
    if not files:
        print("No human labels yet (data/eval/track1/ps1/human_validation/labels_<rater>.jsonl). "
              "Agreement cannot be computed without them.")
        return
    result = ps1.agreement(key, files)
    (HV / "agreement.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(func=cmd_build)
    for name, func in (("run", cmd_run), ("judge", cmd_judge)):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        if name == "judge":
            s.add_argument("--judge", required=True)
        s.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
        s.add_argument("--api-key-env", default=None)
        s.add_argument("--no-reasoning-effort", action="store_true")
        s.add_argument("--limit", type=int, default=0)
        s.set_defaults(func=func)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--judge", default="qwen-voice-9b")
    s.set_defaults(func=cmd_score)
    s = sub.add_parser("sample")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--per-model", type=int, default=24)
    s.set_defaults(func=cmd_sample)
    s = sub.add_parser("label")
    s.add_argument("--rater", required=True)
    s.set_defaults(func=cmd_label)
    sub.add_parser("agree").set_defaults(func=cmd_agree)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
