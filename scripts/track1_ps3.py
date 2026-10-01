"""PS-3 command line.

    python scripts/track1_ps3.py build                      # write the suite JSONL
    python scripts/track1_ps3.py run --model qwen-voice-4b   # real model calls (resumable)
    python scripts/track1_ps3.py score --model qwen-voice-4b # re-score, no model calls

A hosted baseline runs the same way against any OpenAI-compatible endpoint:

    python scripts/track1_ps3.py run --model <id> --base-url https://.../v1 \
        --api-key-env BASELINE_API_KEY --no-reasoning-effort

Challenge rule (section 8): challenge data may go to one declared hosted
baseline only, and not to any model API hosted outside India.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.evaluation import ps3  # noqa: E402
from app.evaluation.challenge import CALL_CONTEXT, CHALLENGE_VERSION  # noqa: E402
from app.evaluation.client import (  # noqa: E402
    ChatClient,
    ModelEndpoint,
    ollama_model_digest,
    ollama_version,
)
from app.evaluation.ps3_suite import build_cases  # noqa: E402

OUT = REPO / "data" / "eval" / "track1" / "ps3"
SUITE = OUT / "suite.jsonl"


def cmd_build(_: argparse.Namespace) -> None:
    cases = build_cases()
    OUT.mkdir(parents=True, exist_ok=True)
    with SUITE.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {SUITE.relative_to(REPO)} sha256={ps3.suite_digest(cases)}")


def _load() -> list[dict]:
    return [json.loads(line) for line in SUITE.read_text(encoding="utf-8").splitlines() if line]


def cmd_run(args: argparse.Namespace) -> None:
    cases = _load()
    if args.limit:
        cases = cases[: args.limit]
    endpoint = ModelEndpoint(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        reasoning_effort=None if args.no_reasoning_effort else "none",
    )
    run_dir = OUT / "runs" / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "challenge_version": CHALLENGE_VERSION,
        "problem": "PS-3",
        "model": args.model,
        "base_url": args.base_url,
        "ollama_version": ollama_version() if "11434" in args.base_url else None,
        "ollama_model": ollama_model_digest(args.model) if "11434" in args.base_url else None,
        "request_settings": {
            "reasoning_effort": endpoint.reasoning_effort,
            "temperature": endpoint.temperature,
            "seed": endpoint.seed,
            "max_tokens": ps3.MAX_TOKENS,
            "stream": True,
        },
        "call_context_message": CALL_CONTEXT,
        "suite_sha256": ps3.suite_digest(_load()),
        "suite_cases": len(_load()),
    })
    manifest.setdefault("runs", []).append(
        {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "limit": args.limit}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    client = ChatClient(endpoint)
    try:
        ps3.run(cases, client, run_dir / "raw.jsonl")
    finally:
        client.close()
    manifest["runs"][-1]["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def cmd_score(args: argparse.Namespace) -> None:
    cases = _load()
    run_dir = OUT / "runs" / args.model
    scores = ps3.score_run(cases, run_dir / "raw.jsonl")
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as fh:
        for s in scores:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    metrics = ps3.aggregate(scores)
    metrics["model"] = args.model
    metrics["scored_cases"] = len(scores)
    metrics["suite_cases"] = len(cases)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    o = metrics["overall"]
    print(json.dumps({
        "model": args.model,
        "scored": len(scores),
        "correct_tool": o["correct_tool_rate"]["rate"],
        "arg_accuracy": o["argument_accuracy"]["rate"],
        "strict": o["strict_accuracy"]["rate"],
        "spurious": o["spurious_call_rate"]["rate"],
        "missed": o["missed_call_rate"]["rate"],
        "malformed": o["malformed_argument_rate"]["rate"],
        "en_vs_hinglish_correct_tool": metrics["language_deltas"]["en_vs_hi-en"]["correct_tool"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(func=cmd_build)
    run = sub.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    run.add_argument("--api-key-env", default=None)
    run.add_argument("--no-reasoning-effort", action="store_true")
    run.add_argument("--limit", type=int, default=0)
    run.set_defaults(func=cmd_run)
    score = sub.add_parser("score")
    score.add_argument("--model", required=True)
    score.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
