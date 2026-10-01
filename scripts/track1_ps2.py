"""PS-2 command line.

    python scripts/track1_ps2.py build
    python scripts/track1_ps2.py run   --model qwen-voice-4b    # 6 calls x 8 turns
    python scripts/track1_ps2.py score --model qwen-voice-4b    # text metrics + TTS pass
    python scripts/track1_ps2.py sheet --models qwen-voice-4b qwen-voice-9b
    python scripts/track1_ps2.py agree                           # after two raters fill sheets

The TTS pass looks for a TTS/STT engine on PATH and records ``blocked`` when there is
none; it never fabricates audio.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.evaluation import ps2  # noqa: E402
from app.evaluation.challenge import CHALLENGE_VERSION, REGISTER_RUBRIC  # noqa: E402
from app.evaluation.client import (  # noqa: E402
    ChatClient,
    ModelEndpoint,
    ollama_model_digest,
    ollama_version,
)
from app.evaluation.ps3 import suite_digest  # noqa: E402

OUT = REPO / "data" / "eval" / "track1" / "ps2"
SUITE = OUT / "calls.jsonl"
HR = OUT / "human_rating"


def _load() -> list[dict]:
    return [json.loads(l) for l in SUITE.read_text(encoding="utf-8").splitlines() if l]


def cmd_build(_):
    calls = ps2.build_calls()
    OUT.mkdir(parents=True, exist_ok=True)
    with SUITE.open("w", encoding="utf-8") as fh:
        for c in calls:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    (OUT / "rubric.json").write_text(json.dumps(ps2.rubric_document(REGISTER_RUBRIC), indent=2,
                                                ensure_ascii=False))
    print(f"wrote {len(calls)} calls to {SUITE.relative_to(REPO)} sha256={suite_digest(calls)}")


def cmd_run(args):
    if not SUITE.exists():
        cmd_build(args)
    calls = _load()
    run_dir = OUT / "runs" / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    endpoint = ModelEndpoint(model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                             reasoning_effort=None if args.no_reasoning_effort else "none")
    local = "11434" in args.base_url
    manifest = run_dir / "manifest.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    data.update({
        "challenge_version": CHALLENGE_VERSION, "problem": "PS-2", "model": args.model,
        "base_url": args.base_url,
        "ollama_version": ollama_version() if local else None,
        "ollama_model": ollama_model_digest(args.model) if local else None,
        "request_settings": {"reasoning_effort": endpoint.reasoning_effort,
                             "temperature": endpoint.temperature, "seed": endpoint.seed,
                             "max_tokens": ps2.MAX_TOKENS, "tools": "none offered (spoken text only)"},
        "suite_sha256": suite_digest(calls),
        "run_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    client = ChatClient(endpoint)
    try:
        ps2.run(calls, client, run_dir / "raw.jsonl")
    finally:
        client.close()
    data["run_finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_score(args):
    run_dir = OUT / "runs" / args.model
    raw = ps2.load_raw(run_dir / "raw.jsonl")
    scored = [ps2.score_call(c, raw) for c in _load()]
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as fh:
        for c in scored:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    metrics = ps2.aggregate(scored)
    metrics["model"] = args.model
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    engines = ps2.detect_engines()
    tts = ps2.tts_pass(scored, None, None, run_dir / "audio") if not (engines["tts"] or engines["stt"]) else None
    if tts is None:
        tts = {"engines_found": engines, "status": "engine present but no adapter wired; not run"}
    (run_dir / "tts.json").write_text(json.dumps(tts, indent=2, ensure_ascii=False))
    print(json.dumps({"model": args.model, "calls": metrics["calls"], "turns": metrics["turns"],
                      "script_by_language": metrics["script_by_language"],
                      "failure_modes": {k: {kk: vv for kk, vv in v.items() if kk != "first_turn_by_bucket"}
                                        for k, v in metrics["failure_modes"].items()},
                      "tts": tts.get("status_counts", tts.get("status"))}, indent=2, ensure_ascii=False))


def cmd_sheet(args):
    by_model = {}
    for model in args.models:
        path = OUT / "runs" / model / "scores.jsonl"
        by_model[model] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    sheet, key = ps2.rating_template(by_model)
    HR.mkdir(parents=True, exist_ok=True)
    (HR / "sheet_blank.json").write_text(json.dumps(sheet, indent=2, ensure_ascii=False))
    (HR / "key.json").write_text(json.dumps(key, indent=2))
    print(f"wrote {len(sheet)} blinded calls to {(HR / 'sheet_blank.json').relative_to(REPO)}; "
          f"each rater copies it to sheet_<name>.json and fills in scores 1-5")


def cmd_agree(_):
    sheets = {p.stem.removeprefix("sheet_"): json.loads(p.read_text(encoding="utf-8"))
              for p in sorted(HR.glob("sheet_*.json")) if p.stem != "sheet_blank"}
    if len(sheets) < 2:
        print(f"{len(sheets)} filled rater sheet(s) found; inter-rater agreement needs two.")
        return
    result = ps2.rater_agreement(sheets)
    (HR / "agreement.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(func=cmd_build)
    s = sub.add_parser("run")
    s.add_argument("--model", required=True)
    s.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    s.add_argument("--api-key-env", default=None)
    s.add_argument("--no-reasoning-effort", action="store_true")
    s.set_defaults(func=cmd_run)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.set_defaults(func=cmd_score)
    s = sub.add_parser("sheet")
    s.add_argument("--models", nargs="+", required=True)
    s.set_defaults(func=cmd_sheet)
    sub.add_parser("agree").set_defaults(func=cmd_agree)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
