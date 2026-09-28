"""Measure a decision provider: accuracy by threshold, and latency.

Run from the repository root::

    # Local overhead only. No network, no key: the real Jev adapter and the
    # real typesafe-sdk run against an in-process transport that answers
    # instantly with each case's expected label.
    python scripts/eval_decisions.py --mock

    # The configured provider, for real. Needs DECISION_PROVIDER=jev,
    # JEV_ENABLED=true, JEV_API_KEY and JEV_MODEL in the environment or .env.
    python scripts/eval_decisions.py --live

Why this script exists
----------------------
The decision layer exists to replace slower or costlier calls, and that is a
claim about latency and accuracy that only measurement can support. The
thresholds in ``app/config.py`` are provisional; this prints the
coverage/accuracy table they should be chosen from, per decision, and the
latency percentiles to set ``JEV_TIMEOUT_SECONDS`` against.

``--live`` sends every case's utterance to the configured provider. The shipped
cases (``data/eval/decision_cases.json``) are synthetic. Do not point this at a
file of real call transcripts without the data-processing approval that sending
them to a third party requires.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evaluation.decisions import load_cases, run_cases, summarise  # noqa: E402

DEFAULT_CASES = ROOT / "data" / "eval" / "decision_cases.json"


def _mock_service(cases, timeout_seconds: float):
    """The real Jev adapter and SDK over an instant, in-process transport."""
    import httpx2

    from app.services.decision_jev import JevDecisionService
    from app.services.decisions.registry import mask_identifiers

    # Keyed as the provider sees each case: by question, and by the utterance
    # after masking.
    expected = {
        (case.decision.value, mask_identifiers(case.context.utterance)): case.expected
        for case in cases
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        key = next(iter(body["questions"]))
        label = expected[(key, body["state"]["customer_utterance"])]
        return httpx2.Response(
            200,
            json={
                "model": "mock",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "answers": {
                    key: {"type": "choice", "choice": label, "confidence": 0.99, "probabilities": {label: 0.99}}
                },
            },
        )

    return JevDecisionService(
        api_key="mock-key", model="mock", timeout_seconds=timeout_seconds, transport=httpx2.MockTransport(handler)
    )


async def _main(args: argparse.Namespace) -> dict:
    cases = load_cases(args.cases)
    if args.mock:
        service = _mock_service(cases, args.timeout)
        timeout = args.timeout
    else:
        from app.config import get_settings
        from app.runtime import build_decision_service
        from app.services.decision import DisabledDecisionService

        settings = get_settings()
        service = build_decision_service(settings)
        if isinstance(service, DisabledDecisionService):
            raise SystemExit(
                "The decision provider is disabled. Set DECISION_PROVIDER=jev and JEV_ENABLED=true "
                "(plus JEV_API_KEY and JEV_MODEL) to measure it, or use --mock."
            )
        timeout = settings.jev_timeout_seconds
    try:
        outcomes = await run_cases(
            service, cases, timeout_seconds=timeout, concurrency=args.concurrency, repeat=args.repeat
        )
    finally:
        await service.aclose()
    return {
        "mode": "mock (local overhead only; no network, answers are the expected labels)" if args.mock else "live",
        "provider": service.provider,
        "model": getattr(service, "model", None),
        "cases": len(cases),
        "repeat": args.repeat,
        "concurrency": args.concurrency,
        "results": summarise(outcomes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="measure local overhead only, offline")
    mode.add_argument("--live", action="store_true", help="call the configured provider")
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    parser.add_argument("--repeat", type=int, default=1, help="ask every case this many times")
    parser.add_argument("--concurrency", type=int, default=1, help="decisions in flight at once")
    parser.add_argument("--timeout", type=float, default=0.5, help="--mock deadline, seconds")
    print(json.dumps(asyncio.run(_main(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
