"""Talk to the agent from a terminal, one typed turn at a time.

Text only: no STT or TTS engine is integrated, so "talking" means typing what a
borrower would say. Each turn goes through the full orchestrator (policy gates,
model, tool registry, validator, TTS normalisation) against a local model.

The account is the in-memory sample used by the test suite, and the clock is
fixed inside calling hours so the calling-hours rule does not stop the call.

    .venv/bin/python scripts/chat.py --model qwen-voice-4b --language hi-en
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402
from app.core.clock import FixedClock  # noqa: E402
from app.core.session import EventSignals  # noqa: E402
from app.models.customer import ComplianceContext  # noqa: E402
from app.models.enums import ConversationStage, EventKind, Language  # noqa: E402
from app.orchestrator import ConversationOrchestrator  # noqa: E402
from app.runtime import build_runtime  # noqa: E402
from app.services.stt import TranscriptSegment  # noqa: E402
from tests.fakes import InMemoryBankingBackend, sample_account, sample_customer  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen-voice-4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--language", default="hi-en", choices=[lang.value for lang in Language])
    parser.add_argument("--dpd", type=int, default=35, help="Days past due on the sample account.")
    parser.add_argument("--grievance-pending", action="store_true", help="Policy should stop every turn.")
    parser.add_argument("--now", default="2026-09-30T11:00", help="Fixed call time, IST.")
    args = parser.parse_args()

    language = Language(args.language)
    settings = Settings(
        _env_file=None,
        app_env="dev",
        log_level="WARNING",
        default_timezone="Asia/Kolkata",
        regulatory_rules_path=REPO_ROOT / "data" / "regulatory" / "rules" / "recovery_rules.json",
        model_base_url=args.base_url,
        model_name=args.model,
        model_reasoning_effort="none",
        model_temperature=0.0,
        model_timeout_seconds=300,
    )
    account = sample_account(dpd=args.dpd)
    compliance = ComplianceContext(grievance_pending=args.grievance_pending)
    backend = InMemoryBankingBackend(
        customers={"CUST-1": sample_customer()},
        accounts={account.account_ref: account},
        compliance={account.account_ref: compliance},
    )
    now = datetime.fromisoformat(args.now).replace(tzinfo=IST)
    runtime = build_runtime(settings, clock=FixedClock(now), backend=backend)
    orchestrator = ConversationOrchestrator(runtime)

    session = runtime.sessions.create(
        now=now,
        customer=sample_customer(),
        account=account,
        compliance=compliance,
        language=language,
    )
    started = runtime.sessions.next_event(
        session,
        kind=EventKind.SESSION_STARTED,
        now=now,
        language=language,
        data=EventSignals(
            identity_verified=True,
            disclosed_recording=True,
            identified_agent=True,
            stage=ConversationStage.ACCOUNT_DISCUSSION,
        ).model_dump(mode="json", exclude_none=True),
    )
    runtime.sessions.append_event(session.session_id, started)

    print(
        f"Model {args.model} | language {language.value} | account {account.account_ref}, "
        f"Rs {account.outstanding_minor / 100:,.2f} outstanding, {account.dpd} DPD | call time {now:%Y-%m-%d %H:%M} IST"
    )
    print("Type what the borrower says. Replies take 20-60 s on CPU. Ctrl-D or 'quit' to stop.\n")

    try:
        while True:
            try:
                text = input("borrower> ").strip()
            except EOFError:
                break
            if text.lower() in {"quit", "exit"}:
                break
            if not text:
                continue
            result = orchestrator.process_turn(
                session.session_id, TranscriptSegment(text=text, is_final=True, language=language)
            )
            seconds = result.latency.total_ms / 1000
            if result.speakable:
                print(f"agent>    {result.response_text}")
            else:
                print(f"agent>    [not spoken: {result.outcome.value}]")
                if result.draft_text:
                    print(f"          model draft: {result.draft_text}")
                for error in result.errors:
                    print(f"          {error.category.value}: {error.detail}")
            tools = ", ".join(f"{t.tool_name}({t.status.value if t.status else 'refused'})" for t in result.tools)
            print(f"          [{seconds:.1f}s | tools: {tools or 'none'}]\n")
    finally:
        close = getattr(runtime.llm, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
