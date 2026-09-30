"""Stage 3A review: the orchestrator's and the registry's own hardening (key ORCH).

What is pinned, and why
------------------------
Four gaps the peer audit found in :mod:`app.orchestrator.pipeline` and
:mod:`app.tools.base`, none of them at the model boundary:

1. **Repeated-write authorisation (C1).** A payment promise, a dispute and an
   escalation are each gated by a state flag (``payment_promise``, ``dispute``,
   ``escalation_required``) that, once raised by an upstream NLU signal, never
   clears. Before the fix that flag alone authorised the write forever: the
   same round, a later round of the same turn, or any later turn could all
   write again from one "yes". The fix, :meth:`ConversationOrchestrator._already_written`,
   reads the session's own audit trail newest-first and refuses a write if one
   already *succeeded* since the customer's most recent signal of that kind.
   A write that failed, or came back ``NOT_IMPLEMENTED``, does not spend the
   authorisation - only a write that actually reached the ledger does - and a
   signal for one kind of write (``DISPUTE``) must not be mistaken for a fresh
   authorisation of a different kind (``PAYMENT_PROMISE``).
2. **A cut-off generation from *any* model service.** The HTTP adapter already
   refuses its own truncated replies; ``_generate`` now holds the same line
   for every :class:`~app.services.llm.LlmService`, via
   :func:`app.services.llm.incomplete_reason`. A :class:`ScriptedLlmService`
   that simply returns ``finish_reason="length"`` (or any word not on the
   allow-list) must not be able to hand the pipeline a fragment marked
   complete.
3. **A tool payload that does not fit the account context.** ``_absorb`` folds
   a successful tool result into :class:`~app.models.customer.AccountContext`
   by re-validating it. A payload the model itself cannot see is malformed -
   ``dpd=-5`` say - is now reported to the model as unusable rather than as an
   update ("FACTS is now up to date") that never happened.
4. **Nothing model-authored reaches a log line unfiltered.** An unregistered
   tool name, an argument key no tool declared, and a model tool-call id that
   is not a short plain token are all text the model copied out of the
   transcript - potentially a PAN, an Aadhaar number, a phone number. The
   registry's log line, the orchestrator's turn log line, and the audit trail
   record ``"<unregistered>"`` for the first, count-but-never-name the second,
   and use a minted id in place of the third - consistently in
   ``ToolAttempt.request_id``, the audit events, and the messages sent back to
   the model.

Every generation is scripted (:class:`ScriptedLlmService`); no network, no real
model and no wall clock is involved. ``make_runtime``, ``open_session`` and
``say`` are the harness :mod:`tests.test_orchestrator` already uses.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date

import pytest

from app.core.session import EventSignals
from app.models.customer import AccountContext
from app.models.enums import EventKind, Intent, ToolStatus
from app.orchestrator import ConversationOrchestrator, TurnErrorCategory, TurnOutcome
from app.services.llm import LlmGeneration, ScriptedLlmService
from app.tools.base import ToolNotImplemented
from tests.fakes import (
    InMemoryBankingBackend,
    sample_account,
    sample_customer,
    text_generation,
    tool_generation,
)
from tests.test_orchestrator import make_runtime, make_settings, open_session, say


def _records(stream, event: str) -> list[dict]:
    """Every JSON log line with ``"event": event``, parsed."""
    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return [line for line in lines if line.get("event") == event]


# --- 1. repeated-write authorisation ----------------------------------------


@dataclass(frozen=True)
class _WriteToolCase:
    """One of the three write tools, and what authorises and records it."""

    tool_name: str
    intent: Intent
    backend_attr: str
    needs_promise_date: bool
    base_arguments: dict = field(default_factory=dict)

    def arguments(self) -> dict:
        """Everything the model would send, minus what ``_bind`` overwrites.

        ``account_ref`` (and ``promise_date``, for a promise) is bound from
        session state regardless of what is passed here, so it is omitted.
        """
        return dict(self.base_arguments)

    def signal(self, promise_date: date | None = None) -> EventSignals:
        if self.needs_promise_date:
            return EventSignals(intent=self.intent, promise_date=promise_date or date(2026, 10, 5))
        return EventSignals(intent=self.intent)

    def written(self, backend: InMemoryBankingBackend) -> list:
        return getattr(backend, self.backend_attr)


WRITE_TOOL_CASES: tuple[_WriteToolCase, ...] = (
    _WriteToolCase(
        "record_payment_promise",
        Intent.PAYMENT_PROMISE,
        "promises",
        True,
        {"amount_minor": 250_000},
    ),
    _WriteToolCase(
        "create_dispute",
        Intent.DISPUTE,
        "disputes",
        False,
        {"reason_code": "BILLING_ERROR"},
    ),
    _WriteToolCase(
        "escalate_case",
        Intent.ESCALATION_REQUEST,
        "escalations",
        False,
        {"reason_code": "CUSTOMER_REQUEST"},
    ),
)


class _FlakyWriteBackend(InMemoryBankingBackend):
    """Fails ``method`` the first ``failures`` times it is called, then works.

    Proves that a write which never reached the ledger - because the backend
    raised, or because it is not wired up yet - does not spend the customer's
    authorisation. Nothing else about :class:`InMemoryBankingBackend` changes.
    """

    def __init__(self, *, method: str, failures: int, as_not_implemented: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._method = method
        self._remaining = failures
        self._as_not_implemented = as_not_implemented

    def _maybe_fail(self, name: str) -> None:
        if name == self._method and self._remaining > 0:
            self._remaining -= 1
            if self._as_not_implemented:
                raise ToolNotImplemented("backend maintenance window")
            raise RuntimeError("ledger temporarily unavailable")

    def record_payment_promise(self, account_ref: str, promise_date: date, amount_minor: int) -> str:
        self._maybe_fail("record_payment_promise")
        return super().record_payment_promise(account_ref, promise_date, amount_minor)

    def create_dispute(self, account_ref: str, reason_code: str) -> str:
        self._maybe_fail("create_dispute")
        return super().create_dispute(account_ref, reason_code)

    def escalate_case(self, account_ref: str, reason_code: str) -> str:
        self._maybe_fail("escalate_case")
        return super().escalate_case(account_ref, reason_code)


@pytest.mark.parametrize("case", WRITE_TOOL_CASES, ids=lambda c: c.tool_name)
def test_a_second_write_in_the_same_turn_is_refused_but_the_first_stands(case: _WriteToolCase) -> None:
    """Two rounds of one turn ask for the same write; only the first reaches the ledger."""
    backend = InMemoryBankingBackend()
    runtime = make_runtime(
        backend=backend,
        settings=make_settings(max_tool_calls_per_turn=2),
        llm=ScriptedLlmService([
            tool_generation(case.tool_name, case.arguments(), call_id="first"),
            tool_generation(case.tool_name, case.arguments(), call_id="second"),
        ]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("Yes, go ahead."), signals=case.signal()
    )

    assert [t.status for t in result.tools if t.dispatched] == [ToolStatus.OK]
    refused = [t for t in result.tools if not t.dispatched]
    assert len(refused) == 1
    assert refused[0].tool_name == case.tool_name
    assert "already recorded" in (refused[0].refusal_reason or "")
    assert TurnErrorCategory.INVALID_TOOL_REQUEST in result.error_categories
    assert result.outcome is TurnOutcome.FAILED
    assert len(case.written(backend)) == 1


@pytest.mark.parametrize("case", WRITE_TOOL_CASES, ids=lambda c: c.tool_name)
def test_repeating_a_write_on_a_later_turn_needs_a_fresh_signal(case: _WriteToolCase) -> None:
    backend = InMemoryBankingBackend()
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation(case.tool_name, case.arguments(), call_id="t1"),
            text_generation("Noted, thank you."),
            tool_generation(case.tool_name, case.arguments(), call_id="t2"),
            tool_generation(case.tool_name, case.arguments(), call_id="t3"),
            text_generation("Noted again, thank you."),
        ]),
    )
    session = open_session(runtime)
    orchestrator = ConversationOrchestrator(runtime)

    first = orchestrator.process_turn(session.session_id, say("Yes."), signals=case.signal())
    assert first.outcome is TurnOutcome.COMPLETED
    assert len(case.written(backend)) == 1

    # No new signal on this later turn: the state flag is still set (it never
    # clears), but the write already happened once, so it must be refused.
    repeat = orchestrator.process_turn(session.session_id, say("Same as before."))
    attempt = repeat.tools[0]
    assert attempt.dispatched is False
    assert "already recorded" in (attempt.refusal_reason or "")
    assert repeat.outcome is TurnOutcome.FAILED
    assert len(case.written(backend)) == 1

    # A fresh signal re-authorises exactly one more write.
    again = orchestrator.process_turn(
        session.session_id,
        say("Yes, once more."),
        signals=case.signal(promise_date=date(2026, 10, 12)),
    )
    assert again.outcome is TurnOutcome.COMPLETED
    assert len(case.written(backend)) == 2


@pytest.mark.parametrize("case", WRITE_TOOL_CASES, ids=lambda c: c.tool_name)
def test_a_write_on_the_turn_after_the_signal_is_the_legitimate_next_turn_case(
    case: _WriteToolCase,
) -> None:
    backend = InMemoryBankingBackend()
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            text_generation("Understood, one moment."),
            tool_generation(case.tool_name, case.arguments()),
            text_generation("Done, thank you."),
        ]),
    )
    session = open_session(runtime)
    orchestrator = ConversationOrchestrator(runtime)

    # Turn 1: the signal arrives, but the model asks for no tool.
    first = orchestrator.process_turn(session.session_id, say("Yes, I agree to that."), signals=case.signal())
    assert first.outcome is TurnOutcome.COMPLETED
    assert len(case.written(backend)) == 0

    # Turn 2: the write, with no fresh signal of its own, is still authorised.
    second = orchestrator.process_turn(session.session_id, say("Please go ahead."))
    assert second.tools[0].dispatched is True
    assert second.tools[0].status is ToolStatus.OK
    assert second.outcome is TurnOutcome.COMPLETED
    assert len(case.written(backend)) == 1


@pytest.mark.parametrize("case", WRITE_TOOL_CASES, ids=lambda c: c.tool_name)
def test_a_write_the_backend_rejected_does_not_consume_the_authorisation(case: _WriteToolCase) -> None:
    """A write that failed never reached the ledger, so it has not been used."""
    backend = _FlakyWriteBackend(method=case.tool_name, failures=1)
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation(case.tool_name, case.arguments(), call_id="fails"),
            tool_generation(case.tool_name, case.arguments(), call_id="retried"),
            text_generation("Done, thank you."),
        ]),
    )
    session = open_session(runtime)
    orchestrator = ConversationOrchestrator(runtime)

    first = orchestrator.process_turn(session.session_id, say("Yes."), signals=case.signal())
    assert first.tools[0].status is ToolStatus.BACKEND_ERROR
    assert first.outcome is TurnOutcome.FAILED
    assert len(case.written(backend)) == 0

    # No new signal: the failed attempt did not spend the one that already exists.
    retry = orchestrator.process_turn(session.session_id, say("Please try again."))
    assert retry.tools[0].dispatched is True
    assert retry.tools[0].status is ToolStatus.OK
    assert retry.outcome is TurnOutcome.COMPLETED
    assert len(case.written(backend)) == 1


@pytest.mark.parametrize("case", WRITE_TOOL_CASES, ids=lambda c: c.tool_name)
def test_a_write_the_backend_has_not_implemented_does_not_consume_the_authorisation(
    case: _WriteToolCase,
) -> None:
    backend = _FlakyWriteBackend(method=case.tool_name, failures=1, as_not_implemented=True)
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation(case.tool_name, case.arguments(), call_id="unavailable"),
            tool_generation(case.tool_name, case.arguments(), call_id="retried"),
            text_generation("Done, thank you."),
        ]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("Yes."), signals=case.signal()
    )

    assert [t.status for t in result.tools] == [ToolStatus.NOT_IMPLEMENTED, ToolStatus.OK]
    assert result.outcome is TurnOutcome.COMPLETED
    assert len(case.written(backend)) == 1


def test_writes_of_different_kinds_do_not_consume_each_other() -> None:
    """A DISPUTE signal must not be read as a fresh PAYMENT_PROMISE authorisation."""
    promise_case = WRITE_TOOL_CASES[0]
    dispute_case = WRITE_TOOL_CASES[1]
    backend = InMemoryBankingBackend()
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation(promise_case.tool_name, promise_case.arguments(), call_id="p1"),
            text_generation("Noted."),
            tool_generation(dispute_case.tool_name, dispute_case.arguments(), call_id="d1"),
            text_generation("Noted."),
            tool_generation(promise_case.tool_name, promise_case.arguments(), call_id="p2"),
        ]),
    )
    session = open_session(runtime)
    orchestrator = ConversationOrchestrator(runtime)

    promised = orchestrator.process_turn(
        session.session_id, say("I'll pay on the fifth."), signals=promise_case.signal()
    )
    assert promised.outcome is TurnOutcome.COMPLETED

    disputed = orchestrator.process_turn(
        session.session_id, say("Actually I dispute this."), signals=dispute_case.signal()
    )
    assert disputed.outcome is TurnOutcome.COMPLETED
    assert len(promise_case.written(backend)) == 1
    assert len(dispute_case.written(backend)) == 1

    # No fresh PAYMENT_PROMISE signal since turn 1: the dispute's own signal
    # must not be mistaken for one, so this is still refused.
    again = orchestrator.process_turn(session.session_id, say("Record my promise again."))
    assert again.tools[0].dispatched is False
    assert again.outcome is TurnOutcome.FAILED
    assert len(promise_case.written(backend)) == 1


class _SlowWriteBackend(InMemoryBankingBackend):
    """Widens the window between the authorisation check and the audit record.

    ``SessionStore.get()`` and ``append_event()`` are each locked only for
    their own instant; ``_already_written`` reads the log, the registry calls
    the backend, and the result is appended, as three separate steps. Without
    something serialising that whole sequence, two overlapping requests for
    one session could both read "not yet written" before either's result
    lands. The sleep here is what makes that window wide enough to hit
    deterministically instead of by chance.
    """

    def __init__(self, *, method: str, delay_seconds: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._method = method
        self._delay = delay_seconds

    def _maybe_delay(self, name: str) -> None:
        if name == self._method:
            time.sleep(self._delay)

    def record_payment_promise(self, account_ref: str, promise_date: date, amount_minor: int) -> str:
        self._maybe_delay("record_payment_promise")
        return super().record_payment_promise(account_ref, promise_date, amount_minor)

    def create_dispute(self, account_ref: str, reason_code: str) -> str:
        self._maybe_delay("create_dispute")
        return super().create_dispute(account_ref, reason_code)

    def escalate_case(self, account_ref: str, reason_code: str) -> str:
        self._maybe_delay("escalate_case")
        return super().escalate_case(account_ref, reason_code)


@pytest.mark.parametrize("case", WRITE_TOOL_CASES, ids=lambda c: c.tool_name)
def test_two_concurrent_turns_authorised_by_one_signal_still_write_only_once(
    case: _WriteToolCase,
) -> None:
    """The sequential re-use tests above send one round at a time; this sends two at once.

    One customer statement must authorise one write, not one per request that
    happens to race it. Two ``process_turn`` calls for the same session, on
    two threads, both asking for the same write with no new signal between
    them: exactly one may reach the backend, and the other must see the
    ordinary "already recorded" refusal, not a second, unauthorised write.
    """
    backend = _SlowWriteBackend(method=case.tool_name, delay_seconds=0.2)
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([text_generation("Noted.")]),
    )
    session = open_session(runtime)
    ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("I'll take care of it."), signals=case.signal()
    )
    assert len(case.written(backend)) == 0  # the signal alone writes nothing

    outcomes: dict[str, TurnOutcome] = {}
    barrier = threading.Barrier(2)

    def fire(name: str, call_id: str) -> None:
        orchestrator = ConversationOrchestrator(
            dataclasses.replace(
                runtime,
                llm=ScriptedLlmService([tool_generation(case.tool_name, case.arguments(), call_id=call_id)]),
            )
        )
        barrier.wait()
        outcomes[name] = orchestrator.process_turn(session.session_id, say("please go ahead")).outcome

    threads = [
        threading.Thread(target=fire, args=("first", "call-1")),
        threading.Thread(target=fire, args=("second", "call-2")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    # Exactly one write reached the ledger, whichever request happened to win.
    assert len(case.written(backend)) == 1
    # And exactly one of the two outcomes is the ordinary refusal - the write
    # was consumed once, not zero or two times.
    assert TurnOutcome.COMPLETED in outcomes.values() or TurnOutcome.FAILED in outcomes.values()


# --- 2. a cut-off generation from any LlmService, not only the HTTP adapter --


@pytest.mark.parametrize("finish_reason", ["length", "abort", "LENGTH"])
def test_any_llm_service_reporting_an_incomplete_generation_fails_the_turn(finish_reason: str) -> None:
    """A ScriptedLlmService, not only the HTTP adapter, is held to this line."""
    runtime = make_runtime(
        llm=ScriptedLlmService(
            [LlmGeneration(text="You do not need to", finish_reason=finish_reason, model="scripted")]
        )
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert result.response_text is None
    assert TurnErrorCategory.LLM_INCOMPLETE_RESPONSE in result.error_categories
    stored = runtime.sessions.get(session.session_id)
    assert [e for e in stored.events if e.kind is EventKind.AGENT_UTTERANCE] == []


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_any_llm_service_reporting_a_finished_generation_completes(finish_reason: str | None) -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService(
            [LlmGeneration(text="Understood, thank you.", finish_reason=finish_reason, model="scripted")]
        )
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True


# --- 3. a tool payload that does not fit the account context ----------------


class _NegativeDpdBackend(InMemoryBankingBackend):
    """A backend whose ``get_account`` hands back a fact the account model rejects.

    ``GetDpd``/``GetAccountStatus`` only read attributes off whatever
    ``get_account`` returns - they never validate it - so this is exactly what
    a payload that "does not fit" the account context looks like: something a
    real backend could send, that :class:`AccountContext` itself would refuse.
    """

    def get_account(self, account_ref: str) -> AccountContext:
        return super().get_account(account_ref).model_copy(update={"dpd": -5})


def test_a_tool_payload_that_does_not_fit_the_account_context_is_reported_as_unusable() -> None:
    backend = _NegativeDpdBackend(
        customers={"CUST-1": sample_customer()},
        accounts={"ACC-1": sample_account(dpd=35)},
    )

    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1"}),
            text_generation("I will check on that and call you back."),
        ]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    tool_messages = [
        m for request in runtime.llm.requests[1:] for m in request.messages if m.role == "tool"
    ]
    assert len(tool_messages) == 1
    assert "could not be used" in tool_messages[0].content
    assert "up to date" not in tool_messages[0].content

    system_message = runtime.llm.requests[-1].messages[0].content
    assert "days_past_due: 35" in system_message
    assert "days_past_due: -5" not in system_message

    assert result.outcome is TurnOutcome.COMPLETED
    assert TurnErrorCategory.TOOL_EXECUTION_FAILED in result.error_categories


def test_a_normal_tool_payload_is_absorbed_and_reported_up_to_date() -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1"}),
            text_generation("Understood."),
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    tool_messages = [
        m for request in runtime.llm.requests[1:] for m in request.messages if m.role == "tool"
    ]
    assert len(tool_messages) == 1
    assert "ok" in tool_messages[0].content
    assert "up to date" in tool_messages[0].content
    assert "could not be used" not in tool_messages[0].content
    assert result.outcome is TurnOutcome.COMPLETED
    assert TurnErrorCategory.TOOL_EXECUTION_FAILED not in result.error_categories


# --- 4. nothing model-authored reaches a log line unfiltered -----------------


@pytest.mark.parametrize(
    "tool_name",
    ["ABCDE1234F_lookup", "my PAN is ABCDE1234F"],
    ids=["adapter_shaped_name", "free_text_name"],
)
def test_an_unregistered_tool_name_that_looks_like_pii_never_reaches_a_log_line(
    tool_name: str, log_stream
) -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([tool_generation(tool_name, {"account_ref": "ACC-1"})])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.tools[0].dispatched is True
    assert result.tools[0].status is ToolStatus.NOT_FOUND
    assert result.outcome is TurnOutcome.FAILED

    logged = log_stream.getvalue()
    assert "ABCDE1234F" not in logged
    assert "PAN" not in logged

    [tool_call_record] = _records(log_stream, "tool_call")
    assert tool_call_record["tool_name"] == "<unregistered>"
    [turn_record] = _records(log_stream, "turn")
    assert turn_record["tool_name"] == "<unregistered>"


def test_invented_argument_keys_never_reach_a_log_line_but_declared_ones_do(log_stream) -> None:
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation(
                "get_account_status",
                {"account_ref": "ACC-1", "aadhaar_123412341234": "1234 5678 9012"},
            )
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.tools[0].status is ToolStatus.INVALID_REQUEST

    logged = log_stream.getvalue()
    assert "aadhaar_123412341234" not in logged
    assert "1234 5678 9012" not in logged

    [record] = _records(log_stream, "tool_call")
    assert record["tool_name"] == "get_account_status"
    assert record["argument_keys"] == ["account_ref"]
    assert record["undeclared_argument_count"] == 1


def test_an_unusable_model_call_id_never_reaches_a_log_line_and_a_minted_one_is_used_throughout(
    log_stream,
) -> None:
    bad_id = "call for 9876543210 customer"
    runtime = make_runtime(
        llm=ScriptedLlmService([
            tool_generation("get_dpd", {"account_ref": "ACC-1"}, call_id=bad_id),
            text_generation("Understood."),
        ])
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    logged = log_stream.getvalue()
    assert bad_id not in logged
    assert "9876543210" not in logged

    minted = result.tools[0].request_id
    assert minted != bad_id
    assert re.fullmatch(r"[0-9a-f]{32}", minted)

    stored = runtime.sessions.get(session.session_id)
    tool_events = [
        e for e in stored.events if e.kind in (EventKind.TOOL_CALL, EventKind.TOOL_RESULT)
    ]
    assert tool_events
    assert all(e.data.get("request_id") == minted for e in tool_events)
    assert all(e.data.get("request_id") != bad_id for e in tool_events)

    [record] = _records(log_stream, "tool_call")
    assert record["request_id"] == minted

    # On the wire: the assistant turn's own tool call, and the tool result
    # answering it, both carry the minted id - never the model's original text.
    second_request = runtime.llm.requests[1]
    assistant_messages = [
        m for m in second_request.messages if m.role == "assistant" and m.tool_calls
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].tool_calls[0].call_id == minted
    tool_messages = [m for m in second_request.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == minted
