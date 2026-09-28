"""Stage 3A, the tool loop: every request follows the chat-completions tool-call contract.

What is pinned, and why
-----------------------
After a tool round the orchestrator calls the model again. The chat-completions
contract says what that request must look like. The model's own tool-requesting
turn goes back first, as an assistant message carrying its ``tool_calls``. After
it comes one ``tool`` message per call, each answering one of that message's
ids. Before the fix the assistant turn went back as ``{"content": ""}`` with no
calls, so every ``tool`` message answered a call that was not on record. A
server that checks the contract, OpenAI's included, answers that request with
HTTP 400, and the turn ended as an upstream error on every round that used a
tool.

``_StrictEndpoint`` below is stricter than the ``_Endpoint`` double in
``test_llm_http_integration``. It checks every request's ``messages`` against
the contract and answers 400 when a rule is broken. A turn that completes here
has completed against a server that checks.

The fix must also keep these properties:

- call ids are unique across the whole turn, whatever the server sent (no id,
  a repeated id, or index-style ``call_0`` restarting every round), and the id
  on the wire is the id in the audit trail;
- the echoed arguments leave out what the orchestrator binds itself
  (``account_ref``, ``customer_ref``, ``promise_date``). The echo can't tell the
  model a reference, and it can't confirm one the model made up;
- a round that fails ends the turn before another request could carry an
  unanswered id;
- tools still run only through the registry. The HTTP adapter and the echo
  don't run them.

No model, network or clock is involved. The endpoint is an
``httpx2.MockTransport`` handler in this process.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, NoReturn

import httpx2
import pytest
from pydantic import ValidationError

from app.core.session import EventSignals
from app.models.customer import ComplianceContext
from app.models.enums import Intent, ToolStatus
from app.orchestrator import ConversationOrchestrator, TurnErrorCategory, TurnOutcome, TurnResult
from app.services.llm import LlmMessage, LlmRequest, LlmToolCall, LlmUpstreamError
from app.services.llm_openai import _render_message
from app.tools.banking import NullBankingBackend
from tests.fakes import (
    InMemoryBankingBackend,
    openai_text_completion,
    openai_tool_completion,
    sample_account,
    sample_customer,
)
from tests.test_llm_http_integration import http_llm
from tests.test_orchestrator import GROUNDED_REPLY, make_settings, make_runtime, open_session, say

#: Must never appear in any request of any turn: the session's account and
#: customer references, the borrower's name, and the outstanding amount in paise.
_NEVER_ON_THE_WIRE = ("ACC-1", "CUST-1", "Test Borrower", "1234500")

_PROMISE = EventSignals(intent=Intent.PAYMENT_PROMISE, promise_date=date(2026, 10, 5))


# --- the strict endpoint ------------------------------------------------------

_ROLES = frozenset({"system", "user", "assistant", "tool"})
_MESSAGE_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})
_TOOL_CALL_KEYS = frozenset({"id", "type", "function"})
_FUNCTION_KEYS = frozenset({"name", "arguments"})


def _strict_json(text: str) -> Any:
    """Decode ``text`` as JSON. ``NaN``, ``Infinity`` and an overflowing number are refused.

    Python's parser accepts all three, and JSON has none of them.
    """

    def refuse(token: str) -> NoReturn:
        raise ValueError(f"{token} is not a JSON number")

    def finite(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"{token} is not a finite number")
        return value

    return json.loads(text, parse_constant=refuse, parse_float=finite)


def _tool_call_violation(call: Any) -> str | None:
    """Rule r3, for one ``tool_calls`` entry."""
    if not isinstance(call, dict):
        return "a tool_calls entry is not an object"
    if set(call) - _TOOL_CALL_KEYS:
        return f"a tool_calls entry has unknown key(s) {sorted(set(call) - _TOOL_CALL_KEYS)}"
    if not isinstance(call.get("id"), str) or not call["id"]:
        return "a tool_calls entry has no id"
    if call.get("type") != "function":
        return "a tool_calls entry is not of type 'function'"
    function = call.get("function")
    if not isinstance(function, dict):
        return "a tool_calls entry has no function object"
    if set(function) - _FUNCTION_KEYS:
        return f"a function has unknown key(s) {sorted(set(function) - _FUNCTION_KEYS)}"
    if not isinstance(function.get("name"), str) or not function["name"]:
        return "a function has no name"
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        return "function.arguments is not a string"
    try:
        decoded = _strict_json(arguments)
    except ValueError:
        return "function.arguments is not JSON"
    if not isinstance(decoded, dict):
        return "function.arguments is not a JSON object"
    return None


def _protocol_violation(messages: Any) -> str | None:
    """The first tool-call contract rule ``messages`` breaks, or ``None``.

    r1  a tool message follows an assistant message with tool_calls, with only
        other tool messages in between, and answers one of its unanswered ids;
    r2  every id of such a message is answered before the next non-tool message
        and before the array ends;
    r3  each tool_calls entry has a non-empty string id, type "function", a
        function name, and arguments that are a JSON string decoding to an
        object; ids are unique within the message; tool_calls is never empty;
    r4  an assistant message with tool_calls has null or string content;
    r5  a message has no key but role, content, tool_calls, tool_call_id, name.
    """
    if not isinstance(messages, list) or not messages:
        return "messages is not a non-empty array"
    # The ids the most recent assistant tool_calls message is still waiting on.
    # None once any other message intervenes: no tool message may follow then.
    open_ids: set[str] | None = None
    for index, message in enumerate(messages):
        where = f"messages[{index}]"
        if not isinstance(message, dict):
            return f"{where} is not an object"
        unknown = set(message) - _MESSAGE_KEYS
        if unknown:
            return f"{where} has unknown key(s) {sorted(unknown)}"  # r5
        role = message.get("role")
        if role not in _ROLES:
            return f"{where} has no valid role"
        content = message.get("content")

        if role == "tool":
            if open_ids is None:
                return f"{where} is a tool message that follows no assistant tool_calls"  # r1
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in open_ids:
                return (
                    f"{where} answers {call_id!r}, which is not an unanswered call "
                    "of the preceding assistant message"
                )  # r1
            open_ids.discard(call_id)
            if "tool_calls" in message:
                return f"{where} is a tool message carrying tool_calls"
            if not isinstance(content, str):
                return f"{where} content is not a string"
            continue

        if open_ids:
            return f"{where} arrives while tool call(s) {sorted(open_ids)} are unanswered"  # r2
        open_ids = None
        if "tool_call_id" in message:
            return f"{where} carries tool_call_id but is not a tool message"
        if "tool_calls" in message:
            if role != "assistant":
                return f"{where} carries tool_calls but is not an assistant message"
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not calls:
                return f"{where} has an empty or non-array tool_calls"  # r3
            if content is not None and not isinstance(content, str):
                return f"{where} content is neither null nor a string"  # r4
            for call in calls:
                problem = _tool_call_violation(call)
                if problem is not None:
                    return f"{where}: {problem}"  # r3
            ids = [call["id"] for call in calls]
            if len(set(ids)) != len(ids):
                return f"{where} repeats a tool call id"  # r3
            open_ids = set(ids)
        elif not isinstance(content, str):
            return f"{where} content is not a string"

    if open_ids:
        return f"tool call(s) {sorted(open_ids)} are never answered"  # r2
    return None


class _StrictEndpoint:
    """An OpenAI-compatible endpoint that enforces the tool-call message contract.

    Scripted like ``_Endpoint``: each accepted request consumes the next
    response. A request that breaks the contract consumes nothing. It gets a 400
    with an OpenAI-style ``{"error": ...}`` body, as a server that validates
    would send. Every request body is recorded, rejected or not.
    """

    def __init__(self, *responses: dict) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.rejections: list[str] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        request.read()
        try:
            body = _strict_json(request.content.decode("utf-8"))
        except ValueError:
            return self._reject("the request body is not JSON")
        self.requests.append(body)
        problem = _protocol_violation(body.get("messages") if isinstance(body, dict) else None)
        if problem is not None:
            return self._reject(problem)
        if not self._responses:
            raise AssertionError("the orchestrator made more model calls than were scripted")
        return httpx2.Response(200, json=self._responses.pop(0))

    def _reject(self, problem: str) -> httpx2.Response:
        self.rejections.append(problem)
        return httpx2.Response(
            400,
            json={"error": {"message": problem, "type": "invalid_request_error", "code": None}},
        )


# --- harness ------------------------------------------------------------------


class _CountingBackend:
    """Any banking backend, recording every call that reaches it."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._inner, name)
        if not callable(target):
            return target

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args))
            return target(*args, **kwargs)

        return counted


class _ExplodingBackend(InMemoryBankingBackend):
    """A backend whose account lookup fails the way a database outage does."""

    def get_account(self, account_ref: str):
        raise RuntimeError("postgres://user:hunter2@db is unreachable")


def _sample_backend() -> InMemoryBankingBackend:
    return InMemoryBankingBackend(
        customers={"CUST-1": sample_customer()},
        accounts={"ACC-1": sample_account()},
        compliance={"ACC-1": ComplianceContext(grievance_pending=False)},
    )


@dataclass
class _Turn:
    result: TurnResult
    endpoint: _StrictEndpoint
    backend: _CountingBackend

    def messages(self, request: int) -> list[dict]:
        return self.endpoint.requests[request]["messages"]


def _run_turn(
    endpoint: _StrictEndpoint,
    *,
    backend: object | None = None,
    settings=None,
    text: str = "How much do I have to pay?",
    signals: EventSignals | None = None,
) -> _Turn:
    """Run one turn through the orchestrator and the real HTTP adapter, against ``endpoint``.

    Also checks what must hold for every turn, whatever it did:

    - the strict endpoint accepted every request the turn sent;
    - no request carried a value from ``_NEVER_ON_THE_WIRE``;
    - the registry ran every tool the model asked for, and nothing else did.
      Registry executions, dispatched attempts and backend calls all agree.
    """
    counting = _CountingBackend(backend if backend is not None else _sample_backend())
    runtime = make_runtime(llm=http_llm(endpoint), backend=counting, settings=settings)
    executed: list[str] = []
    registry_execute = runtime.tools.execute

    def execute(request):
        executed.append(request.request_id)
        return registry_execute(request)

    runtime.tools.execute = execute
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say(text), signals=signals
    )

    assert endpoint.rejections == [], f"a request broke the protocol: {endpoint.rejections}"
    for index, body in enumerate(endpoint.requests):
        wire = json.dumps(body)
        for secret in _NEVER_ON_THE_WIRE:
            assert secret not in wire, f"request {index} carries {secret!r}"
    dispatched = [attempt.request_id for attempt in result.tools if attempt.dispatched]
    assert executed == dispatched
    assert len(counting.calls) == len(dispatched)
    return _Turn(result, endpoint, counting)


def _roles(messages: list[dict]) -> list[str]:
    return [message["role"] for message in messages]


def _echoed_arguments(call: dict) -> dict:
    arguments = call["function"]["arguments"]
    assert isinstance(arguments, str)
    return json.loads(arguments)


def _without_ids(body: dict) -> dict:
    """A tool-call completion from a server that sends no call ids."""
    for call in body["choices"][0]["message"]["tool_calls"]:
        del call["id"]
    return body


# --- 1. the strict endpoint, checked on its own -------------------------------
#
# The endpoint is what the turn tests below rely on. These tests show it
# rejects what it should, so a turn that passes it has passed a real check.


def _call(call_id: Any = "call_a", name: Any = "get_dpd", arguments: Any = "{}", **extra) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
        **extra,
    }


def _exchange(*calls: Any, content: Any = None, answers: tuple | None = None) -> list[dict]:
    """system, user, an assistant turn carrying ``calls``, then one tool result per answer."""
    if answers is None:
        answers = tuple(call["id"] for call in calls if isinstance(call, dict))
    return [
        {"role": "system", "content": "CONSTRAINTS"},
        {"role": "user", "content": "how much do I owe?"},
        {"role": "assistant", "content": content, "tool_calls": list(calls)},
        *({"role": "tool", "tool_call_id": answer, "content": "ok."} for answer in answers),
    ]


#: The second request as the orchestrator sent it before the fix.
_PRE_FIX_SHAPE = [
    {"role": "system", "content": "CONSTRAINTS"},
    {"role": "user", "content": "how much do I owe?"},
    {"role": "assistant", "content": ""},
    {"role": "tool", "tool_call_id": "call_0", "content": "get_outstanding_amount ok."},
]


def test_the_strict_endpoint_accepts_a_well_formed_tool_exchange() -> None:
    assert _protocol_violation(_exchange(_call("call_a"), _call("call_b"))) is None
    assert _protocol_violation(_exchange(_call("call_a"), content="Let me check.")) is None
    assert _protocol_violation(_exchange(_call(arguments='{"amount_minor": 250000}'))) is None


def test_the_strict_endpoint_rejects_the_pre_fix_shape() -> None:
    """A tool result after an assistant turn that made no call answers nothing on record."""
    assert "follows no assistant tool_calls" in (_protocol_violation(_PRE_FIX_SHAPE) or "")


def test_the_pre_fix_shape_comes_back_from_the_endpoint_as_http_400() -> None:
    endpoint = _StrictEndpoint(openai_text_completion("never sent"))
    with httpx2.Client(transport=httpx2.MockTransport(endpoint)) as client:
        response = client.post(
            "http://model.invalid:8000/v1/chat/completions",
            json={"model": "m", "messages": _PRE_FIX_SHAPE},
        )

    assert response.status_code == 400
    assert "follows no assistant tool_calls" in response.json()["error"]["message"]


def test_through_the_adapter_the_pre_fix_shape_is_an_upstream_400() -> None:
    """This is how the pre-fix orchestrator failed every tool round against such a server."""
    endpoint = _StrictEndpoint(openai_text_completion("never sent"))
    request = LlmRequest(
        messages=(
            LlmMessage(role="system", content="CONSTRAINTS"),
            LlmMessage(role="user", content="how much do I owe?"),
            LlmMessage(role="assistant", content=""),
            LlmMessage(role="tool", content="get_outstanding_amount ok.", tool_call_id="call_0"),
        )
    )

    with pytest.raises(LlmUpstreamError) as raised:
        http_llm(endpoint).generate(request)

    assert raised.value.status_code == 400
    assert len(endpoint.rejections) == 1


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(
            _exchange(_call("call_a"), _call("call_b"), answers=("call_a",)),
            "never answered",
            id="r2-an-id-never-answered",
        ),
        pytest.param(
            [
                *_exchange(_call("call_a"), answers=()),
                {"role": "user", "content": "hello?"},
            ],
            "unanswered",
            id="r2-a-user-message-before-the-answer",
        ),
        pytest.param(
            _exchange(_call("call_a"), answers=("call_zzz",)),
            "not an unanswered call",
            id="r1-an-answer-to-an-unknown-id",
        ),
        pytest.param(
            _exchange(_call("call_a"), answers=("call_a", "call_a")),
            "not an unanswered call",
            id="r1-the-same-id-answered-twice",
        ),
    ],
)
def test_the_strict_endpoint_rejects_an_unanswered_or_misanswered_call(
    messages: list[dict], expected: str
) -> None:
    assert expected in (_protocol_violation(messages) or "")


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(_exchange(), "empty or non-array tool_calls", id="r3-empty-tool-calls"),
        pytest.param(_exchange(_call("")), "has no id", id="r3-empty-id"),
        pytest.param(_exchange(_call(None), answers=()), "has no id", id="r3-null-id"),
        pytest.param(
            _exchange(_call("call_a"), _call("call_a")), "repeats a tool call id", id="r3-dup-ids"
        ),
        pytest.param(
            _exchange({**_call(), "type": "tool"}), "not of type 'function'", id="r3-wrong-type"
        ),
        pytest.param(_exchange(_call(name=None)), "has no name", id="r3-no-name"),
        pytest.param(
            _exchange(_call(arguments={"amount_minor": 1})), "not a string", id="r3-args-object"
        ),
        pytest.param(_exchange(_call(arguments="{broken")), "not JSON", id="r3-args-not-json"),
        pytest.param(
            _exchange(_call(arguments="[1, 2]")), "not a JSON object", id="r3-args-array"
        ),
        pytest.param(
            _exchange(_call(arguments='{"x": NaN}')), "not JSON", id="r3-args-nan"
        ),
        pytest.param(
            _exchange(_call(arguments='{"x": Infinity}')), "not JSON", id="r3-args-infinity"
        ),
        pytest.param(
            _exchange(_call(arguments='{"x": -Infinity}')), "not JSON", id="r3-args-minus-inf"
        ),
        pytest.param(
            _exchange(_call(arguments='{"x": 1e400}')), "not JSON", id="r3-args-overflow"
        ),
        pytest.param(
            _exchange(_call(), content=0), "neither null nor a string", id="r4-numeric-content"
        ),
        pytest.param(
            [{"role": "system", "content": "C", "metadata": {}}, {"role": "user", "content": "u"}],
            "unknown key",
            id="r5-unknown-message-key",
        ),
        pytest.param(
            _exchange(_call(extra_field=1)), "unknown key", id="r3-unknown-tool-call-key"
        ),
        pytest.param(
            [
                {"role": "system", "content": "C"},
                {"role": "user", "content": "u", "tool_calls": [_call()]},
            ],
            "not an assistant message",
            id="tool-calls-on-a-user-message",
        ),
    ],
)
def test_the_strict_endpoint_enforces_every_shape_rule(messages: list[dict], expected: str) -> None:
    assert expected in (_protocol_violation(messages) or "")


# --- 2. a plain reply is untouched --------------------------------------------


def test_a_plain_reply_sends_one_request_of_system_and_user_only() -> None:
    endpoint = _StrictEndpoint(openai_text_completion(GROUNDED_REPLY))

    turn = _run_turn(endpoint)

    assert turn.result.outcome is TurnOutcome.COMPLETED
    assert len(endpoint.requests) == 1
    assert _roles(turn.messages(0)) == ["system", "user"]
    assert all(set(message) == {"role", "content"} for message in turn.messages(0))


# --- 3. tool rounds, against a server that checks -----------------------------


def test_one_tool_call_is_echoed_before_its_result_and_the_turn_completes() -> None:
    endpoint = _StrictEndpoint(
        openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-1"})),
        openai_text_completion(GROUNDED_REPLY),
    )

    turn = _run_turn(endpoint)

    assert turn.result.outcome is TurnOutcome.COMPLETED
    assert turn.result.speakable is True
    assert turn.result.llm_calls == 2
    assert len(endpoint.requests) == 2
    second = turn.messages(1)
    assert _roles(second) == ["system", "user", "assistant", "tool"]
    assert second[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_outstanding_amount", "arguments": "{}"},
            }
        ],
    }
    assert second[3]["tool_call_id"] == "call_0"
    # The id the model is answered with is the id in the audit trail.
    assert [attempt.request_id for attempt in turn.result.tools] == ["call_0"]


def test_two_calls_in_one_round_are_echoed_together_and_answered_in_order() -> None:
    endpoint = _StrictEndpoint(
        openai_tool_completion(
            ("get_outstanding_amount", {"account_ref": "ACC-1"}),
            ("get_account_status", {"account_ref": "ACC-1"}),
        ),
        openai_text_completion(GROUNDED_REPLY),
    )

    turn = _run_turn(endpoint)

    assert turn.result.outcome is TurnOutcome.COMPLETED
    assert [attempt.status for attempt in turn.result.tools] == [ToolStatus.OK, ToolStatus.OK]
    second = turn.messages(1)
    assert _roles(second) == ["system", "user", "assistant", "tool", "tool"]
    echoed = [(call["id"], call["function"]["name"]) for call in second[2]["tool_calls"]]
    assert echoed == [("call_0", "get_outstanding_amount"), ("call_1", "get_account_status")]
    assert [message["tool_call_id"] for message in second[3:]] == ["call_0", "call_1"]
    assert [attempt.request_id for attempt in turn.result.tools] == ["call_0", "call_1"]


def test_an_index_style_id_reused_in_the_next_round_is_replaced_so_ids_stay_unique_in_the_turn() -> None:
    """Some servers number calls per response, so every round starts again at ``call_0``."""
    endpoint = _StrictEndpoint(
        openai_tool_completion(("get_dpd", {"account_ref": "ACC-1"})),
        openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-1"})),
        openai_text_completion(GROUNDED_REPLY),
    )

    turn = _run_turn(endpoint, settings=make_settings(max_tool_calls_per_turn=2))

    assert turn.result.outcome is TurnOutcome.COMPLETED
    assert len(endpoint.requests) == 3
    final = turn.messages(2)
    assert _roles(final) == ["system", "user", "assistant", "tool", "assistant", "tool"]
    first_round, second_round = final[2]["tool_calls"], final[4]["tool_calls"]
    assert [call["function"]["name"] for call in first_round] == ["get_dpd"]
    assert [call["function"]["name"] for call in second_round] == ["get_outstanding_amount"]

    first_id, second_id = first_round[0]["id"], second_round[0]["id"]
    assert first_id == "call_0"  # a usable model id is kept
    assert second_id and second_id != first_id  # a reused one is not
    # Each result answers its own round, not the other round's call.
    assert final[3]["tool_call_id"] == first_id
    assert final[5]["tool_call_id"] == second_id
    # Ids are settled once, when the round runs. The later request carries the
    # first round exactly as the earlier one did.
    assert turn.messages(1)[2:] == final[2:4]
    assert [attempt.request_id for attempt in turn.result.tools] == [first_id, second_id]


@pytest.mark.parametrize(
    "calls",
    [
        pytest.param((("get_outstanding_amount", {"account_ref": "ACC-1"}),), id="one-call"),
        pytest.param(
            (
                ("get_outstanding_amount", {"account_ref": "ACC-1"}),
                ("get_account_status", {"account_ref": "ACC-1"}),
            ),
            id="two-calls",
        ),
    ],
)
def test_a_call_sent_without_an_id_gets_one_that_its_echo_and_its_result_share(calls) -> None:
    endpoint = _StrictEndpoint(
        _without_ids(openai_tool_completion(*calls)),
        openai_text_completion(GROUNDED_REPLY),
    )

    turn = _run_turn(endpoint)

    assert turn.result.outcome is TurnOutcome.COMPLETED
    second = turn.messages(1)
    minted = [call["id"] for call in second[2]["tool_calls"]]
    assert len(minted) == len(calls)
    assert all(isinstance(call_id, str) and call_id for call_id in minted)
    assert len(set(minted)) == len(minted)
    assert [message["tool_call_id"] for message in second[3:]] == minted
    assert [attempt.request_id for attempt in turn.result.tools] == minted


def test_duplicate_ids_within_one_round_are_made_unique_before_they_go_back() -> None:
    endpoint = _StrictEndpoint(
        openai_tool_completion(
            ("get_outstanding_amount", {"account_ref": "ACC-1"}),
            ("get_account_status", {"account_ref": "ACC-1"}),
            call_ids=("dup", "dup"),
        ),
        openai_text_completion(GROUNDED_REPLY),
    )

    turn = _run_turn(endpoint)

    assert turn.result.outcome is TurnOutcome.COMPLETED
    second = turn.messages(1)
    ids = [call["id"] for call in second[2]["tool_calls"]]
    assert ids[0] == "dup"
    assert ids[1] and ids[1] != "dup"
    assert [message["tool_call_id"] for message in second[3:]] == ids
    assert [attempt.request_id for attempt in turn.result.tools] == ids


@pytest.mark.parametrize(
    ("spoken", "echoed"),
    [
        pytest.param("Let me check that for you.", "Let me check that for you.", id="text"),
        pytest.param(None, None, id="null"),
        pytest.param("", None, id="empty"),
        pytest.param("   ", None, id="whitespace"),
    ],
)
def test_what_the_model_said_with_its_tool_call_goes_back_as_the_assistant_content(
    spoken: str | None, echoed: str | None
) -> None:
    endpoint = _StrictEndpoint(
        openai_tool_completion(
            ("get_outstanding_amount", {"account_ref": "ACC-1"}), content=spoken
        ),
        openai_text_completion(GROUNDED_REPLY),
    )

    turn = _run_turn(endpoint)

    assert turn.result.outcome is TurnOutcome.COMPLETED
    assistant = turn.messages(1)[2]
    assert "content" in assistant
    assert assistant["content"] == echoed


# --- 4. a round that fails ends the turn before the next request --------------


def test_a_tool_call_whose_arguments_are_not_json_fails_the_turn_on_the_first_request() -> None:
    body = openai_tool_completion(("get_dpd", {}))
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{broken"
    endpoint = _StrictEndpoint(body, openai_text_completion("never requested"))

    turn = _run_turn(endpoint)

    assert turn.result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.LLM_INVALID_TOOL_CALL in turn.result.error_categories
    assert len(endpoint.requests) == 1
    assert turn.result.tools == ()


@pytest.mark.parametrize(
    "calls",
    [
        pytest.param((("get_account_status", {"account_ref": "ACC-1"}),), id="alone"),
        pytest.param(
            (
                ("get_account_status", {"account_ref": "ACC-1"}),
                ("get_customer_context", {"customer_ref": "CUST-1"}),
            ),
            id="first-of-two",
        ),
    ],
)
def test_a_tool_whose_backend_raises_ends_the_turn_before_another_request(calls) -> None:
    """With two calls, the second is never run. Another request would carry its id unanswered."""
    endpoint = _StrictEndpoint(
        openai_tool_completion(*calls), openai_text_completion("never requested")
    )
    backend = _ExplodingBackend(
        customers={"CUST-1": sample_customer()}, accounts={"ACC-1": sample_account()}
    )

    turn = _run_turn(endpoint, backend=backend)

    assert turn.result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.TOOL_EXECUTION_FAILED in turn.result.error_categories
    assert [attempt.status for attempt in turn.result.tools] == [ToolStatus.BACKEND_ERROR]
    assert len(endpoint.requests) == 1


@pytest.mark.parametrize(
    ("calls", "signals", "refusal"),
    [
        pytest.param(
            (
                (
                    "record_payment_promise",
                    {"account_ref": "ACC-1", "promise_date": "2026-10-05", "amount_minor": 250000},
                ),
            ),
            None,
            "has not made a payment promise",
            id="no-promise-in-state",
        ),
        pytest.param(
            (
                ("get_account_status", {"account_ref": "ACC-1"}),
                ("record_payment_promise", {"amount_minor": 250000}),
            ),
            _PROMISE,
            "only tool call in its round",
            id="write-beside-a-read",
        ),
    ],
)
def test_a_refused_write_ends_the_turn_without_a_second_request(calls, signals, refusal) -> None:
    backend = _sample_backend()
    endpoint = _StrictEndpoint(
        openai_tool_completion(*calls), openai_text_completion("never requested")
    )

    turn = _run_turn(endpoint, backend=backend, signals=signals)

    assert turn.result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.INVALID_TOOL_REQUEST in turn.result.error_categories
    refused = [attempt for attempt in turn.result.tools if not attempt.dispatched]
    assert [attempt.tool_name for attempt in refused] == ["record_payment_promise"]
    assert refusal in (refused[0].refusal_reason or "")
    assert backend.promises == []
    assert len(endpoint.requests) == 1


# --- 5. a tool with nothing behind it keeps the turn going --------------------


@pytest.mark.parametrize(
    "backend_factory",
    [
        pytest.param(NullBankingBackend, id="null-backend"),
        pytest.param(InMemoryBankingBackend, id="unknown-account"),
    ],
)
def test_a_tool_with_nothing_behind_it_is_answered_and_the_next_request_is_valid(
    backend_factory,
) -> None:
    endpoint = _StrictEndpoint(
        openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-1"})),
        openai_text_completion("Let me check that and call you back."),
    )

    turn = _run_turn(endpoint, backend=backend_factory())

    assert turn.result.tools[0].status is ToolStatus.NOT_IMPLEMENTED
    assert TurnErrorCategory.TOOL_BACKEND_NOT_IMPLEMENTED in turn.result.error_categories
    assert turn.result.outcome is TurnOutcome.COMPLETED
    assert len(endpoint.requests) == 2
    second = turn.messages(1)
    assert _roles(second) == ["system", "user", "assistant", "tool"]
    assert second[3]["tool_call_id"] == second[2]["tool_calls"][0]["id"]
    assert "not_implemented" in second[3]["content"]


def test_a_round_mixing_an_unavailable_tool_and_a_working_one_answers_both() -> None:
    # Knows the account but not the customer, so one call works and one cannot.
    backend = InMemoryBankingBackend(accounts={"ACC-1": sample_account()})
    endpoint = _StrictEndpoint(
        openai_tool_completion(
            ("get_customer_context", {"customer_ref": "CUST-1"}),
            ("get_outstanding_amount", {"account_ref": "ACC-1"}),
        ),
        openai_text_completion(GROUNDED_REPLY),
    )

    turn = _run_turn(endpoint, backend=backend)

    assert [attempt.status for attempt in turn.result.tools] == [
        ToolStatus.NOT_IMPLEMENTED,
        ToolStatus.OK,
    ]
    assert turn.result.outcome is TurnOutcome.COMPLETED
    second = turn.messages(1)
    assert _roles(second) == ["system", "user", "assistant", "tool", "tool"]
    assert [message["tool_call_id"] for message in second[3:]] == ["call_0", "call_1"]


# --- 6. the echo tells the model nothing it was not already told --------------
#
# ``_run_turn`` already checks, for every request of every turn above, that no
# account or customer reference, borrower name or paise figure is on the wire.
# In most of those turns the model named ``ACC-1`` itself, so that check also
# covers what is stripped from the echo. The tests below cover the rounds where
# a leak would do the most harm.


def test_a_customer_context_round_sends_the_model_no_customer_reference_or_name() -> None:
    endpoint = _StrictEndpoint(
        openai_tool_completion(("get_customer_context", {"customer_ref": "CUST-1"})),
        openai_text_completion(GROUNDED_REPLY),
    )

    turn = _run_turn(endpoint)

    assert turn.result.tools[0].status is ToolStatus.OK
    assert turn.backend.calls == [("get_customer", ("CUST-1",))]  # the backend did answer
    assert turn.result.outcome is TurnOutcome.COMPLETED
    echoed = turn.messages(1)[2]["tool_calls"][0]
    assert echoed["function"]["name"] == "get_customer_context"
    assert _echoed_arguments(echoed) == {}
    wire = json.dumps(endpoint.requests)
    assert "CUST-1" not in wire
    assert "Test Borrower" not in wire


def test_a_foreign_account_the_model_named_is_neither_read_nor_echoed_back() -> None:
    """The model's own account_ref is dropped from the echo as well.

    Sending it back would tell the model its choice of account was honoured.
    """
    endpoint = _StrictEndpoint(
        openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-VICTIM"})),
        openai_text_completion(GROUNDED_REPLY),
    )

    turn = _run_turn(endpoint)

    assert turn.backend.calls == [("get_account", ("ACC-1",))]  # the session's, not the model's
    assert turn.result.tools[0].status is ToolStatus.OK
    assert turn.result.outcome is TurnOutcome.COMPLETED
    echoed = turn.messages(1)[2]["tool_calls"][0]
    assert _echoed_arguments(echoed) == {}
    assert "ACC-VICTIM" not in json.dumps(endpoint.requests[1])


def test_a_payment_promise_round_echoes_only_the_models_own_amount() -> None:
    """The date is taken from state and the account is bound, so neither goes back in the echo."""
    backend = _sample_backend()
    endpoint = _StrictEndpoint(
        openai_tool_completion(
            (
                "record_payment_promise",
                {"account_ref": "ACC-1", "promise_date": "2099-01-01", "amount_minor": 250000},
            )
        ),
        openai_text_completion("Noted for 2026-10-05. Thank you."),
    )

    turn = _run_turn(
        endpoint, backend=backend, text="I can pay on the fifth of October.", signals=_PROMISE
    )

    assert turn.result.tools[0].status is ToolStatus.OK
    assert backend.promises == [("ACC-1", date(2026, 10, 5), 250000)]
    assert turn.result.outcome is TurnOutcome.COMPLETED
    echoed = turn.messages(1)[2]["tool_calls"][0]
    assert echoed["function"]["name"] == "record_payment_promise"
    assert _echoed_arguments(echoed) == {"amount_minor": 250000}
    assert "2099-01-01" not in json.dumps(endpoint.requests[1])


# --- 7. the message contract, below the orchestrator --------------------------


def _calls(*ids: str) -> tuple[LlmToolCall, ...]:
    return tuple(LlmToolCall(call_id=call_id, tool_name="get_dpd") for call_id in ids)


def test_an_assistant_message_carries_its_tool_calls() -> None:
    message = LlmMessage(role="assistant", content="", tool_calls=_calls("call_a", "call_b"))
    assert [call.call_id for call in message.tool_calls] == ["call_a", "call_b"]


@pytest.mark.parametrize("role", ["user", "tool", "system"])
def test_only_an_assistant_message_may_carry_tool_calls(role: str) -> None:
    with pytest.raises(ValidationError, match="only an assistant message can carry tool calls"):
        LlmMessage(role=role, content="x", tool_calls=_calls("call_a"))


def test_a_tool_call_sent_back_to_the_model_must_carry_an_id() -> None:
    with pytest.raises(ValidationError, match="must carry an id"):
        LlmMessage(role="assistant", content="", tool_calls=_calls("call_a", ""))


def test_tool_call_ids_in_one_assistant_message_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        LlmMessage(role="assistant", content="", tool_calls=_calls("call_a", "call_a"))


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        pytest.param(
            LlmMessage(role="system", content="CONSTRAINTS"),
            {"role": "system", "content": "CONSTRAINTS"},
            id="system",
        ),
        pytest.param(
            LlmMessage(role="user", content="kitna bakaya hai?"),
            {"role": "user", "content": "kitna bakaya hai?"},
            id="user",
        ),
        pytest.param(
            LlmMessage(role="assistant", content=""),
            {"role": "assistant", "content": ""},
            id="assistant-empty",
        ),
        pytest.param(
            LlmMessage(role="tool", content="get_dpd ok.", tool_call_id="call_a"),
            {"role": "tool", "content": "get_dpd ok.", "tool_call_id": "call_a"},
            id="tool",
        ),
    ],
)
def test_a_message_without_tool_calls_is_rendered_exactly_as_before(
    message: LlmMessage, expected: dict
) -> None:
    """No ``tool_calls`` key at all. OpenAI rejects an empty array."""
    assert _render_message(message) == expected


@pytest.mark.parametrize(
    ("content", "wire_content"),
    [pytest.param("", None, id="silent"), pytest.param("Let me check.", "Let me check.", id="text")],
)
def test_an_assistant_turn_with_tool_calls_renders_in_the_openai_wire_shape(
    content: str, wire_content: str | None
) -> None:
    message = LlmMessage(
        role="assistant",
        content=content,
        tool_calls=(
            LlmToolCall(call_id="call_a", tool_name="get_dpd"),
            LlmToolCall(
                call_id="call_b",
                tool_name="record_payment_promise",
                arguments={"amount_minor": 250000},
            ),
        ),
    )

    rendered = _render_message(message)

    assert rendered == {
        "role": "assistant",
        "content": wire_content,
        "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "get_dpd", "arguments": "{}"}},
            {
                "id": "call_b",
                "type": "function",
                "function": {
                    "name": "record_payment_promise",
                    "arguments": '{"amount_minor": 250000}',
                },
            },
        ],
    }
    # With both calls answered, the strict endpoint accepts it.
    assert _protocol_violation(
        [
            {"role": "system", "content": "CONSTRAINTS"},
            {"role": "user", "content": "I can pay on the fifth."},
            rendered,
            {"role": "tool", "tool_call_id": "call_a", "content": "ok."},
            {"role": "tool", "tool_call_id": "call_b", "content": "ok."},
        ]
    ) is None
