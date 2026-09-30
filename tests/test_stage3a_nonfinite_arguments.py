"""Tool arguments holding a number JSON cannot represent.

Python's ``json`` module reads the non-standard literals ``NaN``, ``Infinity``
and ``-Infinity`` as floats, and reads a number that overflows - ``1e400`` - as
infinity. So "it parsed" never meant "it is JSON". Before this was pinned a
model could send ``{"amount_minor": NaN}`` as a tool call, the adapter handed it
on as a perfectly ordinary argument, and any tool with a float or a free-form
field ran with it. Two further cracks let a server take the adapter down rather
than fail a turn: arguments nested deeper than the parser's recursion limit
raised ``RecursionError`` straight out of the adapter, and so did a body nested
that deep.

What is being pinned
--------------------
- The adapter refuses a non-finite number anywhere in a tool call's arguments -
  top level, in a dict, in a list, deep inside both - whether the server sent
  the arguments as a JSON string or as an object. The refusal is
  ``LlmInvalidToolCall``, never a generation, never another exception type,
  never retried, and it carries none of the arguments.
- It over-refuses nothing: zero, negatives, fractions, ``1e308``, a 31-digit
  integer, booleans, null, nested containers, and the *strings* ``"NaN"``,
  ``"Infinity"`` and ``"1e400"`` all come through exactly as sent.
- Nesting the parser cannot handle is a typed failure: ``LlmInvalidToolCall``
  inside the arguments string, ``LlmMalformedResponse`` for the body itself.
- ``contains_non_finite_number`` finds NaN and ±Infinity - a ``float`` or a
  ``Decimal`` - in dicts, lists, tuples and sets, terminates on a cycle, and
  walks 200 000 levels without recursing. A string is text, never a number, and
  nothing is converted.
- The registry refuses non-finite arguments itself - ``INVALID_REQUEST``,
  logged as ``NonFiniteArgument`` - and the tool's ``run`` is never called. It
  checks twice. Before validation, on the raw arguments, because a lax schema
  could otherwise turn a NaN into the text ``"nan"`` or drop it unseen. After
  validation, on what validation produced, because a float field turns the JSON
  *string* ``"NaN"`` into a float NaN, and a set, a tuple or a ``Decimal`` field
  can hold one just as well. Finite floats and the ``datetime.date`` the
  orchestrator binds pass through untouched.
- The orchestrator never lets such a call reach the banking backend, ends the
  turn as ``invalid_tool_request``, and never sends another model request whose
  assistant echo could carry the number. A NaN in an argument the application
  binds is overwritten and left out of the echo. An echo that does go on the
  wire is JSON proper.

Nothing sleeps and no socket is opened: the endpoint is ``httpx2.MockTransport``
and the model is either that or a script. Bodies are written as raw text where
the literal ``NaN`` token matters, because ``json.dumps`` of a Python ``nan``
is exactly the thing being tested for.

Writing this file found cases the first fix missed. A tool with a float field
accepts the JSON *string* ``"NaN"`` (or ``"1e400"``, or a ``Decimal``) and
pydantic converts it to a non-finite float after the registry's check of the
raw arguments. The registry now checks the validated arguments too; the tests
that found it are kept below. So is the one that found the walk ignoring dict
*keys* (``dict[float, int]``) and ``deque[float]``, which the same conversion
reaches, and a ``complex`` field, which turns ``"nan"`` into ``nan+0j``.
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from datetime import date
from decimal import Decimal
from typing import Any, NoReturn

import httpx2
import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.core.session import EventSignals
from app.models.customer import ComplianceContext
from app.models.enums import Intent, ToolStatus
from app.models.tools import ToolRequest
from app.orchestrator import ConversationOrchestrator, TurnErrorCategory, TurnOutcome
from app.services import llm as llm_module
from app.services.llm import (
    LlmError,
    LlmGeneration,
    LlmInvalidToolCall,
    LlmMalformedResponse,
    LlmMessage,
    LlmRequest,
    LlmToolCall,
    ScriptedLlmService,
)
from app.services.llm_openai import OpenAiCompatibleLlmService
from app.tools.base import Tool, ToolRegistry
from tests.fakes import (
    InMemoryBankingBackend,
    openai_text_completion,
    sample_account,
    sample_customer,
    tool_generation,
)
from tests.test_llm_http_integration import _Endpoint, http_llm
from tests.test_orchestrator import GROUNDED_REPLY, make_runtime, open_session, say

BASE_URL = "http://model.invalid:8000/v1"

_PROMISE = EventSignals(intent=Intent.PAYMENT_PROMISE, promise_date=date(2026, 10, 5))

#: Deeper than the parser's recursion limit, in this process or any other.
_TOO_DEEP = 100_000

#: Deep enough to matter, shallow enough that the parser accepts it from inside
#: a test, an HTTP client and the adapter's own frames.
_ACCEPTED_DEPTH = 400


def _contains_non_finite_number(value: Any) -> bool:
    """The helper under test, looked up when called rather than imported."""
    return llm_module.contains_non_finite_number(value)


def _strict_json(text: str | bytes) -> Any:
    """Decode as JSON proper. ``NaN``, ``Infinity`` and an overflowing number are refused."""

    def refuse(token: str) -> NoReturn:
        raise ValueError(f"{token} is not a JSON number")

    def finite(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"{token} is not a finite number")
        return value

    return json.loads(text, parse_constant=refuse, parse_float=finite)


def _captured(stream) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# --- the adapter: raw bodies on an in-process transport ---------------------

#: Stands in for the arguments while the rest of the body is serialised, then
#: is replaced by the raw text, so the literal tokens reach the parser as sent.
_PLACEHOLDER = "__RAW_ARGUMENTS__"


def _tool_body(arguments: Any, tool_name: str = "record_payment_promise") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _string_form(arguments_text: str, tool_name: str = "record_payment_promise") -> str:
    """A body whose arguments are the JSON *string* ``arguments_text``, as OpenAI sends them."""
    return json.dumps(_tool_body(arguments_text, tool_name))


def _object_form(arguments_text: str) -> str:
    """A body whose arguments are an object, written into the body verbatim.

    What some compatible servers emit. The tokens are then part of the body
    itself, and reach the adapter through the parse of the whole response.
    """
    return json.dumps(_tool_body(_PLACEHOLDER)).replace(json.dumps(_PLACEHOLDER), arguments_text)


_FORMS = pytest.mark.parametrize("form", [_string_form, _object_form], ids=["string", "object"])


def _raw_service(raw: str, **kwargs) -> tuple[OpenAiCompatibleLlmService, list[bytes]]:
    """An adapter whose endpoint answers every request with ``raw``. No socket is opened."""
    seen: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        seen.append(request.content)
        return httpx2.Response(200, text=raw, headers={"Content-Type": "application/json"})

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("model", "configured-model")
    return OpenAiCompatibleLlmService(client=client, **kwargs), seen


def _request() -> LlmRequest:
    return LlmRequest(
        messages=(
            LlmMessage(role="system", content="CONSTRAINTS\n- tone: neutral"),
            LlmMessage(role="user", content="main paanch tareekh ko de dunga"),
        )
    )


def _generate(raw: str, **kwargs) -> LlmGeneration:
    service, _ = _raw_service(raw, **kwargs)
    return service.generate(_request())


_NON_FINITE_ARGUMENTS = [
    pytest.param('{"amount_minor": NaN}', id="nan"),
    pytest.param('{"amount_minor": Infinity}', id="infinity"),
    pytest.param('{"amount_minor": -Infinity}', id="minus-infinity"),
    pytest.param('{"amount_minor": 1e400}', id="overflow-1e400"),
    pytest.param('{"amount_minor": -1e400}', id="overflow-minus-1e400"),
    pytest.param('{"amount_minor": 1E400}', id="overflow-capital-e"),
    pytest.param('{"amount_minor": 2e308}', id="overflow-just-past-float-max"),
    pytest.param('{"amount": {"minor": NaN}}', id="nested-in-a-dict"),
    pytest.param('{"amount": {"minor": -Infinity}}', id="minus-infinity-in-a-dict"),
    pytest.param('{"amounts": [1, 2, Infinity]}', id="in-a-list"),
    pytest.param('{"amounts": [{"minor": 1e400}, 3]}', id="overflow-in-a-list-of-dicts"),
    pytest.param('{"a": {"b": [1, {"c": [NaN]}]}}', id="deep-inside"),
    pytest.param('{"ok": 250000, "also": "fine", "bad": [[[[-Infinity]]]]}', id="beside-valid-keys"),
    pytest.param(
        '{"a": ' + "[" * _ACCEPTED_DEPTH + "NaN" + "]" * _ACCEPTED_DEPTH + "}",
        id="at-the-bottom-of-accepted-nesting",
    ),
]


@_FORMS
@pytest.mark.parametrize("arguments", _NON_FINITE_ARGUMENTS)
def test_a_non_finite_number_in_tool_arguments_is_an_invalid_tool_call(form, arguments) -> None:
    with pytest.raises(LlmInvalidToolCall) as caught:
        _generate(form(arguments))

    assert type(caught.value) is LlmInvalidToolCall
    assert caught.value.category == "llm_invalid_tool_call"


def _several_calls(arguments: list[str], *, as_object: bool) -> str:
    """A body with one tool call per entry of ``arguments``, each written as in ``_string_form``/``_object_form``."""
    body = _tool_body("")
    template = body["choices"][0]["message"]["tool_calls"][0]
    calls = []
    for index, text in enumerate(arguments):
        call = json.loads(json.dumps(template))
        call["id"] = f"call_{index}"
        call["function"]["arguments"] = f"__RAW_{index}__" if as_object else text
        calls.append(call)
    body["choices"][0]["message"]["tool_calls"] = calls
    raw = json.dumps(body)
    if as_object:
        for index, text in enumerate(arguments):
            raw = raw.replace(json.dumps(f"__RAW_{index}__"), text)
    return raw


@pytest.mark.parametrize("as_object", [False, True], ids=["string", "object"])
@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(['{"amount_minor": 250000}', '{"amount_minor": NaN}'], id="last"),
        pytest.param(['{"amount_minor": -Infinity}', '{"amount_minor": 250000}'], id="first"),
        pytest.param(['{"a": 1}', '{"b": [1e400]}', '{"c": "ok"}'], id="middle"),
    ],
)
def test_one_non_finite_call_among_several_refuses_the_whole_response(
    arguments, as_object
) -> None:
    """No partial generation: the valid calls beside it are not handed on either."""
    with pytest.raises(LlmInvalidToolCall):
        _generate(_several_calls(arguments, as_object=as_object))


@pytest.mark.parametrize("as_object", [False, True], ids=["string", "object"])
def test_several_finite_calls_all_come_through(as_object) -> None:
    """The control for the test above."""
    generation = _generate(
        _several_calls(['{"a": 1}', '{"b": [1e308]}', '{"c": "NaN"}'], as_object=as_object)
    )
    assert [call.arguments for call in generation.tool_calls] == [
        {"a": 1},
        {"b": [1e308]},
        {"c": "NaN"},
    ]


@_FORMS
def test_a_non_finite_argument_is_never_retried(form) -> None:
    """The same request would produce the same arguments."""
    service, seen = _raw_service(form('{"amount_minor": NaN}'), max_retries=3)
    with pytest.raises(LlmInvalidToolCall):
        service.generate(_request())
    assert len(seen) == 1


@_FORMS
def test_a_non_finite_argument_is_logged_as_a_category_and_carries_no_argument(
    form, log_stream
) -> None:
    raw = form('{"amount_minor": Infinity, "note_from_model": "sk-live-abc123"}')
    with pytest.raises(LlmInvalidToolCall) as caught:
        _generate(raw)

    assert "amount_minor" not in str(caught.value)
    assert "sk-live-abc123" not in str(caught.value)
    record = [r for r in _captured(log_stream) if r.get("event") == "llm_call"][-1]
    assert record["outcome"] == "failed"
    assert record["error_category"] == "llm_invalid_tool_call"
    assert record["http_status"] == 200
    assert record["attempt"] == 1
    logged = log_stream.getvalue()
    assert "amount_minor" not in logged
    assert "sk-live-abc123" not in logged


# --- the adapter: valid values are preserved exactly ------------------------


_FINITE_ARGUMENTS = [
    pytest.param('{"n": 0}', {"n": 0}, id="zero"),
    pytest.param('{"n": -0.0}', {"n": -0.0}, id="negative-zero"),
    pytest.param('{"n": -1}', {"n": -1}, id="minus-one"),
    pytest.param('{"n": 1.5}', {"n": 1.5}, id="fraction"),
    pytest.param('{"n": -2.25}', {"n": -2.25}, id="negative-fraction"),
    pytest.param('{"n": 1e308}', {"n": 1e308}, id="1e308"),
    pytest.param('{"n": 1.7976931348623157e308}', {"n": sys.float_info.max}, id="float-max"),
    pytest.param('{"n": 5e-324}', {"n": 5e-324}, id="smallest-subnormal"),
    pytest.param(
        '{"n": 1000000000000000000000000000000}', {"n": 10**30}, id="thirty-one-digit-integer"
    ),
    pytest.param('{"s": "NaN"}', {"s": "NaN"}, id="string-nan"),
    pytest.param('{"s": "Infinity"}', {"s": "Infinity"}, id="string-infinity"),
    pytest.param('{"s": "-Infinity"}', {"s": "-Infinity"}, id="string-minus-infinity"),
    pytest.param('{"s": "1e400"}', {"s": "1e400"}, id="string-1e400"),
    pytest.param(
        '{"t": true, "f": false, "z": null}', {"t": True, "f": False, "z": None}, id="literals"
    ),
    pytest.param(
        '{"l": [1, [2.5, {"k": "v"}]], "d": {"e": {}, "f": []}}',
        {"l": [1, [2.5, {"k": "v"}]], "d": {"e": {}, "f": []}},
        id="nested-containers",
    ),
    pytest.param(
        '{"a": ' + "[" * _ACCEPTED_DEPTH + "7" + "]" * _ACCEPTED_DEPTH + "}",
        None,
        id="accepted-nesting-with-a-finite-leaf",
    ),
]


@_FORMS
@pytest.mark.parametrize(("arguments", "expected"), _FINITE_ARGUMENTS)
def test_finite_arguments_are_preserved_exactly(form, arguments, expected) -> None:
    """The refusal is for non-finite numbers only. Nothing else is rejected or rewritten."""
    if expected is None:
        expected = json.loads(arguments)

    generation = _generate(form(arguments))

    assert len(generation.tool_calls) == 1
    parsed = generation.tool_calls[0].arguments
    assert parsed == expected
    # Same values *and* same types: 10**30 is still an int, 1e308 still a
    # float, "NaN" still a string. allow_nan=False also proves nothing
    # non-finite slipped through.
    assert json.dumps(parsed, allow_nan=False, sort_keys=True) == json.dumps(
        expected, allow_nan=False, sort_keys=True
    )


def test_a_thirty_one_digit_integer_stays_an_integer() -> None:
    generation = _generate(_object_form('{"n": 1000000000000000000000000000000}'))
    value = generation.tool_calls[0].arguments["n"]
    assert type(value) is int and value == 10**30


def test_a_non_finite_number_elsewhere_in_the_body_does_not_fail_a_text_reply() -> None:
    """Only tool arguments are refused. An unreadable usage count is dropped, as before."""
    raw = json.dumps(openai_text_completion("ok", usage={"prompt_tokens": 1})).replace(
        '"prompt_tokens": 1', '"prompt_tokens": NaN'
    )
    generation = _generate(raw)
    assert generation.text == "ok"
    assert generation.usage is None


# --- the adapter: nesting the parser cannot handle --------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param('{"a": ' + "[" * _TOO_DEEP + "]" * _TOO_DEEP + "}", id="inside-an-object"),
        pytest.param("[" * _TOO_DEEP + "]" * _TOO_DEEP, id="bare-array"),
        pytest.param('{"a": ' * _TOO_DEEP + "1" + "}" * _TOO_DEEP, id="objects"),
    ],
)
def test_arguments_nested_past_the_recursion_limit_are_an_invalid_tool_call(arguments) -> None:
    """Not a RecursionError escaping the adapter, and not chained to one."""
    with pytest.raises(LlmInvalidToolCall) as caught:
        _generate(_string_form(arguments))

    assert type(caught.value) is LlmInvalidToolCall
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"choices": ' + "[" * _TOO_DEEP + "]" * _TOO_DEEP + "}", id="choices"),
        pytest.param("[" * _TOO_DEEP + "]" * _TOO_DEEP, id="bare-array"),
        pytest.param(
            _object_form('{"a": ' + "[" * _TOO_DEEP + "]" * _TOO_DEEP + "}"),
            id="object-form-arguments",
        ),
    ],
)
def test_a_body_nested_past_the_recursion_limit_is_a_malformed_response(raw) -> None:
    """Object-form arguments are part of the body, so their nesting is the body's."""
    with pytest.raises(LlmMalformedResponse) as caught:
        _generate(raw)

    assert type(caught.value) is LlmMalformedResponse
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_a_deeply_nested_body_is_never_retried() -> None:
    service, seen = _raw_service("[" * _TOO_DEEP + "]" * _TOO_DEEP, max_retries=3)
    with pytest.raises(LlmMalformedResponse):
        service.generate(_request())
    assert len(seen) == 1


# --- contains_non_finite_number ---------------------------------------------


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), float("1e400"), -float("1e400")],
    ids=["nan", "inf", "minus-inf", "overflow", "minus-overflow"],
)
def test_a_non_finite_float_is_found(value) -> None:
    assert _contains_non_finite_number(value) is True


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        1.5,
        -2.25,
        1e308,
        sys.float_info.max,
        5e-324,
        0,
        -1,
        10**30,
        True,
        False,
        None,
        "NaN",
        "Infinity",
        "1e400",
        b"nan",
        {},
        [],
        (),
        {"a": [1, 2.5, {"b": (3, "inf")}]},
    ],
)
def test_finite_numbers_and_everything_else_are_not(value) -> None:
    assert _contains_non_finite_number(value) is False


@pytest.mark.parametrize(
    "value",
    [
        {"amount_minor": float("nan")},
        {"a": {"b": {"c": float("inf")}}},
        [1, 2, float("-inf")],
        (1.0, (2.0, (float("nan"),))),
        {"a": [1, {"b": (2, [float("inf")])}]},
        [{"ok": 1.5}, {"ok": 2.5}, {"bad": float("nan")}],
    ],
    ids=["dict", "nested-dicts", "list", "nested-tuples", "mixed", "last-of-many"],
)
def test_a_non_finite_float_is_found_inside_dicts_lists_and_tuples(value) -> None:
    assert _contains_non_finite_number(value) is True


def test_a_cyclic_list_terminates() -> None:
    finite: list[Any] = [1.0, 2.0]
    finite.append(finite)
    assert _contains_non_finite_number(finite) is False

    poisoned: list[Any] = [1.0]
    poisoned.append(poisoned)
    poisoned.append(float("nan"))
    assert _contains_non_finite_number(poisoned) is True


def test_a_cyclic_dict_terminates() -> None:
    finite: dict[str, Any] = {"x": 1.5}
    finite["self"] = finite
    finite["list"] = [finite, (finite,)]
    assert _contains_non_finite_number(finite) is False

    poisoned: dict[str, Any] = {"x": 1.5}
    poisoned["self"] = poisoned
    poisoned["later"] = {"back": poisoned, "bad": float("inf")}
    assert _contains_non_finite_number(poisoned) is True


def test_a_container_shared_between_branches_is_still_searched() -> None:
    """Visited-once bookkeeping must not skip the first visit."""
    shared = [float("nan")]
    assert _contains_non_finite_number({"a": shared, "b": shared, "c": [shared]}) is True


def _nest(depth: int, leaf: Any) -> Any:
    """``depth`` levels of alternating list, dict and tuple around ``leaf``, built without recursion."""
    value = leaf
    for level in range(depth):
        kind = level % 3
        value = [value] if kind == 0 else {"k": value} if kind == 1 else (value,)
    return value


@pytest.mark.parametrize(("leaf", "expected"), [(float("nan"), True), (1.5, False)])
def test_two_hundred_thousand_levels_of_nesting_are_walked_without_recursion(
    leaf, expected
) -> None:
    """The parser accepts nesting near the recursion limit; the walk must not fail on it."""
    deep = _nest(200_000, leaf)
    assert _contains_non_finite_number(deep) is expected


def test_only_numbers_count_and_nothing_is_converted() -> None:
    """Floats and Decimals are numbers; sets are entered; anything else passes.

    JSON produces neither a Decimal nor a set, so the model boundary never sees
    one, but validating arguments into a tool's model can produce both, and the
    registry checks what validation produced.
    """
    assert _contains_non_finite_number(date(2026, 10, 5)) is False
    assert _contains_non_finite_number(Decimal("NaN")) is True
    assert _contains_non_finite_number(Decimal("Infinity")) is True
    assert _contains_non_finite_number(Decimal("-Infinity")) is True
    assert _contains_non_finite_number(Decimal("1.5")) is False
    assert _contains_non_finite_number({float("nan")}) is True
    assert _contains_non_finite_number(frozenset({float("inf")})) is True
    assert _contains_non_finite_number({1.5, 2.0}) is False
    assert _contains_non_finite_number(object()) is False
    assert _contains_non_finite_number("NaN") is False

    class _Rate(float):
        pass

    assert _contains_non_finite_number(_Rate("nan")) is True
    assert _contains_non_finite_number(_Rate("1.5")) is False


def test_the_walk_does_not_change_what_it_walks() -> None:
    value = {"a": [1.5, {"b": (2, "NaN")}], "c": date(2026, 10, 5)}
    before = repr(value)
    _contains_non_finite_number(value)
    assert repr(value) == before


# --- the registry refuses non-finite arguments itself -----------------------


class _RateArgs(BaseModel):
    """A tool schema with the two kinds of field a non-finite number could land in."""

    model_config = ConfigDict(extra="forbid")

    rate: float = 0.0
    details: dict[str, Any] = {}
    on: date | None = None


class _RateTool(Tool):
    """A tool that only counts. Not a banking tool: none of those has a float field."""

    name = "set_rate"
    description = "Record a rate. Test double."
    args_model = _RateArgs

    def __init__(self) -> None:
        self.calls: list[_RateArgs] = []

    def run(self, args: BaseModel) -> dict[str, Any]:
        self.calls.append(args)  # type: ignore[arg-type]
        return {"recorded": True}


def _registry() -> tuple[ToolRegistry, _RateTool]:
    tool = _RateTool()
    registry = ToolRegistry()
    registry.register(tool)
    return registry, tool


def _execute(registry: ToolRegistry, arguments: dict[str, Any], tool_name: str = "set_rate"):
    return registry.execute(
        ToolRequest(
            request_id="req-1",
            session_id="sess-1",
            turn_id=0,
            tool_name=tool_name,
            arguments=arguments,
        )
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"rate": float("nan")},
        {"rate": float("inf")},
        {"rate": float("-inf")},
        {"details": {"x": float("nan")}},
        {"details": {"x": [1, {"y": float("inf")}]}},
        {"details": {"pair": (1.0, float("-inf"))}},
        {"rate": 1.5, "details": {"deep": _nest(1_000, float("nan"))}},
        # Not from JSON, but an LlmService need not parse JSON, and a free-form
        # field keeps whatever it is given - through validation and into run().
        {"details": {"x": Decimal("NaN")}},
        {"details": {"x": [Decimal("-Infinity")]}},
        {"details": {"x": {1.5, float("nan")}}},
        {"details": {"x": frozenset({float("inf")})}},
    ],
    ids=[
        "top-level-nan",
        "top-level-inf",
        "top-level-minus-inf",
        "free-form-dict",
        "free-form-deep",
        "free-form-tuple",
        "beside-a-valid-float",
        "free-form-decimal-nan",
        "free-form-decimal-minus-infinity",
        "free-form-set",
        "free-form-frozenset",
    ],
)
def test_the_registry_refuses_a_non_finite_argument_and_never_runs_the_tool(
    arguments, log_stream
) -> None:
    registry, tool = _registry()

    result = _execute(registry, arguments)

    assert result.status is ToolStatus.INVALID_REQUEST
    assert result.data is None
    assert tool.calls == []
    record = [r for r in _captured(log_stream) if r.get("event") == "tool_call"][-1]
    assert record["status"] == ToolStatus.INVALID_REQUEST.value
    assert record["error_type"] == "NonFiniteArgument"


def test_the_non_finite_check_comes_before_schema_validation(log_stream) -> None:
    """A request that is also invalid for the schema is still reported as non-finite."""
    registry, tool = _registry()

    result = _execute(registry, {"rate": float("nan"), "not_in_schema": 1})

    assert result.status is ToolStatus.INVALID_REQUEST
    assert tool.calls == []
    record = [r for r in _captured(log_stream) if r.get("event") == "tool_call"][-1]
    assert record["error_type"] == "NonFiniteArgument"


def test_finite_floats_and_a_bound_date_reach_the_tool_unchanged() -> None:
    registry, tool = _registry()

    result = _execute(
        registry,
        {
            "rate": -2.25,
            "details": {"big": 1e308, "int": 10**30, "text": "NaN", "list": [0, 1.5, None]},
            "on": date(2026, 10, 5),
        },
    )

    assert result.status is ToolStatus.OK
    assert len(tool.calls) == 1
    args = tool.calls[0]
    assert args.rate == -2.25
    assert args.on == date(2026, 10, 5)
    assert args.details == {"big": 1e308, "int": 10**30, "text": "NaN", "list": [0, 1.5, None]}


def test_a_string_that_spells_nan_in_a_free_form_field_is_text_and_runs() -> None:
    registry, tool = _registry()

    result = _execute(registry, {"details": {"note": "NaN", "other": "Infinity"}})

    assert result.status is ToolStatus.OK
    assert tool.calls[0].details == {"note": "NaN", "other": "Infinity"}


@pytest.mark.parametrize(
    "value",
    ["NaN", "Infinity", "-Infinity", "1e400", "-1e400", Decimal("NaN"), Decimal("Infinity")],
    ids=[
        "str-nan",
        "str-infinity",
        "str-minus-infinity",
        "str-1e400",
        "str-minus-1e400",
        "decimal-nan",
        "decimal-inf",
    ],
)
def test_a_float_field_that_validation_makes_non_finite_is_still_refused(
    value, log_stream
) -> None:
    """The string is legitimate JSON; the float pydantic makes of it is not."""
    registry, tool = _registry()

    result = _execute(registry, {"rate": value})

    assert tool.calls == [], f"run() received rate={tool.calls[0].rate!r}"
    assert result.status is ToolStatus.INVALID_REQUEST
    assert result.data is None
    record = [r for r in _captured(log_stream) if r.get("event") == "tool_call"][-1]
    assert record["error_type"] == "NonFiniteArgument"


class _ContainerArgs(BaseModel):
    """Fields validation fills with floats or Decimals from JSON *text*."""

    model_config = ConfigDict(extra="forbid")

    bag: set[float] = set()
    frozen: frozenset[float] = frozenset()
    rates: tuple[float, ...] = ()
    exact: Decimal = Field(default=Decimal(0), allow_inf_nan=True)
    weights: dict[float, int] = {}
    window: deque[float] = deque()
    phase: complex = 0j


class _ContainerTool(Tool):
    name = "set_rates"
    description = "Record several rates. Test double."
    args_model = _ContainerArgs

    def __init__(self) -> None:
        self.calls: list[_ContainerArgs] = []

    def run(self, args: BaseModel) -> dict[str, Any]:
        self.calls.append(args)  # type: ignore[arg-type]
        return {"recorded": True}


def _container_registry() -> tuple[ToolRegistry, _ContainerTool]:
    tool = _ContainerTool()
    registry = ToolRegistry()
    registry.register(tool)
    return registry, tool


@pytest.mark.parametrize(
    "arguments",
    [
        {"bag": [1.5, "NaN"]},
        {"frozen": ["Infinity"]},
        {"rates": [1.5, "-Infinity"]},
        {"rates": ["1e400"]},
        {"exact": "NaN"},
        {"exact": "Infinity"},
        {"exact": "-Infinity"},
    ],
    ids=[
        "set",
        "frozenset",
        "tuple",
        "tuple-overflow",
        "decimal-nan",
        "decimal-infinity",
        "decimal-minus-infinity",
    ],
)
def test_validation_that_puts_a_non_finite_number_in_a_set_a_tuple_or_a_decimal_is_refused(
    arguments, log_stream
) -> None:
    """Only strings go in, so only the check of what validation produced can see it."""
    json.dumps(arguments, allow_nan=False)  # the raw arguments are JSON proper
    registry, tool = _container_registry()

    result = _execute(registry, arguments, tool_name="set_rates")

    assert tool.calls == [], f"run() received {tool.calls[0].model_dump(exclude_defaults=True)!r}"
    assert result.status is ToolStatus.INVALID_REQUEST
    record = [r for r in _captured(log_stream) if r.get("event") == "tool_call"][-1]
    assert record["error_type"] == "NonFiniteArgument"


def test_finite_values_in_those_fields_reach_the_tool() -> None:
    """The control for the test above: a set, a tuple, a Decimal and a float key are not suspect in themselves."""
    registry, tool = _container_registry()

    result = _execute(
        registry,
        {
            "bag": [1.5, "2.5"],
            "frozen": [0, -0.0],
            "rates": [1e308, "-2.25"],
            "exact": "1.25",
            "weights": {"1.5": 1},
            "window": [2.0, "3"],
        },
        tool_name="set_rates",
    )

    assert result.status is ToolStatus.OK
    assert len(tool.calls) == 1
    args = tool.calls[0]
    assert args.bag == {1.5, 2.5}
    assert args.frozen == frozenset({0.0})
    assert args.rates == (1e308, -2.25)
    assert args.exact == Decimal("1.25")
    assert args.weights == {1.5: 1}
    assert args.window == deque([2.0, 3.0])


@pytest.mark.parametrize(
    "arguments",
    [
        {"weights": {"NaN": 1}},
        {"weights": {"1.5": 1, "-Infinity": 2}},
        {"window": ["NaN"]},
        {"window": [1.5, "Infinity"]},
        {"phase": "nan"},
        {"phase": "1+infj"},
    ],
    ids=[
        "dict-key-nan",
        "dict-key-minus-infinity",
        "deque-nan",
        "deque-infinity",
        "complex-nan",
        "complex-infinite-imaginary",
    ],
)
def test_a_non_finite_number_validation_puts_in_a_dict_key_a_deque_or_a_complex_is_refused(
    arguments,
) -> None:
    """The same coercion as the float field above, into places the first walk did not look.

    pydantic turns ``{"NaN": 1}`` into ``{nan: 1}`` for a ``dict[float, int]``
    field, ``["NaN"]`` into ``deque([nan])`` for a ``deque[float]``, and
    ``"nan"`` into ``nan+0j`` for a ``complex``. The walk now reads dict keys,
    enters deques and treats a complex as a number. Latent before: no banking
    tool has such a field.
    """
    registry, tool = _container_registry()

    result = _execute(registry, arguments, tool_name="set_rates")

    assert tool.calls == [], f"run() received {tool.calls[0].model_dump(exclude_defaults=True)!r}"
    assert result.status is ToolStatus.INVALID_REQUEST


class _LaxArgs(BaseModel):
    """A schema that would launder a NaN: it drops unknown keys and turns numbers into text."""

    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)

    note: str = ""


class _LaxTool(Tool):
    name = "take_note"
    description = "Record a note. Test double."
    args_model = _LaxArgs

    def __init__(self) -> None:
        self.calls: list[_LaxArgs] = []

    def run(self, args: BaseModel) -> dict[str, Any]:
        self.calls.append(args)  # type: ignore[arg-type]
        return {"recorded": True}


@pytest.mark.parametrize(
    "arguments",
    [
        {"note": float("nan")},
        {"note": float("inf")},
        {"note": float("-inf")},
        {"note": "fine", "dropped": float("nan")},
    ],
    ids=["nan-becomes-text", "inf-becomes-text", "minus-inf-becomes-text", "dropped-key"],
)
def test_a_raw_non_finite_value_that_validation_would_hide_is_still_refused(
    arguments, log_stream
) -> None:
    """Why the raw check exists at all: after validation this is the string "nan", or nothing."""
    tool = _LaxTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = _execute(registry, arguments, tool_name="take_note")

    assert tool.calls == [], f"run() received {tool.calls[0].model_dump()!r}"
    assert result.status is ToolStatus.INVALID_REQUEST
    record = [r for r in _captured(log_stream) if r.get("event") == "tool_call"][-1]
    assert record["error_type"] == "NonFiniteArgument"


# --- the orchestrator -------------------------------------------------------


class _RecordingBackend(InMemoryBankingBackend):
    """The sample borrower's backend, noting every call made to it."""

    def __init__(self) -> None:
        super().__init__(
            customers={"CUST-1": sample_customer()},
            accounts={"ACC-1": sample_account()},
            compliance={"ACC-1": ComplianceContext(grievance_pending=False)},
        )
        self.calls: list[str] = []
        self.account_refs: list[Any] = []

    def get_customer(self, customer_ref):
        self.calls.append("get_customer")
        return super().get_customer(customer_ref)

    def get_account(self, account_ref):
        self.calls.append("get_account")
        self.account_refs.append(account_ref)
        return super().get_account(account_ref)

    def get_compliance(self, account_ref):
        self.calls.append("get_compliance")
        return super().get_compliance(account_ref)

    def record_payment_promise(self, account_ref, promise_date, amount_minor):
        self.calls.append("record_payment_promise")
        return super().record_payment_promise(account_ref, promise_date, amount_minor)

    def create_dispute(self, account_ref, reason_code):
        self.calls.append("create_dispute")
        return super().create_dispute(account_ref, reason_code)

    def escalate_case(self, account_ref, reason_code):
        self.calls.append("escalate_case")
        return super().escalate_case(account_ref, reason_code)


def _refused_as_non_finite(log_stream) -> bool:
    return any(
        r.get("event") == "tool_call" and r.get("error_type") == "NonFiniteArgument"
        for r in _captured(log_stream)
    )


def test_a_read_tool_with_a_nan_argument_never_reaches_the_backend(log_stream) -> None:
    backend = _RecordingBackend()
    llm = ScriptedLlmService([
        tool_generation("get_dpd", {"account_ref": "ACC-1", "x": float("nan")}),
        # Never used: the turn must end before a second model call.
        tool_generation("get_dpd", {"account_ref": "ACC-1"}),
    ])
    runtime = make_runtime(backend=backend, llm=llm)
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert backend.calls == []
    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert result.response_text is None
    assert result.tools[0].status is ToolStatus.INVALID_REQUEST
    assert TurnErrorCategory.INVALID_TOOL_REQUEST in result.error_categories
    assert len(llm.requests) == 1
    assert _refused_as_non_finite(log_stream)


@pytest.mark.parametrize(
    "amount", [float("inf"), float("-inf"), float("nan")], ids=["inf", "minus-inf", "nan"]
)
def test_a_payment_promise_with_a_non_finite_amount_is_never_written(amount, log_stream) -> None:
    """The write's precondition is met, so only the amount stands between it and the backend."""
    backend = _RecordingBackend()
    llm = ScriptedLlmService([
        tool_generation(
            "record_payment_promise",
            {"account_ref": "ACC-1", "promise_date": "2026-10-05", "amount_minor": amount},
        ),
        tool_generation("get_dpd", {"account_ref": "ACC-1"}),
    ])
    runtime = make_runtime(backend=backend, llm=llm)
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("I can pay on the fifth of October."), signals=_PROMISE
    )

    assert backend.calls == []
    assert backend.promises == []
    assert result.tools[0].dispatched is True  # the precondition held; the registry refused
    assert result.tools[0].status is ToolStatus.INVALID_REQUEST
    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert TurnErrorCategory.INVALID_TOOL_REQUEST in result.error_categories
    assert len(llm.requests) == 1
    assert _refused_as_non_finite(log_stream)


def test_the_same_promise_with_a_finite_amount_is_written(log_stream) -> None:
    """The control for the test above: nothing but the amount differs."""
    backend = _RecordingBackend()
    runtime = make_runtime(
        backend=backend,
        llm=ScriptedLlmService([
            tool_generation(
                "record_payment_promise",
                {"account_ref": "ACC-1", "promise_date": "2026-10-05", "amount_minor": 250_000},
            ),
            LlmGeneration(text="Noted for 2026-10-05. Thank you.", model="scripted"),
        ]),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("I can pay on the fifth of October."), signals=_PROMISE
    )

    assert result.tools[0].status is ToolStatus.OK
    assert backend.promises == [("ACC-1", date(2026, 10, 5), 250_000)]
    assert not _refused_as_non_finite(log_stream)


@_FORMS
def test_over_http_a_non_finite_amount_ends_the_turn_at_the_model_boundary(form) -> None:
    """The adapter refuses it first, so the registry and the backend never see it."""
    backend = _RecordingBackend()
    endpoint = _Endpoint(form('{"amount_minor": Infinity}'))
    runtime = make_runtime(backend=backend, llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("I can pay on the fifth of October."), signals=_PROMISE
    )

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert TurnErrorCategory.LLM_INVALID_TOOL_CALL in result.error_categories
    assert result.tools == ()
    assert backend.calls == []
    assert len(endpoint.requests) == 1


# --- no non-finite number in an assistant echo on the wire ------------------


class _FirstRoundScripted:
    """Round one from a script, every later round over HTTP.

    The HTTP adapter refuses a non-finite argument, so a script is the only way
    to hand one to the orchestrator; what the orchestrator then sends back to a
    model is observed on a real request body.
    """

    def __init__(self, first: LlmGeneration, http: OpenAiCompatibleLlmService) -> None:
        self._first = first
        self._http = http
        self.requests: list[LlmRequest] = []

    def generate(self, request: LlmRequest) -> LlmGeneration:
        self.requests.append(request)
        if len(self.requests) == 1:
            return self._first
        return self._http.generate(request)


def _with_rate_tool(first: LlmGeneration, *responses: str):
    """A runtime whose registry also holds a tool with a float field, and its raw endpoint."""
    raw_requests: list[bytes] = []
    pending = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        raw_requests.append(request.content)
        if not pending:
            raise AssertionError("more model calls were made than were scripted")
        return httpx2.Response(
            200, text=pending.pop(0), headers={"Content-Type": "application/json"}
        )

    http = OpenAiCompatibleLlmService(
        base_url=BASE_URL,
        model="gemma-stand-in",
        client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    backend = _RecordingBackend()
    runtime = make_runtime(backend=backend, llm=_FirstRoundScripted(first, http))
    tool = _RateTool()
    runtime.tools.register(tool)
    return runtime, tool, backend, raw_requests


def test_a_non_finite_argument_to_a_float_tool_sends_no_echo_to_the_model() -> None:
    runtime, tool, backend, raw_requests = _with_rate_tool(
        tool_generation("set_rate", {"rate": float("nan")}),
        json.dumps(openai_text_completion(GROUNDED_REPLY)),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert tool.calls == []
    assert backend.calls == []
    assert result.outcome is TurnOutcome.FAILED
    assert result.tools[0].status is ToolStatus.INVALID_REQUEST
    assert TurnErrorCategory.INVALID_TOOL_REQUEST in result.error_categories
    assert raw_requests == []  # no second model call, so no echo at all


def test_one_non_finite_call_in_a_round_stops_the_round_and_sends_no_echo() -> None:
    runtime, tool, _backend, raw_requests = _with_rate_tool(
        LlmGeneration(
            tool_calls=(
                LlmToolCall(call_id="a", tool_name="set_rate", arguments={"rate": 1.5}),
                LlmToolCall(call_id="b", tool_name="set_rate", arguments={"rate": float("inf")}),
            ),
            model="scripted",
        ),
        json.dumps(openai_text_completion(GROUNDED_REPLY)),
    )
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert [args.rate for args in tool.calls] == [1.5]
    assert [t.status for t in result.tools] == [ToolStatus.OK, ToolStatus.INVALID_REQUEST]
    assert result.outcome is TurnOutcome.FAILED
    assert raw_requests == []


def test_an_echo_that_does_go_on_the_wire_is_strict_json_with_finite_arguments() -> None:
    runtime, tool, _backend, raw_requests = _with_rate_tool(
        tool_generation(
            "set_rate",
            {"rate": -2.25, "details": {"big": 1e308, "int": 10**30, "text": "NaN"}},
            call_id="call-rate",
        ),
        json.dumps(openai_text_completion(GROUNDED_REPLY)),
    )
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert len(tool.calls) == 1
    assert len(raw_requests) == 1
    body = _strict_json(raw_requests[0])  # raises on NaN, Infinity or an overflow
    echoes = [m for m in body["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(echoes) == 1
    call = echoes[0]["tool_calls"][0]
    assert call["id"] == "call-rate"
    assert _strict_json(call["function"]["arguments"]) == {
        "rate": -2.25,
        "details": {"big": 1e308, "int": 10**30, "text": "NaN"},
    }


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "minus-inf"]
)
def test_the_adapter_never_puts_a_non_finite_echo_argument_on_the_wire(value) -> None:
    """Unreachable from the orchestrator, which ends the turn first; refused here regardless.

    What is pinned is that no request leaves. The exception is a bare
    ``ValueError`` from ``json.dumps(allow_nan=False)`` today, not an ``LlmError``;
    either is accepted here.
    """
    echo = LlmMessage(
        role="assistant",
        content="",
        tool_calls=(LlmToolCall(call_id="c1", tool_name="set_rate", arguments={"rate": value}),),
    )
    request = LlmRequest(
        messages=(
            LlmMessage(role="user", content="x"),
            echo,
            LlmMessage(role="tool", content="set_rate ok.", tool_call_id="c1"),
        )
    )
    service, seen = _raw_service(json.dumps(openai_text_completion("ok")))

    with pytest.raises((ValueError, LlmError)):
        service.generate(request)
    assert seen == []


def test_over_http_a_string_nan_for_a_float_field_never_reaches_the_tool(log_stream) -> None:
    """The adapter keeps the string "NaN" (correctly); the registry refuses the float it becomes."""
    endpoint = _Endpoint(
        _string_form('{"rate": "NaN"}', tool_name="set_rate"),
        json.dumps(openai_text_completion(GROUNDED_REPLY)),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    tool = _RateTool()
    runtime.tools.register(tool)
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert tool.calls == [], f"run() received rate={tool.calls[0].rate!r}"
    assert result.tools[0].status is ToolStatus.INVALID_REQUEST
    assert result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.INVALID_TOOL_REQUEST in result.error_categories
    assert len(endpoint.requests) == 1  # the turn ended; nothing was echoed
    assert _refused_as_non_finite(log_stream)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), Decimal("NaN")], ids=["nan", "inf", "decimal-nan"]
)
def test_a_non_finite_value_in_an_app_bound_argument_is_overwritten_and_never_echoed(
    value,
) -> None:
    """The one path where the registry does not refuse: the application's value replaces it first.

    ``account_ref`` is bound from the session, so the backend reads the
    borrower's own account, and the echo leaves bound arguments out, so the
    model's NaN never goes back on the wire.
    """
    endpoint = _Endpoint(json.dumps(openai_text_completion(GROUNDED_REPLY)))
    backend = _RecordingBackend()
    llm = _FirstRoundScripted(
        tool_generation("get_dpd", {"account_ref": value}, call_id="call-dpd"),
        http_llm(endpoint),
    )
    runtime = make_runtime(backend=backend, llm=llm)
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.tools[0].status is ToolStatus.OK
    assert backend.account_refs == ["ACC-1"]
    assert len(endpoint.requests) == 1
    body = _strict_json(json.dumps(endpoint.requests[0]))
    echoes = [m for m in body["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
    assert [call["id"] for call in echoes[0]["tool_calls"]] == ["call-dpd"]
    assert _strict_json(echoes[0]["tool_calls"][0]["function"]["arguments"]) == {}
