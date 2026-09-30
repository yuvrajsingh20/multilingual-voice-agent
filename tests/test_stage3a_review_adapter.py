"""Stage 3A review, agent ADAPTER: the OpenAI-compatible adapter and its config.

This pins the fixes made to :mod:`app.services.llm_openai` (and the matching
:class:`~app.config.Settings` bounds) in response to the adversarial review and
the peer audit. Each section below corresponds to one reviewed finding.

What is pinned
--------------
1. ``finish_reason`` is read through an *allowlist* (never a denylist of the two
   words ``length``/``content_filter``): only a reason known to mean "finished"
   passes, case- and whitespace-insensitively, and what is stored - and what a
   success log carries - is always the normalised word, never the server's own
   spelling.
2. The legacy ``message.function_call`` field is never silently dropped: it is
   ``LlmInvalidToolCall``, whether or not the message also carries text.
3. ``finish_reason: "tool_calls"`` with no tool call in the message is a
   malformed body, not a wordless success.
4. Tool-call markup a server failed to parse and left in the text - a chat
   template token, or a reply that is nothing but a JSON object or array - is
   never spoken; ordinary speech with braces or brackets that is not JSON is.
5. A tool name is checked exactly, never repaired: no stripping, no
   ASCII-length exception, no ``type`` other than ``function`` or absent.
6. A tool-call id that is not a short plain token - spaced, oversized,
   non-ASCII, or free text a model should not have echoed - is never kept: the
   adapter reports it as no id, and the orchestrator mints one that is used
   consistently and never leaks the model's original text.
7. A repeated JSON key inside tool arguments - string or object form, nested,
   sent by a model to smuggle a non-finite number past the last write wins -
   fails the call. The same repetition anywhere *else* in the body does not.
8. Nothing unexpected escapes ``generate()`` as itself: a non-``LlmError``
   exception - a bug in the adapter, or an odd transport failure - becomes a
   plain, uncaused ``LlmError``; a lone surrogate in a transcript reaches the
   wire as escaped, valid ASCII JSON instead of raising an encoding error; and
   a closed owned client raises ``LlmNotConfigured`` rather than trying to call
   out.
9. The constructor refuses a base URL httpx2 could not use, a stray credential
   alongside ``api_key``, an unusable temperature, output cap, or timeout, and
   a SOCKS proxy the environment configures without the package to speak it -
   all without ever putting the rejected value in the exception.
10. ``Settings`` never echoes a rejected value - ``hide_input_in_errors`` - and
    the two model timeouts are capped at 600 seconds.

Every test drives the adapter through ``httpx2.MockTransport`` or a raw text
body; nothing sleeps, and no socket is opened. Names the fix introduced -
``_TOOL_CALL_MARKUP``, ``_require_usable_url``, and the like - are imported
directly, so a build that removed them fails on import rather than silently
collecting nothing.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.orchestrator import ConversationOrchestrator, TurnErrorCategory, TurnOutcome
from app.services.llm import (
    LlmConfigurationError,
    LlmError,
    LlmIncompleteResponse,
    LlmInvalidToolCall,
    LlmMalformedResponse,
    LlmMessage,
    LlmNotConfigured,
    LlmRequest,
)
from app.services.llm_openai import (
    _TOOL_CALL_MARKUP,
    OpenAiCompatibleLlmService,
    _is_unparsed_tool_call,
    _parse_tool_calls,
)
from tests.fakes import openai_text_completion, openai_tool_completion
from tests.test_llm_http_integration import _Endpoint, http_llm
from tests.test_orchestrator import GROUNDED_REPLY, make_runtime, open_session, say

BASE_URL = "http://model.invalid:8000/v1"

#: A sentinel distinguishing "the key was left out" from "the key was given as None".
_ABSENT = object()


# --- shared harness ----------------------------------------------------------


def _request() -> LlmRequest:
    return LlmRequest(
        messages=(
            LlmMessage(role="system", content="CONSTRAINTS\n- tone: neutral"),
            LlmMessage(role="user", content="kitna bakaya hai?"),
        )
    )


def _service(handler, **kwargs) -> OpenAiCompatibleLlmService:
    """An adapter wired to an in-process transport. No socket is opened."""
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("model", "configured-model")
    return OpenAiCompatibleLlmService(
        client=httpx2.Client(transport=httpx2.MockTransport(handler)), **kwargs
    )


def _responder(*bodies: Any, status: int = 200):
    """A handler that replays ``bodies`` (the last one repeats)."""
    remaining = list(bodies)

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        body = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return httpx2.Response(status, json=body)

    return handler


def _raw_service(raw: str, **kwargs) -> OpenAiCompatibleLlmService:
    """An adapter whose endpoint answers every request with the raw text ``raw``."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        return httpx2.Response(200, text=raw, headers={"Content-Type": "application/json"})

    return _service(handler, **kwargs)


def _body(message: dict, finish_reason: Any) -> dict:
    return {
        "id": "chatcmpl-review",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def _llm_records(stream) -> list[dict]:
    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return [r for r in records if r.get("event") == "llm_call"]


# --- 1. finish_reason: an allowlist, stored normalised -----------------------

#: (sent, normalised, whether the message must carry a tool call for it to be legal)
_ALLOWLIST_VARIANTS = [
    pytest.param("stop", "stop", False, id="stop"),
    pytest.param("STOP", "stop", False, id="stop-upper"),
    pytest.param("  stop  ", "stop", False, id="stop-padded"),
    pytest.param("tool_calls", "tool_calls", True, id="tool_calls"),
    pytest.param(" tool_calls ", "tool_calls", True, id="tool_calls-padded"),
    pytest.param("TOOL_CALLS", "tool_calls", True, id="tool_calls-upper"),
    pytest.param("function_call", "function_call", False, id="function_call"),
    pytest.param("eos_token", "eos_token", False, id="eos_token"),
    pytest.param("eos", "eos", False, id="eos"),
    pytest.param("stop_sequence", "stop_sequence", False, id="stop_sequence"),
    pytest.param("end_turn", "end_turn", False, id="end_turn"),
    pytest.param("End_Turn", "end_turn", False, id="end_turn-mixed-case"),
]


def _finish_reason_body(finish_reason: str, *, needs_tool_call: bool) -> dict:
    if needs_tool_call:
        return openai_tool_completion(
            ("get_dpd", {"account_ref": "ACC-1"}), finish_reason=finish_reason
        )
    return openai_text_completion("Namaste", finish_reason=finish_reason)


@pytest.mark.parametrize(("finish_reason", "normalised", "needs_tool_call"), _ALLOWLIST_VARIANTS)
def test_every_allowlisted_finish_reason_is_accepted_and_stored_normalised(
    finish_reason, normalised, needs_tool_call, log_stream
) -> None:
    """Case and surrounding whitespace never survive into the stored value."""
    generation = _service(
        _responder(_finish_reason_body(finish_reason, needs_tool_call=needs_tool_call))
    ).generate(_request())

    assert generation.finish_reason == normalised
    [record] = _llm_records(log_stream)
    assert record["outcome"] == "ok"
    assert record["finish_reason"] == normalised


def test_a_null_finish_reason_is_still_accepted_as_complete() -> None:
    """Control: absence of an opinion is not a cut-off. Not every server reports one."""
    generation = _service(
        _responder(openai_text_completion("Namaste", finish_reason=None))
    ).generate(_request())

    assert generation.finish_reason is None


@pytest.mark.parametrize("finish_reason", ["length", "LENGTH", " length", "max_tokens", "model_length"])
def test_every_spelling_of_a_length_cutoff_is_incomplete_with_reason_length(finish_reason) -> None:
    with pytest.raises(LlmIncompleteResponse) as caught:
        _service(
            _responder(openai_text_completion("frag", finish_reason=finish_reason))
        ).generate(_request())

    assert caught.value.reason == "length"


def test_content_filter_is_incomplete_with_its_own_reason() -> None:
    with pytest.raises(LlmIncompleteResponse) as caught:
        _service(
            _responder(openai_text_completion("frag", finish_reason="content_filter"))
        ).generate(_request())

    assert caught.value.reason == "content_filter"


@pytest.mark.parametrize("finish_reason", ["", "abort", "error", "banana"])
def test_every_unrecognised_finish_reason_is_incomplete_with_reason_unrecognised(
    finish_reason,
) -> None:
    """The allowlist means an unfamiliar word is never assumed to mean "finished"."""
    with pytest.raises(LlmIncompleteResponse) as caught:
        _service(
            _responder(openai_text_completion("frag", finish_reason=finish_reason))
        ).generate(_request())

    assert caught.value.reason == "unrecognised"


def test_an_odd_or_long_upstream_finish_reason_never_appears_in_a_log_line(log_stream) -> None:
    """The category, never the string that produced it, is what gets logged."""
    canary = "CANARY-" + "z" * 300
    with pytest.raises(LlmIncompleteResponse):
        _service(_responder(openai_text_completion("frag", finish_reason=canary))).generate(
            _request()
        )

    [record] = _llm_records(log_stream)
    assert record["finish_reason"] == "unrecognised"
    assert canary not in log_stream.getvalue()


# --- 2. legacy message.function_call ------------------------------------------


def _function_call_body(content: Any) -> dict:
    return _body(
        {
            "role": "assistant",
            "content": content,
            "function_call": {"name": "get_dpd", "arguments": "{}"},
        },
        "function_call",
    )


@pytest.mark.parametrize(
    "content", [None, "Sure, let me check that."], ids=["no-content", "with-content"]
)
def test_a_legacy_function_call_field_is_always_an_invalid_tool_call(content) -> None:
    with pytest.raises(LlmInvalidToolCall) as caught:
        _service(_responder(_function_call_body(content))).generate(_request())

    assert caught.value.category == "llm_invalid_tool_call"


# --- 3. finish_reason "tool_calls" with nothing to run ------------------------


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "Let me check."},
        {"role": "assistant", "content": None, "tool_calls": []},
    ],
    ids=["no-tool-calls-key", "empty-tool-calls-array"],
)
def test_a_tool_calls_finish_reason_with_nothing_to_run_is_malformed(message) -> None:
    with pytest.raises(LlmMalformedResponse):
        _service(_responder(_body(message, "tool_calls"))).generate(_request())


def test_a_tool_calls_finish_reason_that_is_not_lower_case_is_still_checked() -> None:
    """The check reads the normalised value, so a server's odd casing cannot bypass it."""
    with pytest.raises(LlmMalformedResponse):
        _service(
            _responder(_body({"role": "assistant", "content": "hi"}, "TOOL_CALLS"))
        ).generate(_request())


# --- 4. unparsed tool-call markup ---------------------------------------------


@pytest.mark.parametrize("marker", _TOOL_CALL_MARKUP)
@pytest.mark.parametrize("case", ["as-sent", "upper", "lower"])
def test_every_tool_call_markup_marker_inside_normal_text_is_malformed(marker, case) -> None:
    variant = {"as-sent": marker, "upper": marker.upper(), "lower": marker.lower()}[case]
    text = f"Sure, one moment {variant} thanks for waiting."

    with pytest.raises(LlmMalformedResponse):
        _service(_responder(openai_text_completion(text))).generate(_request())


def test_a_whole_json_object_reply_is_malformed() -> None:
    with pytest.raises(LlmMalformedResponse):
        _service(_responder(openai_text_completion('{"amount_minor": 250000}'))).generate(
            _request()
        )


def test_a_whole_json_array_reply_is_malformed() -> None:
    with pytest.raises(LlmMalformedResponse):
        _service(_responder(openai_text_completion("[1, 2, 3]"))).generate(_request())


@pytest.mark.parametrize("text", ["I will call you {soon}", "[laughs] okay"])
def test_ordinary_speech_with_braces_or_brackets_that_is_not_json_is_accepted(text) -> None:
    """The control: a brace or a bracket alone is not what the markup check refuses."""
    generation = _service(_responder(openai_text_completion(text))).generate(_request())

    assert generation.text == text


def test_the_markup_check_reads_the_unit_helper_directly_for_a_marker_and_its_case_variants() -> None:
    for marker in _TOOL_CALL_MARKUP:
        assert _is_unparsed_tool_call(f"reply {marker} more")
        assert _is_unparsed_tool_call(f"reply {marker.upper()} more")
    assert _is_unparsed_tool_call('{"a": 1}') is True
    assert _is_unparsed_tool_call("[1]") is True
    assert _is_unparsed_tool_call("I will call you {soon}") is False
    assert _is_unparsed_tool_call("[laughs] okay") is False


def test_over_http_a_markup_reply_fails_the_turn_with_nothing_spoken_and_no_tool_run() -> None:
    endpoint = _Endpoint(
        openai_text_completion('Sure, <tool_call>{"name": "get_dpd"}</tool_call>')
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert result.response_text is None
    assert TurnErrorCategory.LLM_MALFORMED_RESPONSE in result.error_categories
    assert result.tools == ()
    assert len(endpoint.requests) == 1


# --- 5. tool names: checked exactly, never repaired ---------------------------


def _tool_call_entry(name: Any, *, type_: Any = _ABSENT, call_id: str = "call_0") -> dict:
    entry: dict[str, Any] = {"id": call_id, "function": {"name": name, "arguments": "{}"}}
    if type_ is not _ABSENT:
        entry["type"] = type_
    return entry


@pytest.mark.parametrize(
    "name",
    ["  get_dpd ", "get dpd", "a" * 65, "café", ""],
    ids=["leading-trailing-space", "inner-space", "65-chars", "non-ascii", "empty"],
)
def test_an_unusable_tool_name_is_an_invalid_tool_call_and_is_never_repaired(name) -> None:
    with pytest.raises(LlmInvalidToolCall):
        _parse_tool_calls([_tool_call_entry(name)])


def test_a_64_character_tool_name_is_the_longest_one_accepted() -> None:
    name = "a" * 64
    [call] = _parse_tool_calls([_tool_call_entry(name)])
    assert call.tool_name == name


def test_a_tool_call_of_type_custom_is_invalid() -> None:
    with pytest.raises(LlmInvalidToolCall):
        _parse_tool_calls([_tool_call_entry("get_dpd", type_="custom")])


def test_a_tool_call_of_type_function_is_accepted() -> None:
    [call] = _parse_tool_calls([_tool_call_entry("get_dpd", type_="function")])
    assert call.tool_name == "get_dpd"


def test_a_missing_type_is_accepted_as_a_function_call() -> None:
    [call] = _parse_tool_calls([_tool_call_entry("get_dpd", type_=_ABSENT)])
    assert call.tool_name == "get_dpd"


def test_an_unusable_tool_name_over_http_fails_the_turn_as_an_invalid_tool_call() -> None:
    endpoint = _Endpoint(openai_tool_completion(("get dpd", {"account_ref": "ACC-1"})))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.LLM_INVALID_TOOL_CALL in result.error_categories
    assert result.tools == ()


# --- 6. tool-call ids: unusable ones are minted, never kept -------------------


def _call_with_id(call_id: Any) -> dict:
    return {"id": call_id, "function": {"name": "get_dpd", "arguments": "{}"}}


@pytest.mark.parametrize(
    "call_id",
    ["call with spaces", "x" * 129, "café-id", "Priya Sharma called re ACC-9981"],
    ids=["spaces", "over-128-chars", "non-ascii", "pii-like-free-text"],
)
def test_an_unusable_call_id_is_parsed_as_empty_for_the_orchestrator_to_mint(call_id) -> None:
    [call] = _parse_tool_calls([_call_with_id(call_id)])
    assert call.call_id == ""


def test_a_128_character_call_id_is_the_longest_kept_as_is() -> None:
    call_id = "x" * 128
    [call] = _parse_tool_calls([_call_with_id(call_id)])
    assert call.call_id == call_id


def test_a_plain_call_id_is_kept_exactly() -> None:
    [call] = _parse_tool_calls([_call_with_id("call_0")])
    assert call.call_id == "call_0"


def test_over_http_an_unusable_call_id_is_minted_and_used_consistently_and_never_logged(
    log_stream,
) -> None:
    """The model's free text never reaches the wire again, and never the logs."""
    pii_text = "Priya Sharma called re ACC-9981"
    endpoint = _Endpoint(
        openai_tool_completion(("get_outstanding_amount", {}), call_ids=(pii_text,)),
        openai_text_completion(GROUNDED_REPLY),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    minted = result.tools[0].request_id
    assert minted != pii_text

    second_request = endpoint.requests[1]
    echoed = [
        m for m in second_request["messages"] if m["role"] == "assistant" and m.get("tool_calls")
    ]
    tool_results = [m for m in second_request["messages"] if m["role"] == "tool"]
    assert [c["id"] for c in echoed[0]["tool_calls"]] == [minted]
    assert [m["tool_call_id"] for m in tool_results] == [minted]

    wire = json.dumps(endpoint.requests)
    assert pii_text not in wire
    assert pii_text not in log_stream.getvalue()


# --- 7. a repeated JSON key inside tool arguments ------------------------------


def _tool_call_raw(arguments_text: str, tool_name: str = "get_dpd") -> str:
    """A body whose one tool call's ``arguments`` is the JSON *string* ``arguments_text``."""
    return json.dumps(
        _body(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments_text},
                    }
                ],
            },
            "tool_calls",
        )
    )


def _object_form_raw(arguments_literal: str, tool_name: str = "get_dpd") -> str:
    """The same body, but ``arguments`` is a raw object literal, as some servers emit."""
    placeholder = "__ARGS__"
    return _tool_call_raw(placeholder, tool_name).replace(
        json.dumps(placeholder), arguments_literal
    )


_REPEATED_KEY_FORMS = pytest.mark.parametrize(
    "raw_factory", [_tool_call_raw, _object_form_raw], ids=["string-form", "object-form"]
)


@_REPEATED_KEY_FORMS
def test_a_top_level_repeated_key_in_arguments_is_an_invalid_tool_call(raw_factory) -> None:
    with pytest.raises(LlmInvalidToolCall):
        _raw_service(raw_factory('{"amount_minor": NaN, "amount_minor": 1}')).generate(_request())


@_REPEATED_KEY_FORMS
def test_a_nested_repeated_key_in_arguments_is_an_invalid_tool_call(raw_factory) -> None:
    nested = '{"outer": {"amount_minor": 1, "amount_minor": 2}}'
    with pytest.raises(LlmInvalidToolCall):
        _raw_service(raw_factory(nested)).generate(_request())


def test_a_repeated_key_elsewhere_in_the_body_does_not_fail_the_call() -> None:
    """Only tool arguments are checked for repeats. A usage block is observability, not a call."""
    raw = (
        '{"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, '
        '"finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "prompt_tokens": 2}}'
    )

    generation = _raw_service(raw).generate(_request())

    assert generation.text == "hi"
    assert generation.usage is not None
    assert generation.usage.prompt_tokens == 2


def test_a_repeated_key_beside_an_unrepeated_tool_call_still_fails_the_whole_response() -> None:
    raw = _tool_call_raw('{"amount_minor": 250000, "amount_minor": 1}')
    with pytest.raises(LlmInvalidToolCall):
        _raw_service(raw).generate(_request())


# --- 8. nothing unexpected escapes generate() as itself ------------------------


def test_a_transcript_with_a_lone_surrogate_reaches_the_endpoint_as_valid_ascii_json() -> None:
    """A surrogate is escaped, not encoded, so it can never raise a UnicodeEncodeError."""
    raw_bodies: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        raw_bodies.append(request.content)
        return httpx2.Response(200, json=openai_text_completion(GROUNDED_REPLY))

    runtime = make_runtime(llm=http_llm(handler))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("\ud83d how much do I owe?")
    )

    assert result.outcome is TurnOutcome.COMPLETED
    [raw] = raw_bodies
    assert all(byte < 128 for byte in raw), "the request body was not plain ASCII"
    parsed = json.loads(raw)  # must not raise
    contents = [
        m["content"]
        for m in parsed["messages"]
        if isinstance(m.get("content"), str)
    ]
    assert any("\ud83d" in text for text in contents)


def test_a_non_httpx_transport_exception_becomes_a_plain_uncaused_llm_error(log_stream) -> None:
    """A defect in the transport - not one of httpx2's own exceptions - is still caught."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        raise RuntimeError("boom sk-SECRET")

    with pytest.raises(LlmError) as caught:
        _service(handler).generate(_request())

    error = caught.value
    assert type(error) is LlmError
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "boom" not in str(error)
    assert "sk-SECRET" not in str(error)

    [record] = _llm_records(log_stream)
    assert record["outcome"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert record["error_category"] == "llm_failed"
    assert "sk-SECRET" not in log_stream.getvalue()


def test_generate_after_close_on_an_owned_client_is_not_configured() -> None:
    service = OpenAiCompatibleLlmService(base_url=BASE_URL, model="m")
    service.close()

    with pytest.raises(LlmNotConfigured):
        service.generate(_request())


def test_closing_an_injected_client_is_a_no_op_and_generate_still_works() -> None:
    """Control: an injected client's lifecycle is not the adapter's to end."""
    client = httpx2.Client(
        transport=httpx2.MockTransport(_responder(openai_text_completion("ok")))
    )
    service = OpenAiCompatibleLlmService(base_url=BASE_URL, model="m", client=client)

    service.close()
    assert not client.is_closed
    generation = service.generate(_request())
    assert generation.text == "ok"
    client.close()


# --- 9. constructor validation --------------------------------------------------


_BAD_BASE_URLS = [
    pytest.param("http://model\x07.invalid/v1", id="control-char"),
    pytest.param("http://model .invalid/v1", id="space"),
    pytest.param("http://mödel.invalid/v1", id="non-ascii"),
    pytest.param("ftp://x/v1", id="wrong-scheme"),
    pytest.param("http:///v1", id="no-host"),
    pytest.param("http://host:70000/v1", id="port-too-large"),
    pytest.param("http://host:0/v1", id="port-zero"),
    pytest.param("https://user:p/ss#word@host/v1", id="password-with-slash-and-hash"),
]


@pytest.mark.parametrize("base_url", _BAD_BASE_URLS)
def test_an_unusable_base_url_is_refused_without_ever_quoting_it(base_url) -> None:
    with pytest.raises(ValueError) as caught:
        OpenAiCompatibleLlmService(base_url=base_url, model="m")

    message = str(caught.value)
    assert base_url not in message
    assert all(base_url not in str(arg) for arg in caught.value.args)


def test_a_password_with_a_slash_and_a_hash_never_appears_in_the_refusal() -> None:
    """The exact case the module docstring names: an unencoded '/' and '#' in userinfo."""
    with pytest.raises(ValueError) as caught:
        OpenAiCompatibleLlmService(base_url="https://user:p/ss#word@host/v1", model="m")

    assert "p/ss#word" not in str(caught.value)


def test_a_base_url_with_an_empty_dns_label_is_refused() -> None:
    """'a..b' has an empty label between the dots and can never resolve."""
    with pytest.raises(ValueError):
        OpenAiCompatibleLlmService(base_url="http://a..b/v1", model="m")


@pytest.mark.parametrize("host", ["a.", "b.example.com.", "127.0.0.1", "[::1]"])
def test_a_base_url_whose_host_is_not_an_empty_dns_label_is_still_accepted(host: str) -> None:
    """The fix must not reject a bare trailing dot (FQDN root) or IP literals."""
    OpenAiCompatibleLlmService(base_url=f"http://{host}/v1", model="m").close()


def test_userinfo_in_the_base_url_together_with_an_api_key_is_refused() -> None:
    with pytest.raises(ValueError) as caught:
        OpenAiCompatibleLlmService(
            base_url="https://admin:hunter2@model.invalid/v1", model="m", api_key="sk-live-x"
        )

    message = str(caught.value)
    assert "hunter2" not in message
    assert "sk-live-x" not in message


def test_userinfo_in_the_base_url_alone_is_still_allowed() -> None:
    """The pre-existing behaviour this check must not break: a bare userinfo, no api_key."""
    service = OpenAiCompatibleLlmService(
        base_url="https://admin:hunter2@model.invalid/v1", model="m"
    )
    service.close()


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -0.1, 2.1, True])
def test_an_unusable_temperature_is_refused(temperature) -> None:
    with pytest.raises(ValueError):
        OpenAiCompatibleLlmService(base_url=BASE_URL, model="m", temperature=temperature)


@pytest.mark.parametrize("value", [0.0, 2.0])
def test_the_temperature_bounds_themselves_are_accepted(value) -> None:
    service = OpenAiCompatibleLlmService(base_url=BASE_URL, model="m", temperature=value)
    service.close()


@pytest.mark.parametrize("max_output_tokens", [0, -1, True, 1.5])
def test_an_unusable_max_output_tokens_is_refused(max_output_tokens) -> None:
    with pytest.raises(ValueError):
        OpenAiCompatibleLlmService(
            base_url=BASE_URL, model="m", max_output_tokens=max_output_tokens
        )


@pytest.mark.parametrize("field", ["timeout_seconds", "connect_timeout_seconds"])
def test_a_timeout_over_600_seconds_is_refused(field) -> None:
    kwargs = {"connect_timeout_seconds": 600} if field == "timeout_seconds" else {}
    with pytest.raises(ValueError):
        OpenAiCompatibleLlmService(base_url=BASE_URL, model="m", **{field: 601, **kwargs})


def test_600_seconds_is_the_longest_timeout_accepted() -> None:
    service = OpenAiCompatibleLlmService(
        base_url=BASE_URL, model="m", timeout_seconds=600, connect_timeout_seconds=600
    )
    service.close()


_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "REQUEST_METHOD",
)


def test_a_socks_proxy_the_environment_configures_refuses_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No socksio is installed, so a SOCKS proxy in the environment is unusable, not silently skipped."""
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1")

    with pytest.raises(LlmConfigurationError):
        OpenAiCompatibleLlmService(base_url=BASE_URL, model="m")


def test_with_no_proxy_configured_construction_still_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control for the test above."""
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    service = OpenAiCompatibleLlmService(base_url=BASE_URL, model="m")
    service.close()


# --- 10. Settings: never echoes a rejected value, and caps the two timeouts ----


@pytest.fixture
def clean_model_environ(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every environment variable ``Settings`` would read. See test_stage3a_env_example.py."""
    fields = set(Settings.model_fields)
    for field in fields:
        monkeypatch.delenv(field.upper(), raising=False)
    for name in list(os.environ):
        lowered = name.lower()
        if lowered in fields or lowered.startswith("model_"):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_a_password_in_model_base_url_never_appears_in_the_validation_error(
    clean_model_environ: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, model_base_url="htps://user:hunter2@host/v1")

    assert "hunter2" not in str(caught.value)
    assert "hunter2" not in repr(caught.value)


def test_model_timeout_seconds_over_600_is_rejected(clean_model_environ: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_timeout_seconds=601)


def test_model_timeout_seconds_of_600_is_accepted(clean_model_environ: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None, model_timeout_seconds=600).model_timeout_seconds == 600.0


def test_model_connect_timeout_seconds_over_600_is_rejected(
    clean_model_environ: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_connect_timeout_seconds=601)
