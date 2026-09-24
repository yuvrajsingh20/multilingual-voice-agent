"""The OpenAI-compatible HTTP adapter.

Nothing here reaches the internet, a cloud, a model registry or a GPU. Most tests
drive the adapter through an in-process transport stub; a handful use a real HTTP
server bound to 127.0.0.1 on an ephemeral port, for the properties that only an
actual socket can demonstrate - that the configured URL is the one dialled, that
the ``Authorization`` header leaves the process, and that a slow server trips the
configured timeout.

What is being pinned
--------------------
Two things, repeatedly. First, that every way a model server can misbehave ends
as a typed, categorised failure rather than as a plausible-looking generation:
the adapter never repairs a body, never invents a tool call and never lets an
empty answer through as an empty turn. Second, that nothing sensitive escapes -
not the API key, not the response body, not the prompt.
"""

from __future__ import annotations

import json
import logging

import httpx2
import pytest

from app.observability import configure_logging
from app.services.llm import (
    LlmConnectionFailed,
    LlmEmptyResponse,
    LlmInvalidToolCall,
    LlmMalformedResponse,
    LlmMessage,
    LlmRequest,
    LlmTimeout,
    LlmToolSpec,
    LlmUpstreamError,
)
from app.services.llm_openai import OpenAiCompatibleLlmService
from app.tools.banking import build_registry
from tests.fakes import (
    FakeOpenAiServer,
    openai_text_completion,
    openai_tool_completion,
    unused_loopback_url,
)

BASE_URL = "http://model.invalid:8000/v1"


def _request(**kwargs) -> LlmRequest:
    return LlmRequest(
        messages=(
            LlmMessage(role="system", content="CONSTRAINTS\n- tone: neutral"),
            LlmMessage(role="user", content="kitna bakaya hai?"),
        ),
        **kwargs,
    )


def _service(handler, **kwargs) -> OpenAiCompatibleLlmService:
    """An adapter wired to an in-process transport. No socket is opened."""
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("model", "configured-model")
    return OpenAiCompatibleLlmService(client=client, **kwargs)


def _responder(body, *, status: int = 200, raw: str | None = None):
    """A handler that answers every request identically, recording what it saw."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        seen.append(request)
        if raw is not None:
            return httpx2.Response(status, text=raw, headers={"Content-Type": "application/json"})
        return httpx2.Response(status, json=body)

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


# --- 1. normal text response ------------------------------------------------


def test_a_normal_assistant_reply_becomes_a_generation() -> None:
    handler = _responder(openai_text_completion("Namaste, main bank se baat kar raha hoon."))
    generation = _service(handler).generate(_request())

    assert generation.text == "Namaste, main bank se baat kar raha hoon."
    assert generation.tool_calls == ()
    assert generation.finish_reason == "stop"
    assert generation.model == "test-model"


def test_latency_is_measured_at_the_boundary() -> None:
    generation = _service(_responder(openai_text_completion("ok"))).generate(_request())
    assert generation.latency_ms > 0


def test_usage_is_carried_when_the_server_reports_it() -> None:
    body = openai_text_completion(
        "ok", usage={"prompt_tokens": 310, "completion_tokens": 24, "total_tokens": 334}
    )
    usage = _service(_responder(body)).generate(_request()).usage
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (310, 24, 334)


def test_a_missing_usage_block_is_absent_rather_than_zero() -> None:
    assert _service(_responder(openai_text_completion("ok"))).generate(_request()).usage is None


def test_an_unreadable_usage_block_does_not_cost_the_turn() -> None:
    """Token accounting is observability. Losing it must not fail a customer turn."""
    body = openai_text_completion("ok", usage={"prompt_tokens": "lots"})
    generation = _service(_responder(body)).generate(_request())
    assert generation.text == "ok"
    assert generation.usage is None


# --- 2 & 3. tool calls ------------------------------------------------------


def test_a_tool_call_response_is_parsed_into_the_projects_tool_call_type() -> None:
    body = openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-9"}))
    generation = _service(_responder(body)).generate(_request())

    assert generation.text is None
    assert len(generation.tool_calls) == 1
    call = generation.tool_calls[0]
    assert call.tool_name == "get_outstanding_amount"
    assert call.arguments == {"account_ref": "ACC-9"}
    assert call.call_id == "call_0"
    assert generation.finish_reason == "tool_calls"


def test_multiple_tool_calls_are_all_preserved_in_order() -> None:
    body = openai_tool_completion(
        ("get_outstanding_amount", {"account_ref": "ACC-1"}),
        ("get_account_status", {"account_ref": "ACC-1"}),
    )
    calls = _service(_responder(body)).generate(_request()).tool_calls
    assert [c.tool_name for c in calls] == ["get_outstanding_amount", "get_account_status"]


def test_text_alongside_a_tool_call_is_kept() -> None:
    body = openai_tool_completion(("get_dpd", {}), content="Ek minute, main check karta hoon.")
    generation = _service(_responder(body)).generate(_request())
    assert generation.text == "Ek minute, main check karta hoon."
    assert len(generation.tool_calls) == 1


def test_arguments_serialised_as_an_object_are_accepted_too() -> None:
    """Some OpenAI-compatible servers emit the object rather than a JSON string."""
    body = {
        "model": "m",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_dpd", "arguments": {"account_ref": "A"}},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    call = _service(_responder(body)).generate(_request()).tool_calls[0]
    assert call.arguments == {"account_ref": "A"}


def test_absent_arguments_become_an_empty_mapping_not_a_guess() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "get_dpd"}}
                    ],
                }
            }
        ]
    }
    assert _service(_responder(body)).generate(_request()).tool_calls[0].arguments == {}


def test_a_tool_call_with_no_id_is_tolerated_and_left_for_the_orchestrator() -> None:
    """The orchestrator mints a request id when the model supplies none."""
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"type": "function", "function": {"name": "get_dpd", "arguments": "{}"}}
                    ],
                }
            }
        ]
    }
    assert _service(_responder(body)).generate(_request()).tool_calls[0].call_id == ""


@pytest.mark.parametrize(
    "arguments",
    [
        "{not json",           # the model produced a broken string
        "[1, 2, 3]",           # valid JSON, but not an object
        '"ACC-1"',             # valid JSON, but a bare string
        42,                    # not a string and not an object
    ],
)
def test_unreadable_tool_arguments_raise_rather_than_being_repaired(arguments) -> None:
    """A repaired tool call is a tool call the application invented."""
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_dpd", "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }
    with pytest.raises(LlmInvalidToolCall):
        _service(_responder(body)).generate(_request())


@pytest.mark.parametrize(
    "tool_calls",
    [
        "get_dpd",                                                      # not an array
        [{"id": "c1", "type": "function"}],                             # no function object
        [{"id": "c1", "function": {"arguments": "{}"}}],                # no name
        [{"id": "c1", "function": {"name": "", "arguments": "{}"}}],    # empty name
        [{"id": 7, "function": {"name": "get_dpd", "arguments": "{}"}}],  # id of the wrong type
    ],
)
def test_a_malformed_tool_call_structure_is_rejected(tool_calls) -> None:
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}}
        ]
    }
    with pytest.raises(LlmInvalidToolCall):
        _service(_responder(body)).generate(_request())


# --- 4 & 5. malformed and incomplete bodies ---------------------------------


def test_a_body_that_is_not_json_is_a_malformed_response() -> None:
    with pytest.raises(LlmMalformedResponse):
        _service(_responder(None, raw="<html>502 Bad Gateway</html>")).generate(_request())


@pytest.mark.parametrize(
    "body",
    [
        {},                                                   # no choices
        {"choices": []},                                      # empty choices
        {"choices": "nope"},                                  # choices of the wrong type
        {"choices": ["text"]},                                # choice is not an object
        {"choices": [{}]},                                    # no message
        {"choices": [{"message": "hello"}]},                  # message is not an object
        {"choices": [{"message": {"content": 12345}}]},       # content is not a string
        [1, 2, 3],                                            # body is not an object
    ],
)
def test_a_body_missing_expected_fields_is_a_malformed_response(body) -> None:
    with pytest.raises(LlmMalformedResponse):
        _service(_responder(body)).generate(_request())


def test_a_malformed_response_never_carries_the_upstream_body() -> None:
    """An upstream body is attacker-influenceable and routinely echoes the request."""
    secret_ish = "<html>token=sk-live-abc123 account=ACC-1</html>"
    with pytest.raises(LlmMalformedResponse) as caught:
        _service(_responder(None, raw=secret_ish)).generate(_request())
    assert "sk-live-abc123" not in str(caught.value)
    assert "ACC-1" not in str(caught.value)


# --- 6-10. HTTP error statuses ----------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429, 500, 502, 503, 504])
def test_every_http_error_status_becomes_a_typed_upstream_error(status: int) -> None:
    with pytest.raises(LlmUpstreamError) as caught:
        _service(_responder({"error": {"message": "nope"}}, status=status)).generate(_request())
    assert caught.value.status_code == status
    assert caught.value.category == "llm_upstream_error"


def test_an_http_error_body_is_discarded_at_the_boundary() -> None:
    body = {"error": {"message": "invalid api key sk-live-abc123 for org ACME"}}
    with pytest.raises(LlmUpstreamError) as caught:
        _service(_responder(body, status=401)).generate(_request())
    assert "sk-live-abc123" not in str(caught.value)
    assert "ACME" not in str(caught.value)


# --- 11 & 12. transport failures --------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [httpx2.ReadTimeout("slow"), httpx2.ConnectTimeout("slow"), httpx2.PoolTimeout("slow")],
)
def test_a_timeout_becomes_an_llm_timeout(exc: Exception) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exc

    with pytest.raises(LlmTimeout):
        _service(handler).generate(_request())


@pytest.mark.parametrize(
    "exc",
    [
        httpx2.ConnectError("refused"),
        httpx2.RemoteProtocolError("truncated"),
        httpx2.ProxyError("no proxy"),
    ],
)
def test_a_transport_failure_becomes_a_connection_failure(exc: Exception) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exc

    with pytest.raises(LlmConnectionFailed):
        _service(handler).generate(_request())


def test_a_real_closed_port_is_a_connection_failure() -> None:
    """Not a simulated transport error: nothing is listening on this port."""
    service = OpenAiCompatibleLlmService(
        base_url=unused_loopback_url(), model="m", timeout_seconds=2.0
    )
    try:
        with pytest.raises(LlmConnectionFailed):
            service.generate(_request())
    finally:
        service.close()


# --- 13. empty model response -----------------------------------------------


@pytest.mark.parametrize("content", [None, "", "   \n\t "])
def test_a_model_that_says_nothing_is_a_failure_not_an_empty_turn(content) -> None:
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    with pytest.raises(LlmEmptyResponse):
        _service(_responder(body)).generate(_request())


def test_empty_content_with_a_tool_call_is_not_an_empty_response() -> None:
    body = openai_tool_completion(("get_dpd", {}), content="")
    assert len(_service(_responder(body)).generate(_request()).tool_calls) == 1


# --- 14-16. what actually goes on the wire ----------------------------------


def test_the_configured_model_name_is_what_is_sent() -> None:
    handler = _responder(openai_text_completion("ok"))
    _service(handler, model="gemma-stand-in-abc").generate(_request())
    assert json.loads(handler.seen[0].content)["model"] == "gemma-stand-in-abc"


def test_the_configured_base_url_is_what_is_dialled() -> None:
    handler = _responder(openai_text_completion("ok"))
    _service(handler, base_url="https://elsewhere.invalid:9443/openai/v1").generate(_request())
    assert str(handler.seen[0].url) == "https://elsewhere.invalid:9443/openai/v1/chat/completions"


def test_a_trailing_slash_on_the_base_url_does_not_double_up() -> None:
    handler = _responder(openai_text_completion("ok"))
    _service(handler, base_url="http://m.invalid:8000/v1/").generate(_request())
    assert str(handler.seen[0].url) == "http://m.invalid:8000/v1/chat/completions"


def test_the_configured_api_key_is_sent_as_a_bearer_token() -> None:
    handler = _responder(openai_text_completion("ok"))
    _service(handler, api_key="sk-test-secret").generate(_request())
    assert handler.seen[0].headers["authorization"] == "Bearer sk-test-secret"


def test_no_authorization_header_is_sent_when_no_key_is_configured() -> None:
    """A local vLLM server needs no key, and `Bearer ` is worse than nothing."""
    handler = _responder(openai_text_completion("ok"))
    _service(handler, api_key=None).generate(_request())
    assert "authorization" not in handler.seen[0].headers
    handler2 = _responder(openai_text_completion("ok"))
    _service(handler2, api_key="   ").generate(_request())
    assert "authorization" not in handler2.seen[0].headers


def test_messages_are_translated_into_openai_role_content_objects() -> None:
    handler = _responder(openai_text_completion("ok"))
    request = LlmRequest(
        messages=(
            LlmMessage(role="system", content="rules"),
            LlmMessage(role="user", content="kitna?"),
            LlmMessage(role="assistant", content=""),
            LlmMessage(role="tool", content="get_dpd ok.", tool_call_id="req-7"),
        )
    )
    _service(handler).generate(request)
    sent = json.loads(handler.seen[0].content)["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[3]["tool_call_id"] == "req-7"
    assert "tool_call_id" not in sent[0]


def test_tool_specs_are_sent_in_the_openai_function_shape_unaltered() -> None:
    handler = _responder(openai_text_completion("ok"))
    specs = build_registry().specs()
    _service(handler).generate(_request(tools=specs))

    sent = json.loads(handler.seen[0].content)["tools"]
    assert [t["type"] for t in sent] == ["function"] * len(specs)
    assert [t["function"]["name"] for t in sent] == [s.name for s in specs]
    # The registry's own schema, not a rewritten one: the model must not be
    # offered a looser contract than the registry will enforce.
    assert sent[0]["function"]["parameters"] == specs[0].parameters


def test_no_tools_key_is_sent_when_no_tools_are_offered() -> None:
    handler = _responder(openai_text_completion("ok"))
    _service(handler).generate(_request())
    assert "tools" not in json.loads(handler.seen[0].content)


def test_streaming_is_never_requested() -> None:
    handler = _responder(openai_text_completion("ok"))
    _service(handler).generate(_request())
    assert json.loads(handler.seen[0].content)["stream"] is False


# --- sampling configuration -------------------------------------------------


def test_configuration_supplies_the_output_cap_and_temperature() -> None:
    handler = _responder(openai_text_completion("ok"))
    _service(handler, max_output_tokens=96, temperature=0.05).generate(_request())
    sent = json.loads(handler.seen[0].content)
    assert sent["max_tokens"] == 96
    assert sent["temperature"] == 0.05


def test_an_explicit_request_value_wins_over_configuration() -> None:
    handler = _responder(openai_text_completion("ok"))
    _service(handler, max_output_tokens=96, temperature=0.05).generate(
        _request(max_output_tokens=1024, temperature=0.9)
    )
    sent = json.loads(handler.seen[0].content)
    assert sent["max_tokens"] == 1024
    assert sent["temperature"] == 0.9


def test_without_configuration_the_requests_own_defaults_are_sent() -> None:
    handler = _responder(openai_text_completion("ok"))
    _service(handler).generate(_request())
    sent = json.loads(handler.seen[0].content)
    assert sent["max_tokens"] == 512
    assert sent["temperature"] == 0.2


# --- 17. timeouts are configured and respected ------------------------------


def test_the_configured_timeout_is_applied_to_the_client() -> None:
    service = OpenAiCompatibleLlmService(
        base_url=BASE_URL, model="m", timeout_seconds=7.5, connect_timeout_seconds=1.25
    )
    try:
        timeout = service._timeout  # noqa: SLF001 - asserting configuration reached the client
        assert timeout.read == 7.5
        assert timeout.connect == 1.25
    finally:
        service.close()


def test_the_connect_timeout_falls_back_to_the_request_timeout() -> None:
    service = OpenAiCompatibleLlmService(base_url=BASE_URL, model="m", timeout_seconds=3.0)
    try:
        assert service._timeout.connect == 3.0  # noqa: SLF001
    finally:
        service.close()


def test_a_slow_server_trips_the_configured_timeout() -> None:
    """A real socket, a real stall, a real deadline."""
    with FakeOpenAiServer(body=openai_text_completion("too late"), delay_seconds=3.0) as server:
        service = OpenAiCompatibleLlmService(
            base_url=server.base_url, model="m", timeout_seconds=0.2
        )
        try:
            with pytest.raises(LlmTimeout):
                service.generate(_request())
        finally:
            service.close()


# --- retries: bounded, and never on a request defect ------------------------


def test_retries_are_off_by_default() -> None:
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        return httpx2.Response(503, json={})

    with pytest.raises(LlmUpstreamError):
        _service(handler).generate(_request())
    assert len(attempts) == 1


def test_a_transient_status_is_retried_up_to_the_configured_bound() -> None:
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        return httpx2.Response(503, json={})

    with pytest.raises(LlmUpstreamError):
        _service(handler, max_retries=2).generate(_request())
    assert len(attempts) == 3  # the original plus two retries, and then it stops


def test_a_retry_that_succeeds_returns_the_generation() -> None:
    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx2.Response(429, json={})
        return httpx2.Response(200, json=openai_text_completion("second time lucky"))

    assert _service(handler, max_retries=1).generate(_request()).text == "second time lucky"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_a_client_error_is_never_retried(status: int) -> None:
    """A 400 or a 401 is a bug or a credential problem; repeating it fixes neither."""
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        return httpx2.Response(status, json={})

    with pytest.raises(LlmUpstreamError):
        _service(handler, max_retries=3).generate(_request())
    assert len(attempts) == 1


def test_a_read_timeout_is_not_retried() -> None:
    """The turn has already spent its whole latency budget once."""
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        raise httpx2.ReadTimeout("slow")

    with pytest.raises(LlmTimeout):
        _service(handler, max_retries=3).generate(_request())
    assert len(attempts) == 1


def test_a_connection_failure_is_retried() -> None:
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        raise httpx2.ConnectError("refused")

    with pytest.raises(LlmConnectionFailed):
        _service(handler, max_retries=2).generate(_request())
    assert len(attempts) == 3


def test_a_malformed_body_is_never_retried() -> None:
    """The same request would produce the same body."""
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        return httpx2.Response(200, text="<html>", headers={"Content-Type": "application/json"})

    with pytest.raises(LlmMalformedResponse):
        _service(handler, max_retries=3).generate(_request())
    assert len(attempts) == 1


def test_a_negative_retry_bound_is_rejected() -> None:
    with pytest.raises(ValueError):
        _service(_responder({}), max_retries=-1)


@pytest.mark.parametrize("kwargs", [{"base_url": ""}, {"model": ""}, {"model": "  "}])
def test_the_adapter_refuses_to_be_built_without_an_endpoint_and_a_model(kwargs) -> None:
    with pytest.raises(ValueError):
        _service(_responder({}), **kwargs)


# --- logging ----------------------------------------------------------------


def _captured(stream) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_a_successful_call_logs_categories_and_measurements_only(log_stream) -> None:
    body = openai_text_completion(
        "Namaste", usage={"prompt_tokens": 310, "completion_tokens": 24, "total_tokens": 334}
    )
    _service(
        _responder(body), api_key="sk-live-supersecret", model="gemma-stand-in"
    ).generate(_request())

    records = [r for r in _captured(log_stream) if r.get("event") == "llm_call"]
    assert len(records) == 1
    record = records[0]
    assert record["provider"] == "openai-compatible"
    assert record["model"] == "gemma-stand-in"
    assert record["endpoint"] == "http://model.invalid:8000"
    assert record["outcome"] == "ok"
    assert record["http_status"] == 200
    assert record["tool_call_count"] == 0
    assert record["model_latency_ms"] >= 0
    assert record["usage_prompt"] == 310
    assert record["usage_completion"] == 24
    assert record["usage_total"] == 334
    assert record["request_id"]


def test_the_api_key_never_appears_in_a_log_line(log_stream) -> None:
    _service(
        _responder(openai_text_completion("Namaste")), api_key="sk-live-supersecret"
    ).generate(_request())
    assert "sk-live-supersecret" not in log_stream.getvalue()
    assert "Bearer" not in log_stream.getvalue()
    assert "authorization" not in log_stream.getvalue().lower()


def test_the_prompt_and_the_draft_never_appear_in_a_log_line(log_stream) -> None:
    secret_prompt = "outstanding: INR 12,345.00 for Priya Sharma on ACC-77"
    request = LlmRequest(messages=(LlmMessage(role="system", content=secret_prompt),))
    _service(_responder(openai_text_completion("aapka bakaya INR 12,345.00 hai"))).generate(request)

    logged = log_stream.getvalue()
    assert "Priya Sharma" not in logged
    assert "ACC-77" not in logged
    assert "12,345.00" not in logged


def test_a_failure_is_logged_as_a_category_and_a_status(log_stream) -> None:
    with pytest.raises(LlmUpstreamError):
        _service(_responder({"error": {"message": "key sk-live-xyz revoked"}}, status=401)).generate(
            _request()
        )

    record = [r for r in _captured(log_stream) if r.get("event") == "llm_call"][-1]
    assert record["outcome"] == "failed"
    assert record["error_category"] == "llm_upstream_error"
    assert record["http_status"] == 401
    assert "sk-live-xyz" not in log_stream.getvalue()


def test_each_retry_is_logged_so_a_degrading_endpoint_is_visible(log_stream) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503, json={})

    with pytest.raises(LlmUpstreamError):
        _service(handler, max_retries=2).generate(_request())

    records = [r for r in _captured(log_stream) if r.get("event") == "llm_call"]
    assert [r["outcome"] for r in records] == ["retrying", "retrying", "failed"]
    assert [r["attempt"] for r in records] == [1, 2, 3]


def test_credentials_embedded_in_the_base_url_are_not_logged(log_stream) -> None:
    """A base URL may legitimately carry userinfo. Logging it verbatim publishes it."""
    _service(
        _responder(openai_text_completion("ok")),
        base_url="https://admin:hunter2@model.invalid:8443/v1",
    ).generate(_request())

    logged = log_stream.getvalue()
    assert "hunter2" not in logged
    assert "admin" not in logged
    assert "https://model.invalid:8443" in logged


def test_the_repr_does_not_carry_the_api_key() -> None:
    service = _service(_responder({}), api_key="sk-live-supersecret")
    assert "sk-live-supersecret" not in repr(service)


def test_a_field_named_like_a_credential_is_still_redacted(log_stream) -> None:
    """The adapter relies on app.observability.redact as a last line of defence."""
    from app.observability import get_logger, log_event

    log_event(get_logger("test"), "probe", api_key="sk-live-x", authorization="Bearer y", model="m")
    record = _captured(log_stream)[-1]
    assert record["api_key"] == "[redacted]"
    assert record["authorization"] == "[redacted]"
    assert record["model"] == "m"


# --- the real server: base URL, headers and model name over an actual socket -


def test_over_a_real_socket_the_url_key_and_model_all_arrive() -> None:
    with FakeOpenAiServer(body=openai_text_completion("Namaste")) as server:
        service = OpenAiCompatibleLlmService(
            base_url=server.base_url,
            model="gemma-stand-in-e4b",
            api_key="sk-test-secret",
            timeout_seconds=5.0,
        )
        try:
            generation = service.generate(_request(tools=build_registry().specs()))
        finally:
            service.close()

    assert generation.text == "Namaste"
    assert len(server.requests) == 1
    received = server.requests[0]
    assert received.path == "/v1/chat/completions"
    assert received.headers["authorization"] == "Bearer sk-test-secret"
    assert received.headers["content-type"] == "application/json"
    body = received.json()
    assert body["model"] == "gemma-stand-in-e4b"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["tools"][0]["type"] == "function"


def test_over_a_real_socket_a_tool_call_round_trips() -> None:
    body = openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-1"}))
    with FakeOpenAiServer(body=body) as server:
        service = OpenAiCompatibleLlmService(
            base_url=server.base_url, model="m", timeout_seconds=5.0
        )
        try:
            generation = service.generate(_request())
        finally:
            service.close()

    assert generation.tool_calls[0].tool_name == "get_outstanding_amount"
    assert generation.tool_calls[0].arguments == {"account_ref": "ACC-1"}


def test_over_a_real_socket_a_500_is_an_upstream_error() -> None:
    with FakeOpenAiServer(status=500, body={"error": "boom"}) as server:
        service = OpenAiCompatibleLlmService(
            base_url=server.base_url, model="m", timeout_seconds=5.0
        )
        try:
            with pytest.raises(LlmUpstreamError) as caught:
                service.generate(_request())
        finally:
            service.close()
    assert caught.value.status_code == 500


def test_over_a_real_socket_a_non_json_body_is_malformed() -> None:
    with FakeOpenAiServer(raw_body=b"<html>gateway</html>") as server:
        service = OpenAiCompatibleLlmService(
            base_url=server.base_url, model="m", timeout_seconds=5.0
        )
        try:
            with pytest.raises(LlmMalformedResponse):
                service.generate(_request())
        finally:
            service.close()


# --- the adapter is a transport, and nothing more ---------------------------


def test_the_adapter_never_executes_a_tool() -> None:
    """A tool call leaves here as data. Only the registry runs anything."""
    executed: list[str] = []
    registry = build_registry()
    original = registry.execute

    def tripwire(request):  # pragma: no cover - the point is that it is not called
        executed.append(request.tool_name)
        return original(request)

    registry.execute = tripwire  # type: ignore[method-assign]

    body = openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-1"}))
    generation = _service(_responder(body)).generate(_request(tools=registry.specs()))

    assert generation.tool_calls  # the request was parsed
    assert executed == []  # and nothing was run


def test_the_adapter_holds_no_conversation_state_between_calls() -> None:
    """Two identical calls produce two identical requests. Nothing accumulates."""
    handler = _responder(openai_text_completion("ok"))
    service = _service(handler)
    service.generate(_request())
    service.generate(_request())
    first, second = (json.loads(r.content) for r in handler.seen)
    assert first == second


def test_closing_an_injected_client_is_left_to_its_owner() -> None:
    client = httpx2.Client(transport=httpx2.MockTransport(_responder(openai_text_completion("ok"))))
    service = OpenAiCompatibleLlmService(base_url=BASE_URL, model="m", client=client)
    service.close()
    assert not client.is_closed


def test_the_service_closes_its_own_client_as_a_context_manager() -> None:
    with OpenAiCompatibleLlmService(base_url=BASE_URL, model="m") as service:
        client = service._client  # noqa: SLF001
    assert client.is_closed
