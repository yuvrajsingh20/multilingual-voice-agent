"""The orchestrator, driven by a model that answers over HTTP.

The whole chain, end to end and offline:

    ConversationOrchestrator
        -> LlmService
        -> OpenAiCompatibleLlmService
        -> an OpenAI-compatible HTTP endpoint (in-process or on loopback)
        -> LlmGeneration
        -> the orchestrator carries on exactly as before

The point of these tests is that swapping a scripted stand-in for a real HTTP
adapter changes *nothing* about who is in charge. Tools are still executed only
by the registry, identity is still bound by the orchestrator, policy is still
evaluated at all three checkpoints, and a figure the model was not given is still
blocked before it can be spoken - now with the model's words arriving over a
socket instead of from a list.

No model, no GPU, no cloud and no internet is involved. The endpoint is either
``httpx2.MockTransport`` in this process or a ``http.server`` bound to
127.0.0.1.
"""

from __future__ import annotations

import json

import httpx2
import pytest

from app.config import Settings
from app.models.customer import ComplianceContext
from app.models.enums import ToolStatus
from app.orchestrator import (
    ConversationOrchestrator,
    GroundingSource,
    PolicyCheckpoint,
    TurnErrorCategory,
    TurnOutcome,
)
from app.runtime import build_llm_service, build_runtime
from app.services.llm import LlmConfigurationError, NotConfiguredLlmService
from app.services.llm_openai import OpenAiCompatibleLlmService
from tests.conftest import RULES_PATH
from tests.fakes import (
    FakeOpenAiServer,
    openai_text_completion,
    openai_tool_completion,
    unused_loopback_url,
)
from tests.test_orchestrator import (
    GROUNDED_REPLY,
    OUTSTANDING,
    make_runtime,
    make_settings,
    open_session,
    say,
)

ENDPOINT = "http://model.invalid:8000/v1"


class _Endpoint:
    """An in-process OpenAI-compatible endpoint that replies from a script.

    Records every request body so a test can assert what the orchestrator
    actually put on the wire.
    """

    def __init__(self, *responses: object, status: int = 200) -> None:
        self._responses = list(responses)
        self._status = status
        self.requests: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        request.read()
        self.requests.append(json.loads(request.content))
        if not self._responses:
            raise AssertionError("the orchestrator made more model calls than were scripted")
        body = self._responses.pop(0)
        if isinstance(body, Exception):
            raise body
        if isinstance(body, str):
            return httpx2.Response(
                self._status, text=body, headers={"Content-Type": "application/json"}
            )
        return httpx2.Response(self._status, json=body)


def http_llm(endpoint, **kwargs) -> OpenAiCompatibleLlmService:
    """An adapter whose transport is ``endpoint``. No socket is opened."""
    kwargs.setdefault("base_url", ENDPOINT)
    kwargs.setdefault("model", "gemma-stand-in")
    return OpenAiCompatibleLlmService(
        client=httpx2.Client(transport=httpx2.MockTransport(endpoint)), **kwargs
    )


# --- 1. a complete turn, with the model answering over HTTP -----------------


def test_a_turn_completes_with_the_model_answering_over_http() -> None:
    endpoint = _Endpoint(openai_text_completion(GROUNDED_REPLY))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True
    assert result.llm_calls == 1
    assert result.errors == ()
    assert "twelve thousand three hundred forty-five rupees" in (result.response_text or "")
    assert len(endpoint.requests) == 1


def test_the_prompt_the_orchestrator_built_is_what_goes_on_the_wire() -> None:
    """The adapter translates. It does not author, edit or append to a prompt."""
    endpoint = _Endpoint(openai_text_completion(GROUNDED_REPLY))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say("how much do I owe?"))

    messages = endpoint.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "CONSTRAINTS" in messages[0]["content"]
    assert "FACTS" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "how much do I owe?"}


def test_the_model_is_still_never_told_an_account_or_customer_reference() -> None:
    """A Stage 2 invariant, re-checked at the byte level now that HTTP is real."""
    endpoint = _Endpoint(openai_text_completion(GROUNDED_REPLY))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    wire = json.dumps(endpoint.requests[0])
    assert "ACC-1" not in wire
    assert "CUST-1" not in wire
    assert "Test Borrower" not in wire


def test_latency_at_the_model_boundary_is_recorded_on_the_turn() -> None:
    endpoint = _Endpoint(openai_text_completion(GROUNDED_REPLY))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())
    assert result.latency.llm_ms > 0


def test_the_configured_model_name_reaches_the_endpoint_from_the_orchestrator() -> None:
    endpoint = _Endpoint(openai_text_completion(GROUNDED_REPLY))
    runtime = make_runtime(llm=http_llm(endpoint, model="some-other-model-id"))
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())
    assert endpoint.requests[0]["model"] == "some-other-model-id"


# --- 2. a tool call, over HTTP, executed only by the registry ---------------


def test_a_tool_call_arriving_over_http_is_executed_through_the_registry() -> None:
    endpoint = _Endpoint(
        openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-1"})),
        openai_text_completion(GROUNDED_REPLY),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.llm_calls == 2
    assert len(result.tools) == 1
    attempt = result.tools[0]
    assert attempt.tool_name == "get_outstanding_amount"
    assert attempt.dispatched is True
    assert attempt.status is ToolStatus.OK
    assert GroundingSource.TOOL_RESULT in result.grounding_sources
    assert result.latency.tool_ms > 0


def test_the_http_adapter_is_not_what_reached_the_banking_backend() -> None:
    """The adapter parses a tool call. The registry, and only the registry, runs it."""
    endpoint = _Endpoint(
        openai_tool_completion(("record_payment_promise", {"amount_minor": 100_000})),
        openai_text_completion(GROUNDED_REPLY),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    # No promise was written: the orchestrator refused, because conversation
    # state records no promise from the customer. The adapter, which saw the
    # call first, wrote nothing either.
    backend = runtime.tools._tools["record_payment_promise"].backend  # noqa: SLF001
    assert backend.promises == []


def test_the_policy_checkpoints_are_all_still_evaluated_around_an_http_tool_call() -> None:
    endpoint = _Endpoint(
        openai_tool_completion(("get_account_status", {"account_ref": "ACC-1"})),
        openai_text_completion(GROUNDED_REPLY),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert [e.checkpoint for e in result.policy_evaluations] == [
        PolicyCheckpoint.PRE_LLM,
        PolicyCheckpoint.POST_TOOL,
        PolicyCheckpoint.PRE_TTS,
    ]


def test_the_model_still_cannot_choose_which_account_a_tool_reads_over_http() -> None:
    """Identity binding sits above the transport. A model on the wire gains nothing."""
    endpoint = _Endpoint(
        openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-VICTIM"})),
        openai_text_completion(GROUNDED_REPLY),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    # The session's own account was read, not the one the model named. If the
    # model's argument had survived, the tool would have failed on an unknown
    # account_ref instead of returning this session's figures.
    assert result.tools[0].status is ToolStatus.OK
    assert OUTSTANDING in (result.draft_text or "")


def test_the_tool_result_is_fed_back_over_http_as_an_outcome_not_a_payload() -> None:
    endpoint = _Endpoint(
        openai_tool_completion(
            ("get_outstanding_amount", {"account_ref": "ACC-1"}), call_ids=("call_xyz",)
        ),
        openai_text_completion(GROUNDED_REPLY),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    second = endpoint.requests[1]["messages"]
    tool_messages = [m for m in second if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_xyz"
    # An outcome, not the backend payload: paise must never reach the model.
    assert "1234500" not in json.dumps(second)
    assert "ACC-1" not in json.dumps(second)


def test_two_tool_calls_in_one_http_response_are_both_seen_by_the_orchestrator() -> None:
    endpoint = _Endpoint(
        openai_tool_completion(
            ("get_outstanding_amount", {"account_ref": "ACC-1"}),
            ("get_account_status", {"account_ref": "ACC-1"}),
        ),
        openai_text_completion(GROUNDED_REPLY),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert [t.tool_name for t in result.tools] == [
        "get_outstanding_amount",
        "get_account_status",
    ]
    assert all(t.dispatched and t.status is ToolStatus.OK for t in result.tools)


def test_a_tool_the_registry_does_not_know_is_still_rejected_over_http() -> None:
    endpoint = _Endpoint(openai_tool_completion(("wire_transfer", {"amount": 1})))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert result.tools[0].status is ToolStatus.NOT_FOUND


# --- 3. policy still stops the turn before the model is dialled -------------


def test_a_policy_blocked_turn_makes_no_http_request_at_all() -> None:
    endpoint = _Endpoint(openai_text_completion("should never be produced"))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime, compliance=ComplianceContext(grievance_pending=True))

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.POLICY_BLOCKED
    assert result.llm_calls == 0
    assert endpoint.requests == []


def test_an_ungrounded_figure_from_an_http_model_is_still_blocked() -> None:
    """The transport changes nothing about what may be spoken."""
    endpoint = _Endpoint(
        openai_text_completion("Your outstanding balance is INR 99,999.00, pay today.")
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.RESPONSE_BLOCKED
    assert result.speakable is False
    assert result.response_text is None


# --- 4. every HTTP failure mode, as the orchestrator sees it ----------------


@pytest.mark.parametrize(
    ("response", "category"),
    [
        (httpx2.ReadTimeout("slow"), TurnErrorCategory.LLM_TIMEOUT),
        (httpx2.ConnectError("refused"), TurnErrorCategory.LLM_CONNECTION_FAILED),
        ("<html>bad gateway</html>", TurnErrorCategory.LLM_MALFORMED_RESPONSE),
        ({"choices": []}, TurnErrorCategory.LLM_MALFORMED_RESPONSE),
        (
            {"choices": [{"message": {"role": "assistant", "content": None}}]},
            TurnErrorCategory.LLM_EMPTY_RESPONSE,
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {"name": "get_dpd", "arguments": "{broken"},
                                }
                            ],
                        }
                    }
                ]
            },
            TurnErrorCategory.LLM_INVALID_TOOL_CALL,
        ),
    ],
)
def test_a_boundary_failure_ends_the_turn_with_its_own_category(response, category) -> None:
    runtime = make_runtime(llm=http_llm(_Endpoint(response)))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert result.response_text is None
    assert category in result.error_categories


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 502, 503])
def test_an_upstream_http_error_ends_the_turn_without_leaking_the_status_to_a_customer(
    status: int,
) -> None:
    endpoint = _Endpoint({"error": {"message": "internal detail"}}, status=status)
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert result.response_text is None
    assert TurnErrorCategory.LLM_UPSTREAM_ERROR in result.error_categories
    # The operator sees the status; the customer sees nothing at all.
    error = next(e for e in result.errors if e.category is TurnErrorCategory.LLM_UPSTREAM_ERROR)
    assert str(status) in error.detail
    assert "internal detail" not in error.detail


def test_a_boundary_failure_still_does_not_log_the_transcript_or_the_endpoint_path(
    log_stream,
) -> None:
    runtime = make_runtime(llm=http_llm(_Endpoint(httpx2.ConnectError("refused"))))
    session = open_session(runtime)

    ConversationOrchestrator(runtime).process_turn(
        session.session_id, say("mera naam Priya Sharma hai")
    )

    logged = log_stream.getvalue()
    assert "Priya Sharma" not in logged
    assert "/chat/completions" not in logged


def test_a_dead_endpoint_over_a_real_socket_fails_the_turn_cleanly() -> None:
    service = OpenAiCompatibleLlmService(
        base_url=unused_loopback_url(), model="m", timeout_seconds=2.0
    )
    try:
        runtime = make_runtime(llm=service)
        session = open_session(runtime)
        result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())
    finally:
        service.close()

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert TurnErrorCategory.LLM_CONNECTION_FAILED in result.error_categories


def test_a_full_turn_over_a_real_loopback_server() -> None:
    """The same pipeline, over an actual socket rather than a transport stub."""
    with FakeOpenAiServer(body=openai_text_completion(GROUNDED_REPLY)) as server:
        service = OpenAiCompatibleLlmService(
            base_url=server.base_url, model="gemma-stand-in", timeout_seconds=5.0
        )
        try:
            runtime = make_runtime(llm=service)
            session = open_session(runtime)
            result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())
        finally:
            service.close()

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True
    assert server.requests[0].json()["model"] == "gemma-stand-in"


# --- 5. configuration decides whether there is an endpoint at all -----------


def test_an_unconfigured_runtime_builds_no_http_client() -> None:
    """The default. This is what keeps the suite, and a fresh checkout, offline."""
    runtime = build_runtime(make_settings())
    assert isinstance(runtime.llm, NotConfiguredLlmService)


def test_configuring_a_base_url_and_a_model_builds_the_adapter() -> None:
    settings = make_settings(model_base_url="http://localhost:8000/v1", model_name="some-model")
    service = build_llm_service(settings)
    try:
        assert isinstance(service, OpenAiCompatibleLlmService)
    finally:
        service.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_base_url": "http://localhost:8000/v1"},  # no model name
        {"model_name": "some-model"},                    # no base URL
    ],
)
def test_half_a_configuration_fails_loudly_rather_than_falling_back(overrides) -> None:
    with pytest.raises(LlmConfigurationError):
        build_llm_service(make_settings(**overrides))


def test_a_base_url_that_is_not_http_is_rejected_at_startup() -> None:
    with pytest.raises(ValueError):
        make_settings(model_base_url="localhost:8000/v1", model_name="m")


def test_an_empty_environment_variable_reads_as_unconfigured() -> None:
    """`MODEL_BASE_URL=` in a shipped .env template must not half-configure anything."""
    settings = Settings(
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        regulatory_rules_path=RULES_PATH,
        model_base_url="",
        model_name="",
        model_api_key="",
    )
    assert isinstance(build_llm_service(settings), NotConfiguredLlmService)


def test_configuration_reaches_the_adapter_intact() -> None:
    """Every configured knob shows up in the body that leaves the process."""
    settings = make_settings(
        model_base_url="https://model.example:8443/v1",
        model_name="configured-model",
        model_api_key="sk-configured",
        model_timeout_seconds=12.0,
        model_connect_timeout_seconds=2.0,
        model_max_output_tokens=160,
        model_temperature=0.35,
        model_max_retries=2,
    )
    service = build_llm_service(settings)
    endpoint = _Endpoint(openai_text_completion(GROUNDED_REPLY))
    # Swap the transport, keeping everything configuration built. The socket is
    # the only thing replaced; the URL, key, model and sampling settings are the
    # real ones.
    service.close()
    service._client = httpx2.Client(transport=httpx2.MockTransport(endpoint))  # noqa: SLF001
    service._owns_client = True  # noqa: SLF001

    try:
        runtime = make_runtime(llm=service)
        session = open_session(runtime)
        result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())
    finally:
        service.close()

    assert result.outcome is TurnOutcome.COMPLETED
    sent = endpoint.requests[0]
    assert sent["model"] == "configured-model"
    assert sent["max_tokens"] == 160
    assert sent["temperature"] == 0.35
    assert service._timeout.read == 12.0  # noqa: SLF001
    assert service._timeout.connect == 2.0  # noqa: SLF001
