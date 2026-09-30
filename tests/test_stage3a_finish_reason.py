"""Stage 3A, finding F: a cut-off generation is never presented as a completed one.

What is pinned, and why
-----------------------
An OpenAI-compatible server says why the model stopped in ``finish_reason``.
``stop`` (or ``tool_calls``) means it finished. ``length`` means the output-token
cap cut it off, and ``content_filter`` means the server withheld part of it.
Before the fix the adapter read ``finish_reason`` only to copy it onto the
generation, so a fragment came back as a normal generation. Validation checks
that figures are grounded. It does not check that a sentence is whole, so a
reply cut off at "You do not need to pay INR 12,345.00 because" would pass and
be spoken.

The fix, and what this file holds it to:

- ``length`` and ``content_filter`` raise :class:`LlmIncompleteResponse`
  (category ``llm_incomplete_response``, ``reason`` one of the adapter's own
  constants, ``detail`` ``"llm_incomplete_response (<reason>)"``). This
  happens whatever the message holds: text, a tool call, nothing at all, or a
  tool call whose arguments were truncated mid-JSON. ``finish_reason`` is read
  straight after the message object is found, before content or tool calls, so
  a cut-off reply is reported as cut off rather than as an empty reply or an
  unreadable tool call.
- ``finish_reason`` absent, ``null``, or any other string (TGI's ``eos_token``,
  ``stop_sequence``) is a completed generation, and the string is recorded.
- A ``finish_reason`` that is not a string is a malformed body. Before the fix
  it was quietly dropped to ``None``.
- An incomplete response is never retried, is logged as ``failed`` with its
  ``finish_reason`` and ``http_status`` 200, and carries none of the generated
  text in its message, its ``detail`` or the log.
- In the orchestrator the turn fails with
  ``TurnErrorCategory.LLM_INCOMPLETE_RESPONSE``. Nothing is spoken, no agent
  utterance is recorded, and the duties the caller claimed for the utterance
  (recording disclosure, agent identification) are not marked as done. That
  holds on the first request and on the request after a tool round.
- Every ``LlmError`` subclass in :mod:`app.services.llm` has a turn error
  category of its own, and the orchestrator maps each one to it rather than
  to the ``LLM_FAILED`` fallback.

No model, network or wall clock is involved. The adapter is driven through
``httpx2.MockTransport``, and retries sleep on a fake clock. Names the fix
introduced are looked up inside the tests, so this file still collects
against the code before the fix and fails there on assertions.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest

import app.services.llm as llm_module
import app.services.llm_openai as llm_openai
from app.models.enums import ConversationStage, EventKind, RequiredAction, ToolStatus
from app.orchestrator import ConversationOrchestrator, TurnErrorCategory, TurnOutcome, TurnResult
from app.orchestrator.pipeline import _llm_error_category
from app.services.llm import (
    LlmEmptyResponse,
    LlmError,
    LlmInvalidToolCall,
    LlmMalformedResponse,
    LlmMessage,
    LlmRequest,
)
from app.services.llm_openai import OpenAiCompatibleLlmService
from tests.fakes import openai_text_completion, openai_tool_completion
from tests.test_llm_http_integration import _Endpoint, http_llm
from tests.test_orchestrator import GROUNDED_REPLY, OUTSTANDING, make_runtime, open_session, say

BASE_URL = "http://model.invalid:8000/v1"

#: What a cut-off reply looks like: grounded figures, and a sentence that stops
#: before it has said what it meant. Nothing in this string may reach an error,
#: a log line or a speaker.
FRAGMENT = f"You do not need to pay {OUTSTANDING} because"

#: The greeting the claimed-actions test uses. It passes validation when the
#: caller claims both in-turn duties.
GREETING = "Good morning. This call is recorded. I am calling from the bank."

_ABSENT = object()


# --- helpers ------------------------------------------------------------------


def _incomplete_error() -> type[LlmError]:
    """``LlmIncompleteResponse``, looked up so that this file collects without it."""
    cls = getattr(llm_module, "LlmIncompleteResponse", None)
    assert cls is not None, "app.services.llm defines no LlmIncompleteResponse"
    assert issubclass(cls, LlmError)
    return cls


def _incomplete_category() -> TurnErrorCategory:
    member = getattr(TurnErrorCategory, "LLM_INCOMPLETE_RESPONSE", None)
    assert member is not None, "TurnErrorCategory has no LLM_INCOMPLETE_RESPONSE"
    return member


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
    """A handler that replays ``bodies`` (the last one repeats) and counts requests."""
    remaining = list(bodies)
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        seen.append(json.loads(request.content))
        body = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(body, httpx2.Response):
            return body
        return httpx2.Response(status, json=body)

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


def _body(message: Any, finish_reason: Any = _ABSENT) -> dict:
    """A chat-completions body with one choice. ``_ABSENT`` leaves the key out."""
    choice: dict[str, Any] = {"index": 0, "message": message}
    if finish_reason is not _ABSENT:
        choice["finish_reason"] = finish_reason
    return {"id": "chatcmpl-f", "object": "chat.completion", "model": "m", "choices": [choice]}


def _text(content: Any, finish_reason: Any = _ABSENT) -> dict:
    return _body({"role": "assistant", "content": content}, finish_reason)


def _truncated_tool_call(finish_reason: str = "length") -> dict:
    """A tool call whose JSON arguments stop mid-string, as a token cap leaves them."""
    return _body(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "get_outstanding_amount",
                        "arguments": '{"account_ref": "AC',
                    },
                }
            ],
        },
        finish_reason,
    )


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Record the adapter's waits instead of taking them. Clocks stay real."""
    taken: list[float] = []
    monkeypatch.setattr(
        llm_openai,
        "time",
        SimpleNamespace(
            monotonic=time.monotonic, perf_counter=time.perf_counter, sleep=taken.append
        ),
    )
    return taken


def _cut_off(handler, **kwargs) -> LlmError:
    """Run one call that must end as an incomplete response, and return the error.

    Catches the base class and then names the type, so that a call that returns,
    or that fails as something else, is reported as such.
    """
    with pytest.raises(LlmError) as caught:
        _service(handler, **kwargs).generate(_request())
    error = caught.value
    assert type(error).__name__ == "LlmIncompleteResponse", f"raised {type(error).__name__}"
    assert type(error) is _incomplete_error()
    return error


def _llm_records(stream) -> list[dict]:
    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return [r for r in records if r.get("event") == "llm_call"]


def _agent_utterances(runtime, session_id: str) -> list:
    stored = runtime.sessions.get(session_id)
    return [e for e in stored.events if e.kind is EventKind.AGENT_UTTERANCE]


# --- completed generations are still generations (controls) --------------------


def test_a_stop_reply_is_a_generation_with_its_text_and_finish_reason() -> None:
    generation = _service(_responder(openai_text_completion(GROUNDED_REPLY))).generate(_request())

    assert generation.text == GROUNDED_REPLY
    assert generation.finish_reason == "stop"
    assert generation.tool_calls == ()


def test_a_tool_calls_reply_is_parsed_into_tool_calls() -> None:
    body = openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-9"}))
    generation = _service(_responder(body)).generate(_request())

    assert generation.finish_reason == "tool_calls"
    assert [(c.call_id, c.tool_name, c.arguments) for c in generation.tool_calls] == [
        ("call_0", "get_outstanding_amount", {"account_ref": "ACC-9"})
    ]


@pytest.mark.parametrize("finish_reason", [_ABSENT, None], ids=["absent", "null"])
def test_a_reply_that_does_not_say_why_it_stopped_is_accepted(finish_reason) -> None:
    """Not every compatible server sends ``finish_reason``. Its absence is not a cut-off."""
    generation = _service(_responder(_text("Namaste", finish_reason))).generate(_request())

    assert generation.text == "Namaste"
    assert generation.finish_reason is None


@pytest.mark.parametrize(
    "finish_reason", ["eos_token", "stop_sequence", "function_call", "end_turn"]
)
def test_a_servers_own_word_for_a_normal_stop_is_accepted_and_recorded(
    finish_reason, log_stream
) -> None:
    """TGI says ``eos_token``, others ``stop_sequence``. Only a known cut-off is refused."""
    generation = _service(_responder(_text("Namaste", finish_reason))).generate(_request())

    assert generation.text == "Namaste"
    assert generation.finish_reason == finish_reason
    [record] = _llm_records(log_stream)
    assert record["outcome"] == "ok"
    assert record["finish_reason"] == finish_reason


# --- length and content_filter: a fragment is refused ------------------------


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_a_cut_off_reply_is_an_incomplete_response(finish_reason) -> None:
    """``length``: the output cap cut it off. ``content_filter``: the server withheld part of it."""
    error = _cut_off(_responder(openai_text_completion(FRAGMENT, finish_reason=finish_reason)))

    assert error.category == "llm_incomplete_response"
    assert error.reason == finish_reason
    assert error.detail == f"llm_incomplete_response ({finish_reason})"


def test_a_well_formed_tool_call_cut_off_by_the_output_cap_is_incomplete() -> None:
    """The call parsed, but the model had not finished its turn. It is not handed on."""
    body = openai_tool_completion(
        ("get_outstanding_amount", {"account_ref": "ACC-9"}), finish_reason="length"
    )

    assert _cut_off(_responder(body)).reason == "length"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "   \n"},
        {"role": "assistant"},
        {"role": "assistant", "content": None, "tool_calls": []},
    ],
    ids=["empty", "null", "whitespace", "absent", "no-calls"],
)
def test_a_cut_off_with_nothing_in_it_is_incomplete_not_empty(message) -> None:
    """The cap cut the model off before it said anything. That is not an empty reply."""
    error = _cut_off(_responder(_body(message, "length")))

    assert not isinstance(error, LlmEmptyResponse)
    assert error.category == "llm_incomplete_response"


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_a_truncated_tool_call_is_incomplete_not_an_invalid_tool_call(finish_reason) -> None:
    """Arguments that stop mid-JSON were cut off. They were not written wrongly."""
    error = _cut_off(_responder(_truncated_tool_call(finish_reason)))

    assert not isinstance(error, LlmInvalidToolCall)
    assert error.reason == finish_reason


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": 42},
        {"role": "assistant", "content": None, "tool_calls": "not-an-array"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"arguments": "{}"}}]},
    ],
    ids=["content-not-a-string", "tool-calls-not-an-array", "tool-call-without-a-name"],
)
def test_a_cut_off_reply_is_reported_as_cut_off_before_its_content_is_read(message) -> None:
    """``finish_reason`` is read first, so what the fragment holds does not pick the category."""
    _cut_off(_responder(_body(message, "length")))


def test_a_choice_without_a_message_object_is_still_malformed_whatever_its_finish_reason() -> None:
    """``finish_reason`` is read once the message object is known to be there, not before."""
    body = {"model": "m", "choices": [{"index": 0, "finish_reason": "length"}]}

    with pytest.raises(LlmError) as caught:
        _service(_responder(body)).generate(_request())

    assert type(caught.value) is LlmMalformedResponse


# --- a finish_reason that is not a string ---------------------------------------


@pytest.mark.parametrize(
    "finish_reason",
    [42, 0, 1.5, True, False, {}, [], ["length"], {"reason": "length"}],
    ids=["int", "zero", "float", "true", "false", "empty-object", "empty-array", "array", "object"],
)
def test_a_finish_reason_that_is_not_a_string_is_a_malformed_response(finish_reason) -> None:
    """Before the fix a non-string was dropped to ``None`` and the reply accepted."""
    with pytest.raises(LlmError) as caught:
        _service(_responder(_text(GROUNDED_REPLY, finish_reason))).generate(_request())

    assert type(caught.value) is LlmMalformedResponse


def test_a_non_string_finish_reason_is_malformed_even_when_the_reply_is_empty() -> None:
    """Checked before content: an empty reply with a bad ``finish_reason`` is a broken body."""
    with pytest.raises(LlmError) as caught:
        _service(_responder(_text("", 42))).generate(_request())

    assert type(caught.value) is LlmMalformedResponse


# --- never retried ------------------------------------------------------------


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_an_incomplete_response_is_never_retried(finish_reason, sleeps) -> None:
    """The same request is cut off in the same place. Retrying spends the customer's silence."""
    handler = _responder(openai_text_completion(FRAGMENT, finish_reason=finish_reason))

    _cut_off(handler, max_retries=3)

    assert len(handler.seen) == 1
    assert sleeps == []


def test_a_retry_that_lands_on_a_cut_off_reply_stops_there(sleeps) -> None:
    """Retries are on and do fire for a 503; the cut-off reply that follows ends the call."""
    handler = _responder(
        httpx2.Response(503, json={}),
        openai_text_completion(FRAGMENT, finish_reason="length"),
    )

    _cut_off(handler, max_retries=3)

    assert len(handler.seen) == 2
    assert len(sleeps) == 1


def test_a_malformed_finish_reason_is_never_retried(sleeps) -> None:
    handler = _responder(_text(GROUNDED_REPLY, 42))

    with pytest.raises(LlmMalformedResponse):
        _service(handler, max_retries=3).generate(_request())

    assert len(handler.seen) == 1
    assert sleeps == []


# --- what the error and the log carry --------------------------------------------


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_the_incomplete_error_carries_no_upstream_text(finish_reason) -> None:
    handler = _responder(openai_text_completion(FRAGMENT, finish_reason=finish_reason))

    error = _cut_off(handler, api_key="sk-live-supersecret")

    carried = " ".join((str(error), error.detail, repr(error), repr(error.args)))
    assert FRAGMENT not in carried
    assert OUTSTANDING not in carried
    assert "12,345" not in carried
    assert "because" not in carried
    assert "sk-live-supersecret" not in carried
    assert str(error) == f"model generation ended early ({finish_reason})"
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_a_cut_off_call_is_logged_as_failed_with_its_finish_reason_and_without_its_text(
    finish_reason, log_stream
) -> None:
    handler = _responder(openai_text_completion(FRAGMENT, finish_reason=finish_reason))

    _cut_off(handler, max_retries=3)

    [record] = _llm_records(log_stream)
    assert record["outcome"] == "failed"
    assert record["attempt"] == 1
    assert record["error_type"] == "LlmIncompleteResponse"
    assert record["error_category"] == "llm_incomplete_response"
    assert record["finish_reason"] == finish_reason
    assert record["http_status"] == 200
    logged = log_stream.getvalue()
    assert FRAGMENT not in logged
    assert "12,345" not in logged
    assert "because" not in logged


def test_a_malformed_finish_reason_is_not_copied_into_the_log(log_stream) -> None:
    """The server's non-string value is a defect to report, not a value to record."""
    with pytest.raises(LlmMalformedResponse):
        _service(_responder(_text(GROUNDED_REPLY, {"reason": "canary-7731"}))).generate(_request())

    [record] = _llm_records(log_stream)
    assert record["outcome"] == "failed"
    assert record["error_category"] == "llm_malformed_response"
    assert record.get("finish_reason") is None
    assert "canary-7731" not in log_stream.getvalue()


# --- the orchestrator: a fragment is never spoken --------------------------------


def _assert_failed_as_incomplete(result: TurnResult, finish_reason: str) -> None:
    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert result.response_text is None
    assert "llm_incomplete_response" in {c.value for c in result.error_categories}
    category = _incomplete_category()
    assert category in result.error_categories
    assert TurnErrorCategory.LLM_FAILED not in result.error_categories
    [error] = [e for e in result.errors if e.category is category]
    assert error.detail == f"Model call failed: llm_incomplete_response ({finish_reason})."
    for recorded in result.errors:
        assert "12,345" not in recorded.detail
        assert "because" not in recorded.detail


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_a_cut_off_reply_fails_the_turn_even_when_it_would_pass_validation(finish_reason) -> None:
    """GROUNDED_REPLY is grounded, whole and speakable, except that the model had not finished."""
    endpoint = _Endpoint(openai_text_completion(GROUNDED_REPLY, finish_reason=finish_reason))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    _assert_failed_as_incomplete(result, finish_reason)
    assert result.draft_text is None
    assert result.validation is None
    assert result.llm_calls == 0
    assert len(endpoint.requests) == 1
    assert _agent_utterances(runtime, session.session_id) == []


@pytest.mark.parametrize("finish_reason", ["stop", "eos_token", None])
def test_a_finished_reply_still_completes_the_turn(finish_reason) -> None:
    """Control for the test above: the same reply, finished, is spoken."""
    endpoint = _Endpoint(openai_text_completion(GROUNDED_REPLY, finish_reason=finish_reason))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True
    assert result.errors == ()
    assert "twelve thousand three hundred forty-five rupees" in (result.response_text or "")
    assert len(_agent_utterances(runtime, session.session_id)) == 1


def test_duties_claimed_for_a_cut_off_utterance_are_not_recorded_as_done() -> None:
    """The utterance was never spoken, so it disclosed nothing and identified no one."""
    endpoint = _Endpoint(openai_text_completion(GREETING, finish_reason="length"))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime, stage=ConversationStage.GREETING, disclosed=False)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        say("Hello?"),
        claimed_actions=(
            RequiredAction.DISCLOSE_CALL_RECORDING,
            RequiredAction.IDENTIFY_BANK_AND_AGENT,
        ),
    )

    _assert_failed_as_incomplete(result, "length")
    assert result.state.recording_disclosed is False
    assert result.state.agent_identified is False
    stored = runtime.sessions.get(session.session_id)
    assert stored.state.recording_disclosed is False
    assert stored.state.agent_identified is False
    assert _agent_utterances(runtime, session.session_id) == []


def test_duties_claimed_for_a_finished_greeting_are_recorded() -> None:
    """Control for the test above: the claim is honoured when the reply is spoken."""
    endpoint = _Endpoint(openai_text_completion(GREETING, finish_reason="stop"))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime, stage=ConversationStage.GREETING, disclosed=False)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id,
        say("Hello?"),
        claimed_actions=(
            RequiredAction.DISCLOSE_CALL_RECORDING,
            RequiredAction.IDENTIFY_BANK_AND_AGENT,
        ),
    )

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.state.recording_disclosed is True
    assert result.state.agent_identified is True


def test_a_cut_off_tool_call_is_never_dispatched() -> None:
    endpoint = _Endpoint(
        openai_tool_completion(
            ("get_outstanding_amount", {"account_ref": "ACC-1"}), finish_reason="length"
        )
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    # The endpoint scripts one reply, so a dispatched call would also show up
    # as a second request the endpoint cannot answer.
    assert result.tools == ()
    assert len(endpoint.requests) == 1
    _assert_failed_as_incomplete(result, "length")


def test_a_cut_off_reply_after_a_tool_round_fails_the_turn() -> None:
    """The tool ran and was answered. The model's reply to it was cut off, and is not spoken."""
    endpoint = _Endpoint(
        openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-1"})),
        openai_text_completion(GROUNDED_REPLY, finish_reason="length"),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    _assert_failed_as_incomplete(result, "length")
    assert len(endpoint.requests) == 2
    assert result.llm_calls == 1
    assert [(t.tool_name, t.dispatched, t.status) for t in result.tools] == [
        ("get_outstanding_amount", True, ToolStatus.OK)
    ]
    assert result.draft_text is None
    assert _agent_utterances(runtime, session.session_id) == []


def test_a_finished_reply_after_a_tool_round_still_completes() -> None:
    """Control for the test above."""
    endpoint = _Endpoint(
        openai_tool_completion(("get_outstanding_amount", {"account_ref": "ACC-1"})),
        openai_text_completion(GROUNDED_REPLY, finish_reason="stop"),
    )
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.COMPLETED
    assert result.speakable is True
    assert len(endpoint.requests) == 2


def test_a_malformed_finish_reason_fails_the_turn_as_a_malformed_response() -> None:
    endpoint = _Endpoint(openai_text_completion(GROUNDED_REPLY, finish_reason=42))
    runtime = make_runtime(llm=http_llm(endpoint))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert TurnErrorCategory.LLM_MALFORMED_RESPONSE in result.error_categories


# --- the taxonomy ----------------------------------------------------------------


def _boundary_errors() -> list[type[LlmError]]:
    """Every ``LlmError`` subclass defined in :mod:`app.services.llm` itself."""
    return [
        obj
        for obj in vars(llm_module).values()
        if isinstance(obj, type)
        and issubclass(obj, LlmError)
        and obj is not LlmError
        and obj.__module__ == llm_module.__name__
    ]


def test_every_model_boundary_failure_maps_to_a_turn_error_category_of_its_own() -> None:
    """A failure class that falls back to LLM_FAILED tells the person on call nothing."""
    classes = _boundary_errors()
    assert {cls.__name__ for cls in classes} >= {
        "LlmNotConfigured",
        "LlmConfigurationError",
        "LlmTimeout",
        "LlmConnectionFailed",
        "LlmUpstreamError",
        "LlmMalformedResponse",
        "LlmEmptyResponse",
        "LlmIncompleteResponse",
        "LlmInvalidToolCall",
    }

    values = {member.value for member in TurnErrorCategory}
    for cls in classes:
        assert cls.category in values, f"{cls.__name__}.category has no TurnErrorCategory"
        assert cls.category != TurnErrorCategory.LLM_FAILED.value, cls.__name__
        # Built without __init__: the mapping reads only the class's category,
        # and the constructors take different arguments.
        instance = cls.__new__(cls)
        mapped = _llm_error_category(instance)
        assert mapped is TurnErrorCategory(cls.category), cls.__name__
        assert mapped is not TurnErrorCategory.LLM_FAILED, cls.__name__


def test_an_incomplete_response_maps_to_llm_incomplete_response() -> None:
    incomplete = _incomplete_error()
    category = _incomplete_category()

    assert category.value == "llm_incomplete_response"
    assert _llm_error_category(incomplete("length")) is category
    assert _llm_error_category(incomplete("content_filter")) is category
