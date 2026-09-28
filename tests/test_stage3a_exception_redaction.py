"""Nothing of a failed model exchange travels with the error that reports it.

Before this was pinned, the OpenAI-compatible adapter raised its typed errors
with ``raise ... from exc``. The classification was right and the message was
the adapter's own, but the original exception rode along as ``__cause__``. An
httpx2 exception holds the request it failed on: the ``Authorization: Bearer``
header carrying the API key, and the JSON body carrying the customer's words.
Its message can quote the URL. A body that was not JSON was chained the same
way, as a ``JSONDecodeError`` whose ``doc`` is the whole upstream body, and so
were tool-call arguments that were not JSON. Anything that kept the error - a
standard log formatter, an error tracker, a retry wrapper, a rendered
traceback - kept the key with it.

What is being pinned
--------------------
- For every transport failure httpx2 can raise, through a stub and over a real
  loopback socket, nothing reachable from the surfaced error holds the API key,
  the word ``Bearer``, the transcript or an upstream body. The walk covers
  ``args``, ``__dict__``, ``__slots__``, ``__cause__``, ``__context__`` and
  ``__notes__``, and every container, string and byte buffer beneath them,
  httpx2's own Request, Response, Headers, URL and stream objects included.
- ``__context__`` is always None. ``__cause__`` is either None or a bare
  instance of the same httpx2 class: no request, no message, never raised. The
  class is kept because retry policy decides by it, and still does.
- A body the adapter refused - not JSON, not UTF-8, tool arguments that are not
  JSON, an error status that echoes the key, a cut-off generation - is not held
  by the error either.
- A formatted traceback of the error shows no key, and no traceback behind it
  runs through httpx2 or httpcore2.
- The logs of each failure, and the TurnResult the orchestrator records for it,
  carry none of it.
- The Jev decision adapter, which already raised this way, still chains nothing.

What is deliberately not walked
-------------------------------
Traceback frames. A frame of ``generate`` holds ``self``, and the service must
hold its key in order to send it; a frame is also where the request under
construction lives. Whoever holds a traceback's frames holds the running
program. That residual is documented separately. What is pinned here is every
path to the secret that is not a frame.

One gap is left open by the fix and is pinned as a strict xfail near the end:
a key that httpx2 cannot encode into a header escapes inside an unmapped
``UnicodeEncodeError``.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
import traceback
import types
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import httpx2
import pytest

from app.orchestrator import ConversationOrchestrator, TurnOutcome
from app.services import llm_openai
from app.services.llm import (
    LlmConnectionFailed,
    LlmError,
    LlmMalformedResponse,
    LlmMessage,
    LlmRequest,
    LlmTimeout,
)
from app.services.llm_openai import OpenAiCompatibleLlmService
from tests.fakes import FakeOpenAiServer, openai_text_completion, unused_loopback_url
from tests.test_llm_http_integration import http_llm
from tests.test_orchestrator import make_runtime, open_session, say

BASE_URL = "http://model.invalid:8000/v1"

# --- sentinels, and a request that carries them -----------------------------

#: The configured API key. Its only legitimate home is the service's headers.
API_KEY = "sk-STAGE3A-SECRET-5c1e9d7a42b8"

#: What the customer said. It reaches the wire in the request body, and nowhere
#: else a caller could find it.
TRANSCRIPT = "TRANSCRIPT-SENTINEL-Kavitha-Rao-mera-naam"

#: Text only an upstream body contains. The adapter reads a body and drops it.
BODY = "BODY-SENTINEL-upstream-echo-81f3"

#: Everything that must be unreachable from a surfaced error. ``Bearer`` on its
#: own, because a header value with the key trimmed off is still a header value.
SENTINELS: tuple[str, ...] = (API_KEY, "Bearer", TRANSCRIPT, BODY)

#: What a hostile or merely verbose server sends back: the credential, the
#: customer's words and its own text.
ECHO = f"Authorization: Bearer {API_KEY}; you said: {TRANSCRIPT}; {BODY}"


def _request() -> LlmRequest:
    return LlmRequest(
        messages=(
            LlmMessage(role="system", content="CONSTRAINTS\n- tone: neutral"),
            LlmMessage(role="user", content=f"{TRANSCRIPT}, kitna bakaya hai?"),
        )
    )


def _service(handler, **kwargs) -> OpenAiCompatibleLlmService:
    """An adapter with the key configured, wired to an in-process transport."""
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("model", "configured-model")
    kwargs.setdefault("api_key", API_KEY)
    return OpenAiCompatibleLlmService(
        client=httpx2.Client(transport=httpx2.MockTransport(handler)), **kwargs
    )


def _surfaced(handler, **kwargs) -> LlmError:
    """The error ``generate`` raised, exactly as a caller receives it."""
    with pytest.raises(LlmError) as caught:
        _service(handler, **kwargs).generate(_request())
    return caught.value


# --- the object-graph walker --------------------------------------------------

#: Never descended into. Frames and tracebacks by design (see the module
#: docstring); the rest are code, not data a failed exchange could leave behind,
#: and a class or a function reaches the whole program through its globals.
_OPAQUE: tuple[type, ...] = (
    types.TracebackType,
    types.FrameType,
    types.CodeType,
    types.ModuleType,
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.GeneratorType,
    types.CoroutineType,
    types.AsyncGeneratorType,
    weakref.ReferenceType,
)

#: Exception fields kept at C level rather than in ``__dict__``, which ``args``
#: does not always repeat. ``UnicodeDecodeError.object`` is the whole buffer it
#: failed on.
_EXCEPTION_FIELDS: tuple[str, ...] = ("object", "filename", "filename2", "strerror", "text")

#: A walk that visits more than this many objects has wandered off the error
#: and into the program. It fails loudly rather than returning a partial "clean".
_WALK_LIMIT = 200_000


def _slot_names(cls: type) -> Iterator[str]:
    for klass in cls.__mro__:
        slots = klass.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{klass.__name__.lstrip('_')}{name}"  # private-name mangling
            yield name


def _secrets_reachable_from(root: object, sentinels: tuple[str, ...] = SENTINELS) -> list[tuple[str, str]]:
    """Every ``(path, sentinel)`` reachable from ``root`` without entering a frame.

    Iterative, with an ``id()``-based visited set, so a cyclic graph terminates
    and a deep one cannot exhaust the recursion limit.
    """
    encoded = [(s, s.encode("utf-8")) for s in sentinels]
    found: list[tuple[str, str]] = []
    seen: set[int] = set()
    stack: list[tuple[str, object]] = [("error", root)]
    visited = 0
    while stack:
        path, obj = stack.pop()
        if isinstance(obj, str):
            found.extend((path, s) for s, _ in encoded if s in obj)
            continue
        if isinstance(obj, (bytes, bytearray, memoryview)):
            try:
                data = bytes(obj)
            except ValueError:  # a released memoryview holds nothing any more
                continue
            found.extend((path, s) for s, raw in encoded if raw in data)
            continue
        if obj is None or isinstance(obj, (int, float, complex)) or isinstance(obj, _OPAQUE):
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        visited += 1
        if visited > _WALK_LIMIT:
            raise AssertionError(f"the walk from the error passed {_WALK_LIMIT} objects at {path}")

        children: list[tuple[str, object]] = []
        if isinstance(obj, BaseException):
            children.append((f"{path}.args", obj.args))
            children.append((f"{path}.__cause__", obj.__cause__))
            children.append((f"{path}.__context__", obj.__context__))
            children.append((f"{path}.__notes__", getattr(obj, "__notes__", None)))
            for name in _EXCEPTION_FIELDS:
                children.append((f"{path}.{name}", getattr(obj, name, None)))
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                children.append((f"{path}<key {key!r:.40}>", key))
                children.append((f"{path}[{key!r:.40}]", value))
        elif isinstance(obj, (list, tuple, set, frozenset, collections.deque)):
            children.extend((f"{path}[{i}]", item) for i, item in enumerate(list(obj)))
        try:
            attributes = object.__getattribute__(obj, "__dict__")
        except (AttributeError, TypeError):
            attributes = None
        if isinstance(attributes, dict):
            for key, value in list(attributes.items()):
                children.append((f"{path}.{key}", value))
        for name in _slot_names(type(obj)):
            try:
                children.append((f"{path}.{name}", getattr(obj, name)))
            except Exception:  # noqa: BLE001 - an unset slot, or a descriptor that refuses
                continue
        stack.extend(reversed(children))
    return found


def _chain(error: BaseException) -> list[BaseException]:
    """``error`` and every exception chained behind it, each once."""
    out: list[BaseException] = []
    pending: list[BaseException | None] = [error]
    while pending:
        node = pending.pop()
        if node is None or any(node is known for known in out):
            continue
        out.append(node)
        pending.extend((node.__cause__, node.__context__))
    return out


def _library_frames(error: BaseException) -> list[str]:
    """Frames inside httpx2 or httpcore2 on any traceback in ``error``'s chain."""
    frames: list[str] = []
    for node in _chain(error):
        tb = node.__traceback__
        while tb is not None:
            filename = tb.tb_frame.f_code.co_filename
            if {"httpx2", "httpcore2"} & set(Path(filename).parts):
                frames.append(f"{filename}:{tb.tb_lineno}")
            tb = tb.tb_next
    return frames


def _assert_bare_cause(error: LlmError, expected: type[httpx2.RequestError] | None) -> None:
    """Nothing chained, or only an empty stand-in of the transport class."""
    assert error.__context__ is None
    cause = error.__cause__
    if expected is None:
        assert cause is None
        return
    assert type(cause) is expected
    assert str(cause) == ""
    with pytest.raises(RuntimeError):
        cause.request  # noqa: B018 - the property raises when no request was attached
    assert cause.__cause__ is None
    assert cause.__context__ is None
    assert cause.__traceback__ is None  # built, never raised: it holds no frames


# --- the walker, checked on its own -----------------------------------------
#
# A walker that finds nothing proves nothing unless it is known to find what it
# is looking for. These four hold whatever the adapter does.


def test_the_walker_finds_a_key_and_a_transcript_held_by_a_chained_httpx2_request() -> None:
    request = httpx2.Request(
        "POST",
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"messages": [{"role": "user", "content": TRANSCRIPT}]},
    )
    try:
        try:
            raise httpx2.ConnectError("refused", request=request)
        except httpx2.ConnectError as exc:
            raise LlmConnectionFailed("model server could not be reached") from exc
    except LlmConnectionFailed as chained:
        error = chained

    found = {sentinel for _, sentinel in _secrets_reachable_from(error)}
    assert {API_KEY, "Bearer", TRANSCRIPT} <= found


def test_the_walker_finds_a_body_held_by_an_implicitly_chained_json_error() -> None:
    try:
        try:
            json.loads(f"<html>{BODY}</html>")
        except json.JSONDecodeError:
            raise LlmMalformedResponse("model server returned a body that is not JSON")
    except LlmMalformedResponse as chained:
        error = chained

    paths = [path for path, sentinel in _secrets_reachable_from(error) if sentinel == BODY]
    assert any(path.startswith("error.__context__") for path in paths)


def test_the_walker_reads_slots_and_every_kind_of_container_and_byte_buffer() -> None:
    class _Slotted:
        __slots__ = ("payload", "__hidden")

        def __init__(self) -> None:
            self.payload = [frozenset({("nested", 1)}), {"k": (bytearray(API_KEY.encode()),)}]
            self.__hidden = collections.deque([memoryview(BODY.encode())])

    error = LlmTimeout("x")
    error.holder = {"set": {TRANSCRIPT}, "slotted": _Slotted()}  # type: ignore[attr-defined]
    error.cycle = error  # type: ignore[attr-defined]

    found = {sentinel for _, sentinel in _secrets_reachable_from(error)}
    assert found == {API_KEY, TRANSCRIPT, BODY}


def test_the_walker_does_not_descend_into_traceback_frames() -> None:
    """The one path excluded by design, shown to be excluded and not merely absent."""

    def _raise_with_the_key_in_a_local() -> None:
        secret = API_KEY  # noqa: F841 - held only by this frame
        raise LlmTimeout("model server did not respond within the configured timeout")

    with pytest.raises(LlmTimeout) as caught:
        _raise_with_the_key_in_a_local()

    tb = caught.value.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    assert tb.tb_frame.f_locals["secret"] == API_KEY
    assert _secrets_reachable_from(caught.value) == []


# --- the failures ------------------------------------------------------------


def _raising(error_type: type[httpx2.RequestError]) -> Callable[[httpx2.Request], httpx2.Response]:
    """A transport that fails the way httpx2 does: the request attached.

    The message quotes the URL and the credential, as a verbose transport or a
    proxy's error text can.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        raise error_type(
            f"{request.method} {request.url} failed; sent "
            f"authorization={request.headers.get('authorization')}",
            request=request,
        )

    return handler


def _answering(
    status: int,
    *,
    body: object = None,
    raw: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        if raw is not None:
            return httpx2.Response(
                status, content=raw, headers={"Content-Type": "application/json", **(headers or {})}
            )
        return httpx2.Response(status, json=body, headers=headers)

    return handler


@dataclasses.dataclass(frozen=True)
class _Case:
    name: str
    handler: Callable[[httpx2.Request], httpx2.Response]
    category: str
    #: The bare stand-in expected as ``__cause__``, or None for no cause at all.
    cause: type[httpx2.RequestError] | None


#: Every concrete httpx2 exception a request can raise, and the generic bases.
_TRANSPORT_FAILURES: tuple[type[httpx2.RequestError], ...] = (
    httpx2.ConnectError,
    httpx2.ConnectTimeout,
    httpx2.ReadTimeout,
    httpx2.WriteTimeout,
    httpx2.PoolTimeout,
    httpx2.ReadError,
    httpx2.WriteError,
    httpx2.CloseError,
    httpx2.RemoteProtocolError,
    httpx2.LocalProtocolError,
    httpx2.ProxyError,
    httpx2.UnsupportedProtocol,
    httpx2.DecodingError,
    httpx2.TooManyRedirects,
    httpx2.TransportError,
    httpx2.RequestError,
)

_TRANSPORT_CASES: tuple[_Case, ...] = tuple(
    _Case(
        name=error_type.__name__,
        handler=_raising(error_type),
        category=(
            "llm_timeout" if issubclass(error_type, httpx2.TimeoutException) else "llm_connection_failed"
        ),
        cause=error_type,
    )
    for error_type in _TRANSPORT_FAILURES
)


def _tool_call_body(arguments: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "get_outstanding_amount", "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


_ECHOING_HEADERS = {"WWW-Authenticate": f'Bearer realm="{API_KEY}"', "X-Echo": TRANSCRIPT}

#: Bodies the adapter parsed, or refused to, and the errors that report them.
_BODY_CASES: tuple[_Case, ...] = (
    _Case(
        "body-not-json",
        _answering(200, raw=f"<html>{ECHO}</html>".encode()),
        "llm_malformed_response",
        None,
    ),
    _Case(
        # UnicodeDecodeError keeps the whole buffer in ``.object``.
        "body-not-utf8",
        _answering(200, raw=b'{"echo": "' + ECHO.encode() + b' \xc3\x28"}'),
        "llm_malformed_response",
        None,
    ),
    _Case(
        # An unterminated string: the decode error's ``doc`` is the argument text.
        "tool-arguments-not-json",
        _answering(200, body=_tool_call_body('{"account_ref": "' + ECHO)),
        "llm_invalid_tool_call",
        None,
    ),
    _Case(
        "http-401-echoing-the-key",
        _answering(401, body={"error": {"message": ECHO}}, headers=_ECHOING_HEADERS),
        "llm_upstream_error",
        None,
    ),
    _Case(
        "http-503-echoing-the-key",
        _answering(503, body={"error": {"message": ECHO}}, headers=_ECHOING_HEADERS),
        "llm_upstream_error",
        None,
    ),
    _Case(
        "incomplete-length",
        _answering(200, body=openai_text_completion(ECHO, finish_reason="length")),
        "llm_incomplete_response",
        None,
    ),
    _Case(
        "incomplete-content-filter",
        _answering(200, body=openai_text_completion(ECHO, finish_reason="content_filter")),
        "llm_incomplete_response",
        None,
    ),
)

_ALL_CASES: tuple[_Case, ...] = _TRANSPORT_CASES + _BODY_CASES

#: The cases in which httpx2 itself raised or decoded something.
_HTTPX_CASES: tuple[_Case, ...] = _TRANSPORT_CASES + tuple(
    case for case in _BODY_CASES if case.name in ("body-not-json", "body-not-utf8")
)


def _ids(cases: tuple[_Case, ...]) -> list[str]:
    return [case.name for case in cases]


# --- nothing of the exchange is reachable from the error --------------------


@pytest.mark.parametrize("case", _ALL_CASES, ids=_ids(_ALL_CASES))
def test_nothing_of_the_failed_exchange_is_reachable_from_the_surfaced_error(case: _Case) -> None:
    error = _surfaced(case.handler)

    assert error.category == case.category
    assert _secrets_reachable_from(error) == []
    _assert_bare_cause(error, case.cause)


@pytest.mark.parametrize("case", _ALL_CASES, ids=_ids(_ALL_CASES))
def test_a_formatted_traceback_of_the_surfaced_error_shows_no_key_and_no_customer_words(
    case: _Case,
) -> None:
    """What a standard log formatter or an error tracker renders."""
    rendered = "".join(traceback.format_exception(_surfaced(case.handler)))

    for sentinel in SENTINELS:
        assert sentinel not in rendered


@pytest.mark.parametrize("case", _HTTPX_CASES, ids=_ids(_HTTPX_CASES))
def test_no_traceback_behind_the_surfaced_error_runs_through_httpx2_or_httpcore2(
    case: _Case,
) -> None:
    """A library frame holds the request and the client in its locals.

    The walker does not enter frames, so this is the check that none is kept.
    """
    assert _library_frames(_surfaced(case.handler)) == []


# --- over a real socket -----------------------------------------------------


def _assert_carries_nothing(error: LlmError, cause: type[httpx2.RequestError]) -> None:
    assert _secrets_reachable_from(error) == []
    _assert_bare_cause(error, cause)
    assert _library_frames(error) == []
    rendered = "".join(traceback.format_exception(error))
    assert API_KEY not in rendered
    assert "Bearer" not in rendered


def test_a_real_refused_connection_surfaces_with_nothing_of_the_exchange_attached() -> None:
    """Not a stub: nothing listens on this port, and httpcore2 maps the refusal."""
    service = OpenAiCompatibleLlmService(
        base_url=unused_loopback_url(), model="m", api_key=API_KEY, timeout_seconds=2.0
    )
    try:
        with pytest.raises(LlmConnectionFailed) as caught:
            service.generate(_request())
    finally:
        service.close()

    _assert_carries_nothing(caught.value, httpx2.ConnectError)


def test_a_real_read_timeout_surfaces_with_nothing_of_the_exchange_attached() -> None:
    """A real stall on loopback: 0.2 s budget, 1 s server delay."""
    with FakeOpenAiServer(body=openai_text_completion(ECHO), delay_seconds=1.0) as server:
        service = OpenAiCompatibleLlmService(
            base_url=server.base_url, model="m", api_key=API_KEY, timeout_seconds=0.2
        )
        try:
            with pytest.raises(LlmTimeout) as caught:
                service.generate(_request())
        finally:
            service.close()

    # The key really did leave the process; it just did not come back.
    assert server.requests[0].headers["authorization"] == f"Bearer {API_KEY}"
    _assert_carries_nothing(caught.value, httpx2.ReadTimeout)


# --- retry policy still reads the scrubbed cause ----------------------------


class _FakeClock:
    """The adapter's view of time. ``sleep`` advances it; nothing really waits.

    With ``oversleep_to`` set, the first sleep ends that many seconds into the
    call instead, the way a loaded host oversleeps.
    """

    START = 1_000.0

    def __init__(self, oversleep_to: float | None = None) -> None:
        self.now = self.START
        self.sleeps: list[float] = []
        self._oversleep_to = oversleep_to

    def monotonic(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self._oversleep_to is not None:
            self.now = max(self.now, self.START + self._oversleep_to)
            self._oversleep_to = None


def _install(monkeypatch: pytest.MonkeyPatch, clock: _FakeClock) -> None:
    monkeypatch.setattr(
        llm_openai,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, perf_counter=clock.perf_counter, sleep=clock.sleep),
    )


def _counting(handler: Callable[[httpx2.Request], httpx2.Response]):
    attempts: list[int] = []

    def counted(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        return handler(request)

    return counted, attempts


@pytest.mark.parametrize(
    ("error_type", "attempts"),
    [
        (httpx2.ConnectError, 2),
        (httpx2.ProxyError, 2),
        (httpx2.ConnectTimeout, 2),
        (httpx2.PoolTimeout, 2),
        (httpx2.ReadTimeout, 1),
        (httpx2.WriteTimeout, 1),
        (httpx2.ReadError, 1),
        (httpx2.WriteError, 1),
        (httpx2.RemoteProtocolError, 1),
        (httpx2.RequestError, 1),
        (httpx2.TransportError, 1),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else str(value),
)
def test_retry_policy_still_reads_the_kind_of_failure_from_the_scrubbed_cause(
    monkeypatch: pytest.MonkeyPatch, error_type: type[httpx2.RequestError], attempts: int
) -> None:
    """Dropping the cause outright would have made every transport failure final.

    Only a failure from before the request left - a connect, a proxy, a pool
    wait - may be sent again; the stand-in's class is what says which.
    """
    clock = _FakeClock()
    _install(monkeypatch, clock)
    handler, made = _counting(_raising(error_type))

    with pytest.raises(LlmError):
        _service(handler, max_retries=1, timeout_seconds=1.0).generate(_request())

    assert len(made) == attempts
    assert len(clock.sleeps) == attempts - 1


def test_a_failure_re_raised_after_an_overrunning_wait_is_just_as_bare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error kept across a wait is raised again later, outside its handler.

    It must come back as bare as it went in: no context picked up on the way
    out, and the same empty stand-in as its cause.
    """
    clock = _FakeClock(oversleep_to=0.995)
    _install(monkeypatch, clock)
    handler, made = _counting(_raising(httpx2.ConnectError))

    error = _surfaced(handler, max_retries=1, timeout_seconds=1.0)

    assert isinstance(error, LlmConnectionFailed)
    assert len(made) == 1
    assert _secrets_reachable_from(error) == []
    _assert_bare_cause(error, httpx2.ConnectError)


# --- the logs of a failure --------------------------------------------------


def _llm_records(stream) -> list[dict]:
    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return [record for record in records if record.get("event") == "llm_call"]


@pytest.mark.parametrize("case", _ALL_CASES, ids=_ids(_ALL_CASES))
def test_the_log_of_a_failed_call_holds_no_key_bearer_transcript_or_body(
    case: _Case, log_stream
) -> None:
    _surfaced(case.handler)

    records = _llm_records(log_stream)
    assert records and records[-1]["outcome"] == "failed"
    assert records[-1]["error_category"] == case.category
    logged = log_stream.getvalue()
    for sentinel in SENTINELS:
        assert sentinel not in logged
    assert "authorization" not in logged.lower()


# --- what the orchestrator records ------------------------------------------


@pytest.mark.parametrize("case", _ALL_CASES, ids=_ids(_ALL_CASES))
def test_a_turn_failed_at_the_model_records_and_logs_no_key_and_nothing_the_body_said(
    case: _Case, log_stream
) -> None:
    """The TurnResult legitimately holds the transcript; it holds nothing else."""
    runtime = make_runtime(llm=http_llm(case.handler, api_key=API_KEY))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(
        session.session_id, say(f"{TRANSCRIPT}, kitna bakaya hai?")
    )

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert case.category in {category.value for category in result.error_categories}
    recorded = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    for sentinel in (API_KEY, "Bearer", BODY):
        assert sentinel not in recorded
    logged = log_stream.getvalue()
    for sentinel in SENTINELS:
        assert sentinel not in logged


# --- FIX GAP: a key the header cannot carry ---------------------------------
#
# Marked xfail(strict=True) because the gap is real in the fixed code and the
# test file must still pass. Once the adapter refuses such a key, or maps the
# failure, this XPASSes, the strict marker fails the run, and the marker should
# be removed.


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FIX GAP (finding D): a MODEL_API_KEY with a non-ASCII character passes Settings and "
        "the adapter's constructor. Every generate() then raises an unmapped "
        "UnicodeEncodeError from httpx2's header encoding. Its .object and args[1] are "
        "'Bearer <the whole key>', and its traceback runs through httpx2."
    ),
)
def test_an_api_key_the_header_cannot_encode_never_surfaces_inside_an_exception() -> None:
    """A curly quote pasted into the key: a misconfiguration, but not a reason to hand the key out.

    Either outcome would pass: the constructor refuses the key without quoting
    it, or ``generate`` raises an error with nothing attached.
    """
    pasted = f"{API_KEY}’"
    try:
        _service(_answering(200, body=openai_text_completion("ok")), api_key=pasted).generate(
            _request()
        )
    except Exception as error:  # noqa: BLE001 - whichever error surfaces, it must carry nothing
        surfaced: BaseException = error
    else:  # pragma: no cover - a key that cannot be encoded cannot have been sent
        pytest.fail("a key the header cannot encode was accepted and sent")

    assert _secrets_reachable_from(surfaced) == []
    assert _library_frames(surfaced) == []


# --- the Jev adapter, for comparison ----------------------------------------


def _jev_refusing(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError(
        f"{request.url} refused; sent {request.headers.get('authorization')}", request=request
    )


def _jev_echoing_401(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(401, json={"error": ECHO}, headers=_ECHOING_HEADERS)


@pytest.mark.parametrize(
    "handler", [_jev_refusing, _jev_echoing_401], ids=["connect-error", "401-echo"]
)
def test_the_jev_adapter_chains_nothing_to_the_decision_error_it_raises(handler) -> None:
    """The pattern the model adapter now follows, still holding where it began."""
    from app.services.decision import (
        DecisionContext,
        DecisionError,
        DecisionName,
        DecisionRequest,
    )
    from app.services.decision_jev import JevDecisionService

    service = JevDecisionService(
        api_key=API_KEY,
        model="jev-1.13.0",
        timeout_seconds=0.5,
        transport=httpx2.MockTransport(handler),
    )
    request = DecisionRequest(
        name=DecisionName.CUSTOMER_INTENT,
        context=DecisionContext(utterance=f"{TRANSCRIPT}, maine kal hi pay kar diya"),
    )

    async def _run() -> Any:
        try:
            return await service.decide(request)
        finally:
            await service.aclose()

    with pytest.raises(DecisionError) as caught:
        asyncio.run(_run())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert _secrets_reachable_from(caught.value) == []
