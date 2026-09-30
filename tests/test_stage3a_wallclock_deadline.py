"""Stage 3A, finding G: one wall-clock deadline bounds the whole model call.

What was wrong
--------------
``MODEL_TIMEOUT_SECONDS`` was handed to httpx2 as a per-phase timeout, and
httpx2's timeouts are per phase *and per socket operation*: the read timeout
restarts with every byte that arrives, the write timeout with every ``send``.
The adapter already shared one budget between attempts and the waits between
them, but inside an attempt nothing added the phases up. A connect that took
0.3 s followed by a response that took 0.3 s passed a 0.4 s budget, and a
server that dribbled one byte every 50 ms held the customer's silence for as
long as it liked. The call *succeeded*, far over budget.

Since then, two more layers were added on top of the same deadline, and both
are pinned here too:

- Resolution moved into the backend itself. ``_resolve`` is a module function
  now: an IP literal is returned with no lookup, a name is resolved once and
  deduplicated, and ``_DeadlineBackend.connect_tcp`` tries each address in
  turn, re-clamping ``_time_left`` before every one, so three unreachable
  addresses still cost at most one budget between them rather than one budget
  each. A resolver ``OSError`` becomes ``httpcore2.ConnectError`` rather than
  escaping raw.
- A write no longer always goes through the buffered inner stream. When the
  inner stream's own socket is reachable and not itself wrapped in another TLS
  layer, ``_DeadlineStream.write`` sends straight to it, one clamped
  ``settimeout``/``send`` pair per slice, mapping ``socket.timeout`` and
  ``OSError`` to the httpcore2 timeouts the pool expects. TLS carried inside
  another TLS connection still falls back to the sliced buffered path, because
  the socket there belongs to the outer layer.

What is pinned
--------------
- ``_time_left``: outside a model call a timeout passes through untouched;
  inside one it is shortened to the time the call has left; once the deadline
  has passed it raises the phase's own httpcore2 timeout and never hands a
  socket zero or a negative number.
- ``_DeadlineStream``: every read, write and TLS handshake gets that clamped
  timeout, and none reaches the inner stream once time is up. A write is sliced
  into pieces of at most 16 KiB, each re-clamped, so a slowly draining peer
  cannot stretch one write past the deadline; a plain socket (or one already
  wrapped in TLS by the socket itself) is written to directly instead, one
  clamped ``send`` at a time, while a socket behind a second, outer TLS layer
  still goes through the sliced fallback. ``get_extra_info`` and ``close``
  delegate.
- ``_DeadlineBackend``: connects are clamped; a hostname is resolved once and
  each address tried in turn, each under the time the ones before it left; an
  IP literal skips resolution; a connection that opens after the deadline (a
  slow DNS lookup, which no timeout covers) is closed and reported as a
  connect timeout; ``sleep`` delegates.
- The adapter, end to end on a fake clock: a slow connect, a dribbled response
  and a large request to a slowly draining peer are all cut at the budget.
  Bypassing ``_DeadlineBackend`` alone no longer reproduces the old failure: a
  second, independent guard - the post-parse deadline check below - catches an
  unbounded exchange too, just later and by a different route.
- ``_exchange``: a status other than 200 is decided from the head alone and its
  body is never read, whatever the status; a 200 body is read up to
  ``_MAX_RESPONSE_BYTES`` (8 MiB) *after* decompression, so a small
  gzip-compressed body that inflates past the cap is caught the same as a
  literally huge one, and reading stops as soon as the cap is crossed rather
  than after the whole body is buffered. After the body is parsed, the wall
  clock is checked once more: a reply that only finishes decoding after the
  deadline is reported as a timeout even though the exchange itself read
  cleanly, because decoding time is not covered by any socket timeout either.
- Installation: the client the adapter builds routes its default pool and every
  proxy pool through ``_DeadlineBackend``; an injected client is not touched; a
  client without the private httpx2/httpcore2 2.13.1 structure is refused (and
  closed) rather than run unbounded; the deadline ContextVar is set during the
  request, the same for every attempt of one call, invisible to other threads,
  and unset again however ``generate()`` ends.
- The exact pins in requirements.txt that make the private attributes safe.
- Real sockets on 127.0.0.1, for what only a socket shows: dribbled headers, a
  dribbled body, a slow-then-slow response, and retries enabled all end as
  ``LlmTimeout`` near the 0.4 s budget; a dribbled 503 is decided from the head
  and returns promptly, without waiting out the dribble at all; a fast server
  still answers.

Everything except the last section runs on a fake clock that only the adapter
module sees, with fake httpcore2 streams, backends and sockets; nothing
sleeps. The real-socket probes each finish in well under two seconds with at
least a 2.5x margin over the budget they check. New symbols are looked up at
test time, so against code without them each test fails on an assertion, not
on import.

A handful of tests pass against code that lacks each individual fix too,
deliberately: the premise that the fake server is several budgets long, the
injected client left untouched (kept behaviour), the installed versions (the
venv is shared), and the fast real server that still answers. The two tests
that once reproduced the pre-fix success-far-over-budget by bypassing
``_DeadlineBackend`` now instead pin the post-parse check's independence from
it: bypassing the backend still lets the exchange itself run unbounded - the
elapsed time and the fully-written body prove that - yet the call still ends
as a timeout, from the other guard.
"""

from __future__ import annotations

import contextlib
import gzip
import importlib.metadata
import json
import re
import socket
import ssl
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import httpcore2
import httpx2
import pytest

from app.services import llm_openai
from app.services.llm import (
    LlmConfigurationError,
    LlmConnectionFailed,
    LlmMalformedResponse,
    LlmMessage,
    LlmRequest,
    LlmTimeout,
    LlmUpstreamError,
)
from app.services.llm_openai import OpenAiCompatibleLlmService
from tests.fakes import openai_text_completion

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = "http://model.invalid:8000/v1"

#: ``timeout_seconds`` for every test here: the budget the whole call must fit in.
BUDGET = 0.4

#: Float slack for "never exceeds the budget". The fake clock adds decimal
#: fractions to 1000.0, exact to about 1e-13.
EPSILON = 1e-9

#: The largest piece ``_DeadlineStream.write`` may hand to the inner stream.
SLICE = 16 * 1024

#: The environment variables httpx2 (through urllib) reads proxies from.
_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "REQUEST_METHOD",  # urllib ignores HTTP_PROXY when this is set (CGI)
)


def _request(content: str = "kitna bakaya hai?") -> LlmRequest:
    return LlmRequest(
        messages=(
            LlmMessage(role="system", content="CONSTRAINTS\n- tone: neutral"),
            LlmMessage(role="user", content=content),
        )
    )


def _fixed(name: str) -> Any:
    """A name the fix added to :mod:`app.services.llm_openai`.

    Looked up here rather than imported at the top, so that against code
    without the fix a test fails on this assertion instead of the whole module
    failing to import.
    """
    value = getattr(llm_openai, name, None)
    assert value is not None, (
        f"app.services.llm_openai.{name} does not exist: the call's wall-clock "
        "deadline is not enforced at the socket"
    )
    return value


@pytest.fixture(autouse=True)
def _no_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every adapter here dials directly unless a test sets a proxy itself."""
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _stub_resolve(monkeypatch: pytest.MonkeyPatch) -> Any:
    """No test here ever asks a real resolver about a real name.

    ``_DeadlineBackend.connect_tcp`` now resolves its host itself, so every test
    that drives it with a hostname such as ``model.invalid`` would otherwise
    trigger a genuine DNS lookup. By default every host resolves to itself, one
    address, which keeps every existing fake-clock assertion about *which*
    address a call was made on unchanged. A test about resolution itself
    overrides this with its own fake, and one that wants the real function -
    to check an IP literal or the dedupe rule - gets it back as this fixture's
    value, looked up at test time so a run against code without the fix fails
    on an assertion rather than on collection.
    """
    real_resolve = _fixed("_resolve")
    monkeypatch.setattr(llm_openai, "_resolve", lambda host, port: [host])
    return real_resolve


# --- the fake clock ---------------------------------------------------------


class _FakeClock:
    """The adapter module's view of time, advanced only by the fakes below.

    ``monotonic`` and ``perf_counter`` read one counter. ``sleep`` advances it by
    exactly what was asked and records the request.
    """

    START = 1_000.0

    def __init__(self) -> None:
        self.now = self.START
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds

    @property
    def elapsed(self) -> float:
        return self.now - self.START


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Swap the adapter module's ``time`` for a fresh fake clock."""
    fake = _FakeClock()
    monkeypatch.setattr(
        llm_openai,
        "time",
        SimpleNamespace(monotonic=fake.monotonic, perf_counter=fake.perf_counter, sleep=fake.sleep),
    )
    return fake


@contextlib.contextmanager
def _call_deadline(at: float) -> Iterator[None]:
    """Run the block as if inside a model call whose deadline is ``at``."""
    var = _fixed("_CALL_DEADLINE")
    token = var.set(at)
    try:
        yield
    finally:
        var.reset(token)


def _current_deadline() -> float | None:
    """What the socket layer would read right now; ``None`` if nothing is set."""
    var = getattr(llm_openai, "_CALL_DEADLINE", None)
    return None if var is None else var.get()


# --- recording fakes for the unit tests --------------------------------------


class _InnerStream(httpcore2.NetworkStream):
    """A network stream that records every call, and the timeout it was given.

    Each read and write charges ``cost`` seconds to the fake clock, so a test
    can watch the time left shrink between calls.
    """

    EXTRA = object()

    def __init__(self, clock: _FakeClock | None = None, *, cost: float = 0.0) -> None:
        self._clock = clock
        self._cost = cost
        self.calls: list[tuple] = []
        self.pieces: list[bytes] = []
        self.write_timeouts: list[float | None] = []
        self.closed = False
        self.tls: _InnerStream | None = None

    def _charge(self) -> None:
        if self._clock is not None:
            self._clock.advance(self._cost)

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.calls.append(("read", max_bytes, timeout))
        self._charge()
        return b"x"

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.calls.append(("write", len(buffer), timeout))
        self.pieces.append(bytes(buffer))
        self.write_timeouts.append(timeout)
        self._charge()

    def close(self) -> None:
        self.calls.append(("close",))
        self.closed = True

    def start_tls(
        self, ssl_context: Any, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore2.NetworkStream:
        self.calls.append(("start_tls", ssl_context, server_hostname, timeout))
        self.tls = _InnerStream(self._clock, cost=self._cost)
        return self.tls

    def get_extra_info(self, info: str) -> Any:
        self.calls.append(("get_extra_info", info))
        return self.EXTRA if info == "marker" else None


class _InnerBackend(httpcore2.NetworkBackend):
    """A network backend that records every call.

    ``connect_cost`` is charged to the fake clock *inside* the connect and
    regardless of its timeout, the way a slow DNS lookup spends time before any
    socket timeout applies.
    """

    def __init__(self, clock: _FakeClock, *, connect_cost: float = 0.0) -> None:
        self._clock = clock
        self._connect_cost = connect_cost
        self.calls: list[tuple] = []
        self.streams: list[_InnerStream] = []

    def _open(self) -> _InnerStream:
        self._clock.advance(self._connect_cost)
        stream = _InnerStream(self._clock)
        self.streams.append(stream)
        return stream

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore2.NetworkStream:
        self.calls.append(("connect_tcp", host, port, timeout, local_address, socket_options))
        return self._open()

    def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None
    ) -> httpcore2.NetworkStream:
        self.calls.append(("connect_unix_socket", path, timeout, socket_options))
        return self._open()

    def sleep(self, seconds: float) -> None:
        self.calls.append(("sleep", seconds))


# --- _time_left --------------------------------------------------------------


def test_the_deadline_is_unset_outside_a_model_call() -> None:
    assert _fixed("_CALL_DEADLINE").get() is None


@pytest.mark.parametrize("timeout", [5.0, 0.001, None])
def test_outside_a_model_call_every_timeout_passes_through_untouched(clock, timeout) -> None:
    """With no deadline set the wrapper is inert, even on a clock far in the future."""
    time_left = _fixed("_time_left")
    clock.advance(1e6)
    assert time_left(timeout, httpcore2.ReadTimeout) == timeout


def test_inside_a_call_a_longer_timeout_is_shortened_to_the_time_left(clock) -> None:
    time_left = _fixed("_time_left")
    with _call_deadline(clock.START + 0.25):
        assert time_left(5.0, httpcore2.ReadTimeout) == pytest.approx(0.25)
        clock.advance(0.1)
        assert time_left(5.0, httpcore2.ReadTimeout) == pytest.approx(0.15)


def test_inside_a_call_a_shorter_timeout_is_kept(clock) -> None:
    time_left = _fixed("_time_left")
    with _call_deadline(clock.START + 0.25):
        assert time_left(0.1, httpcore2.ReadTimeout) == pytest.approx(0.1)


def test_inside_a_call_no_timeout_becomes_the_time_left(clock) -> None:
    """``None`` means "wait forever" to a socket. Inside a call it cannot."""
    time_left = _fixed("_time_left")
    with _call_deadline(clock.START + 0.25):
        assert time_left(None, httpcore2.ReadTimeout) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "expired", [httpcore2.ReadTimeout, httpcore2.WriteTimeout, httpcore2.ConnectTimeout]
)
@pytest.mark.parametrize("overrun", [0.0, 1e-6, 0.3, 50.0], ids=["at", "just-past", "past", "long-past"])
@pytest.mark.parametrize("timeout", [5.0, None])
def test_once_the_deadline_is_reached_the_phase_timeout_is_raised(
    clock, expired, overrun, timeout
) -> None:
    """Never a zero or negative timeout.

    ``settimeout(0)`` puts a socket in non-blocking mode and surfaces as a
    connection error, which the adapter retries; a negative value raises an
    error httpx2 does not map.
    """
    time_left = _fixed("_time_left")
    with _call_deadline(clock.START):
        clock.advance(overrun)
        with pytest.raises(expired):
            time_left(timeout, expired)


@pytest.mark.parametrize("left", [1e-9, 1e-3, 0.2])
def test_any_time_left_at_all_is_handed_on_as_a_positive_timeout(clock, left) -> None:
    time_left = _fixed("_time_left")
    with _call_deadline(clock.START + left):
        given = time_left(5.0, httpcore2.ReadTimeout)
    assert given > 0
    assert given == pytest.approx(left, rel=1e-6)


# --- _DeadlineStream ----------------------------------------------------------


def _wrapped_stream(clock: _FakeClock, *, cost: float = 0.0) -> tuple[Any, _InnerStream]:
    inner = _InnerStream(clock, cost=cost)
    return _fixed("_DeadlineStream")(inner), inner


def test_a_read_is_given_the_time_left_when_that_is_shorter(clock) -> None:
    stream, inner = _wrapped_stream(clock)
    with _call_deadline(clock.START + 0.25):
        assert stream.read(4096, 5.0) == b"x"
        assert stream.read(4096, 0.1) == b"x"
        assert stream.read(4096, None) == b"x"
    assert [call[0] for call in inner.calls] == ["read", "read", "read"]
    assert [call[1] for call in inner.calls] == [4096, 4096, 4096]
    assert [call[2] for call in inner.calls] == pytest.approx([0.25, 0.1, 0.25])


def test_outside_a_call_a_stream_passes_every_timeout_through(clock) -> None:
    stream, inner = _wrapped_stream(clock)
    stream.read(10, 5.0)
    stream.write(b"abc", 7.0)
    # A write always asks first whether it can go straight to a socket; ``_InnerStream``
    # exposes none, so it falls back to the sliced path, unaffected by the deadline.
    assert inner.calls == [
        ("read", 10, 5.0),
        ("get_extra_info", "socket"),
        ("write", 3, 7.0),
    ]


def test_each_read_is_clamped_again_as_the_call_runs_down(clock) -> None:
    """The per-read timeout does not restart: every read sees less time than the last."""
    stream, inner = _wrapped_stream(clock, cost=0.06)
    with _call_deadline(clock.START + BUDGET):
        for _ in range(7):  # the seventh read ends at 0.42 s, past the deadline
            stream.read(1, BUDGET)
        with pytest.raises(httpcore2.ReadTimeout):
            stream.read(1, BUDGET)
    given = [call[2] for call in inner.calls]
    assert given == pytest.approx([BUDGET - 0.06 * i for i in range(7)])
    assert all(value > 0 for value in given)
    assert len(inner.calls) == 7, "the read after the deadline reached the socket"


@pytest.mark.parametrize(
    "operation, expired, expected_calls",
    [
        pytest.param(lambda s: s.read(4096, 5.0), httpcore2.ReadTimeout, [], id="read"),
        pytest.param(
            lambda s: s.write(b"payload", 5.0),
            httpcore2.WriteTimeout,
            [("get_extra_info", "socket")],
            id="write",
        ),
        pytest.param(
            lambda s: s.start_tls(object(), "model.invalid", 5.0),
            httpcore2.ConnectTimeout,
            [],
            id="start_tls",
        ),
    ],
)
def test_once_time_is_up_a_stream_operation_raises_without_touching_the_socket(
    clock, operation, expired, expected_calls
) -> None:
    """A write still asks the inner stream what its socket is before it checks the
    clock - deciding *how* to write is not itself I/O - but that inspection is as
    far as it gets: no read, no write, no send reaches the network.
    """
    stream, inner = _wrapped_stream(clock)
    with _call_deadline(clock.START + 0.1):
        clock.advance(0.1)
        with pytest.raises(expired):
            operation(stream)
    assert inner.calls == expected_calls


def test_a_small_write_is_one_piece_with_the_clamped_timeout(clock) -> None:
    stream, inner = _wrapped_stream(clock)
    with _call_deadline(clock.START + 0.25):
        stream.write(b"POST /v1/chat/completions HTTP/1.1\r\n", 5.0)
    assert inner.pieces == [b"POST /v1/chat/completions HTTP/1.1\r\n"]
    assert inner.write_timeouts == pytest.approx([0.25])


def test_a_large_write_is_sliced_into_16_kib_pieces_each_clamped_again(clock) -> None:
    """httpcore2 sends one buffer in a loop under one timeout. Slicing re-reads the clock."""
    stream, inner = _wrapped_stream(clock, cost=0.01)
    buffer = bytes(range(256)) * 400  # 102,400 bytes: six full slices and a remainder
    with _call_deadline(clock.START + BUDGET):
        stream.write(buffer, 5.0)

    assert b"".join(inner.pieces) == buffer
    assert all(0 < len(piece) <= SLICE for piece in inner.pieces)
    assert len(inner.pieces) == -(-len(buffer) // SLICE)
    assert inner.write_timeouts == pytest.approx(
        [BUDGET - 0.01 * i for i in range(len(inner.pieces))]
    )


def test_a_large_write_outside_a_call_is_delivered_intact(clock) -> None:
    stream, inner = _wrapped_stream(clock)
    buffer = b"q" * (3 * SLICE + 5)
    stream.write(buffer, 5.0)
    assert b"".join(inner.pieces) == buffer
    assert all(len(piece) <= SLICE for piece in inner.pieces)
    assert set(inner.write_timeouts) == {5.0}


def test_a_slowly_draining_peer_stops_the_write_at_the_deadline(clock) -> None:
    """Each 16 KiB piece takes 0.1 s: the fifth would start after the 0.4 s deadline."""
    stream, inner = _wrapped_stream(clock, cost=0.1)
    buffer = b"w" * (10 * SLICE)
    with _call_deadline(clock.START + BUDGET):
        with pytest.raises(httpcore2.WriteTimeout):
            stream.write(buffer, 5.0)

    assert 0 < len(inner.pieces) < 10
    assert sum(map(len, inner.pieces)) < len(buffer)
    assert all(value > 0 for value in inner.write_timeouts)
    # Stopped at the deadline, give or take the one piece already under way.
    assert clock.elapsed <= BUDGET + 0.1 + EPSILON


def test_start_tls_is_clamped_and_returns_a_stream_that_is_still_bounded(clock) -> None:
    deadline_stream = _fixed("_DeadlineStream")
    stream, inner = _wrapped_stream(clock)
    context = object()
    with _call_deadline(clock.START + 0.25):
        secured = stream.start_tls(context, "model.invalid", 5.0)
        assert isinstance(secured, deadline_stream)
        assert inner.calls[0][:3] == ("start_tls", context, "model.invalid")
        assert inner.calls[0][3] == pytest.approx(0.25)

        secured.read(100, 5.0)
        assert inner.tls is not None
        assert inner.tls.calls[0][2] == pytest.approx(0.25)

        clock.advance(0.25)
        with pytest.raises(httpcore2.ReadTimeout):
            secured.read(100, 5.0)
    assert len(inner.tls.calls) == 1


def test_get_extra_info_and_close_are_delegated(clock) -> None:
    """The pool asks ``is_readable`` to expire idle connections; that must reach the socket."""
    stream, inner = _wrapped_stream(clock)
    with _call_deadline(clock.START):
        clock.advance(5.0)  # well past the deadline: neither is a timed operation
        assert stream.get_extra_info("marker") is _InnerStream.EXTRA
        assert stream.get_extra_info("ssl_object") is None
        stream.close()
    assert inner.closed is True
    assert inner.calls == [("get_extra_info", "marker"), ("get_extra_info", "ssl_object"), ("close",)]


# --- the direct-socket write path ---------------------------------------------


class _FakeSocket(socket.socket):
    """Stands in for the raw socket a write can reach directly.

    A real (never connected, never bound) ``socket.socket`` subclass, because
    ``_own_socket`` checks ``isinstance(sock, socket.socket)``; no byte here
    ever reaches an interface. ``send`` accepts at most ``chunk_limit`` bytes at
    a time and charges ``cost`` seconds to the fake clock; when the timeout it
    was given is shorter than that cost it raises ``socket.timeout`` instead,
    the way a real send does when the peer stops reading.
    """

    def __init__(
        self, clock: _FakeClock, *, chunk_limit: int | None = None, cost: float = 0.0
    ) -> None:
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        self._clock = clock
        self._chunk_limit = chunk_limit
        self._cost = cost
        self._timeout: float | None = None
        self.timeouts: list[float | None] = []
        self.sent = bytearray()

    def settimeout(self, value: float | None) -> None:  # type: ignore[override]
        self.timeouts.append(value)
        self._timeout = value

    def send(self, data: bytes) -> int:  # type: ignore[override]
        if self._timeout is not None and self._cost > self._timeout:
            self._clock.advance(self._timeout)
            raise socket.timeout("fake socket: send timed out")
        self._clock.advance(self._cost)
        piece = bytes(data[: self._chunk_limit] if self._chunk_limit else data)
        self.sent.extend(piece)
        return len(piece)

    def close(self) -> None:  # type: ignore[override]
        try:
            super().close()
        except OSError:  # pragma: no cover - never actually opened a connection
            pass


class _SocketBackedStream(httpcore2.NetworkStream):
    """A stream whose ``get_extra_info`` exposes a socket - real or make-believe.

    ``ssl_object`` set to a sentinel with a plain (non-``SSLSocket``) ``sock``
    models TLS carried inside another TLS connection: the socket belongs to the
    outer layer, so a write must fall back to ``fallback.write`` in slices
    rather than sending on it directly.
    """

    def __init__(
        self,
        sock: socket.socket | None,
        *,
        ssl_object: Any = None,
        fallback: httpcore2.NetworkStream | None = None,
    ) -> None:
        self._sock = sock
        self._ssl_object = ssl_object
        self._fallback = fallback
        self.calls: list[tuple[Any, ...]] = []

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        raise AssertionError("not exercised by these tests")  # pragma: no cover

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.calls.append(("write", len(buffer), timeout))
        assert self._fallback is not None, "the direct path must not fall back to write()"
        self._fallback.write(buffer, timeout)

    def close(self) -> None:
        self.calls.append(("close",))

    def start_tls(
        self, ssl_context: Any, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore2.NetworkStream:
        raise AssertionError("not exercised by these tests")  # pragma: no cover

    def get_extra_info(self, info: str) -> Any:
        self.calls.append(("get_extra_info", info))
        if info == "socket":
            return self._sock
        if info == "ssl_object":
            return self._ssl_object
        return None


def test_a_plain_socket_is_written_to_directly_each_send_given_the_time_left(clock) -> None:
    """No ``ssl_object``: the socket carries the bytes itself, not the buffered fallback."""
    sock = _FakeSocket(clock, chunk_limit=4096, cost=0.01)
    inner = _SocketBackedStream(sock)
    stream = _fixed("_DeadlineStream")(inner)
    buffer = b"z" * (4096 * 3 + 10)  # four sends: three full, one remainder
    try:
        with _call_deadline(clock.START + BUDGET):
            stream.write(buffer, 5.0)
        assert bytes(sock.sent) == buffer
        assert sock.timeouts == pytest.approx([BUDGET - 0.01 * i for i in range(4)])
        # The socket was inspected, but never asked to buffer a write itself.
        assert all(call[0] == "get_extra_info" for call in inner.calls)
    finally:
        sock.close()


def test_a_socket_behind_another_tls_layer_falls_back_to_sliced_writes(clock) -> None:
    """A real socket is present, but an ``ssl_object`` too: TLS-in-TLS, so it is not used."""
    sock = _FakeSocket(clock)
    fallback = _InnerStream(clock, cost=0.01)
    inner = _SocketBackedStream(sock, ssl_object=object(), fallback=fallback)
    stream = _fixed("_DeadlineStream")(inner)
    buffer = bytes(range(256)) * 100  # 25,600 bytes: two 16 KiB-bounded slices
    with _call_deadline(clock.START + BUDGET):
        stream.write(buffer, 5.0)

    assert not sock.sent, "the outer TLS layer's own socket must not see raw bytes"
    assert b"".join(fallback.pieces) == buffer
    assert all(0 < len(piece) <= SLICE for piece in fallback.pieces)
    assert fallback.write_timeouts == pytest.approx([BUDGET, BUDGET - 0.01])


def test_a_socket_send_that_would_run_past_the_deadline_raises_write_timeout(clock) -> None:
    """A send too slow to finish in what is left times out, mid-write, not after it."""
    sock = _FakeSocket(clock, chunk_limit=1024, cost=0.15)
    inner = _SocketBackedStream(sock)
    stream = _fixed("_DeadlineStream")(inner)
    buffer = b"w" * (1024 * 10)
    try:
        with _call_deadline(clock.START + BUDGET):
            with pytest.raises(httpcore2.WriteTimeout) as caught:
                stream.write(buffer, 5.0)
        assert isinstance(caught.value.__cause__, socket.timeout)
        assert sock.timeouts == pytest.approx([BUDGET, BUDGET - 0.15, BUDGET - 0.30])
        assert len(sock.sent) == 1024 * 2, "two sends should have completed before the third timed out"
        assert clock.elapsed == pytest.approx(BUDGET)
    finally:
        sock.close()


# --- _DeadlineBackend -----------------------------------------------------------


def _wrapped_backend(clock: _FakeClock, *, connect_cost: float = 0.0) -> tuple[Any, _InnerBackend]:
    inner = _InnerBackend(clock, connect_cost=connect_cost)
    return _fixed("_DeadlineBackend")(inner), inner


def test_a_tcp_connect_is_clamped_and_its_other_arguments_pass_through(clock) -> None:
    backend, inner = _wrapped_backend(clock)
    options = [(6, 1, 1)]
    with _call_deadline(clock.START + 0.25):
        stream = backend.connect_tcp("model.invalid", 8000, 5.0, "127.0.0.2", options)
    assert isinstance(stream, _fixed("_DeadlineStream"))
    name, host, port, timeout, local_address, socket_options = inner.calls[0]
    assert (name, host, port, local_address, socket_options) == (
        "connect_tcp",
        "model.invalid",
        8000,
        "127.0.0.2",
        options,
    )
    assert timeout == pytest.approx(0.25)


def test_a_unix_socket_connect_is_clamped(clock) -> None:
    backend, inner = _wrapped_backend(clock)
    with _call_deadline(clock.START + 0.25):
        stream = backend.connect_unix_socket("/run/model.sock", 5.0, None)
    assert isinstance(stream, _fixed("_DeadlineStream"))
    assert inner.calls[0][:2] == ("connect_unix_socket", "/run/model.sock")
    assert inner.calls[0][2] == pytest.approx(0.25)


def test_outside_a_call_the_backend_is_inert(clock) -> None:
    backend, inner = _wrapped_backend(clock)
    stream = backend.connect_tcp("model.invalid", 8000, 5.0)
    stream.read(10, 5.0)
    assert inner.calls[0][3] == 5.0
    assert inner.streams[0].calls == [("read", 10, 5.0)]


def test_the_connection_a_backend_returns_is_bounded_by_the_same_deadline(clock) -> None:
    backend, inner = _wrapped_backend(clock)
    with _call_deadline(clock.START + 0.25):
        stream = backend.connect_tcp("model.invalid", 8000, 5.0)
        clock.advance(0.25)
        with pytest.raises(httpcore2.ReadTimeout):
            stream.read(10, 5.0)
    assert inner.streams[0].calls == []


@pytest.mark.parametrize("kind", ["tcp", "unix"])
def test_a_connect_after_the_deadline_never_reaches_the_network(clock, kind) -> None:
    backend, inner = _wrapped_backend(clock)
    with _call_deadline(clock.START):
        with pytest.raises(httpcore2.ConnectTimeout):
            if kind == "tcp":
                backend.connect_tcp("model.invalid", 8000, 5.0)
            else:
                backend.connect_unix_socket("/run/model.sock", 5.0)
    assert inner.calls == []


@pytest.mark.parametrize("kind", ["tcp", "unix"])
def test_a_connection_that_opens_after_the_deadline_is_closed_and_is_a_connect_timeout(
    clock, kind
) -> None:
    """DNS resolution runs before any socket timeout applies, so it can overrun."""
    backend, inner = _wrapped_backend(clock, connect_cost=0.5)
    with _call_deadline(clock.START + 0.25):
        with pytest.raises(httpcore2.ConnectTimeout):
            if kind == "tcp":
                backend.connect_tcp("model.invalid", 8000, 5.0)
            else:
                backend.connect_unix_socket("/run/model.sock", 5.0)
    assert len(inner.streams) == 1
    assert inner.streams[0].closed is True


def test_a_connection_that_opens_just_in_time_is_kept(clock) -> None:
    backend, inner = _wrapped_backend(clock, connect_cost=0.2)
    with _call_deadline(clock.START + 0.25):
        stream = backend.connect_tcp("model.invalid", 8000, 5.0)
    assert inner.streams[0].closed is False
    assert isinstance(stream, _fixed("_DeadlineStream"))


def test_a_backend_sleep_is_delegated(clock) -> None:
    backend, inner = _wrapped_backend(clock)
    backend.sleep(0.2)
    assert inner.calls == [("sleep", 0.2)]


# --- resolving the host before connecting ------------------------------------


def test_resolve_returns_an_ip_literal_with_no_lookup_at_all(_stub_resolve, monkeypatch) -> None:
    """An IP address names itself; asking the resolver about it would be wrong."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a lookup ran for an address that was already an IP literal")

    monkeypatch.setattr(llm_openai.socket, "getaddrinfo", _boom)
    assert _stub_resolve("203.0.113.5", 8000) == ["203.0.113.5"]
    assert _stub_resolve("::1", 443) == ["::1"]


def test_resolve_deduplicates_addresses_in_resolver_order(_stub_resolve, monkeypatch) -> None:
    def _fake_getaddrinfo(host: str, port: int, type: int | None = None) -> list[tuple]:  # noqa: A002
        return [
            (2, 1, 6, "", ("203.0.113.5", port)),
            (2, 1, 6, "", ("203.0.113.9", port)),
            (2, 1, 6, "", ("203.0.113.5", port)),  # a repeat, from a second record family
        ]

    monkeypatch.setattr(llm_openai.socket, "getaddrinfo", _fake_getaddrinfo)
    assert _stub_resolve("model.invalid", 8000) == ["203.0.113.5", "203.0.113.9"]


def test_outside_a_call_no_resolution_happens(clock, monkeypatch) -> None:
    """The backend is inert outside a call, so it must not even ask for addresses."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a hostname was resolved outside a model call")

    monkeypatch.setattr(llm_openai, "_resolve", _boom)
    backend, inner = _wrapped_backend(clock)
    stream = backend.connect_tcp("model.invalid", 8000, 5.0)
    assert isinstance(stream, _InnerStream)
    assert inner.calls[0][:2] == ("connect_tcp", "model.invalid")


def test_a_resolver_failure_becomes_a_connect_error(clock, monkeypatch) -> None:
    """``getaddrinfo``'s own ``OSError`` (``gaierror`` and friends) is not left to escape."""

    def _boom(host: str, port: int) -> list[str]:
        raise OSError("fake: name or service not known")

    monkeypatch.setattr(llm_openai, "_resolve", _boom)
    backend, inner = _wrapped_backend(clock)
    with _call_deadline(clock.START + 0.25):
        with pytest.raises(httpcore2.ConnectError):
            backend.connect_tcp("model.invalid", 8000, 5.0)
    assert inner.calls == [], "a failed lookup must not still try to connect"


def test_end_to_end_a_resolver_failure_is_reported_as_connection_failed(monkeypatch) -> None:
    def _boom(host: str, port: int) -> list[str]:
        raise OSError("fake: name or service not known")

    monkeypatch.setattr(llm_openai, "_resolve", _boom)
    with _owned_service() as service:
        with pytest.raises(LlmConnectionFailed) as caught:
            service.generate(_request())
    assert isinstance(caught.value.__cause__, httpx2.ConnectError)


class _ScriptedAddressBackend(httpcore2.NetworkBackend):
    """One outcome per resolved address: a quick refusal, a hang, or a fast connect.

    A "hang" eats exactly the clamp it was given, the way a real connect that
    never completes spends whatever timeout it was handed. A "refused" outcome
    fails fast, spending only ``cost`` regardless of the clamp - a closed port
    answers immediately, it does not wait out its timeout.
    """

    def __init__(self, clock: _FakeClock, script: dict[str, tuple[str, float]]) -> None:
        self._clock = clock
        self._script = script
        self.calls: list[tuple[str, float | None]] = []
        self.streams: list[_InnerStream] = []

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore2.NetworkStream:
        self.calls.append((host, timeout))
        kind, cost = self._script[host]
        if kind == "refused":
            self._clock.advance(cost)
            raise httpcore2.ConnectError("fake: connection refused")
        if kind == "hangs":
            self._clock.advance(timeout if timeout is not None else cost)
            raise httpcore2.ConnectTimeout("fake: connect timed out")
        self._clock.advance(cost)
        stream = _InnerStream(self._clock)
        self.streams.append(stream)
        return stream

    def connect_unix_socket(self, *args: Any, **kwargs: Any) -> httpcore2.NetworkStream:
        raise AssertionError("the adapter dials TCP")  # pragma: no cover

    def sleep(self, seconds: float) -> None:
        raise AssertionError("not used here")  # pragma: no cover


def test_each_resolved_address_is_tried_under_the_time_the_one_before_it_left(
    clock, monkeypatch
) -> None:
    """Two addresses refuse quickly; the third connects with what is still left.

    A standard-library ``create_connection`` would give every address the whole
    timeout - three unreachable addresses would then cost three budgets. Here
    each gets only what the call has left after the ones before it.
    """
    monkeypatch.setattr(
        llm_openai, "_resolve", lambda host, port: ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    )
    backend = _ScriptedAddressBackend(
        clock,
        {
            "10.0.0.1": ("refused", 0.05),
            "10.0.0.2": ("refused", 0.05),
            "10.0.0.3": ("ok", 0.05),
        },
    )
    wrapped = _fixed("_DeadlineBackend")(backend)
    with _call_deadline(clock.START + BUDGET):
        stream = wrapped.connect_tcp("model.invalid", 8000, 5.0)

    assert [addr for addr, _ in backend.calls] == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    assert [t for _, t in backend.calls] == pytest.approx([BUDGET, BUDGET - 0.05, BUDGET - 0.1])
    assert isinstance(stream, _fixed("_DeadlineStream"))
    assert clock.elapsed <= BUDGET


def test_a_hang_on_every_address_ends_as_a_connect_timeout_at_the_deadline(
    clock, monkeypatch
) -> None:
    monkeypatch.setattr(llm_openai, "_resolve", lambda host, port: ["10.0.0.1", "10.0.0.2"])
    backend = _ScriptedAddressBackend(
        clock, {"10.0.0.1": ("hangs", 0.0), "10.0.0.2": ("hangs", 0.0)}
    )
    wrapped = _fixed("_DeadlineBackend")(backend)
    with _call_deadline(clock.START + BUDGET):
        with pytest.raises(httpcore2.ConnectTimeout):
            wrapped.connect_tcp("model.invalid", 8000, 5.0)

    assert clock.elapsed == pytest.approx(BUDGET)
    # The first address alone exhausted the clamp; the second's own clamp is
    # already zero, so it is refused by ``_time_left`` without a second attempt.
    assert len(backend.calls) == 1


# --- a slow server on the fake clock, behind the adapter's own client ----------


class _WireStream(httpcore2.NetworkStream):
    """One connection to a slow model server, on the fake clock.

    Behaves as a socket does under ``settimeout``. Each read returns the next
    scripted chunk after its delay; each write goes out in ``send_chunk``-byte
    sends, as httpcore2's ``SyncStream.write`` loops over ``socket.send``, each
    send taking ``send_cost``. The timeout applies to each operation on its own,
    which is exactly why a dribbling server beats per-phase timeouts: an
    operation that completes within its timeout advances the clock by its delay
    and succeeds, and one that would not advances by the timeout and raises the
    phase's timeout. With ``honour_timeouts=False`` every operation takes its
    full delay whatever it was given.
    """

    def __init__(
        self,
        clock: _FakeClock,
        reads: list[tuple[float, bytes]],
        *,
        send_chunk: int,
        send_cost: float,
        honour_timeouts: bool,
    ) -> None:
        self._clock = clock
        self._reads = list(reads)
        self._send_chunk = send_chunk
        self._send_cost = send_cost
        self._honour = honour_timeouts
        self.timeouts: list[float | None] = []
        self.bytes_written = 0
        self.closed = False

    def _wait(self, delay: float, timeout: float | None, expired: type[Exception]) -> None:
        self.timeouts.append(timeout)
        if self._honour and timeout is not None and delay > timeout:
            self._clock.advance(timeout)
            raise expired("fake socket: timed out")
        self._clock.advance(delay)

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if not self._reads:
            return b""
        delay, chunk = self._reads[0]
        self._wait(delay, timeout, httpcore2.ReadTimeout)
        self._reads.pop(0)
        return chunk

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        for start in range(0, len(buffer), self._send_chunk):
            self._wait(self._send_cost, timeout, httpcore2.WriteTimeout)
            self.bytes_written += len(buffer[start : start + self._send_chunk])

    def close(self) -> None:
        self.closed = True

    def start_tls(
        self, ssl_context: Any, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore2.NetworkStream:
        return self

    def get_extra_info(self, info: str) -> Any:
        return None


class _WireBackend(httpcore2.NetworkBackend):
    """A slow model server: a connect that takes ``connect_delay``, then ``reads``.

    Records, per connection, the deadline the socket layer could see, so a test
    can check that every attempt of one call ran against the same one.
    """

    def __init__(
        self,
        clock: _FakeClock,
        *,
        connect_delay: float = 0.0,
        reads: list[tuple[float, bytes]] | None = None,
        responses: list[list[tuple[float, bytes]]] | None = None,
        send_chunk: int = 4096,
        send_cost: float = 0.0,
        honour_timeouts: bool = True,
    ) -> None:
        self._clock = clock
        self._connect_delay = connect_delay
        self._responses = responses if responses is not None else [reads or []]
        self._send_chunk = send_chunk
        self._send_cost = send_cost
        self._honour = honour_timeouts
        self.connect_timeouts: list[float | None] = []
        self.deadlines_seen: list[float | None] = []
        self.connections: list[_WireStream] = []

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore2.NetworkStream:
        self.connect_timeouts.append(timeout)
        self.deadlines_seen.append(_current_deadline())
        if self._honour and timeout is not None and self._connect_delay > timeout:
            self._clock.advance(timeout)
            raise httpcore2.ConnectTimeout("fake socket: connect timed out")
        self._clock.advance(self._connect_delay)
        script = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        stream = _WireStream(
            self._clock,
            script,
            send_chunk=self._send_chunk,
            send_cost=self._send_cost,
            honour_timeouts=self._honour,
        )
        self.connections.append(stream)
        return stream

    def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None
    ) -> httpcore2.NetworkStream:  # pragma: no cover - the adapter dials TCP
        raise AssertionError("the adapter should dial TCP")

    def sleep(self, seconds: float) -> None:  # pragma: no cover - pool retries are off
        raise AssertionError("httpcore2's own connect retries should be off")


def _http_response(
    status: int = 200, body: bytes | None = None, *, reason: str = "OK"
) -> tuple[bytes, bytes]:
    """A complete HTTP/1.1 response as (head, body), one per connection."""
    if body is None:
        body = json.dumps(openai_text_completion("Aapka bakaya baarah hazaar hai.")).encode()
    lines = [
        f"HTTP/1.1 {status} {reason}",
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
        "Connection: close",
    ]
    return ("\r\n".join(lines) + "\r\n\r\n").encode(), body


def _in_chunks(data: bytes, size: int, gap: float) -> list[tuple[float, bytes]]:
    return [(gap, data[i : i + size]) for i in range(0, len(data), size)]


def _owned_service(**kwargs: Any) -> OpenAiCompatibleLlmService:
    """An adapter that builds its own client, as production does. No client injected."""
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("model", "configured-model")
    kwargs.setdefault("timeout_seconds", BUDGET)
    return OpenAiCompatibleLlmService(**kwargs)


def _default_pool(service: OpenAiCompatibleLlmService) -> httpcore2.ConnectionPool:
    return service._client._transport._pool  # noqa: SLF001 - the structure under test


def _install_behind_deadline(service: OpenAiCompatibleLlmService, wire: _WireBackend) -> None:
    """Put ``wire`` where the real socket backend was, *behind* the adapter's wrapper."""
    deadline_backend = _fixed("_DeadlineBackend")
    installed = _default_pool(service)._network_backend  # noqa: SLF001
    assert isinstance(installed, deadline_backend), (
        "the adapter's own client does not route its connections through _DeadlineBackend"
    )
    installed._inner = wire  # noqa: SLF001


#: A dribbling server: one 8-byte chunk every 50 ms, each read well inside any
#: read timeout, and the whole response several budgets long.
DRIBBLE_GAP = 0.05
SLOW_CONNECT = 0.3


def _dribbled_response() -> list[tuple[float, bytes]]:
    head, body = _http_response()
    return _in_chunks(head + body, 8, DRIBBLE_GAP)


def test_the_fake_dribbling_server_takes_several_budgets_to_answer() -> None:
    """The premise of the next tests: the whole exchange is far over budget."""
    total = SLOW_CONNECT + sum(delay for delay, _ in _dribbled_response())
    assert total > 3 * BUDGET
    assert all(delay < BUDGET for delay, _ in _dribbled_response())


def test_without_the_deadline_wrapper_the_post_parse_deadline_check_still_catches_it(
    clock,
) -> None:
    """Bypassing the socket wrapper alone no longer lets a slow reply through.

    httpx2's per-phase timeouts alone would let the connect and every read
    through - the pre-fix failure - but a second, independent guard now checks
    the wall clock once more after the body is parsed. Nothing here reaches a
    socket bound by ``_DeadlineBackend`` - the fake backend is swapped in below
    it, exactly as the old pre-fix control did - yet the call still ends as a
    timeout, because that check does not depend on the backend at all.
    """
    wire = _WireBackend(clock, connect_delay=SLOW_CONNECT, reads=_dribbled_response())
    with _owned_service() as service:
        _default_pool(service)._network_backend = wire  # noqa: SLF001 - bypass on purpose
        with pytest.raises(LlmTimeout) as caught:
            service.generate(_request())

    # The exchange really did run unbounded - every dribbled read completed -
    # which is what makes this a meaningful check of the *second* guard rather
    # than a restatement of the socket-level one.
    assert clock.elapsed > 3 * BUDGET
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "honour_timeouts, overrun",
    [
        pytest.param(True, 0.0, id="socket-honours-its-timeout"),
        pytest.param(False, DRIBBLE_GAP, id="socket-overruns-by-one-read"),
    ],
)
def test_a_slow_connect_then_a_dribbled_response_is_cut_at_the_call_deadline(
    clock, honour_timeouts, overrun
) -> None:
    """Phase accumulation: 0.3 s to connect plus 50 ms reads must stop at 0.4 s in all.

    Each phase, and each read, is inside its own timeout; only their sum is not.
    A socket that honours what it is given stops exactly on the budget; one
    that overruns can exceed it by at most the one read already under way.
    """
    wire = _WireBackend(
        clock,
        connect_delay=SLOW_CONNECT,
        reads=_dribbled_response(),
        honour_timeouts=honour_timeouts,
    )
    with _owned_service() as service:
        _install_behind_deadline(service, wire)
        with pytest.raises(LlmTimeout):
            service.generate(_request())

    assert clock.elapsed <= BUDGET + overrun + EPSILON, f"spent {clock.elapsed:.3f}s"
    assert len(wire.connections) == 1
    assert wire.connections[0].closed is True
    stream_timeouts = wire.connections[0].timeouts
    assert stream_timeouts, "no read or write reached the socket"
    assert all(t is not None and 0 < t <= BUDGET + EPSILON for t in stream_timeouts)
    assert wire.connect_timeouts[0] == pytest.approx(BUDGET)


def test_a_dribbled_response_with_a_fast_connect_is_cut_at_the_call_deadline(clock) -> None:
    wire = _WireBackend(clock, connect_delay=0.0, reads=_dribbled_response())
    with _owned_service() as service:
        _install_behind_deadline(service, wire)
        with pytest.raises(LlmTimeout):
            service.generate(_request())
    assert clock.elapsed <= BUDGET + EPSILON


def test_a_slow_connect_then_slow_headers_is_cut_at_the_call_deadline(clock) -> None:
    """0.3 s to connect, 0.3 s to the first byte: each fits 0.4 s, together they do not."""
    head, body = _http_response()
    wire = _WireBackend(clock, connect_delay=SLOW_CONNECT, reads=[(0.3, head), (0.0, body)])
    with _owned_service() as service:
        _install_behind_deadline(service, wire)
        with pytest.raises(LlmTimeout):
            service.generate(_request())
    assert clock.elapsed == pytest.approx(BUDGET)


def test_a_large_request_to_a_slowly_draining_server_is_cut_at_the_call_deadline(clock) -> None:
    """The write phase: 4 KiB per ``send``, 25 ms per send, a ~200 KB body.

    Every send is inside its timeout, so one ``write`` of the body would take
    over a second. Sliced at 16 KiB and re-clamped, it stops within one slice
    (four sends) of the budget.
    """
    head, body = _http_response()
    send_cost = 0.025
    wire = _WireBackend(
        clock,
        reads=[(0.0, head + body)],
        send_chunk=4096,
        send_cost=send_cost,
    )
    request = _request("ek " * 70_000)
    with _owned_service() as service:
        _install_behind_deadline(service, wire)
        with pytest.raises(LlmTimeout):
            service.generate(request)

    assert wire.connections[0].bytes_written < 200_000
    assert clock.elapsed <= BUDGET + (SLICE // 4096) * send_cost + EPSILON


def test_without_the_deadline_wrapper_a_slow_write_is_also_caught_by_the_post_parse_check(
    clock,
) -> None:
    """The write-side counterpart: an unbounded write still ends as a timeout.

    Per-send timeouts alone would let this write run on and the call "succeed"
    - the pre-fix failure - but the reply is still parsed after the deadline,
    and the post-parse check catches it regardless of which phase was slow.
    """
    head, body = _http_response()
    wire = _WireBackend(clock, reads=[(0.0, head + body)], send_chunk=4096, send_cost=0.025)
    with _owned_service() as service:
        _default_pool(service)._network_backend = wire  # noqa: SLF001 - bypass on purpose
        with pytest.raises(LlmTimeout) as caught:
            service.generate(_request("ek " * 70_000))

    # The whole body really went out unbounded - unlike the wrapped write test
    # above, which stops within one 16 KiB slice of the budget.
    assert wire.connections[0].bytes_written > 200_000
    assert clock.elapsed > 2 * BUDGET
    assert caught.value.__cause__ is None


def test_a_connection_that_opens_after_the_deadline_is_closed_and_not_retried(clock) -> None:
    """A slow DNS lookup: the connect itself overruns, whatever timeout it was given."""
    head, body = _http_response()
    wire = _WireBackend(
        clock, connect_delay=0.5, reads=[(0.0, head + body)], honour_timeouts=False
    )
    with _owned_service(max_retries=3) as service:
        _install_behind_deadline(service, wire)
        with pytest.raises(LlmTimeout) as caught:
            service.generate(_request())

    assert len(wire.connections) == 1
    assert wire.connections[0].closed is True
    assert wire.connections[0].bytes_written == 0, "a request went out on a late connection"
    assert isinstance(caught.value.__cause__, httpx2.ConnectTimeout)
    assert clock.sleeps == []


def test_a_fast_server_behind_the_wrapper_still_answers(clock) -> None:
    head, body = _http_response()
    wire = _WireBackend(clock, connect_delay=0.01, reads=[(0.02, head), (0.02, body)])
    with _owned_service() as service:
        _install_behind_deadline(service, wire)
        generation = service.generate(_request())
    assert generation.text == "Aapka bakaya baarah hazaar hai."
    assert clock.elapsed == pytest.approx(0.05)


def test_every_attempt_of_one_call_sees_the_same_deadline_at_the_socket(clock) -> None:
    """A retry does not get a fresh deadline: the second attempt runs against the first's."""
    busy_head, busy_body = _http_response(503, b'{"error":"busy"}', reason="Service Unavailable")
    head, body = _http_response()
    wire = _WireBackend(
        clock,
        connect_delay=0.05,
        responses=[[(0.05, busy_head + busy_body)], [(0.05, head + body)]],
    )
    with _owned_service(max_retries=1) as service:
        _install_behind_deadline(service, wire)
        generation = service.generate(_request())

    assert generation.text == "Aapka bakaya baarah hazaar hai."
    assert len(wire.connections) == 2
    assert wire.deadlines_seen == pytest.approx([clock.START + BUDGET] * 2)
    # The second connect was given only what the first attempt and the wait left.
    assert wire.connect_timeouts[1] == pytest.approx(BUDGET - 0.1 - clock.sleeps[0])


# --- installation on the adapter's own client -------------------------------------


def test_the_client_the_adapter_builds_routes_its_pool_through_the_deadline_backend() -> None:
    deadline_backend = _fixed("_DeadlineBackend")
    with _owned_service() as service:
        backend = _default_pool(service)._network_backend  # noqa: SLF001
        assert isinstance(backend, deadline_backend)
        assert isinstance(backend._inner, httpcore2.NetworkBackend)  # noqa: SLF001
        assert not isinstance(backend._inner, deadline_backend)  # noqa: SLF001


def test_with_proxies_in_the_environment_every_proxy_pool_is_bounded_too(monkeypatch) -> None:
    """httpx2 honours HTTP(S)_PROXY only for a client built without a transport.

    That is why the adapter swaps backends instead of passing its own
    transport, and why every proxy mount must be wrapped as well.
    """
    deadline_backend = _fixed("_DeadlineBackend")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")

    with _owned_service() as service:
        mounts = [t for t in service._client._mounts.values() if t is not None]  # noqa: SLF001
        assert len(mounts) == 3, "the environment's proxies were not mounted"
        for transport in [service._client._transport, *mounts]:  # noqa: SLF001
            assert isinstance(transport._pool._network_backend, deadline_backend)  # noqa: SLF001
        assert any(isinstance(t._pool, httpcore2.HTTPProxy) for t in mounts)  # noqa: SLF001


def test_a_no_proxy_exclusion_does_not_stop_the_adapter_from_starting(monkeypatch) -> None:
    """``NO_PROXY`` becomes a mount of ``None``, which routes a pattern to no proxy."""
    deadline_backend = _fixed("_DeadlineBackend")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    with _owned_service() as service:
        mounts = service._client._mounts  # noqa: SLF001
        assert None in mounts.values()
        for transport in [service._client._transport, *mounts.values()]:  # noqa: SLF001
            if transport is not None:
                assert isinstance(transport._pool._network_backend, deadline_backend)  # noqa: SLF001


def test_an_injected_client_is_used_as_given_and_not_altered(monkeypatch) -> None:
    """Kept behaviour: the adapter does not reach into a client it did not build."""
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    client = httpx2.Client()
    default_backend = client._transport._pool._network_backend  # noqa: SLF001
    mount_backends = {
        key: t._pool._network_backend  # noqa: SLF001
        for key, t in client._mounts.items()  # noqa: SLF001
        if t is not None
    }
    try:
        service = OpenAiCompatibleLlmService(base_url=BASE_URL, model="m", client=client)
        assert service._client is client  # noqa: SLF001
        assert client._transport._pool._network_backend is default_backend  # noqa: SLF001
        for key, transport in client._mounts.items():  # noqa: SLF001
            if transport is not None:
                assert transport._pool._network_backend is mount_backends[key]  # noqa: SLF001
        deadline_backend = getattr(llm_openai, "_DeadlineBackend", None)
        if deadline_backend is not None:
            assert not isinstance(default_backend, deadline_backend)
        service.close()
        assert not client.is_closed, "closing the adapter closed a client it does not own"
    finally:
        client.close()


def test_wrapping_twice_does_not_nest_the_deadline_backend() -> None:
    enforce = _fixed("_enforce_deadline")
    deadline_backend = _fixed("_DeadlineBackend")
    with _owned_service() as service:
        enforce(service._client)  # noqa: SLF001
        backend = _default_pool(service)._network_backend  # noqa: SLF001
        assert isinstance(backend, deadline_backend)
        assert not isinstance(backend._inner, deadline_backend)  # noqa: SLF001


def _handler(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover - never sent
    return httpx2.Response(200, json=openai_text_completion("ok"))


def _no_pool(client: httpx2.Client) -> None:
    client._transport = httpx2.MockTransport(_handler)  # noqa: SLF001


def _pool_without_backend(client: httpx2.Client) -> None:
    del client._transport._pool._network_backend  # noqa: SLF001


def _backend_of_the_wrong_kind(client: httpx2.Client) -> None:
    client._transport._pool._network_backend = object()  # noqa: SLF001


def _pool_of_the_wrong_kind(client: httpx2.Client) -> None:
    client._transport._pool = SimpleNamespace(  # noqa: SLF001
        _network_backend=httpcore2.SyncBackend(), close=lambda: None
    )


def _proxy_mount_without_a_pool(client: httpx2.Client) -> None:
    from httpx2._utils import URLPattern  # the key type of the private mount table

    client._mounts = {URLPattern("all://"): httpx2.MockTransport(_handler)}  # noqa: SLF001


_BROKEN_STRUCTURES = [
    pytest.param(_no_pool, id="transport-without-a-pool"),
    pytest.param(_pool_without_backend, id="pool-without-a-network-backend"),
    pytest.param(_backend_of_the_wrong_kind, id="backend-of-the-wrong-kind"),
    pytest.param(_pool_of_the_wrong_kind, id="pool-of-the-wrong-kind"),
    pytest.param(_proxy_mount_without_a_pool, id="proxy-mount-without-a-pool"),
]


@pytest.mark.parametrize("breakage", _BROKEN_STRUCTURES)
def test_a_client_without_the_expected_structure_is_refused(breakage) -> None:
    enforce = _fixed("_enforce_deadline")
    client = httpx2.Client()
    try:
        breakage(client)
        with pytest.raises(LlmConfigurationError):
            enforce(client)
    finally:
        client.close()


@pytest.mark.parametrize("breakage", _BROKEN_STRUCTURES)
def test_an_adapter_whose_client_cannot_be_bounded_refuses_to_start_and_closes_it(
    monkeypatch, breakage
) -> None:
    """Refuse rather than degrade: an httpx2 that moved its internals must not run unbounded."""
    real_client = httpx2.Client
    built: list[httpx2.Client] = []
    closes: list[httpx2.Client] = []

    class _TrackedClient(real_client):
        def close(self) -> None:
            closes.append(self)
            super().close()

    def factory(*args: Any, **kwargs: Any) -> httpx2.Client:
        client = _TrackedClient(*args, **kwargs)
        breakage(client)
        built.append(client)
        return client

    monkeypatch.setattr(llm_openai.httpx2, "Client", factory)
    with pytest.raises(LlmConfigurationError) as caught:
        _owned_service()

    assert len(built) == 1
    assert closes == built, "the refused client was left open"
    assert built[0].is_closed
    assert "MODEL_TIMEOUT_SECONDS" in str(caught.value)


# --- the deadline ContextVar ------------------------------------------------------


def _injected_service(handler, **kwargs: Any) -> OpenAiCompatibleLlmService:
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("model", "configured-model")
    kwargs.setdefault("timeout_seconds", BUDGET)
    return OpenAiCompatibleLlmService(
        client=httpx2.Client(transport=httpx2.MockTransport(handler)), **kwargs
    )


def _ok(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=openai_text_completion("ok"))


def _busy(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(503, json={"error": "busy"})


def _garbled(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, text="{not json", headers={"Content-Type": "application/json"})


def _refused(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("refused")


def _slow(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("slow")


@pytest.mark.parametrize(
    "respond, expected",
    [
        pytest.param(_ok, None, id="returns"),
        pytest.param(_busy, LlmUpstreamError, id="raises-upstream"),
        pytest.param(_garbled, LlmMalformedResponse, id="raises-malformed"),
        pytest.param(_refused, LlmConnectionFailed, id="raises-connection"),
        pytest.param(_slow, LlmTimeout, id="raises-timeout"),
    ],
)
def test_the_deadline_is_set_during_the_request_and_unset_however_the_call_ends(
    clock, respond, expected
) -> None:
    var = _fixed("_CALL_DEADLINE")
    seen: list[float | None] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(var.get())
        return respond(request)

    service = _injected_service(handler)
    if expected is None:
        service.generate(_request())
    else:
        with pytest.raises(expected):
            service.generate(_request())

    assert seen == [clock.START + BUDGET]
    assert var.get() is None


def test_the_deadline_is_unset_after_a_call_cut_off_at_the_socket(clock) -> None:
    var = _fixed("_CALL_DEADLINE")
    wire = _WireBackend(clock, connect_delay=SLOW_CONNECT, reads=_dribbled_response())
    with _owned_service() as service:
        _install_behind_deadline(service, wire)
        with pytest.raises(LlmTimeout):
            service.generate(_request())
    assert var.get() is None


def test_attempts_and_waits_all_run_against_the_one_deadline(clock) -> None:
    var = _fixed("_CALL_DEADLINE")
    seen: list[float | None] = []
    replies = [_busy, _busy, _ok]

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(var.get())
        clock.advance(0.02)
        return replies.pop(0)(request)

    generation = _injected_service(handler, max_retries=2).generate(_request())

    assert generation.text == "ok"
    assert seen == [clock.START + BUDGET] * 3
    assert len(clock.sleeps) == 2
    assert var.get() is None


def test_a_deadline_set_by_one_call_is_invisible_on_another_thread(clock) -> None:
    """One client is shared by concurrent calls, each on its own worker thread."""
    var = _fixed("_CALL_DEADLINE")
    inside_first = threading.Event()
    second_done = threading.Event()
    seen_first: list[float | None] = []
    seen_second: list[float | None] = []
    outcome: dict[str, Any] = {}

    def first_handler(request: httpx2.Request) -> httpx2.Response:
        seen_first.append(var.get())
        inside_first.set()
        second_done.wait(5.0)
        seen_first.append(var.get())
        return _ok(request)

    def second_handler(request: httpx2.Request) -> httpx2.Response:
        seen_second.append(var.get())
        return _ok(request)

    first = _injected_service(first_handler, timeout_seconds=1.0)
    second = _injected_service(second_handler, timeout_seconds=5.0)

    def run_first() -> None:
        try:
            outcome["generation"] = first.generate(_request())
        except BaseException as exc:  # noqa: BLE001 - reported below
            outcome["error"] = exc

    worker = threading.Thread(target=run_first, daemon=True)
    worker.start()
    try:
        assert inside_first.wait(5.0)
        assert var.get() is None, "a call on another thread leaked its deadline into this one"
        second.generate(_request())
    finally:
        second_done.set()
        worker.join(5.0)

    assert "error" not in outcome, outcome.get("error")
    assert seen_first == [clock.START + 1.0, clock.START + 1.0]
    assert seen_second == [clock.START + 5.0]
    assert var.get() is None


# --- the exchange: a status is decided before a body, which has a size cap ----


class _CountingStream(httpx2.SyncByteStream):
    """A response body that remembers how much of itself was actually pulled.

    Built with ``stream=`` rather than ``content=`` so nothing is read eagerly:
    a non-200 status must be decidable without ever touching this.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.iterations = 0
        self.chunks_pulled = 0

    def __iter__(self) -> Iterator[bytes]:
        self.iterations += 1
        for chunk in self._chunks:
            self.chunks_pulled += 1
            yield chunk

    def close(self) -> None:
        pass


@pytest.mark.parametrize("status", [201, 204, 301, 302])
def test_a_non_200_status_is_decided_before_the_body_is_read(status) -> None:
    """Whatever the body behind it says, only 200 is a completion.

    A 3xx is not followed - ``follow_redirects`` is not set - and a 201 or 204
    answers nothing; none of them are worth the cost of reading a body that
    will be thrown away regardless of what it contains.
    """
    body = _CountingStream([json.dumps(openai_text_completion("should never be spoken")).encode()])

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, headers={"Content-Type": "application/json"}, stream=body)

    with pytest.raises(LlmUpstreamError) as caught:
        _injected_service(handler).generate(_request())
    assert caught.value.status_code == status
    assert body.iterations == 0, "the body behind a non-200 status must never be iterated"


def test_a_response_over_the_size_cap_is_rejected_without_buffering_all_of_it() -> None:
    """Reading stops the moment the decoded body crosses the cap, not after all of it arrives."""
    one_mib = 1024 * 1024
    body = _CountingStream([b"a" * one_mib] * 20)  # 20 MiB, if all of it were ever read

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"Content-Type": "application/json"}, stream=body)

    with pytest.raises(LlmMalformedResponse):
        _injected_service(handler).generate(_request())
    # 8 MiB is the cap: the 9th one-MiB chunk is what crosses it, and nothing
    # past that point is worth pulling off the wire.
    assert body.chunks_pulled == 9


def test_a_gzip_body_that_inflates_past_the_cap_is_rejected_too() -> None:
    """The cap is on the *decoded* size: a small compressed body is no loophole.

    The padding makes an otherwise well-formed, parseable reply: were the cap
    not applied (or applied to the compressed size instead), this would decode
    cleanly and ``generate`` would return a generation, not raise.
    """
    reply = openai_text_completion("Aapka bakaya baarah hazaar hai.")
    reply["padding"] = "0" * (9 * 1024 * 1024)
    compressed = gzip.compress(json.dumps(reply).encode())
    assert len(compressed) < 100_000, "the point is a small body that inflates past the cap"

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
            content=compressed,
        )

    with pytest.raises(LlmMalformedResponse):
        _injected_service(handler).generate(_request())


def test_a_reply_parsed_after_the_deadline_is_reported_as_a_timeout(clock) -> None:
    """Reading and decoding happen between socket operations and are not themselves
    bounded by anything but the wall clock; a reply that lands too late is not
    handed back just because it arrived well formed.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        clock.advance(BUDGET + 0.05)  # time passes while the body is "read"
        return _ok(request)

    with pytest.raises(LlmTimeout) as caught:
        _injected_service(handler).generate(_request())
    assert caught.value.__cause__ is None


# --- the pins that make the private attributes safe to use ------------------------


def _requirement_lines(package: str) -> list[str]:
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    named = []
    for line in lines:
        text = line.split("#", 1)[0].strip()
        name = re.split(r"[\s\[<>=!~;]", text, maxsplit=1)[0] if text else ""
        if name.lower() == package:
            named.append(text)
    return named


@pytest.mark.parametrize("package", ["httpx2", "httpcore2"])
def test_requirements_pin_the_http_stack_exactly(package) -> None:
    assert _requirement_lines(package) == [f"{package}==2.13.1"]


@pytest.mark.parametrize("package", ["httpx2", "httpcore2"])
def test_the_installed_http_stack_is_the_pinned_version(package) -> None:
    assert importlib.metadata.version(package) == "2.13.1"


# --- real sockets on 127.0.0.1 -------------------------------------------------------


class _ScriptedSocketServer:
    """A raw TCP listener on 127.0.0.1 that answers every connection from one script.

    ``steps`` is a list of ``(delay, bytes)``: wait, then send. Used where the
    property is about a real socket's per-operation timeouts, which no fake can
    vouch for. Stops dribbling as soon as the client hangs up or the test ends.
    """

    def __init__(self, steps: list[tuple[float, bytes]]) -> None:
        self._steps = list(steps)
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._listener.settimeout(0.05)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.connections = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._listener.getsockname()[1]}/v1"

    def __enter__(self) -> "_ScriptedSocketServer":
        self._spawn(self._accept)
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=1.0)
        self._listener.close()

    def _spawn(self, target, *args: Any) -> None:
        thread = threading.Thread(target=target, args=args, daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                self.connections += 1
            self._spawn(self._answer, conn)

    def _answer(self, conn: socket.socket) -> None:
        with conn:
            try:
                conn.settimeout(1.0)
                self._read_request(conn)
                for delay, data in self._steps:
                    if self._stop.wait(delay):
                        return
                    conn.sendall(data)
            except OSError:
                return

    @staticmethod
    def _read_request(conn: socket.socket) -> None:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(65536)
            if not chunk:
                return
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        match = re.search(rb"(?i)content-length:\s*(\d+)", head)
        length = int(match.group(1)) if match else 0
        while len(rest) < length:
            chunk = conn.recv(65536)
            if not chunk:
                return
            rest += chunk


def _dribble(data: bytes, gap: float = DRIBBLE_GAP) -> list[tuple[float, bytes]]:
    return [(gap, data[i : i + 1]) for i in range(len(data))]


def _probe(server: _ScriptedSocketServer, **kwargs: Any) -> tuple[BaseException | Any, float]:
    """One call against ``server``; returns (generation or error, seconds taken)."""
    started = time.monotonic()
    with _owned_service(base_url=server.base_url, **kwargs) as service:
        try:
            outcome: Any = service.generate(_request())
        except Exception as exc:  # noqa: BLE001 - the caller asserts on it
            outcome = exc
    return outcome, time.monotonic() - started


def test_real_socket_headers_dribbled_a_byte_at_a_time_time_out_near_the_budget() -> None:
    """Each byte resets httpx2's read timeout. Pre-fix this took over 3 s and succeeded."""
    head, body = _http_response()
    assert len(head) * DRIBBLE_GAP > 3.0
    with _ScriptedSocketServer([*_dribble(head), (0.0, body)]) as server:
        outcome, elapsed = _probe(server)
    assert isinstance(outcome, LlmTimeout), outcome
    assert elapsed < 1.3


def test_real_socket_a_body_dribbled_after_prompt_headers_times_out_near_the_budget() -> None:
    head, body = _http_response()
    steps = [(0.0, head), (0.0, body[:-64]), *_dribble(body[-64:])]
    with _ScriptedSocketServer(steps) as server:
        outcome, elapsed = _probe(server)
    assert isinstance(outcome, LlmTimeout), outcome
    assert elapsed < 1.3


def test_real_socket_slow_headers_then_a_slow_body_add_up_to_a_timeout() -> None:
    """0.3 s then 0.3 s: each inside the 0.4 s read timeout, together not.

    Pre-fix this took about 0.6 s and returned a generation.
    """
    head, body = _http_response()
    with _ScriptedSocketServer([(0.3, head), (0.3, body)]) as server:
        outcome, elapsed = _probe(server)
    assert isinstance(outcome, LlmTimeout), outcome
    assert elapsed < 1.0


def test_real_socket_a_dribble_with_retries_enabled_is_still_one_bounded_attempt() -> None:
    """A read timeout is not retried: the request may already have reached the model."""
    head, body = _http_response()
    with _ScriptedSocketServer([*_dribble(head), (0.0, body)]) as server:
        outcome, elapsed = _probe(server, max_retries=3)
        connections = server.connections
    assert isinstance(outcome, LlmTimeout), outcome
    assert elapsed < 1.3
    assert connections == 1


def test_real_socket_a_503_whose_body_is_dribbled_is_never_read_and_returns_promptly() -> None:
    """An error status is decided from the head alone; the dribbled body behind it is
    never drained, so a 503 no longer waits anywhere near the deadline to be reported.

    Pre-fix (and pre-``_exchange``) this read through the whole dribble and only
    then timed out, close to the budget; the better behaviour is to fail fast.
    """
    body = json.dumps({"error": {"message": "overloaded " + "z" * 60}}).encode()
    head, _ = _http_response(503, body, reason="Service Unavailable")
    steps = [(0.0, head), (0.0, body[:-64]), *_dribble(body[-64:])]
    assert len(_dribble(body[-64:])) * DRIBBLE_GAP > BUDGET, "the dribble alone outlasts the budget"
    with _ScriptedSocketServer(steps) as server:
        outcome, elapsed = _probe(server)
    assert isinstance(outcome, LlmUpstreamError), outcome
    assert outcome.status_code == 503
    assert elapsed < BUDGET, f"spent {elapsed:.3f}s waiting on a body that is never read"


def test_real_socket_a_fast_server_still_returns_a_generation() -> None:
    head, body = _http_response()
    with _ScriptedSocketServer([(0.0, head + body)]) as server:
        outcome, elapsed = _probe(server)
        connections = server.connections
    assert not isinstance(outcome, BaseException), outcome
    assert outcome.text == "Aapka bakaya baarah hazaar hai."
    assert connections == 1
    assert elapsed < 1.0
