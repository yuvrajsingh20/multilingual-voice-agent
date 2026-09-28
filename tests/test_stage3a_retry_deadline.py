"""Retry waits against the one shared deadline of a model call.

The adapter gives a model call a single wall-clock budget, spent by every
attempt and by every wait between attempts. Before this was pinned the adapter
would take a wait whenever the wait itself fitted in the time left, even when
nothing could follow it: a 503 with 55 ms left bought a 50 ms sleep, after which
the loop found under 10 ms remaining and raised a brand-new ``LlmTimeout``. The
customer's silence was spent for nothing, and the operator was told the model
server was slow when it had in fact said "503, later". A ``Retry-After`` date
the server garbled escaped the adapter altogether, as a ``ValueError``.

What is being pinned
--------------------
- A wait is taken only when an attempt can still follow it
  (``delay + _MIN_ATTEMPT_SECONDS <= time left``). Otherwise the failure that
  actually happened is raised, with no sleep and no second request.
- A wait that overruns - a sleep is a lower bound, not an exact duration - and
  leaves no room for an attempt surfaces the failure that preceded it, logged
  against the attempt that actually ran, never a fresh timeout.
- When there is room, the retry is made, is given only the time actually left,
  and can succeed.
- Retries are bounded (at most three, none by default), happen only on the
  statuses a server uses for "later" and on transport failures from before the
  request left, and never on anything that may already have reached the model.
- ``Retry-After`` is honoured when it is a usable future delay, falls back to
  backoff when it is zero, negative or in the past, and never raises.
- Across every schedule of failures, attempts plus waits never exceed the
  configured ``timeout_seconds``.

Every test runs on a fake clock: only the adapter module's view of ``time`` is
replaced, and the scripted endpoint charges each attempt to that clock. Nothing
sleeps for real and no socket is opened.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import httpx2
import pytest
from pydantic import ValidationError

from app.orchestrator import ConversationOrchestrator, TurnErrorCategory, TurnOutcome
from app.services import llm_openai
from app.services.llm import (
    LlmConnectionFailed,
    LlmError,
    LlmMessage,
    LlmRequest,
    LlmTimeout,
    LlmUpstreamError,
)
from app.services.llm_openai import OpenAiCompatibleLlmService, _parse_retry_after
from tests.fakes import openai_text_completion
from tests.test_llm_http_integration import http_llm
from tests.test_orchestrator import make_runtime, make_settings, open_session, say

BASE_URL = "http://model.invalid:8000/v1"

#: The configured ``timeout_seconds`` for every test unless one says otherwise.
BUDGET = 1.0

#: Float slack for "never exceeds the budget". The fake clock adds decimal
#: fractions to 1000.0, which is exact to about 1e-13; this is far above that
#: and far below the millisecond distances the tests are about.
EPSILON = 1e-9

#: The backoff the adapter uses for its first three retries.
BACKOFF = [0.05, 0.1, 0.2]


def _request() -> LlmRequest:
    return LlmRequest(
        messages=(
            LlmMessage(role="system", content="CONSTRAINTS\n- tone: neutral"),
            LlmMessage(role="user", content="kitna bakaya hai?"),
        )
    )


# --- the fake clock and a scripted endpoint ---------------------------------


class _FakeClock:
    """The adapter's view of time, advanced only by the test.

    ``monotonic`` and ``perf_counter`` read one counter. ``sleep`` records what
    the adapter asked for and advances the counter by exactly that much - unless
    ``wake_at`` is set, in which case that one sleep overruns and ends at
    ``wake_at`` seconds into the call, the way a loaded host oversleeps.
    """

    START = 1_000.0

    def __init__(self) -> None:
        self.now = self.START
        self.sleeps: list[float] = []
        self.wake_at: float | None = None

    def monotonic(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.wake_at is not None:
            self.now = max(self.now, self.START + self.wake_at)
            self.wake_at = None

    @property
    def elapsed(self) -> float:
        return self.now - self.START


def _install_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Swap the adapter module's ``time`` for a fresh fake clock."""
    clock = _FakeClock()
    monkeypatch.setattr(
        llm_openai,
        "time",
        SimpleNamespace(
            monotonic=clock.monotonic, perf_counter=clock.perf_counter, sleep=clock.sleep
        ),
    )
    return clock


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    return _install_clock(monkeypatch)


@dataclasses.dataclass(frozen=True)
class _Step:
    """One scripted attempt: how long it takes, and how it ends."""

    cost: float = 0.0
    status: int = 200
    retry_after: str | None = None
    error: type[Exception] | None = None
    text: str = "ok"


def _fails(status: int, *, cost: float = 0.0, retry_after: str | None = None) -> _Step:
    return _Step(cost=cost, status=status, retry_after=retry_after)


def _raises(error: type[Exception], *, cost: float = 0.0) -> _Step:
    return _Step(cost=cost, error=error)


def _answers(text: str = "ok", *, cost: float = 0.0) -> _Step:
    return _Step(cost=cost, text=text)


class _Transport:
    """A ``MockTransport`` handler that plays a script against the fake clock.

    Each request is charged its step's cost. The last step repeats once the
    script runs out. A step that would take longer than the read timeout the
    adapter put on the request is cut off at that timeout and ends as a read
    timeout, as a real transport would: that keeps the fake clock honest about
    the one thing the adapter controls inside an attempt, the budget it hands
    the request.
    """

    def __init__(self, clock: _FakeClock, *steps: _Step) -> None:
        self._clock = clock
        self._steps = list(steps)
        self.requests: list[httpx2.Request] = []
        self.timeouts: list[dict] = []
        self.cut_off: list[bool] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        request.read()
        self.requests.append(request)
        timeout = request.extensions["timeout"]
        self.timeouts.append(timeout)
        step = self._steps.pop(0) if len(self._steps) > 1 else self._steps[0]
        budget = timeout["read"]
        if budget is not None and step.cost > budget:
            self._clock.advance(budget)
            self.cut_off.append(True)
            raise httpx2.ReadTimeout("fake transport: the attempt outlived its timeout")
        self.cut_off.append(False)
        self._clock.advance(step.cost)
        if step.error is not None:
            raise step.error("fake transport failure")
        if step.status >= 400:
            headers = {"Retry-After": step.retry_after} if step.retry_after is not None else {}
            return httpx2.Response(
                step.status, json={"error": {"message": "busy"}}, headers=headers
            )
        return httpx2.Response(200, json=openai_text_completion(step.text))


def _service(transport: _Transport, **kwargs) -> OpenAiCompatibleLlmService:
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("model", "configured-model")
    kwargs.setdefault("timeout_seconds", BUDGET)
    return OpenAiCompatibleLlmService(
        client=httpx2.Client(transport=httpx2.MockTransport(transport)), **kwargs
    )


def _llm_records(stream) -> list[dict]:
    lines = (json.loads(line) for line in stream.getvalue().splitlines() if line.strip())
    return [record for record in lines if record.get("event") == "llm_call"]


def _assert_within_budget(clock: _FakeClock, budget: float = BUDGET) -> None:
    assert clock.elapsed <= budget + EPSILON, f"spent {clock.elapsed:.6f}s of a {budget}s budget"


# --- a wait is taken only when an attempt can follow it ---------------------


@pytest.mark.parametrize(
    "retry_after",
    [
        pytest.param(None, id="backoff-50ms"),
        pytest.param("0.001", id="retry-after-1ms"),
    ],
)
def test_a_503_with_under_ten_milliseconds_left_is_raised_without_a_wait(
    clock, retry_after
) -> None:
    """5 ms left is not enough for an attempt, however short the wait before it.

    A 1 ms ``Retry-After`` fits in 5 ms, which was once enough to buy a sleep
    followed by a timeout that never touched the server.
    """
    transport = _Transport(clock, _fails(503, cost=0.995, retry_after=retry_after))

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert caught.value.status_code == 503
    assert clock.sleeps == []
    assert len(transport.requests) == 1
    _assert_within_budget(clock)


def test_a_budget_spent_to_exactly_zero_raises_the_failure_that_spent_it(clock) -> None:
    transport = _Transport(clock, _fails(503, cost=BUDGET))

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert caught.value.status_code == 503
    assert clock.sleeps == []
    assert len(transport.requests) == 1
    assert clock.elapsed == pytest.approx(BUDGET)
    _assert_within_budget(clock)


def test_a_retry_after_longer_than_the_time_left_is_not_waited_for(clock) -> None:
    transport = _Transport(clock, _fails(503, cost=0.1, retry_after="2"))

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert caught.value.status_code == 503
    assert caught.value.retry_after_seconds == 2.0
    assert clock.sleeps == []
    assert len(transport.requests) == 1
    _assert_within_budget(clock)


def test_a_backoff_longer_than_the_time_left_is_not_waited_for(clock) -> None:
    """30 ms left, a 50 ms backoff: no wait, the 503 is the answer."""
    transport = _Transport(clock, _fails(503, cost=0.97))

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert caught.value.status_code == 503
    assert clock.sleeps == []
    assert len(transport.requests) == 1
    _assert_within_budget(clock)


@pytest.mark.parametrize(
    ("steps", "max_retries", "status", "sleeps", "outcomes"),
    [
        pytest.param(
            (_fails(503, cost=0.945),),
            1,
            503,
            [],
            ["failed"],
            id="50ms-backoff-with-55ms-left",
        ),
        pytest.param(
            (_fails(429, cost=0.795, retry_after="0.2"),),
            1,
            429,
            [],
            ["failed"],
            id="200ms-retry-after-with-205ms-left",
        ),
        pytest.param(
            (_fails(503, cost=0.1), _fails(503, cost=0.745)),
            2,
            503,
            [0.05],
            ["retrying", "failed"],
            id="second-100ms-backoff-with-105ms-left",
        ),
    ],
)
def test_a_wait_that_fits_but_leaves_no_room_for_an_attempt_is_not_taken(
    clock, log_stream, steps, max_retries, status, sleeps, outcomes
) -> None:
    """The gap: ``delay < time left < delay + 10 ms``.

    The wait fits; the attempt after it would not. Taking the wait used to end
    in a fresh ``LlmTimeout``, so a server that said "503, later" was reported
    as a slow one. The original status is what the caller must get.
    """
    transport = _Transport(clock, *steps)

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=max_retries).generate(_request())

    assert not isinstance(caught.value, LlmTimeout)
    assert caught.value.status_code == status
    assert clock.sleeps == sleeps
    assert len(transport.requests) == len(sleeps) + 1

    records = _llm_records(log_stream)
    assert [r["outcome"] for r in records] == outcomes
    final = records[-1]
    assert final["attempt"] == len(transport.requests)
    assert final["error_type"] == "LlmUpstreamError"
    assert final["http_status"] == status
    _assert_within_budget(clock)


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        pytest.param(httpx2.ConnectError, LlmConnectionFailed, id="connect-error"),
        pytest.param(httpx2.ConnectTimeout, LlmTimeout, id="connect-timeout"),
        pytest.param(httpx2.ProxyError, LlmConnectionFailed, id="proxy-error"),
        pytest.param(httpx2.PoolTimeout, LlmTimeout, id="pool-timeout"),
    ],
)
def test_a_retryable_transport_failure_near_the_deadline_is_raised_as_itself_without_a_wait(
    clock, log_stream, error, raised
) -> None:
    """55 ms left and a 50 ms backoff: the connect failure itself is the answer.

    For a connect timeout that means an ``LlmTimeout`` *caused by* the connect
    timeout - not the cause-less timeout the retry loop raises when it has run
    out of budget, which says nothing about what went wrong.
    """
    transport = _Transport(clock, _raises(error, cost=0.945))

    with pytest.raises(raised) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert type(caught.value.__cause__) is error
    assert clock.sleeps == []
    assert len(transport.requests) == 1

    records = _llm_records(log_stream)
    assert [r["outcome"] for r in records] == ["failed"]
    assert records[-1]["attempt"] == 1
    assert records[-1]["error_type"] == raised.__name__
    _assert_within_budget(clock)


def test_a_budget_too_small_for_any_attempt_sends_nothing_and_times_out(
    clock, log_stream
) -> None:
    """The one case where a cause-less ``LlmTimeout`` is the honest answer."""
    transport = _Transport(clock, _answers())

    with pytest.raises(LlmTimeout) as caught:
        _service(transport, timeout_seconds=0.005, max_retries=3).generate(_request())

    assert caught.value.__cause__ is None
    assert transport.requests == []
    assert clock.sleeps == []
    final = _llm_records(log_stream)[-1]
    assert (final["outcome"], final["attempt"], final["error_type"]) == ("failed", 1, "LlmTimeout")


# --- a wait that overruns ---------------------------------------------------


@pytest.mark.parametrize(
    "left_after_wait",
    [pytest.param(0.005, id="5ms-left"), pytest.param(0.0, id="nothing-left")],
)
def test_a_wait_that_overruns_the_budget_surfaces_the_503_that_preceded_it(
    clock, log_stream, left_after_wait
) -> None:
    """The adapter asked for 50 ms and the host slept far longer.

    No attempt fits afterwards, so no request is sent - and what the caller
    gets is the 503 that the wait was taken after, logged against attempt 1,
    the attempt that actually ran. Not a timeout, and not an attempt 2 that
    never happened.
    """
    transport = _Transport(clock, _fails(503, cost=0.1))
    clock.wake_at = BUDGET - left_after_wait

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert caught.value.status_code == 503
    assert len(transport.requests) == 1
    assert clock.sleeps == [0.05]

    records = _llm_records(log_stream)
    assert [r["outcome"] for r in records] == ["retrying", "failed"]
    final = records[-1]
    assert final["attempt"] == 1
    assert final["error_type"] == "LlmUpstreamError"
    assert final["error_category"] == "llm_upstream_error"
    assert final["http_status"] == 503
    _assert_within_budget(clock)


def test_a_wait_that_overruns_after_a_connect_timeout_surfaces_that_connect_timeout(
    clock, log_stream
) -> None:
    transport = _Transport(clock, _raises(httpx2.ConnectTimeout, cost=0.1))
    clock.wake_at = BUDGET - 0.005

    with pytest.raises(LlmTimeout) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert type(caught.value.__cause__) is httpx2.ConnectTimeout
    assert len(transport.requests) == 1
    assert clock.sleeps == [0.05]

    records = _llm_records(log_stream)
    assert [r["outcome"] for r in records] == ["retrying", "failed"]
    final = records[-1]
    assert final["attempt"] == 1
    assert final["error_type"] == "LlmTimeout"
    assert final.get("http_status") is None
    _assert_within_budget(clock)


# --- when there is room, the retry is made ----------------------------------


@pytest.mark.parametrize(
    ("wake_at", "left"),
    [
        pytest.param(None, 0.015, id="exact-sleep-15ms-left"),
        pytest.param(0.989, 0.011, id="oversleep-11ms-left"),
    ],
)
def test_with_room_left_after_the_wait_the_retry_is_made_and_can_succeed(
    clock, log_stream, wake_at, left
) -> None:
    """65 ms left and a 50 ms backoff: room for the wait and an attempt after it.

    The retry is given only the time actually left once the wait is over -
    including any oversleep - not a fresh budget.
    """
    transport = _Transport(
        clock, _fails(503, cost=0.935), _answers("dobara koshish safal", cost=0.01)
    )
    clock.wake_at = wake_at

    generation = _service(transport, max_retries=1).generate(_request())

    assert generation.text == "dobara koshish safal"
    assert clock.sleeps == [0.05]
    assert len(transport.requests) == 2
    assert transport.timeouts[1]["read"] == pytest.approx(left, abs=1e-9)
    assert transport.timeouts[1]["connect"] == pytest.approx(left, abs=1e-9)

    records = _llm_records(log_stream)
    assert [r["outcome"] for r in records] == ["retrying", "ok"]
    assert [r["attempt"] for r in records] == [1, 2]
    _assert_within_budget(clock)


# --- retries are bounded ----------------------------------------------------


def test_retries_are_off_by_default_so_one_failure_is_one_request(clock) -> None:
    transport = _Transport(clock, _fails(503))

    with pytest.raises(LlmUpstreamError):
        _service(transport).generate(_request())

    assert len(transport.requests) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("max_retries", [0, 1, 2, 3])
def test_the_retry_bound_is_the_number_of_extra_attempts_and_no_more(
    clock, max_retries
) -> None:
    transport = _Transport(clock, _fails(503, cost=0.01))

    with pytest.raises(LlmUpstreamError):
        _service(transport, max_retries=max_retries).generate(_request())

    assert len(transport.requests) == max_retries + 1
    assert clock.sleeps == BACKOFF[:max_retries]
    _assert_within_budget(clock)


@pytest.mark.parametrize("max_retries", [4, 10, -1, True, 1.0, "2"])
def test_a_retry_bound_outside_zero_to_three_is_refused(clock, max_retries) -> None:
    with pytest.raises(ValueError):
        _service(_Transport(clock, _answers()), max_retries=max_retries)


def test_the_adapter_and_the_configuration_agree_on_the_retry_cap() -> None:
    assert llm_openai._MAX_RETRIES == 3
    assert make_settings(model_max_retries=3).model_max_retries == 3
    with pytest.raises(ValidationError):
        make_settings(model_max_retries=4)


# --- which failures are retried ---------------------------------------------


def test_the_retryable_statuses_are_exactly_the_transient_ones() -> None:
    assert llm_openai._RETRYABLE_STATUSES == frozenset({408, 429, 500, 502, 503, 504})


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_a_transient_status_is_retried_once_when_one_retry_is_allowed(clock, status) -> None:
    transport = _Transport(clock, _fails(status, cost=0.01))

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=1).generate(_request())

    assert caught.value.status_code == status
    assert len(transport.requests) == 2
    assert clock.sleeps == [0.05]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 501, 505])
def test_a_status_a_retry_cannot_fix_is_never_retried(clock, status) -> None:
    transport = _Transport(clock, _fails(status, cost=0.01))

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert caught.value.status_code == status
    assert len(transport.requests) == 1
    assert clock.sleeps == []


_BEFORE_THE_REQUEST_LEFT = [
    pytest.param(httpx2.ProxyError, LlmConnectionFailed, id="proxy-error"),
    pytest.param(httpx2.ConnectTimeout, LlmTimeout, id="connect-timeout"),
    pytest.param(httpx2.PoolTimeout, LlmTimeout, id="pool-timeout"),
    pytest.param(httpx2.ConnectError, LlmConnectionFailed, id="connect-error"),
]


@pytest.mark.parametrize(("error", "raised"), _BEFORE_THE_REQUEST_LEFT)
def test_a_failure_before_the_request_left_is_retried_and_the_retry_can_succeed(
    clock, error, raised
) -> None:
    transport = _Transport(clock, _raises(error, cost=0.01), _answers("jud gaya"))

    generation = _service(transport, max_retries=1).generate(_request())

    assert generation.text == "jud gaya"
    assert len(transport.requests) == 2
    assert clock.sleeps == [0.05]


@pytest.mark.parametrize(("error", "raised"), _BEFORE_THE_REQUEST_LEFT)
def test_a_failure_before_the_request_left_keeps_its_kind_through_the_retries(
    clock, error, raised
) -> None:
    """Retry policy reads the cause's type, so the type must survive every attempt."""
    transport = _Transport(clock, _raises(error, cost=0.01))

    with pytest.raises(raised) as caught:
        _service(transport, max_retries=2).generate(_request())

    assert type(caught.value.__cause__) is error
    assert len(transport.requests) == 3
    assert clock.sleeps == BACKOFF[:2]


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        pytest.param(httpx2.ReadTimeout, LlmTimeout, id="read-timeout"),
        pytest.param(httpx2.WriteTimeout, LlmTimeout, id="write-timeout"),
        pytest.param(httpx2.ReadError, LlmConnectionFailed, id="read-error"),
        pytest.param(httpx2.WriteError, LlmConnectionFailed, id="write-error"),
        pytest.param(httpx2.RemoteProtocolError, LlmConnectionFailed, id="remote-protocol-error"),
    ],
)
def test_a_failure_after_the_request_may_have_left_is_never_retried(
    clock, error, raised
) -> None:
    """The model may already have the request; repeating it is worse than ending the turn."""
    transport = _Transport(clock, _raises(error, cost=0.01))

    with pytest.raises(raised) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert type(caught.value.__cause__) is error
    assert len(transport.requests) == 1
    assert clock.sleeps == []


# --- Retry-After ------------------------------------------------------------

_UNREADABLE_RETRY_AFTER = [
    "soon",
    "Mon, 99 Foo 2026",
    "Mon, 32 Jan 2026 00:00:00 GMT",
    "Sat, 29 Feb 2025 00:00:00 GMT",
    "Fri, 31 Dec 99999 23:59:59 GMT",
    "12:00",
]


@pytest.mark.parametrize("header", _UNREADABLE_RETRY_AFTER)
def test_parsing_a_retry_after_header_the_server_garbled_never_raises(header) -> None:
    """The header is read on every error status, and the server controls it."""
    assert _parse_retry_after(header) is None


@pytest.mark.parametrize("header", ["soon", "Mon, 99 Foo 2026"])
@pytest.mark.parametrize("status", [503, 401])
def test_an_unreadable_retry_after_is_still_an_upstream_error_and_is_logged(
    clock, log_stream, status, header
) -> None:
    """Not a ``ValueError`` escaping the adapter with no log line behind it."""
    transport = _Transport(clock, _fails(status, cost=0.01, retry_after=header))

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=1).generate(_request())

    assert caught.value.status_code == status
    assert caught.value.retry_after_seconds is None
    final = _llm_records(log_stream)[-1]
    assert final["outcome"] == "failed"
    assert final["error_type"] == "LlmUpstreamError"
    assert final["http_status"] == status
    if status == 503:
        # Retryable: the unreadable hint is ignored and the backoff used instead.
        assert clock.sleeps == [0.05]
        assert len(transport.requests) == 2
    else:
        assert clock.sleeps == []
        assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "header", ["0", "0.0", "-5", "Wed, 21 Oct 2015 07:28:00 GMT", "nan", "inf", "1e400"]
)
def test_a_retry_after_that_is_not_a_future_delay_falls_back_to_backoff(clock, header) -> None:
    """Zero, negative, past or non-finite: never an immediate retry, never a stall."""
    transport = _Transport(clock, _fails(503, cost=0.01, retry_after=header))

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=1).generate(_request())

    assert caught.value.retry_after_seconds is None
    assert clock.sleeps == [0.05]
    assert len(transport.requests) == 2


def test_a_retry_after_that_fits_the_budget_is_waited_for_as_sent(clock) -> None:
    transport = _Transport(
        clock, _fails(429, cost=0.01, retry_after="0.2"), _answers("thodi der baad")
    )

    generation = _service(transport, max_retries=1).generate(_request())

    assert generation.text == "thodi der baad"
    assert clock.sleeps == [0.2]
    _assert_within_budget(clock)


def test_a_future_http_date_beyond_the_budget_is_not_waited_for(clock) -> None:
    later = format_datetime(datetime.now(timezone.utc) + timedelta(hours=1), usegmt=True)
    transport = _Transport(clock, _fails(503, cost=0.01, retry_after=later))

    with pytest.raises(LlmUpstreamError) as caught:
        _service(transport, max_retries=3).generate(_request())

    assert caught.value.retry_after_seconds > BUDGET
    assert clock.sleeps == []
    assert len(transport.requests) == 1


# --- the budget as a whole --------------------------------------------------

#: How long each attempt takes, as a fraction of the 1 s budget. Chosen to
#: land on both sides of every wait the adapter can take (50/100/200 ms backoff,
#: a 300 ms and a 1 ms ``Retry-After``) and to include attempts the request
#: timeout cuts off. None sits within float noise of a ``delay + 10 ms`` edge.
_COSTS = (0.0, 0.1, 0.25, 0.4, 0.695, 0.8, 0.945, 0.95, 0.99, 0.995, 1.0, 1.5)


@pytest.mark.parametrize(
    ("step", "raised"),
    [
        pytest.param(_fails(503), LlmUpstreamError, id="503-backoff"),
        pytest.param(_fails(429, retry_after="0.3"), LlmUpstreamError, id="429-retry-after-300ms"),
        pytest.param(_fails(503, retry_after="0.001"), LlmUpstreamError, id="503-retry-after-1ms"),
        pytest.param(_raises(httpx2.ConnectError), LlmConnectionFailed, id="connect-error"),
        pytest.param(_raises(httpx2.ConnectTimeout), LlmTimeout, id="connect-timeout"),
    ],
)
def test_no_schedule_of_failures_overdraws_the_budget_or_waits_for_nothing(
    monkeypatch, step, raised
) -> None:
    """Every attempt cost, every retry bound: three invariants hold.

    Attempts plus waits never exceed ``timeout_seconds``; every wait is
    followed by an attempt; and the error raised is the failure that actually
    happened last - never a cause-less timeout manufactured by the retry loop.
    """
    for cost in _COSTS:
        for max_retries in range(4):
            scenario = f"cost={cost} max_retries={max_retries}"
            clock = _install_clock(monkeypatch)
            transport = _Transport(clock, dataclasses.replace(step, cost=cost))

            with pytest.raises(LlmError) as caught:
                _service(transport, max_retries=max_retries).generate(_request())

            _assert_within_budget(clock)
            assert len(transport.requests) == len(clock.sleeps) + 1, scenario
            assert len(transport.requests) <= max_retries + 1, scenario

            error = caught.value
            if transport.cut_off[-1]:
                # The last attempt outlived its timeout: a read timeout is the truth.
                assert type(error) is LlmTimeout, scenario
                assert type(error.__cause__) is httpx2.ReadTimeout, scenario
            else:
                assert type(error) is raised, scenario
                if step.error is not None:
                    assert type(error.__cause__) is step.error, scenario
                else:
                    assert error.status_code == step.status, scenario


# --- what the orchestrator sees ---------------------------------------------


def test_a_503_near_the_deadline_ends_the_turn_as_an_upstream_error_not_a_timeout(
    clock,
) -> None:
    transport = _Transport(clock, _fails(503, cost=0.945))
    runtime = make_runtime(llm=http_llm(transport, timeout_seconds=BUDGET, max_retries=1))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert result.speakable is False
    assert TurnErrorCategory.LLM_UPSTREAM_ERROR in result.error_categories
    assert TurnErrorCategory.LLM_TIMEOUT not in result.error_categories
    assert clock.sleeps == []


def test_an_unreadable_retry_after_ends_the_turn_as_an_upstream_error_not_a_crash(
    clock,
) -> None:
    transport = _Transport(clock, _fails(503, cost=0.01, retry_after="Mon, 99 Foo 2026"))
    runtime = make_runtime(llm=http_llm(transport, timeout_seconds=BUDGET))
    session = open_session(runtime)

    result = ConversationOrchestrator(runtime).process_turn(session.session_id, say())

    assert result.outcome is TurnOutcome.FAILED
    assert TurnErrorCategory.LLM_UPSTREAM_ERROR in result.error_categories
    assert TurnErrorCategory.LLM_FAILED not in result.error_categories
