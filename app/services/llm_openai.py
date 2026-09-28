"""OpenAI-compatible chat-completions transport.

The only module in the application that opens a socket to a model, and the only
one that knows the model speaks HTTP. It implements
:class:`~app.services.llm.LlmService` by translating an
:class:`~app.services.llm.LlmRequest` into an OpenAI-compatible
``POST {base_url}/chat/completions`` body and translating the reply back into an
:class:`~app.services.llm.LlmGeneration`.

What this module is not
-----------------------
It translates; it does not decide. It contains no prompt text - prompts are
built by :mod:`app.orchestrator.prompt` - no policy, no grounding, no
conversation state and, above all, no tool execution. A tool call arrives here as
JSON and leaves as an :class:`~app.services.llm.LlmToolCall`; the registry runs
it later, after the orchestrator has bound the session's identity arguments over
whatever the model asked for. A model that names ``account_ref`` in its
arguments has named a value that will be overwritten before it reaches a backend.

Provider neutrality
-------------------
Nothing here is specific to Gemma, to vLLM or to any vendor. The model
identifier is configuration. The dialect is the OpenAI chat-completions API,
which vLLM, llama.cpp, TGI, Ollama and OpenAI itself all serve, so the deployment
choice stays in the environment rather than in the code.

Failure handling
----------------
Every failure mode is mapped to a class in :mod:`app.services.llm` and nothing
else escapes: a transport error, a 4xx, a 5xx, bad JSON, a body of the wrong
shape, an unreadable tool call, a model that said nothing and a model that was
cut off all become typed, categorised exceptions carrying no upstream text. The
response body is read for parsing and then dropped; it is never logged, never
returned and never spoken.

A raised error carries nothing of the exchange that failed. The transport
exception is not chained to it - an httpx2 exception holds the request it failed
on, ``Authorization`` header included - and neither is a JSON error, which holds
the whole document it could not parse. Only the transport exception's *type*
survives, as an empty stand-in, because retry policy depends on it.

Retries are bounded and off by default. On a live call a retry spends the
customer's silence, so the default is to fail fast and let the orchestrator end
the turn. When enabled, only clearly transient failures are retried - a
connection that never opened, and the statuses a server uses to say "later"
(408, 429, 500, 502, 503, 504). A 400 or a 401 is a bug or a credential problem
and repeating it fixes neither. A read timeout is not retried: the budget has
already been spent waiting.

The configured timeout is one wall-clock budget for the whole call, shared by
every attempt and by the wait between them. It is not a fresh budget per
attempt, and it is not a budget per HTTP phase: see :class:`_DeadlineBackend`.
A wait is taken only when an attempt can still follow it; otherwise the failure
that actually happened is what the caller gets.
"""

from __future__ import annotations

import contextvars
import json
import math
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, NoReturn
from urllib.parse import urlsplit

import httpcore2
import httpx2

from app.observability import get_logger, log_event
from app.services.llm import (
    LlmConfigurationError,
    LlmConnectionFailed,
    LlmEmptyResponse,
    LlmError,
    LlmGeneration,
    LlmIncompleteResponse,
    LlmInvalidToolCall,
    LlmMalformedResponse,
    LlmMessage,
    LlmRequest,
    LlmTimeout,
    LlmToolCall,
    LlmToolSpec,
    LlmUpstreamError,
    LlmUsage,
    contains_non_finite_number,
)

_logger = get_logger(__name__)

#: Statuses a server uses to mean "not now". Everything else - notably every
#: other 4xx - is a request or credential defect that a retry cannot fix.
_RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

#: Matches ``Settings.model_max_retries``. A live call must not be allowed to
#: retry without a bound the configuration layer also enforces.
_MAX_RETRIES = 3

#: Floor under which a further attempt cannot be issued. Also the shortest wait
#: before a retry, so a failure is never followed by another request in the same
#: instant.
_MIN_ATTEMPT_SECONDS = 0.01
_BACKOFF_BASE_SECONDS = 0.05
_BACKOFF_CAP_SECONDS = 1.0

#: The chat-completions path, appended to the configured base URL. The base URL
#: is expected to carry the API version (``.../v1``), as every OpenAI-compatible
#: server documents it.
_CHAT_COMPLETIONS_PATH = "/chat/completions"

#: ``finish_reason`` values that mean the model did not finish: the output-token
#: cap cut it off, or a filter withheld part of it. Mapped to the adapter's own
#: strings so the error records a constant, never the server's text. Any other
#: value - ``stop``, ``tool_calls``, or a server's own word for a normal stop
#: such as TGI's ``eos_token`` - is a completed generation.
_INCOMPLETE_FINISH_REASONS: dict[str, str] = {
    "length": "length",
    "content_filter": "content_filter",
}

#: Largest slice handed to one socket write. httpcore2 sends a buffer in a loop
#: under a single timeout, so a peer that drains slowly could otherwise stretch
#: one write past the deadline; each slice re-reads the time left.
_WRITE_SLICE_BYTES = 16 * 1024


def _raise(error: LlmError, cause: BaseException | None = None) -> NoReturn:
    """Raise ``error`` with nothing of the failed exchange attached to it.

    ``raise ... from exc`` keeps ``exc`` as ``__cause__``, and raising inside an
    ``except`` block keeps it as ``__context__`` even with ``from None``. Either
    way it would travel with the error to every caller. An httpx2 exception
    holds the request it failed on, headers and body included, and so the API
    key and the customer's words; a JSON error holds the whole document it
    could not parse. ``cause``, when given, is a stand-in built by
    :func:`_scrubbed`. The same pattern as :func:`app.services.decision_jev._raise`.
    """
    try:
        raise error from cause
    finally:
        error.__context__ = None


def _scrubbed(exc: httpx2.RequestError) -> httpx2.RequestError:
    """The transport exception's type, and nothing else it carried.

    :meth:`OpenAiCompatibleLlmService._retryable` decides by what *kind* of
    transport failure happened, so the type is kept as the cause. The instance
    is not: it holds the request, and its message can hold the URL.
    """
    try:
        return type(exc)("")
    except Exception:  # noqa: BLE001 - an unexpected constructor; the type is only a hint
        return httpx2.RequestError("")


# -- the wall-clock deadline, enforced at the socket ------------------------

#: When the model call in progress in this context must end, on the monotonic
#: clock. Set by :meth:`OpenAiCompatibleLlmService._attempt` around the one
#: request it sends, and read before every socket operation. A ContextVar, not
#: an attribute, because one client is shared by concurrent calls, each on its
#: own worker thread with its own copied context.
_CALL_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "llm_call_deadline", default=None
)


def _time_left(timeout: float | None, expired: type[Exception]) -> float | None:
    """``timeout``, shortened to the time the current call has left.

    Outside a model call the timeout passes through untouched. Once the
    deadline has passed this raises the phase's own httpcore2 timeout rather
    than returning zero or less: ``settimeout(0)`` would switch the socket to
    non-blocking mode and surface as a connection error - which the adapter
    retries - and a negative value raises an error httpx2 does not map.
    """
    deadline = _CALL_DEADLINE.get()
    if deadline is None:
        return timeout
    left = deadline - time.monotonic()
    if left <= 0:
        raise expired("the model call's deadline has passed")
    return left if timeout is None else min(timeout, left)


class _DeadlineStream(httpcore2.NetworkStream):
    """A connection whose every read and write stops at the call's deadline.

    httpx2's timeouts are per phase and, within a phase, per socket operation:
    the read timeout restarts with every byte that arrives. A server that
    dribbles its response - or a connect that is slow and a read that is slow -
    could otherwise hold one attempt for several times the configured budget.
    """

    def __init__(self, inner: httpcore2.NetworkStream) -> None:
        self._inner = inner

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._inner.read(max_bytes, _time_left(timeout, httpcore2.ReadTimeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        view = memoryview(buffer)
        while view:
            piece = view[:_WRITE_SLICE_BYTES]
            self._inner.write(bytes(piece), _time_left(timeout, httpcore2.WriteTimeout))
            view = view[len(piece):]

    def close(self) -> None:
        self._inner.close()

    def start_tls(
        self,
        ssl_context: Any,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore2.NetworkStream:
        return _DeadlineStream(
            self._inner.start_tls(
                ssl_context, server_hostname, _time_left(timeout, httpcore2.ConnectTimeout)
            )
        )

    def get_extra_info(self, info: str) -> Any:
        # Delegated as-is: the pool asks "is_readable" to expire idle connections.
        return self._inner.get_extra_info(info)


class _DeadlineBackend(httpcore2.NetworkBackend):
    """Opens connections that are bounded by the call's deadline.

    Every connect, TLS handshake, read and write is given the smaller of its
    own httpx2 timeout and the time the call has left, so the attempts and the
    waits between them together stop at the configured budget. Inert outside a
    model call: with no deadline set, every timeout passes through unchanged.

    Not bounded: DNS resolution, which happens inside ``connect_tcp`` before
    any timeout applies. A connection that opens after the deadline because
    the lookup was slow is closed at once and reported as a connect timeout.
    """

    def __init__(self, inner: httpcore2.NetworkBackend) -> None:
        self._inner = inner

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore2.NetworkStream:
        stream = self._inner.connect_tcp(
            host, port, _time_left(timeout, httpcore2.ConnectTimeout), local_address, socket_options
        )
        return self._still_in_time(stream)

    def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None
    ) -> httpcore2.NetworkStream:
        stream = self._inner.connect_unix_socket(
            path, _time_left(timeout, httpcore2.ConnectTimeout), socket_options
        )
        return self._still_in_time(stream)

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)

    @staticmethod
    def _still_in_time(stream: httpcore2.NetworkStream) -> httpcore2.NetworkStream:
        try:
            _time_left(None, httpcore2.ConnectTimeout)
        except httpcore2.ConnectTimeout:
            stream.close()
            raise
        return _DeadlineStream(stream)


def _enforce_deadline(client: httpx2.Client) -> None:
    """Route every connection ``client`` opens through :class:`_DeadlineBackend`.

    httpx2 offers no public hook for a network backend, and passing a custom
    transport instead would silently drop the environment's proxy settings
    (httpx2 honours them only when no transport is given). So the backend is
    swapped on each connection pool the client already built - its default
    transport and any proxy mounts - through attributes private to httpx2 and
    httpcore2 2.13.1, which requirements.txt pins.

    Refuses rather than degrading: if those attributes are not where that
    version keeps them, the adapter cannot promise its deadline, and the
    process fails to start instead of running without one.
    """
    mounts = getattr(client, "_mounts", None)
    transports = [
        getattr(client, "_transport", None),
        *(mounts.values() if isinstance(mounts, dict) else ()),
    ]
    covered = 0
    for transport in transports:
        if transport is None:  # a mount of None routes a pattern to no proxy
            continue
        pool = getattr(transport, "_pool", None)
        backend = getattr(pool, "_network_backend", None)
        if not isinstance(pool, httpcore2.ConnectionPool) or not isinstance(
            backend, httpcore2.NetworkBackend
        ):
            raise LlmConfigurationError(
                "The installed httpx2/httpcore2 do not expose the connection pool this adapter "
                "needs to enforce MODEL_TIMEOUT_SECONDS. Install the versions pinned in "
                "requirements.txt."
            )
        if not isinstance(backend, _DeadlineBackend):
            pool._network_backend = _DeadlineBackend(backend)  # noqa: SLF001
        covered += 1
    if covered == 0:
        raise LlmConfigurationError(
            "The HTTP client has no connection pool to enforce MODEL_TIMEOUT_SECONDS on."
        )


def _require_positive_seconds(name: str, value: float) -> float:
    """Reject a timeout that is missing, non-numeric, non-finite or not positive."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _parse_retry_after(raw: str | None) -> float | None:
    """Seconds to wait, from a ``Retry-After`` header. ``None`` if unusable.

    Accepts a delay in seconds or an HTTP-date. Zero, a past date, and anything
    that is not a delay are ignored so the caller falls back to its own backoff
    instead of retrying immediately.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = None
    else:
        if math.isfinite(seconds) and seconds > 0:
            return seconds
        return None
    # A header the server controls must not be able to raise out of the
    # adapter: depending on the Python version, an unreadable date raises
    # rather than returning None, and the header is read on every error status.
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError, IndexError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = (parsed - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(delta) or delta <= 0:
        return None
    return delta


def _endpoint_label(base_url: str) -> str:
    """``scheme://host:port`` for logs, with any embedded credentials dropped.

    A base URL may legitimately carry userinfo (``https://user:pass@host``).
    Logging the URL verbatim would publish that password, so only the parts an
    operator needs to identify the endpoint survive.
    """
    parts = urlsplit(base_url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}" if host else parts.scheme


class OpenAiCompatibleLlmService:
    """An :class:`~app.services.llm.LlmService` backed by an HTTP model server.

    Construct it with an explicit configuration, or from :class:`Settings` via
    :func:`build_llm_service`. Constructing it opens a connection pool, so it is
    built once per process and only when a model is actually configured.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        connect_timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int = 0,
        provider: str = "openai-compatible",
        client: httpx2.Client | None = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required")
        if not model or not model.strip():
            raise ValueError("model is required")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be an integer")
        if not 0 <= max_retries <= _MAX_RETRIES:
            raise ValueError(f"max_retries must be between 0 and {_MAX_RETRIES}")
        timeout_seconds = _require_positive_seconds("timeout_seconds", timeout_seconds)
        if connect_timeout_seconds is not None:
            connect_timeout_seconds = _require_positive_seconds(
                "connect_timeout_seconds", connect_timeout_seconds
            )

        self._base_url = base_url.strip().rstrip("/")
        self._model = model.strip()
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._connect_timeout_seconds = (
            connect_timeout_seconds if connect_timeout_seconds is not None else timeout_seconds
        )
        self._provider = provider
        self._endpoint_label = _endpoint_label(self._base_url)

        # The key is held for the Authorization header and for nothing else. It
        # is never put on the instance's repr, in a log field or in an exception.
        self._headers = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            self._headers["Authorization"] = f"Bearer {api_key.strip()}"

        self._timeout = httpx2.Timeout(
            self._timeout_seconds,
            connect=self._connect_timeout_seconds,
        )
        self._owns_client = client is None
        if client is None:
            client = httpx2.Client(timeout=self._timeout)
            try:
                _enforce_deadline(client)
            except BaseException:
                client.close()
                raise
        # An injected client is used as given and is not altered: its requests
        # still carry the per-attempt timeout below, but the wall-clock bound
        # inside an attempt holds only for the client the adapter built.
        self._client = client

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the connection pool. A no-op for an injected client."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenAiCompatibleLlmService":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"{type(self).__name__}(endpoint={self._endpoint_label!r}, model={self._model!r})"
        )

    # -- LlmService ---------------------------------------------------------

    def generate(self, request: LlmRequest) -> LlmGeneration:
        """One model call. Raises an :class:`~app.services.llm.LlmError` subclass on any failure."""
        payload = self._build_payload(request)
        correlation_id = uuid.uuid4().hex
        url = f"{self._base_url}{_CHAT_COMPLETIONS_PATH}"

        started = time.perf_counter()
        # One clock for the whole call. Attempts and the waits between them
        # spend the same budget; a retry does not receive a fresh timeout.
        deadline = time.monotonic() + self._timeout_seconds
        attempt = 0
        # The failure a wait was taken after. Kept because the wait can overrun
        # (a sleep is a lower bound, not an exact duration): if no attempt fits
        # afterwards, this is what happened, and it is what the caller gets.
        # A fresh timeout would misreport a 503 as a slow server.
        last_failure: LlmError | None = None
        while True:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining < _MIN_ATTEMPT_SECONDS:
                if last_failure is not None:
                    self._log_failure(correlation_id, attempt - 1, last_failure, started)
                    raise last_failure
                self._log(
                    correlation_id,
                    attempt=attempt,
                    outcome="failed",
                    error_type=LlmTimeout.__name__,
                    error_category=LlmTimeout.category,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
                raise LlmTimeout("model server did not respond within the configured timeout")
            try:
                generation = self._attempt(url, payload, started, deadline)
            except (LlmTimeout, LlmConnectionFailed, LlmUpstreamError) as exc:
                if self._retryable(exc) and attempt <= self._max_retries:
                    delay = self._retry_delay(exc, attempt)
                    # Wait only if a real attempt can still follow the wait.
                    # The complement of the refusal above: a wait that left
                    # less than _MIN_ATTEMPT_SECONDS would be spent for nothing.
                    if delay + _MIN_ATTEMPT_SECONDS <= deadline - time.monotonic():
                        self._log(
                            correlation_id,
                            attempt=attempt,
                            outcome="retrying",
                            error_type=type(exc).__name__,
                            http_status=getattr(exc, "status_code", None),
                            retry_in_ms=round(delay * 1000.0, 3),
                            latency_ms=(time.perf_counter() - started) * 1000.0,
                        )
                        last_failure = exc
                        time.sleep(delay)
                        continue
                self._log_failure(correlation_id, attempt, exc, started)
                raise
            except (
                LlmMalformedResponse,
                LlmEmptyResponse,
                LlmInvalidToolCall,
                LlmIncompleteResponse,
            ) as exc:
                # A well-formed HTTP exchange whose body we could not use. Never
                # retried: the same request would produce the same body, and a
                # second sampling pass is not a fix for a broken serialiser or
                # for an output cap that cuts the answer off.
                self._log(
                    correlation_id,
                    attempt=attempt,
                    outcome="failed",
                    error_type=type(exc).__name__,
                    error_category=exc.category,
                    http_status=200,
                    finish_reason=getattr(exc, "reason", None),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
                raise

            self._log(
                correlation_id,
                attempt=attempt,
                outcome="ok",
                http_status=200,
                latency_ms=generation.latency_ms,
                finish_reason=generation.finish_reason,
                tool_call_count=len(generation.tool_calls),
                usage_prompt=generation.usage.prompt_tokens if generation.usage else None,
                usage_completion=generation.usage.completion_tokens if generation.usage else None,
                usage_total=generation.usage.total_tokens if generation.usage else None,
            )
            return generation

    # -- one HTTP exchange ---------------------------------------------------

    def _attempt(
        self, url: str, payload: dict[str, Any], started: float, deadline: float
    ) -> LlmGeneration:
        remaining = deadline - time.monotonic()
        # Passed on the request, not taken from the client. An injected client
        # may have been built with no timeout at all; this call still stops.
        timeout = httpx2.Timeout(
            min(self._timeout_seconds, remaining),
            connect=min(self._connect_timeout_seconds, remaining),
        )
        # Every socket operation inside the request reads this, through
        # _DeadlineBackend, so the phases cannot add up past the deadline.
        token = _CALL_DEADLINE.set(deadline)
        try:
            response = self._client.post(
                url, json=payload, headers=self._headers, timeout=timeout
            )
        except httpx2.TimeoutException as exc:
            _raise(
                LlmTimeout("model server did not respond within the configured timeout"),
                _scrubbed(exc),
            )
        except httpx2.RequestError as exc:
            # `str(exc)` can contain the URL, so it is deliberately not propagated.
            # Retry policy is decided separately and is narrower than this mapping.
            _raise(LlmConnectionFailed("model server could not be reached"), _scrubbed(exc))
        finally:
            _CALL_DEADLINE.reset(token)

        if response.status_code >= 400:
            raise LlmUpstreamError(
                response.status_code,
                retry_after_seconds=_parse_retry_after(response.headers.get("retry-after")),
            )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError, RecursionError):
            # The decode error holds the whole body it could not parse.
            _raise(LlmMalformedResponse("model server returned a body that is not JSON"))

        generation = _parse_completion(body, fallback_model=self._model)
        return generation.model_copy(
            update={"latency_ms": (time.perf_counter() - started) * 1000.0}
        )

    def _retryable(self, exc: Exception) -> bool:
        """Only failures that happened before the request was sent.

        A read timeout, a write timeout, a mid-body network error and a protocol
        error are not retried. The request may already have reached the model,
        and spending the rest of the turn repeating it is worse than ending it.
        """
        if isinstance(exc, LlmUpstreamError):
            return exc.status_code in _RETRYABLE_STATUSES
        cause = exc.__cause__
        if isinstance(exc, LlmConnectionFailed):
            return isinstance(cause, (httpx2.ConnectError, httpx2.ProxyError))
        if isinstance(exc, LlmTimeout):
            return isinstance(cause, (httpx2.ConnectTimeout, httpx2.PoolTimeout))
        return False

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """How long to wait before the next attempt. Never zero.

        ``Retry-After`` wins when the server sent a positive delay. Otherwise
        the wait grows with the attempt and is capped, so a retry storm cannot
        pause a live call for longer than a second between tries.
        """
        hinted = getattr(exc, "retry_after_seconds", None)
        if isinstance(hinted, (int, float)) and not isinstance(hinted, bool):
            if math.isfinite(hinted) and hinted > 0:
                return float(hinted)
        delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        return min(max(delay, _MIN_ATTEMPT_SECONDS), _BACKOFF_CAP_SECONDS)

    # -- request translation -------------------------------------------------

    def _build_payload(self, request: LlmRequest) -> dict[str, Any]:
        """Render an :class:`LlmRequest` as an OpenAI chat-completions body.

        ``max_output_tokens`` and ``temperature`` come from configuration unless
        the caller set them explicitly on the request; pydantic's
        ``model_fields_set`` is what distinguishes "the caller chose 0.2" from
        "0.2 is the field default". The request wins when it spoke.
        """
        explicit = request.model_fields_set
        max_tokens = (
            request.max_output_tokens
            if "max_output_tokens" in explicit or self._max_output_tokens is None
            else self._max_output_tokens
        )
        temperature = (
            request.temperature
            if "temperature" in explicit or self._temperature is None
            else self._temperature
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_render_message(m) for m in request.messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if request.tools:
            payload["tools"] = [_render_tool(t) for t in request.tools]
        return payload

    # -- logging -------------------------------------------------------------

    def _log_failure(
        self, correlation_id: str, attempt: int, exc: LlmError, started: float
    ) -> None:
        """The final record of a call that failed at the transport or with a status."""
        self._log(
            correlation_id,
            attempt=attempt,
            outcome="failed",
            error_type=type(exc).__name__,
            error_category=exc.category,
            http_status=getattr(exc, "status_code", None),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _log(self, correlation_id: str, **fields: Any) -> None:
        """One structured record per attempt: categories and measurements only.

        Deliberately absent: the prompt, the draft, the tool arguments, the
        response body, the URL path, every header and the API key. What is here
        is what an operator needs to see a model endpoint degrading.

        ``usage_*`` rather than ``*_tokens`` is not cosmetic:
        :func:`~app.observability.redact` blanks any field whose name contains
        ``token``, which is correct for credentials and would otherwise blank
        these counts too.
        """
        latency = fields.pop("latency_ms", None)
        log_event(
            _logger,
            "llm_call",
            provider=self._provider,
            model=self._model,
            endpoint=self._endpoint_label,
            request_id=correlation_id,
            model_latency_ms=round(latency, 3) if latency is not None else None,
            **fields,
        )


# -- translation helpers ----------------------------------------------------


def _render_message(message: LlmMessage) -> dict[str, Any]:
    """An :class:`LlmMessage` as an OpenAI message object.

    An assistant turn that requested tools is sent with those requests, in the
    wire shape the server produced them in, so each ``tool`` message that
    follows answers a call on record. Such a turn with no words carries
    ``"content": null``, which is what the contract specifies, rather than an
    empty string some servers reject. A message with no tool calls is sent
    exactly as before - no empty ``tool_calls`` array, which OpenAI rejects.
    """
    rendered: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        rendered["content"] = message.content or None
        rendered["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.tool_name,
                    # A JSON string, as the model sent it. allow_nan=False: a
                    # non-finite number is not JSON and is never put on the wire.
                    "arguments": json.dumps(
                        call.arguments, ensure_ascii=False, allow_nan=False, default=str
                    ),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        rendered["tool_call_id"] = message.tool_call_id
    return rendered


def _render_tool(spec: LlmToolSpec) -> dict[str, Any]:
    """An :class:`LlmToolSpec` as an OpenAI function-tool declaration.

    ``parameters`` is passed through exactly as the registry published it - it is
    the tool's own pydantic JSON Schema, and rewriting it here would let the
    model be offered a looser contract than the registry will enforce.
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _parse_completion(body: Any, *, fallback_model: str) -> LlmGeneration:
    """Translate an OpenAI-compatible response body into an :class:`LlmGeneration`.

    Strict by design. Every shape this does not recognise raises rather than
    being coerced into something plausible: a body the application had to guess
    at is a body it cannot claim the model produced.
    """
    if not isinstance(body, dict):
        raise LlmMalformedResponse("response body is not a JSON object")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmMalformedResponse("response contains no choices")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise LlmMalformedResponse("response choice is not an object")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise LlmMalformedResponse("response choice contains no message object")

    # Whether the model finished is read before what it said, so that a
    # cut-off tool call or a cut-off silence is reported as cut off rather than
    # as an unreadable tool call or an empty reply. Absent is accepted - not
    # every compatible server sends it - but a value of the wrong type is not.
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise LlmMalformedResponse("response finish_reason is not a string")
    incomplete = _INCOMPLETE_FINISH_REASONS.get(finish_reason) if finish_reason else None
    if incomplete is not None:
        raise LlmIncompleteResponse(incomplete)

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise LlmMalformedResponse("response message content is not a string")

    tool_calls = _parse_tool_calls(message.get("tool_calls"))

    text = content if content else None
    if text is not None and not text.strip():
        text = None
    if text is None and not tool_calls:
        raise LlmEmptyResponse("model returned neither text nor a tool call")

    reported_model = body.get("model")
    model = reported_model if isinstance(reported_model, str) and reported_model else fallback_model

    return LlmGeneration(
        text=text,
        tool_calls=tool_calls,
        model=model,
        finish_reason=finish_reason,
        usage=_parse_usage(body.get("usage")),
    )


def _parse_tool_calls(raw: Any) -> tuple[LlmToolCall, ...]:
    """Read the ``tool_calls`` array. Absent or empty is normal, not an error."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise LlmInvalidToolCall("tool_calls is not an array")

    calls: list[LlmToolCall] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise LlmInvalidToolCall("a tool call is not an object")
        function = entry.get("function")
        if not isinstance(function, dict):
            raise LlmInvalidToolCall("a tool call carries no function object")

        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise LlmInvalidToolCall("a tool call carries no tool name")

        # An absent id is tolerated: the orchestrator mints a request id when the
        # model does not supply one. An id of the wrong type is not tolerated,
        # because it would be carried into the audit record.
        call_id = entry.get("id")
        if call_id is None:
            call_id = ""
        elif not isinstance(call_id, str):
            raise LlmInvalidToolCall("a tool call id is not a string")

        calls.append(
            LlmToolCall(
                call_id=call_id,
                tool_name=name.strip(),
                arguments=_parse_arguments(function.get("arguments")),
            )
        )
    return tuple(calls)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Read a tool call's arguments.

    OpenAI serialises them as a JSON *string*; some compatible servers emit the
    object directly. Both are accepted because both are common in the wild, and
    neither is a guess about what the model meant. Anything else - a list, a
    number, a string that is not JSON, a JSON value that is not an object -
    raises: the registry expects a mapping, and inventing one here would put
    arguments no model asked for in front of a banking backend.

    So does a number JSON cannot represent. Python's parser reads ``NaN``,
    ``Infinity``, ``-Infinity`` and an overflowing ``1e400`` as floats, on both
    paths, and nothing here converts one into something else.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        parsed: Any = dict(raw)
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError, RecursionError):
            # Not chained: the decode error holds the raw arguments string.
            _raise(LlmInvalidToolCall("tool call arguments are not valid JSON"))
        if not isinstance(parsed, dict):
            raise LlmInvalidToolCall("tool call arguments did not decode to an object")
    else:
        raise LlmInvalidToolCall("tool call arguments are neither an object nor a JSON string")
    if contains_non_finite_number(parsed):
        raise LlmInvalidToolCall("tool call arguments contain a number JSON cannot represent")
    return parsed


def _parse_usage(raw: Any) -> LlmUsage | None:
    """Read the optional ``usage`` block. A malformed one is dropped, not fatal.

    Token accounting is observability. Losing it must not cost a customer their
    turn, so unreadable counts are simply absent.
    """
    if not isinstance(raw, dict):
        return None

    def _count(key: str) -> int | None:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    prompt, completion, total = _count("prompt_tokens"), _count("completion_tokens"), _count("total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    return LlmUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
