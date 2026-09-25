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
shape, an unreadable tool call and a model that said nothing all become typed,
categorised exceptions carrying no upstream text. The response body is read for
parsing and then dropped; it is never logged, never returned and never spoken.

Retries are bounded and off by default. On a live call a retry spends the
customer's silence, so the default is to fail fast and let the orchestrator end
the turn. When enabled, only clearly transient failures are retried - a
connection that never opened, and the statuses a server uses to say "later"
(408, 429, 500, 502, 503, 504). A 400 or a 401 is a bug or a credential problem
and repeating it fixes neither. A read timeout is not retried: the budget has
already been spent waiting.

The configured timeout is one budget for the whole call, shared by every
attempt and by the wait between them. It is not a fresh budget per attempt.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx2

from app.observability import get_logger, log_event
from app.services.llm import (
    LlmConnectionFailed,
    LlmEmptyResponse,
    LlmGeneration,
    LlmInvalidToolCall,
    LlmMalformedResponse,
    LlmRequest,
    LlmTimeout,
    LlmToolCall,
    LlmToolSpec,
    LlmUpstreamError,
    LlmUsage,
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
    parsed = parsedate_to_datetime(text)
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
        self._client = client or httpx2.Client(timeout=self._timeout)

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
        while True:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining < _MIN_ATTEMPT_SECONDS:
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
                generation = self._attempt(url, payload, started, remaining)
            except (LlmTimeout, LlmConnectionFailed, LlmUpstreamError) as exc:
                if self._retryable(exc) and attempt <= self._max_retries:
                    delay = self._retry_delay(exc, attempt)
                    if delay < deadline - time.monotonic():
                        self._log(
                            correlation_id,
                            attempt=attempt,
                            outcome="retrying",
                            error_type=type(exc).__name__,
                            http_status=getattr(exc, "status_code", None),
                            retry_in_ms=round(delay * 1000.0, 3),
                            latency_ms=(time.perf_counter() - started) * 1000.0,
                        )
                        time.sleep(delay)
                        continue
                self._log(
                    correlation_id,
                    attempt=attempt,
                    outcome="failed",
                    error_type=type(exc).__name__,
                    error_category=exc.category,
                    http_status=getattr(exc, "status_code", None),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
                raise
            except (LlmMalformedResponse, LlmEmptyResponse, LlmInvalidToolCall) as exc:
                # A well-formed HTTP exchange whose body we could not use. Never
                # retried: the same request would produce the same body, and a
                # second sampling pass is not a fix for a broken serialiser.
                self._log(
                    correlation_id,
                    attempt=attempt,
                    outcome="failed",
                    error_type=type(exc).__name__,
                    error_category=exc.category,
                    http_status=200,
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
        self, url: str, payload: dict[str, Any], started: float, remaining: float
    ) -> LlmGeneration:
        # Passed on the request, not taken from the client. An injected client
        # may have been built with no timeout at all; this call still stops.
        timeout = httpx2.Timeout(
            min(self._timeout_seconds, remaining),
            connect=min(self._connect_timeout_seconds, remaining),
        )
        try:
            response = self._client.post(
                url, json=payload, headers=self._headers, timeout=timeout
            )
        except httpx2.TimeoutException as exc:
            raise LlmTimeout("model server did not respond within the configured timeout") from exc
        except httpx2.RequestError as exc:
            # `str(exc)` can contain the URL, so it is deliberately not propagated.
            # Retry policy is decided separately and is narrower than this mapping.
            raise LlmConnectionFailed("model server could not be reached") from exc

        if response.status_code >= 400:
            raise LlmUpstreamError(
                response.status_code,
                retry_after_seconds=_parse_retry_after(response.headers.get("retry-after")),
            )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            raise LlmMalformedResponse("model server returned a body that is not JSON") from exc

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


def _render_message(message: Any) -> dict[str, Any]:
    """An :class:`LlmMessage` as an OpenAI message object."""
    rendered: dict[str, Any] = {"role": message.role, "content": message.content}
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

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise LlmMalformedResponse("response message content is not a string")

    tool_calls = _parse_tool_calls(message.get("tool_calls"))

    text = content if content else None
    if text is not None and not text.strip():
        text = None
    if text is None and not tool_calls:
        raise LlmEmptyResponse("model returned neither text nor a tool call")

    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = None

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
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        raise LlmInvalidToolCall("tool call arguments are neither an object nor a JSON string")
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LlmInvalidToolCall("tool call arguments are not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise LlmInvalidToolCall("tool call arguments did not decode to an object")
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
