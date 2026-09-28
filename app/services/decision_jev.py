"""Jev (TypeSafe) adapter for :class:`app.services.decision.DecisionService`.

The only module in the application that imports ``typesafe_sdk``. It is itself
imported only when ``DECISION_PROVIDER=jev`` and ``JEV_ENABLED=true`` (see
:func:`app.runtime.build_decision_service`), so a disabled process never loads
the SDK and never constructs an HTTP client.

Contract used
-------------
``typesafe-sdk==0.7.1``, the official Python SDK, and its documented
``AsyncTypeSafeClient.system_one(state, questions)`` call against
``POST {base_url}/v1/systemone``. Every registered decision is put to Jev as one
``Choice`` question; the answer's ``choice``, ``confidence`` and
``probabilities`` are read and nothing else.

What this adapter adds on top of the SDK
----------------------------------------
* **A hard deadline.** The SDK's ``timeout`` is per HTTP phase and its default
  retry policy retries for up to 30 seconds. Here the whole call, retries
  included, is bounded by ``JEV_TIMEOUT_SECONDS``, and retries are off unless
  configured.
* **Answer validation the SDK does not do.** The SDK checks the response's
  *shape*. It does not check that the chosen label is one that was offered,
  that a confidence is finite and within [0, 1], or that the question was
  answered at all. A NaN confidence or a label nobody asked for passes the SDK;
  here it is a malformed response.
* **Error classification without text.** SDK exceptions carry the provider's
  response body, and their ``str`` embeds the provider's error message, which
  can echo the request - that is, the customer's words. They are mapped to
  :class:`~app.services.decision.DecisionError` categories by type and status
  code, and the SDK exception is dropped from the chain.
* **No SDK log output at all.** At DEBUG the SDK logs full request and
  response bodies, ``TYPESAFE_LOG_LEVEL`` can switch that on at import, and its
  one WARNING line quotes provider-controlled text. Every SDK record is dropped
  by a filter installed when this adapter is built.
* **Narrow retries, off by default.** When enabled, a failed connect (nothing
  was sent) and 408/429/500/502/503/504 (the server received the request and
  reported a transient failure, so a retry sends the customer's words again)
  are retried. A read, write or protocol error is not: whether the request was
  delivered is unknown, and a timeout has already spent the budget.
* **No ambient configuration.** Key, model and base URL are always passed
  explicitly, so the SDK's own ``TYPESAFE_*`` environment variables are never
  consulted.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from typing import NoReturn

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

from app.services.decision import (
    DecisionAnswer,
    DecisionAuthenticationFailed,
    DecisionConfigurationError,
    DecisionConnectionFailed,
    DecisionError,
    DecisionMalformedResponse,
    DecisionRateLimited,
    DecisionRequest,
    DecisionRequestRejected,
    DecisionTimeout,
    DecisionUpstreamError,
)
from app.services.decisions.registry import DecisionSpec, build_state, get_spec

PROVIDER = "jev"

#: TypeSafe's documented API root. Used when no base URL is configured, and
#: passed explicitly so ``TYPESAFE_BASE_URL`` is never read.
DEFAULT_BASE_URL = "https://api.typesafe.ai"

#: The SDK's logger. Silenced entirely: at INFO it logs request URLs, at DEBUG
#: the request and response bodies, which carry the customer's words, and at
#: WARNING provider-controlled text. See :class:`_DropEverything`.
SDK_LOGGER = "typesafe_sdk"

#: Statuses retried when ``max_retries`` is raised above zero. Narrower than the
#: SDK default (408, 429 and every 5xx): 501 and 505 are not transient.
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

#: What a reported model name may look like before it is recorded. The field is
#: provider-controlled text that reaches the logs: a short identifier with at
#: least one letter, so a bare number (a phone or account number echoed back) is
#: not recorded. Applied with ``fullmatch``, so a trailing newline fails too.
_MODEL_NAME = re.compile(r"(?=[^A-Za-z]*[A-Za-z])[A-Za-z0-9._:/~-]{1,80}")


def _raise(error: DecisionError) -> NoReturn:
    """Raise ``error`` with no chained SDK exception - and so no response body."""
    try:
        raise error from None
    finally:
        error.__context__ = None


class _DropEverything(logging.Filter):
    """Discards every record the SDK emits.

    Holding the logger at WARNING is not enough: the SDK's one WARNING call
    ("Ignoring answer %r with unrecognized type %r") formats two strings taken
    verbatim from the response body into the message, where
    :func:`app.observability.redact` cannot see them. This adapter reports every
    SDK failure itself, as a category, so the SDK's own lines add nothing. A
    filter, unlike a level, survives ``configure_logging`` being run again.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return False


def _silence_sdk_logging() -> None:
    logger = logging.getLogger(SDK_LOGGER)
    if not any(isinstance(existing, _DropEverything) for existing in logger.filters):
        logger.addFilter(_DropEverything())
    if logger.level < logging.WARNING:
        logger.setLevel(logging.WARNING)


def _connect_failed(error: BaseException) -> bool:
    """True only for a connection that was never established.

    The SDK wraps every non-timeout transport error as
    ``TypeSafeAPIConnectionError`` and keeps the original type as the cause.
    A read, write or protocol error can happen after the request - and the
    customer's words - reached the provider, so only ``ConnectError`` is safe
    to send again.
    """
    return (
        isinstance(error, TypeSafeAPIConnectionError)
        and not isinstance(error, TypeSafeAPITimeoutError)
        and isinstance(error.__cause__, httpx2.ConnectError)
    )


class JevDecisionService:
    """Asks Jev one registered question per decision.

    One :class:`AsyncTypeSafeClient` - and so one connection pool - per
    instance, shared by every concurrent decision. Build it once per process,
    use it and close it with :meth:`aclose` on one event loop: pooled
    connections belong to the loop that opened them, so a second loop (a
    per-call ``asyncio.run``, say) breaks the instance rather than merely
    losing its pool.
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        base_url: str | None = None,
        max_retries: int = 0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        if not model or not model.strip():
            raise DecisionConfigurationError("JEV_MODEL must name the Jev model to use.")
        if not (math.isfinite(timeout_seconds) and timeout_seconds > 0):
            raise DecisionConfigurationError("JEV_TIMEOUT_SECONDS must be a positive number.")
        if max_retries < 0:
            raise DecisionConfigurationError("JEV_MAX_RETRIES must not be negative.")

        _silence_sdk_logging()
        self._model = model.strip()
        self._timeout = timeout_seconds
        retry = RetryPolicy(
            max_retries=max_retries,
            # Bound retries by the same budget as the call, and keep backoff
            # short enough to fit it. A timeout has already spent the budget,
            # and any other transport error but a failed connect may have
            # delivered the request, so neither is retried.
            timeout=timeout_seconds,
            backoff_initial=min(0.05, timeout_seconds / 4),
            backoff_max=min(0.2, timeout_seconds / 2),
            http_statuses=set(_RETRYABLE_STATUSES),
            api_timeout_error=False,
            api_connection_error=False,
            predicate=_connect_failed,
        )
        try:
            self._client = AsyncTypeSafeClient(
                api_key=api_key,
                model=self._model,
                base_url=(base_url or DEFAULT_BASE_URL).rstrip("/"),
                timeout=timeout_seconds,
                retry=retry,
                transport=transport,
            )
        except TypeSafeError:
            # The SDK validates the key's format at construction and, since
            # 0.7.1, keeps its value out of the message. The message is still
            # not reused: this one is authored here.
            _raise(
                DecisionConfigurationError(
                    "JEV_API_KEY is missing or malformed (it must be printable ASCII without "
                    "whitespace)."
                )
            )

    @property
    def model(self) -> str:
        return self._model

    async def decide(self, request: DecisionRequest) -> DecisionAnswer:
        spec = get_spec(request.name)
        state = build_state(spec, request.context)
        question = Choice(instructions=spec.instructions, criteria=dict(spec.criteria))

        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._client.system_one(state=state, questions={spec.key: question}),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            _raise(DecisionTimeout())
        # Order matters: the SDK's timeout error is a connection error, and its
        # response-validation error is an API error.
        except TypeSafeAPITimeoutError:
            _raise(DecisionTimeout())
        except TypeSafeAPIConnectionError:
            _raise(DecisionConnectionFailed())
        except TypeSafeAPIResponseValidationError:
            _raise(DecisionMalformedResponse())
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as exc:
            _raise(DecisionAuthenticationFailed(exc.status))
        except TypeSafeRateLimitError:
            _raise(DecisionRateLimited())
        except (TypeSafeBadRequestError, TypeSafeNotFoundError, TypeSafeUnprocessableEntityError) as exc:
            _raise(DecisionRequestRejected(exc.status))
        except TypeSafeAPIError as exc:
            _raise(DecisionUpstreamError(exc.status))
        except TypeSafeError:
            # Raised before any request is sent: the question or state could
            # not be encoded. A defect here, not at the provider.
            _raise(DecisionRequestRejected())
        latency_ms = (time.perf_counter() - started) * 1000.0

        # Read what is needed and let the response go. The SDK response object
        # keeps the raw HTTP response, body included, and must not escape.
        answer = response.answers.get(spec.key)
        reported_model = response.model
        input_tokens = response.usage.input_tokens
        del response
        return _answer(spec, request, answer, reported_model, input_tokens, latency_ms)

    async def aclose(self) -> None:
        """Release the connection pool."""
        await self._client.aclose()


def _answer(
    spec: DecisionSpec,
    request: DecisionRequest,
    answer: object,
    reported_model: object,
    input_tokens: object,
    latency_ms: float,
) -> DecisionAnswer:
    """Validate one Choice answer against the registered labels."""
    if not isinstance(answer, ChoiceAnswer):
        _raise(DecisionMalformedResponse())
    labels = spec.label_values
    label = answer.choice
    confidence = answer.confidence
    probabilities = answer.probabilities
    if label not in labels:
        _raise(DecisionMalformedResponse())
    if not (isinstance(confidence, float) and math.isfinite(confidence) and 0.0 <= confidence <= 1.0):
        _raise(DecisionMalformedResponse())
    if not set(probabilities) <= labels or not all(
        math.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities.values()
    ):
        _raise(DecisionMalformedResponse())

    model = reported_model if isinstance(reported_model, str) and _MODEL_NAME.fullmatch(reported_model) else None
    return DecisionAnswer(
        name=request.name,
        label=label,
        confidence=confidence,
        probabilities=dict(probabilities),
        provider=PROVIDER,
        model=model,
        latency_ms=latency_ms,
        input_tokens=input_tokens if isinstance(input_tokens, int) and input_tokens >= 0 else None,
    )
