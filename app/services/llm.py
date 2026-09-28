"""Model-service boundary.

This module owns the *contract*: the request and generation shapes, the error
taxonomy, and the :class:`LlmService` protocol. It performs no I/O and imports no
HTTP library. The transport lives next door in
:mod:`app.services.llm_openai`, which speaks the OpenAI-compatible
chat-completions dialect and is the only module in the application that opens a
socket to a model.

No real model is connected. The adapter is wired only when a base URL and a model
name are configured; with neither set the application keeps
:class:`NotConfiguredLlmService`, so nothing here can reach the network by
accident.

The application depends on :class:`LlmService`, never on a provider. Two rules
hold whatever the provider turns out to be:

1. The model produces a *draft*. It never reaches TTS without passing
   :mod:`app.services.validation`.
2. The model requests tools; it does not execute them. Execution goes through
   :class:`~app.tools.base.ToolRegistry`, which validates arguments first.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant", "tool"]


def contains_non_finite_number(value: Any) -> bool:
    """True if ``value`` holds a float that JSON cannot represent: NaN or ±Infinity.

    Python's ``json`` module reads the non-standard literals ``NaN``,
    ``Infinity`` and ``-Infinity`` as floats, and reads an overflowing literal
    such as ``1e400`` as infinity, so "it parsed" does not mean "it is JSON".
    Tool arguments are checked with this at the model boundary and again at the
    registry. Nothing here converts a value: an argument that fails is refused.

    Walked iteratively, because the parser accepts nesting close to the
    recursion limit and a recursive walk could fail on input it had just
    accepted. Only dicts, lists and tuples are descended into. Anything else -
    the ``datetime.date`` the orchestrator binds, say - is not a number and
    passes.
    """
    stack = [value]
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if isinstance(item, float):
            if not math.isfinite(item):
                return True
        elif isinstance(item, (dict, list, tuple)):
            if id(item) in seen:
                continue
            seen.add(id(item))
            stack.extend(item.values() if isinstance(item, dict) else item)
    return False


class LlmToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LlmMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str
    tool_call_id: str | None = None
    # An assistant turn that requested tools carries those requests, so that
    # every tool result sent after it answers a call the model is on record as
    # having made. The OpenAI chat-completions contract requires this; a tool
    # message with no preceding tool call is rejected by a strict server.
    tool_calls: tuple[LlmToolCall, ...] = ()

    @model_validator(mode="after")
    def _tool_calls_are_answerable(self) -> "LlmMessage":
        if not self.tool_calls:
            return self
        if self.role != "assistant":
            raise ValueError("only an assistant message can carry tool calls")
        ids = [call.call_id for call in self.tool_calls]
        if not all(ids):
            raise ValueError("a tool call sent back to the model must carry an id")
        if len(set(ids)) != len(ids):
            raise ValueError("tool call ids in one assistant message must be unique")
        return self


class LlmToolSpec(BaseModel):
    """A tool offered to the model. ``parameters`` is a JSON Schema object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, Any]


class LlmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[LlmMessage, ...]
    tools: tuple[LlmToolSpec, ...] = ()
    max_output_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class LlmUsage(BaseModel):
    """Token accounting, when the server reports it.

    Every field is optional: an OpenAI-compatible server is not obliged to return
    a ``usage`` block, and a missing count is recorded as missing rather than as
    zero. These are counts only - no prompt or completion text is carried here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LlmGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str | None = None
    tool_calls: tuple[LlmToolCall, ...] = ()
    model: str | None = None
    finish_reason: str | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    usage: LlmUsage | None = Field(
        default=None, description="None when the server reported no usage block."
    )


class LlmError(RuntimeError):
    """Base of every failure at the model boundary.

    Failures are classified, not described. :attr:`category` is a stable string a
    caller can branch on and an operator can aggregate; :attr:`detail` is the
    operator-facing phrase the orchestrator records. Neither ever carries an
    upstream response body, a URL, a header or an API key - a model server's
    error text is attacker-influenceable and routinely echoes the request.

    The category strings are deliberately equal to the corresponding
    :class:`~app.orchestrator.result.TurnErrorCategory` values, so the
    orchestrator can map a failure without a lookup table that could drift.
    """

    category: str = "llm_failed"

    @property
    def detail(self) -> str:
        """A short, safe, application-authored description. Never upstream text."""
        return self.category


class LlmNotConfigured(LlmError):
    """No model endpoint is configured. Raised instead of silently degrading."""

    category = "llm_not_configured"


class LlmConfigurationError(LlmNotConfigured):
    """The model configuration is present but unusable.

    A half-configured endpoint - a base URL with no model name, or a URL that is
    not HTTP - is a deployment mistake, not a degraded mode. It is raised while
    the application is being built so the process fails to start, rather than at
    the first customer turn. Silently falling back to another model, or to no
    model, would hide it.
    """


class LlmTimeout(LlmError):
    """The model server did not answer inside the configured budget."""

    category = "llm_timeout"


class LlmConnectionFailed(LlmError):
    """The model server could not be reached at all."""

    category = "llm_connection_failed"


class LlmUpstreamError(LlmError):
    """The model server answered with an HTTP error status.

    Only the status code is retained. The response body is discarded at the
    boundary and never reaches a log, a result or a customer.
    """

    category = "llm_upstream_error"

    def __init__(self, status_code: int, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(f"model server returned HTTP {status_code}")
        self.status_code = status_code
        # A parsed delay, never the raw header. Absent when the server sent none
        # or sent one that could not be read as a positive wait.
        self.retry_after_seconds = retry_after_seconds

    @property
    def detail(self) -> str:
        return f"{self.category} (HTTP {self.status_code})"


class LlmMalformedResponse(LlmError):
    """The body was not the OpenAI-compatible shape: bad JSON, or fields missing."""

    category = "llm_malformed_response"


class LlmEmptyResponse(LlmError):
    """The model returned neither text nor a tool call.

    Treated as a failure rather than as an empty turn: an agent that says nothing
    on a live call is a defect, and an empty draft must not be handed to
    validation as though the model had chosen to say nothing.
    """

    category = "llm_empty_response"


class LlmIncompleteResponse(LlmError):
    """The model stopped before it had finished.

    ``finish_reason`` was ``length`` - the output-token cap cut the generation
    off - or ``content_filter`` - the server withheld part of it. What arrived is
    a fragment, not the answer: a sentence cut off after "you do not need to"
    says something the model never meant. Validation checks that figures are
    grounded, not that a sentence is whole, so a fragment is refused here
    rather than trusted downstream.

    Never retried: the same request is cut off in the same place, and a second
    attempt spends more of the customer's silence to learn that.
    """

    category = "llm_incomplete_response"

    def __init__(self, reason: str) -> None:
        super().__init__(f"model generation ended early ({reason})")
        # One of the adapter's own constants, never the server's string.
        self.reason = reason

    @property
    def detail(self) -> str:
        return f"{self.category} ({self.reason})"


class LlmInvalidToolCall(LlmError):
    """A tool call could not be read: no name, or arguments that are not a JSON object.

    Raised instead of guessing. A repaired tool call is a tool call the
    application invented, and the registry would then validate arguments no model
    actually asked for.
    """

    category = "llm_invalid_tool_call"


class LlmService(Protocol):
    """What the pipeline needs from a language model."""

    def generate(self, request: LlmRequest) -> LlmGeneration: ...


class NotConfiguredLlmService:
    """Default service. Fails loudly so a missing model is never mistaken for a quiet one."""

    def generate(self, request: LlmRequest) -> LlmGeneration:
        raise LlmNotConfigured(
            "No model endpoint configured. Set MODEL_BASE_URL and MODEL_NAME and register "
            "an LlmService implementation."
        )


class ScriptedLlmService:
    """Replays a fixed list of generations, in order. For tests only.

    Kept here rather than in the test package so that pipeline code has a
    deterministic stand-in without importing test helpers.
    """

    def __init__(self, generations: list[LlmGeneration]) -> None:
        self._generations = list(generations)
        self._index = 0
        self.requests: list[LlmRequest] = []

    def generate(self, request: LlmRequest) -> LlmGeneration:
        self.requests.append(request)
        if self._index >= len(self._generations):
            raise AssertionError("ScriptedLlmService ran out of scripted generations")
        generation = self._generations[self._index]
        self._index += 1
        return generation
