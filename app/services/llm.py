"""Model-service boundary.

Nothing is connected. No provider SDK is installed and no network call is made
anywhere in this package. The point of this module is that when Gemma is hosted
behind an OpenAI-compatible endpoint, one adapter class implementing
:class:`LlmService` is the only thing that needs writing.

The application depends on :class:`LlmService`, never on a provider. Two rules
hold whatever the provider turns out to be:

1. The model produces a *draft*. It never reaches TTS without passing
   :mod:`app.services.validation`.
2. The model requests tools; it does not execute them. Execution goes through
   :class:`~app.tools.base.ToolRegistry`, which validates arguments first.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class LlmMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str
    tool_call_id: str | None = None


class LlmToolSpec(BaseModel):
    """A tool offered to the model. ``parameters`` is a JSON Schema object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, Any]


class LlmToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LlmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[LlmMessage, ...]
    tools: tuple[LlmToolSpec, ...] = ()
    max_output_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class LlmGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str | None = None
    tool_calls: tuple[LlmToolCall, ...] = ()
    model: str | None = None
    finish_reason: str | None = None
    latency_ms: float = Field(default=0.0, ge=0)


class LlmNotConfigured(RuntimeError):
    """No model endpoint is configured. Raised instead of silently degrading."""


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
