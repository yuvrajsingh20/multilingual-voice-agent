"""Tool contract and registry.

The model *requests* a tool; the registry *runs* it. Arguments are validated
against the tool's own pydantic model before anything executes, so a malformed
or invented request is rejected rather than reaching a backend.

Auditability: every execution produces a :class:`~app.models.tools.ToolResult`
carrying the request id, status and latency, and emits one structured log event.

Nothing backend-controlled reaches the log. Argument *values* are never logged -
only their keys - and a failure is logged as an error *category* rather than as
the exception's message, because both routinely carry account and customer
identifiers. The descriptive message stays on ``ToolResult.error`` for the
orchestrator; that string is backend-supplied and must never be spoken to the
customer or logged verbatim.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from app.models.enums import ToolStatus
from app.models.tools import ToolRequest, ToolResult
from app.observability import get_logger, log_event
from app.services.llm import LlmToolSpec, contains_non_finite_number

_logger = get_logger(__name__)


class ToolNotImplemented(RuntimeError):
    """The tool exists but no backend is wired behind it."""


class Tool(ABC):
    """One callable capability.

    Subclasses declare a name, a description the model sees, and a pydantic model
    for the arguments.
    """

    name: str
    description: str
    args_model: type[BaseModel]

    @abstractmethod
    def run(self, args: BaseModel) -> dict[str, Any]:
        """Execute with already-validated arguments. Return a JSON-safe payload."""

    def spec(self) -> LlmToolSpec:
        return LlmToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )


class ToolRegistry:
    """The only place a tool is executed."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def specs(self) -> tuple[LlmToolSpec, ...]:
        return tuple(self._tools[name].spec() for name in sorted(self._tools))

    def execute(self, request: ToolRequest) -> ToolResult:
        started = time.perf_counter()
        tool = self._tools.get(request.tool_name)

        if tool is None:
            return self._finish(
                request,
                started,
                ToolStatus.NOT_FOUND,
                error=f"unknown tool: {request.tool_name}",
                error_type="UnknownTool",
            )

        # Checked here as well as at the model boundary, because not every
        # LlmService parses JSON, and a tool whose schema has a float or a
        # free-form field would otherwise take NaN or Infinity as a value.
        if contains_non_finite_number(request.arguments):
            return self._finish(
                request,
                started,
                ToolStatus.INVALID_REQUEST,
                error="invalid arguments: a number JSON cannot represent (NaN or Infinity)",
                error_type="NonFiniteArgument",
            )

        try:
            args = tool.args_model.model_validate(request.arguments)
        except ValidationError as exc:
            return self._finish(
                request,
                started,
                ToolStatus.INVALID_REQUEST,
                error=f"invalid arguments: {exc.error_count()} problem(s)",
                error_type="ValidationError",
            )

        try:
            data = tool.run(args)
        except ToolNotImplemented as exc:
            return self._finish(
                request,
                started,
                ToolStatus.NOT_IMPLEMENTED,
                error=str(exc),
                error_type="ToolNotImplemented",
            )
        except Exception as exc:  # noqa: BLE001 - boundary: never leak a backend stack trace
            return self._finish(
                request,
                started,
                ToolStatus.BACKEND_ERROR,
                error=type(exc).__name__,
                error_type=type(exc).__name__,
            )

        return self._finish(request, started, ToolStatus.OK, data=data)

    def _finish(
        self,
        request: ToolRequest,
        started: float,
        status: ToolStatus,
        *,
        data: dict[str, Any] | None = None,
        error: str | None = None,
        error_type: str | None = None,
    ) -> ToolResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        log_event(
            _logger,
            "tool_call",
            session_id=request.session_id,
            turn_id=request.turn_id,
            request_id=request.request_id,
            tool_name=request.tool_name,
            tool_latency_ms=round(latency_ms, 3),
            status=status.value,
            argument_keys=sorted(request.arguments),
            error_type=error_type,
        )
        return ToolResult(
            request_id=request.request_id,
            tool_name=request.tool_name,
            status=status,
            data=data,
            error=error,
            latency_ms=latency_ms,
        )
