"""Tool call envelope.

Every backend call the agent makes is a :class:`ToolRequest` in and a
:class:`ToolResult` out, so the pair can be written to the event log and replayed
during an audit.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import ToolStatus


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: int = Field(ge=0)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    requested_by: Literal["llm", "policy", "system"] = "llm"


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    tool_name: str
    status: ToolStatus
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = Field(default=0.0, ge=0)

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK
