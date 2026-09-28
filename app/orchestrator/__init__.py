"""Application-level conversation orchestration.

The top layer. It coordinates the deterministic engine (:mod:`app.core`), the
boundary services (:mod:`app.services`) and the tool registry (:mod:`app.tools`)
into one turn pipeline, and owns the control flow that decides when policy is
evaluated and when the turn must stop.

Nothing in :mod:`app.core`, :mod:`app.services`, :mod:`app.models` or
:mod:`app.tools` imports this package; the dependency runs one way only.
"""

from app.orchestrator.decisions import DecisionCoordinator
from app.orchestrator.pipeline import (
    ConversationOrchestrator,
    IncompleteTurn,
    UnknownSession,
)
from app.orchestrator.prompt import build_llm_request, build_system_prompt
from app.orchestrator.result import (
    GroundingSource,
    PolicyCheckpoint,
    PolicyEvaluation,
    ToolAttempt,
    TurnError,
    TurnErrorCategory,
    TurnLatency,
    TurnOutcome,
    TurnResult,
    TurnStage,
)

__all__ = [
    "ConversationOrchestrator",
    "DecisionCoordinator",
    "GroundingSource",
    "IncompleteTurn",
    "PolicyCheckpoint",
    "PolicyEvaluation",
    "ToolAttempt",
    "TurnError",
    "TurnErrorCategory",
    "TurnLatency",
    "TurnOutcome",
    "TurnResult",
    "TurnStage",
    "UnknownSession",
    "build_llm_request",
    "build_system_prompt",
]
