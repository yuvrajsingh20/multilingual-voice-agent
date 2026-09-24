"""Structured result of one orchestrated turn.

A :class:`TurnResult` is the audit record of a single customer turn: what was
heard, what state it produced, every policy decision taken, every tool attempted,
what the model drafted, whether validation let it through, and how long each
stage took.

Why this lives here and not in :mod:`app.models`
------------------------------------------------
A turn result aggregates the output of the policy engine *and* of the response
validator, so it sits above both. Putting it in ``app.models`` would make the
domain-model package import ``app.services``, which imports ``app.core``, which
imports ``app.models`` - a genuine import cycle. The orchestrator package is the
top layer: it depends on everything and nothing depends on it.

What is deliberately kept out
-----------------------------
The result is designed to be auditable without becoming a second copy of the
banking system:

* No ``AccountContext``, ``CustomerContext`` or ``ComplianceContext`` is embedded.
  Amounts, DPD, account references and customer references stay in the backend;
  the result records only *whether* a fact was available and *where* it came
  from (:class:`GroundingSource`), never the value.
* Tool arguments are recorded as key names only, never as values, because
  arguments routinely carry account and customer references.
* A tool failure is recorded as a :class:`~app.models.enums.ToolStatus` and an
  application-authored category. Backend exception messages are not copied in:
  they routinely carry identifiers and are not safe to log or to speak.

``normalized_transcript`` and ``draft_text`` *are* present, because a turn cannot
be audited without them. Both are personal data. They are returned to the caller
and must never be written to a log; see :mod:`app.observability`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.models.conversation import ConversationState
from app.models.enums import (
    Language,
    RequiredAction,
    SpanKind,
    ToolStatus,
)
from app.models.policy import PolicyDecision
from app.services.validation import ValidationResult


class TurnStage(str, Enum):
    """Pipeline stage, used for both latency attribution and error attribution."""

    NORMALIZATION = "normalization"
    STATE_UPDATE = "state_update"
    CONTEXT_LOAD = "context_load"
    POLICY = "policy"
    LLM = "llm"
    TOOLS = "tools"
    VALIDATION = "validation"
    TTS_NORMALIZATION = "tts_normalization"


class TurnOutcome(str, Enum):
    """How the turn ended.

    Exactly one outcome is reported. ``COMPLETED`` is the only outcome that
    produces speakable text.
    """

    COMPLETED = "completed"
    POLICY_BLOCKED = "policy_blocked"
    RESPONSE_BLOCKED = "response_blocked"
    FAILED = "failed"


class TurnErrorCategory(str, Enum):
    """Error *categories*. Never a backend message, never a raw exception string."""

    TRANSCRIPT_NORMALIZATION_FAILED = "transcript_normalization_failed"
    INVALID_CONVERSATION_EVENT = "invalid_conversation_event"
    MISSING_BACKEND_CONTEXT = "missing_backend_context"
    POLICY_CONFIGURATION_ERROR = "policy_configuration_error"
    POLICY_EVALUATION_FAILED = "policy_evaluation_failed"
    LLM_NOT_CONFIGURED = "llm_not_configured"
    LLM_FAILED = "llm_failed"
    INVALID_TOOL_REQUEST = "invalid_tool_request"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    TOOL_BACKEND_NOT_IMPLEMENTED = "tool_backend_not_implemented"
    TOOL_LIMIT_EXCEEDED = "tool_limit_exceeded"
    RESPONSE_VALIDATION_FAILED = "response_validation_failed"
    TTS_NORMALIZATION_FAILED = "tts_normalization_failed"


class PolicyCheckpoint(str, Enum):
    """Where in the turn a policy decision was taken.

    The orchestrator evaluates policy at every one of these points. A decision
    taken before a tool ran is not assumed to survive the tool.
    """

    PRE_LLM = "pre_llm"
    POST_TOOL = "post_tool"
    PRE_TTS = "pre_tts"


class GroundingSource(str, Enum):
    """Where the facts the agent may state came from.

    The LLM is never a source. ``ACCOUNT_CONTEXT`` is the backend context loaded
    for the session; ``TOOL_RESULT`` is a value returned by a tool executed
    through the registry this turn.
    """

    ACCOUNT_CONTEXT = "account_context"
    TOOL_RESULT = "tool_result"


class TurnError(BaseModel):
    """One failure. ``detail`` is authored by this application, not by a backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: TurnErrorCategory
    stage: TurnStage
    detail: str = Field(description="Operator-facing. Never a backend message or customer data.")
    safety_critical: bool = Field(
        default=False,
        description="True when the turn must not continue to speech after this failure.",
    )


class ToolAttempt(BaseModel):
    """One tool the model asked for, whether or not it was allowed to run.

    ``argument_keys`` records the shape of the request without its values.
    ``dispatched`` distinguishes the two ways a tool request can fail: the
    orchestrator refusing it (``False`` - it never reached the registry) from the
    registry rejecting or running it (``True``, with a status).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    tool_name: str
    dispatched: bool = Field(description="False when the orchestrator refused before the registry.")
    status: ToolStatus | None = Field(default=None, description="None when never dispatched.")
    argument_keys: tuple[str, ...] = ()
    latency_ms: float = Field(default=0.0, ge=0)
    refusal_reason: str | None = Field(
        default=None, description="Why the orchestrator refused to execute. Application-authored."
    )


class PolicyEvaluation(BaseModel):
    """A policy decision together with the checkpoint it was taken at."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: PolicyCheckpoint
    decision: PolicyDecision


class TurnLatency(BaseModel):
    """Per-stage wall-clock, in milliseconds.

    Measured, not estimated. Stages that did not run stay at zero, so a zero is
    "did not happen" rather than "was instant"; :attr:`total_ms` covers the whole
    turn including orchestration overhead, so it is not the sum of the parts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    normalization_ms: float = Field(default=0.0, ge=0)
    state_update_ms: float = Field(default=0.0, ge=0)
    context_load_ms: float = Field(default=0.0, ge=0)
    policy_ms: float = Field(default=0.0, ge=0)
    llm_ms: float = Field(default=0.0, ge=0)
    tool_ms: float = Field(default=0.0, ge=0)
    validation_ms: float = Field(default=0.0, ge=0)
    tts_normalization_ms: float = Field(default=0.0, ge=0)
    total_ms: float = Field(default=0.0, ge=0)


class TurnResult(BaseModel):
    """Everything one turn produced.

    ``speakable`` is the single question the caller must ask before playing
    audio. It is true only when a response passed validation and TTS
    normalisation, and it is false for every blocked, failed and policy-stopped
    turn.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    turn_id: int = Field(ge=0)
    outcome: TurnOutcome
    speakable: bool

    # --- what was heard ---------------------------------------------------
    normalized_transcript: str = Field(description="Personal data. Returned to the caller; never logged.")
    transcript_applied: tuple[str, ...] = Field(
        default=(), description="Normalisations actually applied to the transcript."
    )
    language: Language | None = None

    # --- what it did to the call -----------------------------------------
    state: ConversationState = Field(description="State after the turn's events were applied.")
    event_ids: tuple[str, ...] = Field(default=(), description="Events appended by this turn, in order.")

    # --- what policy decided ---------------------------------------------
    policy: PolicyDecision | None = Field(
        default=None,
        description="The last decision taken, i.e. the one that governs. None if the turn failed before policy ran.",
    )
    policy_evaluations: tuple[PolicyEvaluation, ...] = Field(
        default=(), description="Every checkpoint evaluated this turn, in order."
    )
    required_actions: tuple[RequiredAction, ...] = Field(
        default=(), description="What the caller must do next. Categories, not sentences."
    )

    # --- what the backend was asked ---------------------------------------
    tools: tuple[ToolAttempt, ...] = ()
    grounding_sources: tuple[GroundingSource, ...] = Field(
        default=(), description="Where stateable facts came from. Never the values themselves."
    )
    grounded_fact_count: int = Field(default=0, ge=0, description="How many facts, not which.")

    # --- what the model produced ------------------------------------------
    llm_calls: int = Field(default=0, ge=0)
    draft_text: str | None = Field(
        default=None, description="Model output before validation. Personal data; never logged."
    )
    validation: ValidationResult | None = Field(default=None, description="None when validation never ran.")

    # --- what may be spoken ------------------------------------------------
    response_text: str | None = Field(
        default=None, description="TTS-normalised text. Populated only when speakable is true."
    )
    tts_fully_normalized: bool = False
    tts_unrendered_kinds: tuple[SpanKind, ...] = ()

    # --- how it went -------------------------------------------------------
    errors: tuple[TurnError, ...] = ()
    latency: TurnLatency = Field(default_factory=TurnLatency)

    @property
    def failed(self) -> bool:
        return self.outcome is TurnOutcome.FAILED

    @property
    def error_categories(self) -> tuple[TurnErrorCategory, ...]:
        return tuple(error.category for error in self.errors)
