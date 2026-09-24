"""Structured policy inputs and outputs.

The policy engine returns data, never prose. Nothing in this module produces a
sentence for the customer; generation is the LLM's job and happens downstream of
a decision.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.conversation import ConversationState
from app.models.customer import AccountContext, ComplianceContext, CustomerContext
from app.models.enums import (
    CheckStatus,
    DpdStage,
    ProhibitedConduct,
    RequiredAction,
    Severity,
    ToneLevel,
)


class PolicyContext(BaseModel):
    """Everything a decision is computed from.

    ``now`` is supplied by an explicit :class:`~app.core.clock.Clock` and must be
    timezone-aware; the engine rejects a naive instant rather than guessing.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    state: ConversationState
    now: datetime
    customer: CustomerContext | None = None
    account: AccountContext | None = None
    compliance: ComplianceContext = Field(default_factory=ComplianceContext)


class RuleCitation(BaseModel):
    """Where a decision came from. Carried on every violation for audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    paragraph: str
    url: str | None = None


class PolicyViolation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    severity: Severity
    citation: RuleCitation
    conduct: ProhibitedConduct | None = None
    detail: str = Field(description="Operator-facing explanation. Not customer-facing copy.")
    blocks_collection: bool = False


class CheckResult(BaseModel):
    """Outcome of evaluating one encoded rule against one context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    status: CheckStatus
    violation: PolicyViolation | None = None
    required_actions: tuple[RequiredAction, ...] = ()
    prohibited_conduct: tuple[ProhibitedConduct, ...] = ()
    reason: str | None = Field(default=None, description="Why a check was not evaluable.")


class PolicyDecision(BaseModel):
    """Deterministic decision for one turn.

    ``allowed`` is false when any evaluated rule produced a blocking violation.
    ``not_evaluable_rule_ids`` and ``unenforced_rule_ids`` are part of the output
    on purpose: a decision that silently ignored a rule would not be auditable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    escalate: bool
    tone: ToneLevel
    dpd_stage: DpdStage | None

    violations: tuple[PolicyViolation, ...] = ()
    required_actions: tuple[RequiredAction, ...] = ()
    prohibited_conduct: tuple[ProhibitedConduct, ...] = ()

    rule_ids: tuple[str, ...] = Field(default=(), description="Rules actually evaluated.")
    not_evaluable_rule_ids: tuple[str, ...] = Field(
        default=(), description="Active rules whose inputs the backend did not supply."
    )
    unenforced_rule_ids: tuple[str, ...] = Field(
        default=(), description="Active rules recorded for audit but with no automated check."
    )

    evaluated_at: datetime
    policy_version: str
    rules_version: str
