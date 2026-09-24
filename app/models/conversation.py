"""Conversation state and the event log.

Deliberately separated:

* :class:`ConversationState` - the small mutable summary a policy decision needs.
* :class:`ConversationEvent` - the append-only record of what happened.

Transcript text lives on events, not on state. Customer and account facts live in
:mod:`app.models.customer` and are never copied into state, so the backend stays
the single source of truth.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import (
    ConversationStage,
    Emotion,
    EventKind,
    Intent,
    Language,
)


class ConversationState(BaseModel):
    """Current summary of one call.

    Notably *absent*: ``dpd``. Days-past-due is account data and is read from
    :class:`~app.models.customer.AccountContext`; duplicating it here would let
    conversation state drift away from the system of record.

    The three disclosure booleans exist because encoded rules depend on them:
    RBC 2025 paragraph 442(4) (recording intimation) and paragraph 445
    (prohibition on anonymous calls).
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    language: Language | None = None
    intent: Intent | None = None
    emotion: Emotion | None = None

    payment_promise: bool = False
    promise_date: date | None = None

    dispute: bool = False
    wrong_person: bool = False
    escalation_required: bool = False

    current_stage: ConversationStage = ConversationStage.GREETING

    identity_verified: bool = False
    recording_disclosed: bool = False
    agent_identified: bool = False

    turn_count: int = Field(default=0, ge=0)


class ConversationEvent(BaseModel):
    """One immutable entry in the call's audit trail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: int = Field(ge=0)
    occurred_at: datetime
    kind: EventKind
    language: Language | None = None
    text: str | None = Field(default=None, description="Transcript text. Personal data; never logged verbatim.")
    data: dict[str, Any] = Field(default_factory=dict)
