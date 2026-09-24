"""In-memory session store and the state reducer.

The reducer is a pure function: ``apply_event(state, event) -> ConversationState``.
Replaying a session's events from the start reproduces its state exactly, which
is what makes an audit possible.

Storage is a process-local dictionary. There is no database, no persistence
across restarts and no cross-process sharing. That is a deliberate limit of this
foundation, not an oversight.
"""

from __future__ import annotations

import threading
import uuid
from datetime import date, datetime
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.models.conversation import ConversationEvent, ConversationState
from app.models.customer import AccountContext, ComplianceContext, CustomerContext
from app.models.enums import ConversationStage, Emotion, EventKind, Intent, Language

#: Longest transcript fragment accepted on a single event.
MAX_EVENT_TEXT_CHARS = 4000


class SessionLimitReached(RuntimeError):
    """The store is full. Raised rather than silently evicting a live call."""


class EventSignals(BaseModel):
    """Structured signals an event may carry.

    Everything here is an *input* to deterministic state, typically produced by
    upstream NLU. A signal is never a compliance decision: the policy engine
    decides, from state plus backend facts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: Intent | None = None
    emotion: Emotion | None = None
    promise_date: date | None = None
    stage: ConversationStage | None = None
    identity_verified: bool | None = None
    disclosed_recording: bool = False
    identified_agent: bool = False


#: Event kinds whose ``data`` carries state signals.
#:
#: Every other kind's ``data`` is audit metadata written by the orchestrator -
#: tool name, tool status, latency, evaluated rule ids - and is deliberately
#: opaque to the reducer. Conversation state must change because a *signal* said
#: so, never because a tool happened to run; and the audit trail must be able to
#: record a tool call without that record being mistaken for a state signal.
#:
#: Keeping the check on the event *kind* rather than on the shape of ``data``
#: preserves the strictness that matters: a signal-bearing event with an unknown
#: key is still rejected.
SIGNAL_BEARING_KINDS: frozenset[EventKind] = frozenset(
    {
        EventKind.SESSION_STARTED,
        EventKind.USER_UTTERANCE,
        EventKind.AGENT_UTTERANCE,
        EventKind.BARGE_IN,
        EventKind.ACCOUNT_CONTEXT_LOADED,
        EventKind.CALL_ENDED,
    }
)


def parse_signals(data: dict[str, Any]) -> EventSignals:
    """Validate an event's ``data`` payload. Unknown keys are rejected."""
    return EventSignals.model_validate(data)


def apply_event(state: ConversationState, event: ConversationEvent) -> ConversationState:
    """Return the state that results from ``event``. Pure; ``state`` is untouched."""
    signals = (
        parse_signals(event.data) if event.kind in SIGNAL_BEARING_KINDS else EventSignals()
    )
    new = state.model_copy(deep=True)

    if event.language is not None:
        new.language = event.language
    if signals.emotion is not None:
        new.emotion = signals.emotion
    if signals.identity_verified is not None:
        new.identity_verified = signals.identity_verified
    if signals.disclosed_recording:
        new.recording_disclosed = True
    if signals.identified_agent:
        new.agent_identified = True

    if event.kind is EventKind.USER_UTTERANCE:
        new.turn_count = state.turn_count + 1

    intent = signals.intent
    if intent is not None:
        new.intent = intent
        if intent is Intent.IDENTITY_CONFIRMED:
            new.identity_verified = True
        elif intent is Intent.WRONG_PERSON:
            new.wrong_person = True
            new.identity_verified = False
            new.current_stage = ConversationStage.CLOSING
        elif intent is Intent.DISPUTE:
            new.dispute = True
        elif intent is Intent.PAYMENT_PROMISE:
            new.payment_promise = True
        elif intent is Intent.ESCALATION_REQUEST:
            new.escalation_required = True

    if signals.promise_date is not None:
        new.promise_date = signals.promise_date

    # An explicit stage from the orchestrator wins, except that a wrong-person
    # call cannot be steered back into collection.
    if signals.stage is not None and not new.wrong_person:
        new.current_stage = signals.stage

    if event.kind is EventKind.CALL_ENDED:
        new.current_stage = ConversationStage.TERMINATED

    return new


def replay(session_id: str, events: Iterable[ConversationEvent]) -> ConversationState:
    """Rebuild state from an event log. Used for audit and for tests."""
    state = ConversationState(session_id=session_id)
    for event in events:
        state = apply_event(state, event)
    return state


class Session(BaseModel):
    """One call: its state, its event log and the backend context loaded for it."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: datetime
    state: ConversationState
    events: list[ConversationEvent] = Field(default_factory=list)
    customer: CustomerContext | None = None
    account: AccountContext | None = None
    compliance: ComplianceContext = Field(default_factory=ComplianceContext)


class SessionStore:
    """Process-local session store guarded by a lock.

    FastAPI runs sync endpoints on a thread pool, so concurrent access is real
    even in this single-process foundation.
    """

    def __init__(self, max_sessions: int) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._max = max_sessions

    def create(
        self,
        *,
        now: datetime,
        customer: CustomerContext | None = None,
        account: AccountContext | None = None,
        compliance: ComplianceContext | None = None,
        language: Language | None = None,
    ) -> Session:
        session_id = uuid.uuid4().hex
        session = Session(
            session_id=session_id,
            created_at=now,
            state=ConversationState(session_id=session_id, language=language),
            customer=customer,
            account=account,
            compliance=compliance or ComplianceContext(),
        )
        with self._lock:
            if len(self._sessions) >= self._max:
                raise SessionLimitReached(f"session store is full ({self._max})")
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def append_event(self, session_id: str, event: ConversationEvent) -> Session | None:
        """Append ``event`` and advance state atomically."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.state = apply_event(session.state, event)
            session.events.append(event)
            return session

    def next_event(
        self,
        session: Session,
        *,
        kind: EventKind,
        now: datetime,
        language: Language | None = None,
        text: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> ConversationEvent:
        """Build the next event for ``session``, numbering the turn."""
        return ConversationEvent(
            event_id=uuid.uuid4().hex,
            session_id=session.session_id,
            turn_id=len(session.events),
            occurred_at=now,
            kind=kind,
            language=language,
            text=text,
            data=data or {},
        )

    def set_account(self, session_id: str, account: AccountContext) -> Session | None:
        """Replace the session's account context with a freshly-read one.

        A tool reads the *system of record*. Once it has, the session's copy is
        known to be stale, and keeping the stale copy would mean the next turn
        re-decides policy on data this call already disproved.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.account = account
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
