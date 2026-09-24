"""HTTP surface.

Three endpoints plus one read: enough to drive a conversation through the
deterministic layers, and no more. State is in memory and disappears with the
process.

No authentication or authorisation is implemented. Anyone who can reach these
endpoints can create and read sessions. See REPORT.md, "Security Review".
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.session import (
    MAX_EVENT_TEXT_CHARS,
    SIGNAL_BEARING_KINDS,
    EventSignals,
    Session,
    SessionLimitReached,
    parse_signals,
)
from app.models.conversation import ConversationState
from app.models.customer import AccountContext, ComplianceContext, CustomerContext
from app.models.enums import (
    ConversationStage,
    Emotion,
    EventKind,
    Intent,
    Language,
)
from app.models.enums import RequiredAction
from app.models.policy import PolicyContext, PolicyDecision
from app.observability import get_logger, log_event
from app.orchestrator import (
    ConversationOrchestrator,
    IncompleteTurn,
    TurnResult,
    UnknownSession,
)
from app.runtime import Runtime
from app.services.stt import TranscriptSegment

router = APIRouter()
_logger = get_logger(__name__)


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def _orchestrator(request: Request) -> ConversationOrchestrator:
    return request.app.state.orchestrator


class HealthResponse(BaseModel):
    status: str


class CreateSessionRequest(BaseModel):
    """Start a call.

    ``customer``, ``account`` and ``compliance`` are accepted only outside
    production. With no banking backend wired, that is the only way to exercise
    the policy engine end to end; accepting caller-asserted account facts in
    production would let a client dictate its own compliance inputs.
    """

    model_config = ConfigDict(extra="forbid")

    language: Language | None = None
    customer: CustomerContext | None = None
    account: AccountContext | None = None
    compliance: ComplianceContext | None = None


class EventRequest(BaseModel):
    """One conversation event.

    The signal fields are inputs to deterministic state. They are not compliance
    decisions: the policy engine decides, from state plus backend facts.
    """

    model_config = ConfigDict(extra="forbid")

    kind: EventKind
    language: Language | None = None
    text: str | None = Field(default=None, max_length=MAX_EVENT_TEXT_CHARS)
    intent: Intent | None = None
    emotion: Emotion | None = None
    promise_date: str | None = Field(default=None, description="ISO date the customer promised to pay.")
    stage: ConversationStage | None = None
    identity_verified: bool | None = None
    disclosed_recording: bool = False
    identified_agent: bool = False

    def signals(self) -> dict[str, object]:
        payload = self.model_dump(exclude={"kind", "language", "text"}, exclude_none=True)
        return parse_signals(payload).model_dump(mode="json", exclude_none=True)

    @model_validator(mode="after")
    def _signals_belong_to_a_signal_bearing_kind(self) -> "EventRequest":
        """Reject signals the reducer would ignore.

        ``TOOL_CALL``, ``TOOL_RESULT`` and ``POLICY_DECISION`` carry audit
        metadata, not state signals. Accepting a 200 for an intent that will
        never be applied would be worse than refusing it.
        """
        if self.kind in SIGNAL_BEARING_KINDS:
            return self
        supplied = self.model_dump(
            exclude={"kind", "language", "text", "disclosed_recording", "identified_agent"},
            exclude_none=True,
        )
        ignored = sorted(supplied) + [
            name
            for name, value in (
                ("disclosed_recording", self.disclosed_recording),
                ("identified_agent", self.identified_agent),
            )
            if value
        ]
        if ignored:
            raise ValueError(
                f"{self.kind.value} events do not carry state signals; "
                f"remove {', '.join(sorted(ignored))}"
            )
        return self


class SessionResponse(BaseModel):
    session_id: str
    state: ConversationState
    policy: PolicyDecision
    event_count: int


def _decide(runtime: Runtime, session: Session) -> PolicyDecision:
    ctx = PolicyContext(
        state=session.state,
        now=runtime.clock.now(),
        customer=session.customer,
        account=session.account,
        compliance=session.compliance,
    )
    return runtime.policy.evaluate(ctx)


def _response(runtime: Runtime, session: Session) -> SessionResponse:
    decision = _decide(runtime, session)
    log_event(
        _logger,
        "policy_decision",
        session_id=session.session_id,
        turn_id=session.state.turn_count,
        language=session.state.language.value if session.state.language else None,
        dpd_stage=decision.dpd_stage.value if decision.dpd_stage else None,
        intent=session.state.intent.value if session.state.intent else None,
        allowed=decision.allowed,
        escalate=decision.escalate,
        policy_rule_ids=list(decision.rule_ids),
        violation_rule_ids=[v.rule_id for v in decision.violations],
    )
    return SessionResponse(
        session_id=session.session_id,
        state=session.state,
        policy=decision,
        event_count=len(session.events),
    )


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/conversation/session",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["conversation"],
)
def create_session(request: Request, body: CreateSessionRequest) -> SessionResponse:
    runtime = _runtime(request)
    supplied_context = body.customer or body.account or body.compliance
    if supplied_context and runtime.settings.app_env == "prod":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Customer, account and compliance context must come from the banking backend, not the caller.",
        )
    try:
        session = runtime.sessions.create(
            now=runtime.clock.now(),
            customer=body.customer,
            account=body.account,
            compliance=body.compliance,
            language=body.language,
        )
    except SessionLimitReached as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    event = runtime.sessions.next_event(
        session, kind=EventKind.SESSION_STARTED, now=runtime.clock.now(), language=body.language
    )
    updated = runtime.sessions.append_event(session.session_id, event)
    assert updated is not None  # just created under the same store
    return _response(runtime, updated)


@router.get("/conversation/{session_id}", response_model=SessionResponse, tags=["conversation"])
def get_session(request: Request, session_id: str) -> SessionResponse:
    runtime = _runtime(request)
    session = runtime.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown session")
    return _response(runtime, session)


class TurnRequest(BaseModel):
    """One customer turn, as the transcription layer hands it over.

    The signal fields are the same upstream NLU outputs :class:`EventRequest`
    accepts. ``claimed_actions`` is the caller's assertion about what the agent's
    reply will perform - the orchestrator cannot verify an assertion about the
    meaning of a sentence and does not invent one.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=MAX_EVENT_TEXT_CHARS)
    is_final: bool = True
    language: Language | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    intent: Intent | None = None
    emotion: Emotion | None = None
    promise_date: str | None = Field(default=None, description="ISO date the customer promised to pay.")
    stage: ConversationStage | None = None
    identity_verified: bool | None = None
    disclosed_recording: bool = False
    identified_agent: bool = False

    claimed_actions: tuple[RequiredAction, ...] = ()

    def segment(self) -> TranscriptSegment:
        return TranscriptSegment(
            text=self.text, is_final=self.is_final, language=self.language, confidence=self.confidence
        )

    def signals(self) -> EventSignals:
        payload = self.model_dump(
            exclude={"text", "is_final", "language", "confidence", "claimed_actions"},
            exclude_none=True,
        )
        return parse_signals(payload)


@router.post("/conversation/{session_id}/turn", response_model=TurnResult, tags=["conversation"])
def post_turn(request: Request, session_id: str, body: TurnRequest) -> TurnResult:
    """Run one customer turn through the orchestrator.

    The response is the turn's audit record. ``speakable`` is the only field a
    caller needs to consult before playing audio: it is false for every turn that
    policy stopped, validation blocked or a failure ended.
    """
    try:
        return _orchestrator(request).process_turn(
            session_id,
            body.segment(),
            signals=body.signals(),
            claimed_actions=body.claimed_actions,
        )
    except UnknownSession as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown session") from exc
    except IncompleteTurn as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/conversation/{session_id}/event", response_model=SessionResponse, tags=["conversation"])
def post_event(request: Request, session_id: str, body: EventRequest) -> SessionResponse:
    runtime = _runtime(request)
    session = runtime.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown session")

    event = runtime.sessions.next_event(
        session,
        kind=body.kind,
        now=runtime.clock.now(),
        language=body.language,
        text=body.text,
        data=body.signals(),
    )
    updated = runtime.sessions.append_event(session_id, event)
    if updated is None:  # pragma: no cover - deleted between get and append
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown session")
    return _response(runtime, updated)
