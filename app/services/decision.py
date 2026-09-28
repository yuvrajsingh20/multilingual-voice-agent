"""Decision-service boundary.

A *decision* is a bounded semantic classification: which one of a small, fixed
set of labels best describes what the customer just said. It is not a policy
decision, not an authorisation and not a response. This module owns the
provider-neutral contract - request, answer and result shapes, the error
taxonomy and the :class:`DecisionService` protocol. It performs no I/O and
imports no provider SDK; :mod:`app.services.decision_jev` is the only module
that knows Jev exists.

Disabled by default. With ``DECISION_PROVIDER=disabled`` the application keeps
:class:`DisabledDecisionService`, which raises without touching a network, and
every caller falls back to the behaviour it had before this layer existed.

What a decision may never be
----------------------------
The same two rules that bind the model boundary (:mod:`app.services.llm`) bind
this one, whatever the provider:

1. A decision is an *input* to deterministic application logic. It never
   decides whether collection is lawful, whether a disclosure is required,
   whether a customer is who they say they are, or whether a tool may run.
   Those stay with :mod:`app.core.policy`, the session and the tool registry.
2. A decision sees only what it needs. :class:`DecisionContext` has no
   *structured* field for an identity, an account reference, an amount or a
   date, and rejects any it is given. The customer's words are free text and
   can contain identifiers anyway; identifier-shaped spans are masked before
   they leave the process (:func:`app.services.decisions.registry.mask_identifiers`),
   which is best-effort, not a guarantee.
"""

from __future__ import annotations

import asyncio
import math
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from app.models.enums import ConversationStage, Language

#: Longest customer utterance a decision will look at. A backchannel, an intent
#: or an escalation request is carried by one utterance, and TypeSafe documents
#: that Jev's accuracy falls as the state grows with unrelated text. Anything
#: longer is not eligible for a bounded decision and takes the existing path.
MAX_DECISION_TEXT_CHARS = 1000

#: How many earlier customer utterances a decision may see, at most.
MAX_RECENT_UTTERANCES = 3

#: The label reported when no decision was taken - low confidence, a failure,
#: or a disabled provider. Never sent to a provider as an option.
UNCERTAIN = "UNCERTAIN"


class DecisionName(str, Enum):
    """The closed set of decisions a provider may be asked for.

    Adding a member is a code change reviewed alongside its registry entry in
    :mod:`app.services.decisions.registry`; a runtime string never becomes a
    decision.
    """

    BARGE_IN = "barge_in"
    CUSTOMER_INTENT = "customer_intent"
    HUMAN_ESCALATION = "needs_human_escalation"


class DecisionContext(BaseModel):
    """Everything a decision may be told. Deliberately small.

    There is no field for a name, a phone number, an account or customer
    reference, an amount, a DPD or a date, and ``extra="forbid"`` rejects one
    passed anyway. Each registered decision further narrows this to the fields
    it declares; see :func:`app.services.decisions.registry.build_state`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    utterance: str = Field(min_length=1, max_length=MAX_DECISION_TEXT_CHARS)
    language: Language | None = None
    agent_speaking: bool | None = None
    stage: ConversationStage | None = None
    recent_customer_utterances: tuple[str, ...] = Field(
        default=(), max_length=MAX_RECENT_UTTERANCES
    )

    @field_validator("utterance")
    @classmethod
    def _utterance_has_words(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("utterance is blank")
        return stripped

    @field_validator("recent_customer_utterances")
    @classmethod
    def _recent_are_bounded(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item and item.strip())
        if any(len(item) > MAX_DECISION_TEXT_CHARS for item in cleaned):
            raise ValueError(f"a recent utterance exceeds {MAX_DECISION_TEXT_CHARS} characters")
        return cleaned


class DecisionRequest(BaseModel):
    """One decision to take. ``name`` must be a :class:`DecisionName` member.

    ``strict=True`` on ``name`` means the string ``"barge_in"`` is refused just
    as ``"anything_else"`` is: a caller has to name a registered decision in
    code, not pass text through.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: DecisionName = Field(strict=True)
    context: DecisionContext


class DecisionAnswer(BaseModel):
    """What a provider returned, before any threshold is applied.

    Provider-neutral: no SDK object, raw HTTP response or response body crosses
    the boundary. ``label`` is one of the registered labels for ``name``; the
    adapter and the coordinator both check that independently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: DecisionName
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    probabilities: dict[str, float] = Field(default_factory=dict)
    provider: str
    model: str | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)

    @field_validator("probabilities")
    @classmethod
    def _probabilities_are_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        for probability in value.values():
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("every probability must be a finite number between 0 and 1")
        return value


class DecisionOutcome(str, Enum):
    """What became of one attempt to take a decision."""

    DECIDED = "decided"  # a registered label, at or above its threshold
    UNCERTAIN = "uncertain"  # an answer, but below the threshold
    FAILED = "failed"  # the provider failed; see ``failure``
    DISABLED = "disabled"  # no provider is configured


class DecisionResult(BaseModel):
    """One decision attempt, after the threshold, as fusion and the logs see it.

    ``label`` is set only when the outcome is ``DECIDED``. Every other outcome
    is reported as :data:`UNCERTAIN` and uses the fallback: a binary answer is
    never forced out of an uncertain one. ``top_label`` keeps what the provider
    leaned towards, for evaluation only - nothing acts on it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: DecisionName
    outcome: DecisionOutcome
    label: str | None = None
    top_label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float = Field(gt=0.0, le=1.0)
    provider: str
    model: str | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    failure: str | None = Field(
        default=None,
        description="Error category, 'low_confidence', or None when decided. Never upstream text.",
    )

    @model_validator(mode="after")
    def _label_only_when_decided(self) -> "DecisionResult":
        """A label is a decision. It exists exactly when the outcome is DECIDED."""
        if (self.outcome is DecisionOutcome.DECIDED) != (self.label is not None):
            raise ValueError("label must be set if and only if the outcome is DECIDED")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fallback_used(self) -> bool:
        return self.outcome is not DecisionOutcome.DECIDED

    @property
    def decision(self) -> str:
        """The label, or :data:`UNCERTAIN`."""
        return self.label if self.label is not None else UNCERTAIN

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_bucket(self) -> str:
        return confidence_bucket(self.confidence)


def confidence_bucket(confidence: float | None) -> str:
    """Coarse band for dashboards. Buckets, not raw values, reach the logs."""
    if confidence is None:
        return "none"
    for bound, name in ((0.5, "lt_0.50"), (0.7, "0.50_0.70"), (0.8, "0.70_0.80"), (0.9, "0.80_0.90")):
        if confidence < bound:
            return name
    return "ge_0.90"


def result_from_answer(answer: DecisionAnswer, *, threshold: float, latency_ms: float) -> DecisionResult:
    """Apply the decision's threshold to a provider answer."""
    decided = answer.confidence >= threshold
    return DecisionResult(
        name=answer.name,
        outcome=DecisionOutcome.DECIDED if decided else DecisionOutcome.UNCERTAIN,
        label=answer.label if decided else None,
        top_label=answer.label,
        confidence=answer.confidence,
        threshold=threshold,
        provider=answer.provider,
        model=answer.model,
        latency_ms=latency_ms,
        input_tokens=answer.input_tokens,
        failure=None if decided else "low_confidence",
    )


def failed_result(
    name: DecisionName,
    *,
    threshold: float,
    provider: str,
    category: str,
    latency_ms: float,
) -> DecisionResult:
    """A decision that produced no answer."""
    return DecisionResult(
        name=name,
        outcome=(
            DecisionOutcome.DISABLED
            if category == DecisionDisabled.category
            else DecisionOutcome.FAILED
        ),
        threshold=threshold,
        provider=provider,
        latency_ms=latency_ms,
        failure=category,
    )


# --- errors -----------------------------------------------------------------


class DecisionError(RuntimeError):
    """Base of every failure at the decision boundary.

    Classified, not described, exactly as :class:`app.services.llm.LlmError`
    is. :attr:`category` is a stable string; :attr:`detail` is
    application-authored. Neither ever carries a provider response body, a
    header, a URL or a key - a provider's error text can echo the request, and
    the request carries what the customer said.

    A low-confidence answer is **not** an error. It is an answer, and it is
    reported as :attr:`DecisionOutcome.UNCERTAIN`; only a failure to get an
    answer at all lands here.
    """

    category: str = "decision_failed"

    @property
    def detail(self) -> str:
        return self.category


class DecisionDisabled(DecisionError):
    """No decision provider is configured. The caller's existing path applies."""

    category = "decision_disabled"


class DecisionConfigurationError(DecisionError):
    """The decision configuration is present but unusable.

    Raised while the application is built, so the process fails to start rather
    than silently running without the provider an operator asked for.
    """

    category = "decision_configuration_error"


class DecisionTimeout(DecisionError):
    """No answer inside the configured budget."""

    category = "decision_timeout"


class DecisionConnectionFailed(DecisionError):
    """The provider could not be reached."""

    category = "decision_connection_failed"


class DecisionAuthenticationFailed(DecisionError):
    """The provider refused the credentials (401) or the access (403)."""

    category = "decision_authentication_failed"

    def __init__(self, status_code: int) -> None:
        super().__init__(f"decision provider returned HTTP {status_code}")
        self.status_code = status_code

    @property
    def detail(self) -> str:
        return f"{self.category} (HTTP {self.status_code})"


class DecisionRateLimited(DecisionError):
    """The provider's rate limit was hit (429)."""

    category = "decision_rate_limited"


class DecisionRequestRejected(DecisionError):
    """The provider rejected the request itself (400, 404, 422).

    Distinct from an upstream failure because it points at this application:
    a question the provider cannot parse, or a model name it does not serve.
    """

    category = "decision_request_rejected"

    def __init__(self, status_code: int | None = None) -> None:
        super().__init__(
            f"decision provider rejected the request (HTTP {status_code})"
            if status_code is not None
            else "decision request could not be built"
        )
        self.status_code = status_code

    @property
    def detail(self) -> str:
        return f"{self.category} (HTTP {self.status_code})" if self.status_code else self.category


class DecisionUpstreamError(DecisionError):
    """The provider answered with another HTTP error status. Only the code is kept."""

    category = "decision_upstream_error"

    def __init__(self, status_code: int) -> None:
        super().__init__(f"decision provider returned HTTP {status_code}")
        self.status_code = status_code

    @property
    def detail(self) -> str:
        return f"{self.category} (HTTP {self.status_code})"


class DecisionMalformedResponse(DecisionError):
    """An answer that is not a usable answer.

    A label outside the registered set, a confidence that is not a number in
    [0, 1], a missing answer, or a body the provider's schema rejects. The
    answer is discarded rather than repaired: a repaired label is a decision
    this application invented.
    """

    category = "decision_malformed_response"


# --- services ---------------------------------------------------------------


class DecisionService(Protocol):
    """What the application needs from a decision provider.

    Asynchronous, because the caller it exists for - the live audio path - is.
    An implementation must bound its own latency and must never block the event
    loop. The coordinator also enforces a deadline and discards an answer that
    arrives after it, so a slow provider cannot stall a call; a provider that
    blocks the loop defeats every async deadline and is a defect.
    """

    provider: str

    async def decide(self, request: DecisionRequest) -> DecisionAnswer: ...


class DisabledDecisionService:
    """Default service. Opens nothing and raises, so the caller's fallback runs."""

    provider = "disabled"

    async def decide(self, request: DecisionRequest) -> DecisionAnswer:
        raise DecisionDisabled(
            "No decision provider configured. Set DECISION_PROVIDER=jev and JEV_ENABLED=true "
            "to enable one."
        )


class ScriptedDecisionService:
    """Replays a fixed list of answers or errors, in order. For tests only.

    Kept here rather than in the test package for the same reason as
    :class:`app.services.llm.ScriptedLlmService`: orchestration code gets a
    deterministic stand-in without importing test helpers.
    """

    provider = "scripted"

    def __init__(
        self,
        script: list[DecisionAnswer | DecisionError],
        *,
        delay_seconds: float = 0.0,
    ) -> None:
        self._script = list(script)
        self._index = 0
        self._delay = delay_seconds
        self.requests: list[DecisionRequest] = []

    async def decide(self, request: DecisionRequest) -> DecisionAnswer:
        self.requests.append(request)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._index >= len(self._script):
            raise AssertionError("ScriptedDecisionService ran out of scripted answers")
        item = self._script[self._index]
        self._index += 1
        if isinstance(item, DecisionError):
            raise item
        return item
