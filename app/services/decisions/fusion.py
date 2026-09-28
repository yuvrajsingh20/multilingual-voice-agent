"""Decision fusion: a semantic decision, application state and hard signals in;
what the application does out.

Pure functions. No I/O, no provider, no clock. Given the same inputs they
return the same resolution, so every rule here is directly testable and
auditable.

The one principle every function follows: a semantic decision may *refine*
behaviour inside the space the deterministic layers already allow. It may
never widen that space. Concretely:

* It cannot override the acoustic gate or a hard interruption signal -
  sustained speech, or an explicit request to stop.
* It cannot override policy. A policy that requires escalation escalates
  whatever the decision says, and a decision that recommends escalation is not
  an escalation until the application's own escalation path says so.
* It cannot set any conversation-state flag that gates a backend write, ends
  the call or touches identity. Those labels are returned for confirmation,
  never applied.
* When the decision is uncertain, failed or disabled, the result is exactly
  what the application did before this layer existed.
"""

from __future__ import annotations

import unicodedata
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel, ConfigDict

from app.models.conversation import ConversationState
from app.models.enums import BargeInClass, Intent
from app.models.policy import PolicyDecision
from app.services.decision import DecisionName, DecisionOutcome, DecisionResult
from app.services.decisions.registry import BargeInLabel, CustomerIntentLabel, EscalationLabel
from app.services.turn import BargeInDecision, BargeInSignal

#: Default hard-interruption bound, in milliseconds of continuous speech. Equal
#: to :class:`~app.services.turn.HeuristicBargeInClassifier`'s default
#: backchannel window: a semantic decision may keep the agent talking only over
#: speech short enough that the existing classifier already accepts it as
#: acknowledgement-length. Longer speech stops TTS whatever the words mean.
DEFAULT_HARD_INTERRUPTION_MS = 800

#: Explicit requests to stop, in the languages in scope, in Latin and Devanagari
#: script. PROVISIONAL, like :data:`app.services.turn.BACKCHANNEL_TOKENS`:
#: derived from the languages, not from call data. Their only effect is to stop
#: a semantic decision from keeping the agent talking over them - an utterance
#: containing one is never sent for a barge-in decision, and the heuristic's
#: decision stands. A false match costs a pause; a missed one costs talking over
#: a customer who said stop, so stems are listed ("ruk" covers "ruk jao", "ruk
#: jaiye") and the list errs wide.
HALT_PHRASES: frozenset[str] = frozenset({
    # English
    "stop", "wait", "hold on", "hang on", "hold up", "one second", "one sec",
    "one minute", "just a second", "just a sec", "just a minute",
    # Hindi / Hinglish, romanised
    "ruk", "ruko", "rukiye", "rukie", "rukna", "thehro", "thahro", "thehriye",
    "ek minute", "ek minit", "ek min", "ek second", "ek sec", "bas",
    # Marathi, romanised
    "thamb", "thamba", "thambaa", "thamba jara",
    # Devanagari
    "रुक", "रुको", "रुकिए", "रुकिये", "रुकना", "ठहरो", "ठहरिए", "ठहरिये", "बस",
    "एक मिनट", "एक मिनिट", "एक सेकंड", "एक सेकेंड", "थांब", "थांबा",
})


def _normalise(text: str) -> str:
    """Lower-cased, NFC, with punctuation, symbols and format characters removed.

    ``\\w`` is not used: it excludes the combining vowel signs of Devanagari,
    so it would split ``थांबा`` apart. Format characters (zero-width joiner and
    non-joiner) are dropped rather than turned into spaces, because they sit
    inside Devanagari words.
    """
    out = []
    for ch in unicodedata.normalize("NFC", text.lower()):
        category = unicodedata.category(ch)
        if category == "Cf":
            continue
        out.append(" " if category[0] in "PS" else ch)
    return " " + " ".join("".join(out).split()) + " "


#: The phrases in the same normal form the text is compared in.
_HALT_NORMALISED: frozenset[str] = frozenset(_normalise(phrase).strip() for phrase in HALT_PHRASES)


def contains_halt_phrase(text: str) -> bool:
    """True when ``text`` contains an explicit request to stop, as whole words."""
    normalised = _normalise(text)
    return any(f" {phrase} " in normalised for phrase in _HALT_NORMALISED)


class DecisionSource(str, Enum):
    """What determined a resolution."""

    HEURISTIC = "heuristic"  # the existing deterministic behaviour, unchanged
    SEMANTIC = "semantic"  # a decided semantic label, inside deterministic bounds
    POLICY = "policy"  # authoritative application state or policy
    UPSTREAM = "upstream"  # a signal the caller already supplied


def _usable(semantic: DecisionResult | None, name: DecisionName) -> str | None:
    """The decided label, or ``None`` when there is nothing to act on.

    Only a ``DECIDED`` result is ever acted on, whatever else it carries.
    """
    if semantic is None or semantic.name is not name or semantic.outcome is not DecisionOutcome.DECIDED:
        return None
    return semantic.label


def _uncertain_reason(semantic: DecisionResult | None) -> str:
    if semantic is None:
        return "no semantic decision was taken"
    return f"semantic decision {semantic.outcome.value} ({semantic.failure or 'no label'})"


# --- barge-in ---------------------------------------------------------------


class BargeInAction(str, Enum):
    """What the audio path does with the agent's speech."""

    CONTINUE_TTS = "continue_tts"
    STOP_TTS = "stop_tts"
    WAIT = "wait"  # keep speaking for now; re-decide on the next partial transcript


class BargeInResolution(BaseModel):
    """The fused barge-in outcome.

    ``decision`` is a :class:`~app.services.turn.BargeInDecision`, so the result
    drops straight into :meth:`~app.services.turn.TurnStateMachine.on_customer_audio`
    without that machine learning anything new.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: BargeInAction
    decision: BargeInDecision
    source: DecisionSource
    semantic: DecisionResult | None = None
    reason: str


def barge_in_ineligibility(
    signal: BargeInSignal,
    heuristic: BargeInDecision,
    *,
    agent_speaking: bool,
    hard_interruption_ms: int = DEFAULT_HARD_INTERRUPTION_MS,
) -> str | None:
    """Why no semantic decision is needed, or ``None`` when one is.

    This is the routing rule for barge-in, and it runs before any provider is
    called: most customer audio never needs one.
    """
    if not agent_speaking:
        return "agent is not speaking"
    if heuristic.classification is BargeInClass.NOISE:
        return "acoustic gate: no qualifying speech"
    text = (signal.partial_transcript or "").strip()
    if not text:
        return "no transcript yet"
    if signal.vad.duration_ms > hard_interruption_ms:
        return f"hard interruption signal: speech longer than {hard_interruption_ms} ms"
    if contains_halt_phrase(text):
        return "hard interruption signal: an explicit request to stop"
    return None


def _heuristic_action(decision: BargeInDecision) -> BargeInAction:
    if decision.stop_tts:
        return BargeInAction.STOP_TTS
    if decision.classification is BargeInClass.UNKNOWN:
        return BargeInAction.WAIT
    return BargeInAction.CONTINUE_TTS


def fuse_barge_in(
    *,
    heuristic: BargeInDecision,
    signal: BargeInSignal,
    agent_speaking: bool,
    semantic: DecisionResult | None,
    hard_interruption_ms: int = DEFAULT_HARD_INTERRUPTION_MS,
) -> BargeInResolution:
    """Combine the heuristic classifier with a semantic decision.

    The heuristic always runs first and is the fallback. A semantic decision is
    consulted only for eligible audio - the agent speaking, speech past the
    acoustic gate, a transcript, and no hard interruption signal - and only a
    *decided* label changes anything. If ``semantic`` is passed for ineligible
    audio it is ignored: nothing here lets it override the gate.
    """
    ineligible = barge_in_ineligibility(
        signal, heuristic, agent_speaking=agent_speaking, hard_interruption_ms=hard_interruption_ms
    )
    if ineligible is not None:
        return BargeInResolution(
            action=_heuristic_action(heuristic),
            decision=heuristic,
            source=DecisionSource.HEURISTIC,
            reason=f"{ineligible}; heuristic decision applies. {heuristic.reason}",
        )

    label = _usable(semantic, DecisionName.BARGE_IN)
    if label is None:
        return BargeInResolution(
            action=_heuristic_action(heuristic),
            decision=heuristic,
            source=DecisionSource.HEURISTIC,
            semantic=semantic,
            reason=f"{_uncertain_reason(semantic)}; heuristic decision applies. {heuristic.reason}",
        )

    kind = BargeInLabel(label)
    if kind is BargeInLabel.BACKCHANNEL:
        return BargeInResolution(
            action=BargeInAction.CONTINUE_TTS,
            decision=BargeInDecision(
                classification=BargeInClass.BACKCHANNEL,
                stop_tts=False,
                reason="Semantic backchannel within the acknowledgement window.",
            ),
            source=DecisionSource.SEMANTIC,
            semantic=semantic,
            reason="semantic backchannel, no hard interruption signal: the agent keeps speaking",
        )
    reason = (
        "semantic interruption: the customer wants the floor"
        if kind is BargeInLabel.INTERRUPTION
        else "semantic continuation: the customer is finishing their own point"
    )
    return BargeInResolution(
        action=BargeInAction.STOP_TTS,
        decision=BargeInDecision(
            classification=BargeInClass.INTERRUPTION, stop_tts=True, reason=reason.capitalize() + "."
        ),
        source=DecisionSource.SEMANTIC,
        semantic=semantic,
        reason=reason,
    )


# --- customer intent --------------------------------------------------------


class IntentAction(str, Enum):
    """What the application may do with a classified intent."""

    APPLY_SIGNAL = "apply_signal"  # low risk: may be handed to the reducer as EventSignals.intent
    CONFIRM_FIRST = "confirm_first"  # high risk: must be confirmed deterministically first
    EXISTING_PATH = "existing_path"  # nothing from this layer; the turn proceeds as before


#: Labels whose application intent changes nothing that policy, a tool
#: precondition or identity reads. Applying one sets ``ConversationState.intent``
#: and nothing else (see :func:`app.core.session.apply_event`).
LOW_RISK_INTENTS: Mapping[CustomerIntentLabel, Intent] = MappingProxyType({
    CustomerIntentLabel.REFUSAL: Intent.REFUSAL,
    CustomerIntentLabel.CALLBACK_REQUEST: Intent.CALLBACK_REQUEST,
})

#: Labels whose application intent flips a reducer flag with consequences:
#: ``payment_promise`` and ``dispute`` unlock backend writes, ``escalation_required``
#: unlocks ``escalate_case`` and makes policy escalate, and ``wrong_person``
#: clears identity verification and closes the call. None is ever applied on a
#: semantic decision alone.
HIGH_RISK_INTENTS: Mapping[CustomerIntentLabel, Intent] = MappingProxyType({
    CustomerIntentLabel.PAYMENT_PROMISE: Intent.PAYMENT_PROMISE,
    CustomerIntentLabel.DISPUTE: Intent.DISPUTE,
    CustomerIntentLabel.ESCALATION_REQUEST: Intent.ESCALATION_REQUEST,
    CustomerIntentLabel.WRONG_PERSON: Intent.WRONG_PERSON,
})

# PAYMENT_ALREADY_MADE, REQUEST_INFORMATION, FINANCIAL_DIFFICULTY and UNCLEAR
# have no application intent. They are reported for the caller and for
# evaluation, and the turn takes the existing path: answering them needs
# language, and a payment claim needs the system of record, not a classifier.


class IntentResolution(BaseModel):
    """The fused intent outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: IntentAction
    label: CustomerIntentLabel | None = None
    application_intent: Intent | None = None
    source: DecisionSource
    semantic: DecisionResult | None = None
    reason: str

    @property
    def signal_intent(self) -> Intent | None:
        """The intent the caller may pass to the reducer. Only ever a low-risk one."""
        return self.application_intent if self.action is IntentAction.APPLY_SIGNAL else None


def fuse_intent(*, caller_intent: Intent | None, semantic: DecisionResult | None) -> IntentResolution:
    """Turn an intent classification into what the application may do with it."""
    if caller_intent is not None:
        return IntentResolution(
            action=IntentAction.EXISTING_PATH,
            source=DecisionSource.UPSTREAM,
            reason="an upstream intent was supplied; it is not second-guessed",
        )

    label = _usable(semantic, DecisionName.CUSTOMER_INTENT)
    if label is None:
        return IntentResolution(
            action=IntentAction.EXISTING_PATH,
            source=DecisionSource.HEURISTIC,
            semantic=semantic,
            reason=f"{_uncertain_reason(semantic)}; no intent signal, existing path",
        )

    kind = CustomerIntentLabel(label)
    if kind in HIGH_RISK_INTENTS:
        return IntentResolution(
            action=IntentAction.CONFIRM_FIRST,
            label=kind,
            application_intent=HIGH_RISK_INTENTS[kind],
            source=DecisionSource.SEMANTIC,
            semantic=semantic,
            reason=(
                f"{kind.value} changes state that gates a backend write, policy or identity; "
                "it must be confirmed before it is applied"
            ),
        )
    if kind in LOW_RISK_INTENTS:
        return IntentResolution(
            action=IntentAction.APPLY_SIGNAL,
            label=kind,
            application_intent=LOW_RISK_INTENTS[kind],
            source=DecisionSource.SEMANTIC,
            semantic=semantic,
            reason=f"{kind.value} sets only the recorded intent",
        )
    return IntentResolution(
        action=IntentAction.EXISTING_PATH,
        label=kind,
        source=DecisionSource.SEMANTIC,
        semantic=semantic,
        reason=f"{kind.value} has no application intent; existing path",
    )


# --- human escalation -------------------------------------------------------


class EscalationAction(str, Enum):
    ESCALATE = "escalate"  # authoritative: policy or session state already requires it
    RECOMMEND_REVIEW = "recommend_review"  # semantic only: the escalation path must confirm
    NO_CHANGE = "no_change"


class EscalationResolution(BaseModel):
    """The fused escalation outcome. Only ``authoritative`` resolutions escalate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: EscalationAction
    authoritative: bool
    source: DecisionSource
    semantic: DecisionResult | None = None
    reason: str


def escalation_already_required(policy: PolicyDecision | None, state: ConversationState) -> bool:
    """The deterministic escalation signals that exist without this layer."""
    return state.escalation_required or (policy is not None and policy.escalate)


def fuse_escalation(
    *,
    policy: PolicyDecision | None,
    state: ConversationState,
    semantic: DecisionResult | None,
) -> EscalationResolution:
    """Combine deterministic escalation signals with a semantic recommendation.

    Deterministic signals win in both directions: a required escalation stands
    whatever the decision says, and a semantic "yes" is a recommendation for
    the application's escalation path to confirm, never an escalation.
    """
    if escalation_already_required(policy, state):
        return EscalationResolution(
            action=EscalationAction.ESCALATE,
            authoritative=True,
            source=DecisionSource.POLICY,
            reason="policy or session state already requires escalation",
        )

    label = _usable(semantic, DecisionName.HUMAN_ESCALATION)
    if label is not None and EscalationLabel(label) is EscalationLabel.YES:
        return EscalationResolution(
            action=EscalationAction.RECOMMEND_REVIEW,
            authoritative=False,
            source=DecisionSource.SEMANTIC,
            semantic=semantic,
            reason="semantic decision recommends a human; the escalation path must confirm",
        )
    return EscalationResolution(
        action=EscalationAction.NO_CHANGE,
        authoritative=False,
        source=DecisionSource.SEMANTIC if label is not None else DecisionSource.HEURISTIC,
        semantic=semantic,
        reason=(
            "semantic decision: no escalation needed"
            if label is not None
            else f"{_uncertain_reason(semantic)}; deterministic signals only"
        ),
    )
