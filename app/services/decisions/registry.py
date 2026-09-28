"""The closed registry of decisions a provider may be asked for.

Three decisions, and no mechanism for a fourth that does not go through this
file. Each entry fixes, in code:

* the labels a provider may return - anything else is a malformed answer;
* the question wording and a description of every label, written literally,
  because TypeSafe documents that Jev "answers the question you wrote, not the
  one you meant";
* which :class:`~app.services.decision.DecisionContext` fields the decision may
  see - data minimisation is enforced here, not left to each caller;
* which threshold setting gates it.

The registry is provider-neutral. It describes a question; how that question
is put to a particular provider is the adapter's business
(:mod:`app.services.decision_jev`).

What is deliberately *not* registered
-------------------------------------
Nothing that the policy engine, the session or the tool registry already
decides deterministically: calling hours, recording disclosure, agent
identification, identity verification, persistent calling, prohibited conduct,
grievance holds, dispute handling, account access. Registering any of them
here would create a second, probabilistic implementation of a rule that
already has an auditable one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from app.models.enums import Language
from app.services.decision import DecisionContext, DecisionName


class BargeInLabel(str, Enum):
    """What customer speech over the agent is doing, semantically."""

    BACKCHANNEL = "backchannel"
    INTERRUPTION = "interruption"
    CONTINUATION = "continuation"


class CustomerIntentLabel(str, Enum):
    """What the customer is doing in one utterance.

    Deliberately not :class:`app.models.enums.Intent`. That enum is the state
    reducer's input vocabulary, and several of its members change state that
    gates backend writes. The mapping from these labels onto it is explicit,
    tiered by risk, and lives in :mod:`app.services.decisions.fusion`.
    """

    PAYMENT_PROMISE = "payment_promise"
    PAYMENT_ALREADY_MADE = "payment_already_made"
    DISPUTE = "dispute"
    REQUEST_INFORMATION = "request_information"
    REFUSAL = "refusal"
    FINANCIAL_DIFFICULTY = "financial_difficulty"
    WRONG_PERSON = "wrong_person"
    CALLBACK_REQUEST = "callback_request"
    ESCALATION_REQUEST = "escalation_request"
    UNCLEAR = "unclear"


class EscalationLabel(str, Enum):
    """Whether the words suggest a human should take over."""

    YES = "yes"
    NO = "no"


@dataclass(frozen=True)
class DecisionSpec:
    """One registered decision."""

    name: DecisionName
    labels: type[Enum]
    instructions: str
    criteria: Mapping[str, str]
    context_fields: frozenset[str]
    threshold_setting: str
    fallback: str

    @property
    def key(self) -> str:
        """The question's name on the wire. Stable, and equal to the decision name."""
        return self.name.value

    @property
    def label_values(self) -> frozenset[str]:
        return frozenset(self.criteria)


_BARGE_IN = DecisionSpec(
    name=DecisionName.BARGE_IN,
    labels=BargeInLabel,
    instructions=(
        "The automated agent in this phone call is speaking, and while it speaks the customer "
        "says the words in customer_utterance. Decide what the customer is doing with those "
        "words. recent_customer_utterances, when present, is what the customer said before the "
        "agent started speaking."
    ),
    criteria=MappingProxyType({
        BargeInLabel.BACKCHANNEL.value: (
            "Only a brief acknowledgement that the customer is listening and the agent should keep "
            "talking, such as 'hmm', 'yeah', 'uh-huh', 'right', 'okay', 'haan', 'achha' or 'ho'. "
            "It adds no objection, question, request or new point of its own."
        ),
        BargeInLabel.INTERRUPTION.value: (
            "The customer wants the agent to stop talking or wants to speak: telling the agent to "
            "stop or wait (for example 'stop', 'wait', 'no', 'one second', 'ruko', 'ek minute'), "
            "disagreeing or objecting, correcting the agent, asking a question, or starting a new "
            "point. An acknowledgement followed by an objection, such as 'yeah, I don't agree with "
            "that', is an interruption."
        ),
        BargeInLabel.CONTINUATION.value: (
            "The customer is finishing a sentence or point of their own from "
            "recent_customer_utterances, which they had not completed when the agent started "
            "speaking. They are continuing themselves, not reacting to the agent."
        ),
    }),
    context_fields=frozenset({"utterance", "language", "agent_speaking", "recent_customer_utterances"}),
    threshold_setting="jev_backchannel_threshold",
    fallback="HeuristicBargeInClassifier's decision, unchanged.",
)

_CUSTOMER_INTENT = DecisionSpec(
    name=DecisionName.CUSTOMER_INTENT,
    labels=CustomerIntentLabel,
    instructions=(
        "customer_utterance is what a borrower said to an automated agent on a phone call about "
        "repaying a loan or card dues. Choose the one option that best describes what the "
        "customer is doing in customer_utterance."
    ),
    criteria=MappingProxyType({
        CustomerIntentLabel.PAYMENT_PROMISE.value: (
            "The customer commits to paying all or part of the dues at a future time, for example "
            "'I will pay on Friday'."
        ),
        CustomerIntentLabel.PAYMENT_ALREADY_MADE.value: (
            "The customer says the payment has already been made, for example 'I already paid "
            "this yesterday'."
        ),
        CustomerIntentLabel.DISPUTE.value: (
            "The customer says the amount, the charges or the loan itself are wrong or not owed."
        ),
        CustomerIntentLabel.REQUEST_INFORMATION.value: (
            "The customer asks for information: the amount, the due date, the charges, the loan, "
            "or why they are being called."
        ),
        CustomerIntentLabel.REFUSAL.value: (
            "The customer refuses to pay or refuses to continue the conversation, without saying "
            "the dues are wrong."
        ),
        CustomerIntentLabel.FINANCIAL_DIFFICULTY.value: (
            "The customer says they cannot pay now because of money problems, job loss, illness "
            "or a similar hardship."
        ),
        CustomerIntentLabel.WRONG_PERSON.value: (
            "The customer says they are not the person the agent asked for, or that this is the "
            "wrong number."
        ),
        CustomerIntentLabel.CALLBACK_REQUEST.value: (
            "The customer asks to be called back at another time."
        ),
        CustomerIntentLabel.ESCALATION_REQUEST.value: (
            "The customer asks to speak to a human, a supervisor or a manager, or asks to escalate "
            "or complain."
        ),
        CustomerIntentLabel.UNCLEAR.value: (
            "None of the other options clearly applies, or the words are too short, garbled or "
            "ambiguous to tell."
        ),
    }),
    context_fields=frozenset({"utterance", "language", "stage"}),
    threshold_setting="jev_intent_threshold",
    fallback="No intent signal; the turn takes the existing path (upstream signals and the LLM).",
)

_HUMAN_ESCALATION = DecisionSpec(
    name=DecisionName.HUMAN_ESCALATION,
    labels=EscalationLabel,
    instructions=(
        "customer_utterance is what a borrower just said to an automated agent on a phone call "
        "about repaying a loan or card dues; recent_customer_utterances, when present, is what "
        "they said earlier in the call. Should a human agent take over this call?"
    ),
    criteria=MappingProxyType({
        EscalationLabel.YES.value: (
            "Yes: the customer asks for a human, a supervisor or a manager; wants to complain or "
            "escalate; keeps being misunderstood by the agent; is distressed; or describes a "
            "sensitive situation such as illness, a death in the family or serious hardship."
        ),
        EscalationLabel.NO.value: (
            "No: the customer is engaging normally - answering, asking ordinary questions, "
            "acknowledging, or disagreeing in a way the agent can handle."
        ),
    }),
    context_fields=frozenset({"utterance", "language", "recent_customer_utterances"}),
    threshold_setting="jev_escalation_threshold",
    fallback="The policy engine's and the session's escalation signals alone, as before.",
)


#: The registry. Read-only: nothing can register a decision at runtime.
DECISION_REGISTRY: Mapping[DecisionName, DecisionSpec] = MappingProxyType({
    spec.name: spec for spec in (_BARGE_IN, _CUSTOMER_INTENT, _HUMAN_ESCALATION)
})


def get_spec(name: DecisionName) -> DecisionSpec:
    """The registered spec for ``name``. A string, even a valid one, is refused."""
    if not isinstance(name, DecisionName):
        raise TypeError(
            f"decisions are named by DecisionName members, not {type(name).__name__}; "
            "an unregistered decision cannot be requested"
        )
    return DECISION_REGISTRY[name]


#: How a :class:`Language` is named to a provider. A code like ``hi-en`` is
#: something a model has to decode; a name is not.
_LANGUAGE_NAMES: Mapping[Language, str] = MappingProxyType({
    Language.ENGLISH: "English",
    Language.HINDI: "Hindi",
    Language.MARATHI: "Marathi",
    Language.HINGLISH: "Hinglish (code-mixed Hindi and English)",
})


#: Identifier-shaped spans in free text, and what replaces them. None of the
#: three decisions needs the digits of an account, phone or Aadhaar number, a
#: PAN or an email address; the fact that the customer said one is kept.
_IDENTIFIER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Anchored at a token start, so a long word is scanned once, not once per character.
    (re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<email>"),
    (re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b"), "<id>"),
    # Eight or more digits with at most two separator characters (space,
    # comma, period, hyphen) between neighbours: phone, Aadhaar, account and
    # card numbers, including ones read out digit by digit ("9, 8, 7, ...").
    (re.compile(r"\d(?:[ ,.\-]{0,2}\d){7,}"), "<number>"),
)

#: A calendar date written with spaces or hyphens also has eight digits. When a
#: run is exactly one - day-month-year or year-month-day, and a real date - it
#: is what a payment promise is about, and it is kept.
_DMY = re.compile(r"(\d{1,2})[ -](\d{1,2})[ -]((?:19|20)\d{2})")
_YMD = re.compile(r"((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})")


def _is_calendar_date(text: str) -> bool:
    for pattern, order in ((_DMY, (2, 1, 0)), (_YMD, (0, 1, 2))):
        match = pattern.fullmatch(text)
        if match:
            parts = [int(group) for group in match.groups()]
            try:
                date(parts[order[0]], parts[order[1]], parts[order[2]])
            except ValueError:
                return False
            return True
    return False


def _mask(placeholder: str):
    def replace(match: re.Match[str]) -> str:
        if placeholder == "<number>" and _is_calendar_date(match.group(0)):
            return match.group(0)
        return placeholder

    return replace


def mask_identifiers(text: str) -> str:
    """Replace identifier-shaped spans before text leaves the process.

    Best-effort data minimisation, not a guarantee. What it does, exactly:

    * masks PANs, email addresses and runs of eight or more digits separated
      by at most two of space, comma, period or hyphen;
    * keeps shorter numbers (amounts under a crore) and a run that is exactly a
      valid d-m-yyyy or yyyy-mm-dd date;
    * also masks - losing information, in the safe direction - compact dates
      ("12102026"), amounts of eight or more digits, and neighbouring numbers
      that together reach eight digits ("5000 5000");
    * misses names, numbers spoken as words ("nau aath saat"), digits separated
      by anything else or by wider gaps, identifiers shorter than eight digits,
      and an identifier that happens to be a valid date in 2-2-4 grouping.
    """
    for pattern, placeholder in _IDENTIFIER_PATTERNS:
        text = pattern.sub(_mask(placeholder), text)
    return text


def build_state(spec: DecisionSpec, context: DecisionContext) -> dict[str, Any]:
    """The state a provider is shown for ``spec``: only its declared fields.

    Field names are descriptive because the question text refers to them by
    name. Absent values are left out rather than sent as null, so a decision
    does not guess at the meaning of a missing field. Free text is passed
    through :func:`mask_identifiers`.
    """
    fields = spec.context_fields
    state: dict[str, Any] = {}
    if "utterance" in fields:
        state["customer_utterance"] = mask_identifiers(context.utterance)
    if "language" in fields and context.language is not None:
        state["language"] = _LANGUAGE_NAMES[context.language]
    if "agent_speaking" in fields and context.agent_speaking is not None:
        state["agent_is_speaking"] = context.agent_speaking
    if "stage" in fields and context.stage is not None:
        state["conversation_stage"] = context.stage.value.replace("_", " ")
    if "recent_customer_utterances" in fields and context.recent_customer_utterances:
        state["recent_customer_utterances"] = [
            mask_identifiers(item) for item in context.recent_customer_utterances
        ]
    return state


def _check_registry() -> None:
    """Fail at import if the registry and its vocabularies drift apart."""
    if set(DECISION_REGISTRY) != set(DecisionName):
        raise RuntimeError("every DecisionName needs exactly one registry entry")
    context_fields = set(DecisionContext.model_fields)
    for spec in DECISION_REGISTRY.values():
        if set(spec.criteria) != {label.value for label in spec.labels}:
            raise RuntimeError(f"{spec.name.value}: criteria must describe exactly its labels")
        if not spec.context_fields <= context_fields:
            raise RuntimeError(f"{spec.name.value}: context_fields names an unknown field")
        if "utterance" not in spec.context_fields:
            raise RuntimeError(f"{spec.name.value}: a decision must see the utterance it is about")


_check_registry()
