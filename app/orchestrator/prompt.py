"""Construction of the model request.

The orchestrator hands the model three things and nothing else: the policy
decision as structured constraints, the facts it is allowed to state, and the
customer's turn. Everything here is an *instruction to the model*; none of it is
customer-facing copy, and none of it decides anything. The decision was already
taken by :class:`~app.core.policy.PolicyEngine` before this module runs.

Language neutrality
-------------------
The system prompt is written in English because it addresses the model, not the
customer. It names the reply language by code and describes the expected script
and mix in English, and never contains a phrase for the model to echo. No
sentence in this module is ever spoken.

Memory
------
The last few spoken exchanges of the call are replayed as chat messages, so the
model does not re-ask what the customer already answered. The facts it may state
still come only from FACTS; a replayed turn grounds nothing.

Grounding
---------
Amounts and dates are rendered into the prompt from backend facts only. The
prompt states explicitly that nothing outside that list may be said, and
:mod:`app.services.validation` enforces it afterwards - the instruction is a
convenience for the model, not the control.
"""

from __future__ import annotations

from typing import Iterable

from app.models.conversation import ConversationEvent
from app.models.customer import AccountContext
from app.models.enums import EventKind, Language, RequiredAction
from app.models.policy import PolicyDecision
from app.services.llm import LlmMessage, LlmRequest, LlmToolSpec
from app.services.validation import GroundingFacts

#: Spoken exchanges (one customer turn plus one agent turn) replayed to the model.
MAX_REPLAYED_EXCHANGES = 6

#: How each reply language is described to the model. Code-mixed languages need
#: the script spelled out: given only "hi-en", small models answer in English or
#: switch to Devanagari.
_LANGUAGE_INSTRUCTIONS: dict[Language, str] = {
    Language.ENGLISH: "English.",
    Language.HINDI: "Hindi, in Devanagari script.",
    Language.MARATHI: "Marathi, in Devanagari script. Marathi, not Hindi.",
    Language.HINGLISH: (
        "Hinglish: conversational Hindi mixed with common English words, written in "
        "Latin (Roman) script. Not pure English, and not Devanagari."
    ),
    Language.MARATHI_ENGLISH: (
        "Marathi-English: conversational Marathi mixed with common English words, "
        "written in Latin (Roman) script. Use Marathi grammar and words "
        "(tumhi, aahe, kara, nahi), never Hindi (aap, hai, karo)."
    ),
}

#: What each required action asks of this turn, for the actions the model itself
#: carries out. A bare action code was not enough: given only "record_dispute",
#: the model kept quoting the balance to a customer who said they had paid.
_ACTION_INSTRUCTIONS: dict[RequiredAction, str] = {
    RequiredAction.RECORD_DISPUTE: (
        "The customer disputes the dues (for example, says they already paid or the loan "
        "is not theirs). Call create_dispute now. Stop all recovery: do not state any "
        "amount, do not ask for payment, do not argue. Acknowledge, say the dispute is "
        "noted for review, and close politely."
    ),
    RequiredAction.PROVIDE_GRIEVANCE_MECHANISM_DETAILS: (
        "Tell the customer they can raise a complaint through the bank's grievance "
        "process. Do not invent a phone number, address or reference."
    ),
    RequiredAction.ESCALATE_TO_HUMAN: (
        "Call escalate_case now and tell the customer a bank officer will follow up. "
        "Do not ask for payment in this turn."
    ),
    RequiredAction.TERMINATE_COLLECTION_DISCUSSION: (
        "Do not discuss the dues or ask for payment."
    ),
}

_SYSTEM_PREAMBLE = (
    "You are a debt-recovery voice agent for a commercial bank in India. "
    "You speak; you do not decide. Compliance decisions have already been made by "
    "the application and are given to you below as constraints.\n"
    "\n"
    "Hard rules:\n"
    "- State only the facts listed under FACTS. If a fact you need is not listed, "
    "say that you will check and do not invent a value.\n"
    "- Never state an amount, date, account status or days-past-due that is not under FACTS.\n"
    "- Write an amount exactly as FACTS writes it, keeping the currency prefix "
    "(for example INR 12,345.00), and write a date as YYYY-MM-DD. Grounding is checked "
    "on that written form: an amount written as a bare number cannot be verified and "
    "must not be used.\n"
    "- Never offer a waiver, settlement, discount or write-off.\n"
    "- Never threaten, intimidate, abuse, involve third parties, or claim legal or police action.\n"
    "- To read a fact from the bank's systems, request a tool. Do not answer from memory.\n"
    "- Never ask the customer for an account or reference number: the application "
    "already knows which account this call is about.\n"
    "- Do not repeat what you said earlier in the call; respond to what the customer "
    "just said.\n"
    "- Produce one short spoken turn, at most two sentences. No markup, no lists, "
    "no stage directions.\n"
)


def _format_minor(amount_minor: int, currency: str) -> str:
    """Render a minor-unit amount the way the validator parses it back."""
    return f"{currency} {amount_minor // 100:,}.{amount_minor % 100:02d}"


def build_system_prompt(
    decision: PolicyDecision,
    grounding: GroundingFacts,
    *,
    language: Language | None,
    currency: str,
    context_available: bool,
    account: AccountContext | None = None,
) -> str:
    """Render the constraint block. Deterministic: same inputs, same string.

    ``account`` supplies the *names* of the facts. ``grounding`` remains what the
    validator enforces afterwards; this function only decides how the same facts
    are presented to the model.
    """
    lines = [_SYSTEM_PREAMBLE, "CONSTRAINTS"]
    if language is None:
        lines.append("- reply_language: match the customer")
    else:
        lines.append(f"- reply_language: {language.value} - {_LANGUAGE_INSTRUCTIONS[language]}")
    lines.append(f"- tone: {decision.tone.value}")
    lines.append(f"- collection_allowed: {str(decision.allowed).lower()}")
    if decision.escalate:
        lines.append("- escalation_required: true")
    if decision.required_actions:
        lines.append(
            "- required_actions: " + ", ".join(a.value for a in decision.required_actions)
        )
    if decision.prohibited_conduct:
        lines.append(
            "- must_never_express: " + ", ".join(c.value for c in decision.prohibited_conduct)
        )
    instructions = [
        _ACTION_INSTRUCTIONS[a] for a in decision.required_actions if a in _ACTION_INSTRUCTIONS
    ]
    if instructions:
        lines.append("")
        lines.append("THIS TURN")
        lines.extend(f"- {text}" for text in instructions)

    lines.append("")
    lines.append("FACTS")
    if not context_available or account is None:
        lines.append("- none: no account facts have been loaded for this call.")
    else:
        # Each fact is named. An unlabelled list of figures invites the model to
        # present the last payment as the balance, and grounding is set
        # membership - it would not catch that, because the number is genuine.
        lines.append(f"- outstanding: {_format_minor(account.outstanding_minor, currency)}")
        if account.minimum_due_minor is not None:
            lines.append(f"- minimum_due: {_format_minor(account.minimum_due_minor, currency)}")
        if account.last_payment_minor is not None:
            lines.append(
                f"- last_payment_amount: {_format_minor(account.last_payment_minor, currency)}"
            )
        if account.due_date is not None:
            lines.append(f"- due_date: {account.due_date.isoformat()}")
        if account.last_payment_date is not None:
            lines.append(f"- last_payment_date: {account.last_payment_date.isoformat()}")
        lines.append(f"- days_past_due: {account.dpd}")

    account_dates = {
        day
        for day in ((account.due_date, account.last_payment_date) if account else ())
        if day is not None
    }
    for day in grounding.dates:
        if day not in account_dates:
            lines.append(f"- promised_payment_date: {day.isoformat()}")
    lines.append(
        "- concessions_authorised: "
        f"{str(grounding.allow_concession_offer).lower()}"
    )
    return "\n".join(lines)


def build_llm_request(
    *,
    decision: PolicyDecision,
    grounding: GroundingFacts,
    language: Language | None,
    currency: str,
    context_available: bool,
    account: AccountContext | None = None,
    transcript: str,
    prior_turns: tuple[LlmMessage, ...] = (),
    history: tuple[LlmMessage, ...] = (),
    tools: tuple[LlmToolSpec, ...] = (),
) -> LlmRequest:
    """Assemble one model request.

    ``prior_turns`` are earlier spoken exchanges of the call (see
    :func:`replayed_turns`). ``history`` carries this turn's assistant/tool
    exchange back to the model after a tool ran.
    """
    system = LlmMessage(
        role="system",
        content=build_system_prompt(
            decision,
            grounding,
            language=language,
            currency=currency,
            context_available=context_available,
            account=account,
        ),
    )
    customer = LlmMessage(role="user", content=transcript)
    return LlmRequest(messages=(system, *prior_turns, customer, *history), tools=tools)


def replayed_turns(
    events: Iterable[ConversationEvent], *, limit: int = MAX_REPLAYED_EXCHANGES
) -> tuple[LlmMessage, ...]:
    """Earlier spoken turns of the call, oldest first, as chat messages.

    Everything from the most recent customer utterance onwards is the current
    turn and is excluded; the caller sends that transcript itself. Only text
    that was actually heard or spoken is replayed: a customer utterance, or an
    agent utterance, which is recorded (in its written form) only once it
    passed validation.
    """
    events = list(events)
    current = max(
        (i for i, event in enumerate(events) if event.kind is EventKind.USER_UTTERANCE),
        default=len(events),
    )
    messages = [
        LlmMessage(
            role="user" if event.kind is EventKind.USER_UTTERANCE else "assistant",
            content=event.text,
        )
        for event in events[:current]
        if event.kind in (EventKind.USER_UTTERANCE, EventKind.AGENT_UTTERANCE) and event.text
    ]
    return tuple(messages[-2 * limit :]) if limit > 0 else ()
