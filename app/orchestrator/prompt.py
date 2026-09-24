"""Construction of the model request.

The orchestrator hands the model three things and nothing else: the policy
decision as structured constraints, the facts it is allowed to state, and the
customer's turn. Everything here is an *instruction to the model*; none of it is
customer-facing copy, and none of it decides anything. The decision was already
taken by :class:`~app.core.policy.PolicyEngine` before this module runs.

Language neutrality
-------------------
The system prompt is written in English because it addresses the model, not the
customer. It names the reply language as a code (``hi``, ``en``, ``mr``,
``hi-en``) and never contains a phrase for the model to echo. No sentence in
this module is ever spoken.

Grounding
---------
Amounts and dates are rendered into the prompt from backend facts only. The
prompt states explicitly that nothing outside that list may be said, and
:mod:`app.services.validation` enforces it afterwards - the instruction is a
convenience for the model, not the control.
"""

from __future__ import annotations

from app.models.customer import AccountContext
from app.models.enums import Language
from app.models.policy import PolicyDecision
from app.services.llm import LlmMessage, LlmRequest, LlmToolSpec
from app.services.validation import GroundingFacts

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
    "- Produce one short spoken turn. No markup, no lists, no stage directions.\n"
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
    lines.append(f"- reply_language: {language.value if language else 'match the customer'}")
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
    history: tuple[LlmMessage, ...] = (),
    tools: tuple[LlmToolSpec, ...] = (),
) -> LlmRequest:
    """Assemble one model request.

    ``history`` carries this turn's assistant/tool exchange back to the model
    after a tool ran. Earlier turns are not replayed: the conversation's memory
    is :class:`~app.models.conversation.ConversationState` plus backend facts,
    not a transcript buffer.
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
    return LlmRequest(messages=(system, customer, *history), tools=tools)
