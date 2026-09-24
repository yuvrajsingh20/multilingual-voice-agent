"""Response validation: the gate between the model and TTS.

Nothing the model produces reaches the customer without passing through here.
The checks are deterministic and each one names the rule ids it enforces, so a
blocked response can be explained.

What this is not
----------------
It is not a content-safety model. The prohibited-conduct lexicon below is small,
explicit and provisional: it covers English and romanised Hindi constructions
only, has no Devanagari or Marathi coverage, and will miss paraphrases. Its
false-negative rate is high by construction. The grounding checks - every amount
and date the agent states must come from a backend fact - are the strong part of
this module, because they are exact rather than lexical.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.core.rules import RuleSet
from app.models.customer import AccountContext
from app.models.enums import (
    Language,
    ProhibitedConduct,
    RequiredAction,
    Severity,
    SpanKind,
)
from app.models.policy import PolicyDecision
from app.services.llm import LlmToolCall
from app.services.tts import detect_spans


class ValidationCode(str, Enum):
    EMPTY_RESPONSE = "empty_response"
    PROHIBITED_CONTENT = "prohibited_content"
    UNGROUNDED_AMOUNT = "ungrounded_amount"
    UNGROUNDED_DATE = "ungrounded_date"
    UNGROUNDED_NUMBER = "ungrounded_number"
    UNSUPPORTED_PROMISE = "unsupported_promise"
    INVALID_TOOL_REQUEST = "invalid_tool_request"
    MISSING_REQUIRED_ACTION = "missing_required_action"
    COLLECTION_NOT_ALLOWED = "collection_not_allowed"


#: Required actions that must be expressed in the utterance itself, and whose
#: absence blocks the response. The rest are reported but do not block, because
#: they are satisfied by a tool call or by the orchestrator rather than by words.
BLOCKING_IN_TURN_ACTIONS: frozenset[RequiredAction] = frozenset(
    {RequiredAction.DISCLOSE_CALL_RECORDING, RequiredAction.IDENTIFY_BANK_AND_AGENT}
)

REPORTED_IN_TURN_ACTIONS: frozenset[RequiredAction] = frozenset(
    {
        RequiredAction.VERIFY_IDENTITY_BEFORE_DISCLOSURE,
        RequiredAction.PROVIDE_GRIEVANCE_MECHANISM_DETAILS,
        RequiredAction.CONFIRM_PAYMENT_PROMISE_DETAILS,
    }
)

#: Provisional. English and romanised Hindi only; needs replacement by a lexicon
#: reviewed by native Hindi and Marathi speakers before production use.
PROHIBITED_PATTERNS: dict[ProhibitedConduct, tuple[re.Pattern[str], ...]] = {
    ProhibitedConduct.VIOLENCE_THREAT: (
        re.compile(r"\bbreak your (legs|bones|hands)\b", re.I),
        re.compile(r"\bbeat you (up)?\b", re.I),
        re.compile(r"\bhaath\s*pair\s*tod", re.I),
        re.compile(r"\bmaar\s*(dal|denge|dunga)\b", re.I),
    ),
    ProhibitedConduct.THREATENING_CALL: (
        re.compile(r"\b(you|aap)\s+will\s+go\s+to\s+jail\b", re.I),
        re.compile(r"\bjail\s*(bhej|jaoge|jana)\b", re.I),
        re.compile(r"\bpolice\s*(bula|bhej|complaint\s*file)", re.I),
        re.compile(r"\bwe\s+will\s+have\s+you\s+arrested\b", re.I),
        re.compile(r"\bgiraftar\b", re.I),
    ),
    ProhibitedConduct.INTIMIDATION: (
        re.compile(r"\b(final|last)\s+warning\b", re.I),
        re.compile(r"\bdekh\s*lenge\b", re.I),
        re.compile(r"\bbura\s*anjaam\b", re.I),
        re.compile(r"\bface\s+serious\s+consequences\b", re.I),
    ),
    ProhibitedConduct.ABUSIVE_LANGUAGE: (
        re.compile(r"\bshut\s*up\b", re.I),
        re.compile(r"\b(idiot|stupid|useless)\b", re.I),
        re.compile(r"\b(chor|bewakoof|nalayak)\b", re.I),
    ),
    ProhibitedConduct.PUBLIC_HUMILIATION: (
        re.compile(r"\btell\s+your\s+(neighbou?rs|colleagues|office|employer)\b", re.I),
        re.compile(r"\bsociety\s*me(in)?\s*bata", re.I),
        re.compile(r"\binform\s+your\s+employer\b", re.I),
    ),
    ProhibitedConduct.PRIVACY_INTRUSION_THIRD_PARTY: (
        re.compile(r"\b(call|contact)\s+your\s+(family|father|mother|wife|husband|relatives|friends)\b", re.I),
        re.compile(r"\b(ghar\s*walo|rishtedaro)n?\s*ko\b", re.I),
    ),
    ProhibitedConduct.SOCIAL_MEDIA_EXPOSURE: (
        re.compile(r"\bpost\s+(it\s+)?on\s+(facebook|instagram|whatsapp|social\s*media)\b", re.I),
    ),
    ProhibitedConduct.FALSE_OR_MISLEADING_REPRESENTATION: (
        re.compile(r"\bwe\s+are\s+(from\s+)?the\s+(police|court)\b", re.I),
        re.compile(r"\bcourt\s+(order|notice)\s+(has\s+been\s+)?issued\b", re.I),
    ),
}

#: Concessions the agent may not offer unless a backend result authorises it.
_PROMISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwaiv(e|er|ed)\b", re.I),
    re.compile(r"\bwrite[\s-]?off\b", re.I),
    re.compile(r"\bsettle(ment)?\b", re.I),
    re.compile(r"\bdiscount\b", re.I),
    re.compile(r"\b(maaf|maf)\s*(kar|kr)", re.I),
)


class GroundingFacts(BaseModel):
    """The only figures the agent is allowed to state.

    Built from backend tool results. An amount or date in a draft response that is
    not here was invented by the model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    amounts_minor: tuple[int, ...] = ()
    dates: tuple[date, ...] = ()
    numbers: tuple[int, ...] = Field(
        default=(),
        description="Non-monetary figures the agent may state, such as days past due.",
    )
    allow_concession_offer: bool = Field(
        default=False,
        description="True only when a backend result actually authorises a waiver or settlement.",
    )

    @classmethod
    def from_account(cls, account: AccountContext, **extra: object) -> "GroundingFacts":
        amounts = [account.outstanding_minor]
        if account.minimum_due_minor is not None:
            amounts.append(account.minimum_due_minor)
        if account.last_payment_minor is not None:
            amounts.append(account.last_payment_minor)
        dates = [d for d in (account.due_date, account.last_payment_date) if d is not None]
        return cls(
            amounts_minor=tuple(sorted(set(amounts))),
            dates=tuple(sorted(set(dates))),
            numbers=(account.dpd,),
            **extra,  # type: ignore[arg-type]
        )


class DraftResponse(BaseModel):
    """What the model produced, before anything is spoken."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = ""
    language: Language | None = None
    tool_calls: tuple[LlmToolCall, ...] = ()
    claimed_actions: tuple[RequiredAction, ...] = Field(
        default=(), description="Required actions the orchestrator asserts this utterance performs."
    )


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ValidationCode
    severity: Severity
    detail: str
    blocking: bool
    rule_ids: tuple[str, ...] = ()
    conduct: ProhibitedConduct | None = None


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    blocked: bool
    issues: tuple[ValidationIssue, ...] = ()


def _parse_currency_minor(raw: str) -> int | None:
    match = re.search(r"\d[\d,]*(?:\.\d{1,2})?$", raw)
    if match is None:
        return None
    rupees_text, _, paise_text = match.group().replace(",", "").partition(".")
    try:
        rupees = int(rupees_text or "0")
    except ValueError:
        return None
    paise = int((paise_text + "00")[:2]) if paise_text else 0
    return rupees * 100 + paise


def _parse_plain_integer(raw: str) -> int | None:
    """Read a bare number as a whole quantity. ``None`` if it is not one."""
    cleaned = raw.replace(",", "")
    try:
        return int(cleaned)
    except ValueError:
        return None


def _parse_date(raw: str) -> date | None:
    try:
        if "-" in raw:
            return date.fromisoformat(raw)
        day, month, year = raw.split("/")
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


class ResponseValidator:
    """Deterministic gate. Construct once per process; it is stateless per call."""

    def __init__(self, rule_set: RuleSet, allowed_tools: frozenset[str]) -> None:
        self._rule_set = rule_set
        self._allowed_tools = allowed_tools

    def _rules_for(self, conduct: ProhibitedConduct, decision: PolicyDecision) -> tuple[str, ...]:
        """Rule ids that were actually evaluated this turn and prohibit ``conduct``."""
        evaluated = set(decision.rule_ids)
        return tuple(
            rule.rule_id
            for rule in self._rule_set.rules
            if rule.rule_id in evaluated and conduct in rule.prohibited_action
        )

    def validate(
        self,
        draft: DraftResponse,
        decision: PolicyDecision,
        grounding: GroundingFacts,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        text = draft.text or ""

        if not text.strip() and not draft.tool_calls:
            issues.append(
                ValidationIssue(
                    code=ValidationCode.EMPTY_RESPONSE,
                    severity=Severity.HIGH,
                    detail="Draft has neither speech nor a tool call.",
                    blocking=True,
                )
            )

        # 1. Collection blocked by policy.
        if not decision.allowed:
            claimed = set(draft.claimed_actions)
            winding_down = claimed & {
                RequiredAction.TERMINATE_COLLECTION_DISCUSSION,
                RequiredAction.END_CALL,
                RequiredAction.ESCALATE_TO_HUMAN,
            }
            if not winding_down:
                issues.append(
                    ValidationIssue(
                        code=ValidationCode.COLLECTION_NOT_ALLOWED,
                        severity=Severity.CRITICAL,
                        detail="Policy blocked collection, but the draft does not terminate or escalate.",
                        blocking=True,
                        rule_ids=tuple(v.rule_id for v in decision.violations if v.blocks_collection),
                    )
                )

        # 2. Prohibited conduct, for every category the decision put in scope.
        for conduct in decision.prohibited_conduct:
            for pattern in PROHIBITED_PATTERNS.get(conduct, ()):
                match = pattern.search(text)
                if match is None:
                    continue
                issues.append(
                    ValidationIssue(
                        code=ValidationCode.PROHIBITED_CONTENT,
                        severity=Severity.CRITICAL,
                        detail=f"Matched prohibited {conduct.value} pattern: {match.group()!r}",
                        blocking=True,
                        rule_ids=self._rules_for(conduct, decision),
                        conduct=conduct,
                    )
                )
                break

        # 3. Grounding: every amount and date spoken must be a backend fact.
        false_rep_rules = self._rules_for(ProhibitedConduct.FALSE_OR_MISLEADING_REPRESENTATION, decision)
        for span in detect_spans(text):
            if span.kind is SpanKind.CURRENCY:
                minor = _parse_currency_minor(span.raw)
                if minor is not None and minor not in grounding.amounts_minor:
                    issues.append(
                        ValidationIssue(
                            code=ValidationCode.UNGROUNDED_AMOUNT,
                            severity=Severity.CRITICAL,
                            detail=f"Amount {span.raw!r} is not backed by any backend fact.",
                            blocking=True,
                            rule_ids=false_rep_rules,
                            conduct=ProhibitedConduct.FALSE_OR_MISLEADING_REPRESENTATION,
                        )
                    )
            elif span.kind is SpanKind.DATE:
                parsed = _parse_date(span.raw)
                if parsed is not None and parsed not in grounding.dates:
                    issues.append(
                        ValidationIssue(
                            code=ValidationCode.UNGROUNDED_DATE,
                            severity=Severity.HIGH,
                            detail=f"Date {span.raw!r} is not backed by any backend fact.",
                            blocking=True,
                            rule_ids=false_rep_rules,
                            conduct=ProhibitedConduct.FALSE_OR_MISLEADING_REPRESENTATION,
                        )
                    )
            elif span.kind is SpanKind.NUMBER:
                # A figure written without a currency prefix is still a figure the
                # customer hears. Checking only CURRENCY spans would let "you owe
                # 99,999" and "you are 180 days past due" through untouched, and
                # would let an amount in paise be read out as if it were rupees.
                # A bare number is therefore allowed only when it is a known
                # non-monetary fact, or the rupee form of a known amount.
                minor = _parse_currency_minor(span.raw)
                plain = _parse_plain_integer(span.raw)
                grounded = (minor is not None and minor in grounding.amounts_minor) or (
                    plain is not None and plain in grounding.numbers
                )
                if not grounded:
                    issues.append(
                        ValidationIssue(
                            code=ValidationCode.UNGROUNDED_NUMBER,
                            severity=Severity.HIGH,
                            detail=f"Figure {span.raw!r} is not backed by any backend fact.",
                            blocking=True,
                            rule_ids=false_rep_rules,
                            conduct=ProhibitedConduct.FALSE_OR_MISLEADING_REPRESENTATION,
                        )
                    )

        # 4. Concessions the agent is not authorised to offer.
        if not grounding.allow_concession_offer:
            for pattern in _PROMISE_PATTERNS:
                match = pattern.search(text)
                if match is None:
                    continue
                issues.append(
                    ValidationIssue(
                        code=ValidationCode.UNSUPPORTED_PROMISE,
                        severity=Severity.CRITICAL,
                        detail=f"Offered a concession ({match.group()!r}) with no backend authorisation.",
                        blocking=True,
                        rule_ids=false_rep_rules,
                        conduct=ProhibitedConduct.FALSE_OR_MISLEADING_REPRESENTATION,
                    )
                )
                break

        # 5. Tool requests must name a registered tool.
        for call in draft.tool_calls:
            if call.tool_name not in self._allowed_tools:
                issues.append(
                    ValidationIssue(
                        code=ValidationCode.INVALID_TOOL_REQUEST,
                        severity=Severity.HIGH,
                        detail=f"Model requested unknown tool {call.tool_name!r}.",
                        blocking=True,
                    )
                )

        # 6. Mandatory confirmations.
        claimed = set(draft.claimed_actions)
        for action in decision.required_actions:
            if action in claimed:
                continue
            if action in BLOCKING_IN_TURN_ACTIONS:
                issues.append(
                    ValidationIssue(
                        code=ValidationCode.MISSING_REQUIRED_ACTION,
                        severity=Severity.HIGH,
                        detail=f"Required action {action.value} was not performed by this utterance.",
                        blocking=True,
                    )
                )
            elif action in REPORTED_IN_TURN_ACTIONS:
                issues.append(
                    ValidationIssue(
                        code=ValidationCode.MISSING_REQUIRED_ACTION,
                        severity=Severity.MEDIUM,
                        detail=f"Required action {action.value} was not performed by this utterance.",
                        blocking=False,
                    )
                )

        blocked = any(issue.blocking for issue in issues)
        return ValidationResult(valid=not issues, blocked=blocked, issues=tuple(issues))
