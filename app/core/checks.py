"""Deterministic checks that implement encoded regulatory rules.

One function per ``check_id``. Each takes the call context, the rule being
evaluated and the engine's configuration, and returns a
:class:`~app.models.policy.CheckResult`. Checks never produce customer-facing
text and never call a model.

Three outcomes are possible and all three matter:

``PASS``            the rule is satisfied for this turn.
``VIOLATION``       the rule is broken. ``blocks_collection`` says whether that
                    stops collection activity or only records a finding.
``NOT_EVALUABLE``   the rule's input was not supplied. The engine reports this
                    in the decision rather than assuming compliance.
"""

from __future__ import annotations

from datetime import time
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.rules import RegulatoryRule
from app.models.enums import (
    CheckStatus,
    ConversationStage,
    ProhibitedConduct,
    RequiredAction,
)
from app.models.policy import CheckResult, PolicyContext, PolicyViolation, RuleCitation

#: Stages at which the account, the dues or a settlement are actually discussed.
#: Disclosure duties must already be satisfied by the time the call reaches one.
SUBSTANTIVE_STAGES: frozenset[ConversationStage] = frozenset(
    {
        ConversationStage.ACCOUNT_DISCUSSION,
        ConversationStage.NEGOTIATION,
        ConversationStage.RESOLUTION,
    }
)


class PolicyConfig(BaseModel):
    """Bank-set knobs a check may read. Not regulatory text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str
    timezone: str = Field(
        default="Asia/Kolkata",
        description="IANA zone every calling-hour decision is made in.",
    )
    max_contacts_per_day: int | None = Field(
        default=None,
        ge=1,
        description="Bank's own persistent-calling threshold. RBI states no number.",
    )

    @field_validator("timezone")
    @classmethod
    def _zone_must_exist(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value!r}") from exc
        return value

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


CheckFn = Callable[[PolicyContext, RegulatoryRule, PolicyConfig], CheckResult]

_REGISTRY: dict[str, CheckFn] = {}


def register(check_id: str) -> Callable[[CheckFn], CheckFn]:
    def decorator(fn: CheckFn) -> CheckFn:
        if check_id in _REGISTRY:
            raise ValueError(f"duplicate check_id: {check_id}")
        _REGISTRY[check_id] = fn
        return fn

    return decorator


def get_check(check_id: str) -> CheckFn | None:
    return _REGISTRY.get(check_id)


def known_check_ids() -> frozenset[str]:
    return frozenset(_REGISTRY)


# --- helpers ---------------------------------------------------------------


def _citation(rule: RegulatoryRule) -> RuleCitation:
    return RuleCitation(
        document_id=rule.source.document_id,
        paragraph=rule.source.paragraph,
        url=rule.source.url,
    )


def _violation(
    rule: RegulatoryRule,
    detail: str,
    *,
    blocks: bool,
    conduct: ProhibitedConduct | None = None,
) -> PolicyViolation:
    return PolicyViolation(
        rule_id=rule.rule_id,
        severity=rule.severity,
        citation=_citation(rule),
        conduct=conduct,
        detail=detail,
        blocks_collection=blocks,
    )


def _passed(
    rule: RegulatoryRule,
    *,
    required: tuple[RequiredAction, ...] = (),
    conduct: tuple[ProhibitedConduct, ...] = (),
    reason: str | None = None,
) -> CheckResult:
    return CheckResult(
        rule_id=rule.rule_id,
        status=CheckStatus.PASS,
        required_actions=required,
        prohibited_conduct=conduct,
        reason=reason,
    )


def _not_evaluable(rule: RegulatoryRule, reason: str) -> CheckResult:
    return CheckResult(rule_id=rule.rule_id, status=CheckStatus.NOT_EVALUABLE, reason=reason)


def _parse_time(value: object, field: str, rule: RegulatoryRule) -> time:
    if not isinstance(value, str):
        raise ValueError(f"{rule.rule_id}: parameter {field!r} must be an 'HH:MM:SS' string")
    return time.fromisoformat(value)


# --- checks ----------------------------------------------------------------


@register("calling_hours")
def calling_hours(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """Permitted-hours check.

    The window endpoints are inclusive: RBC 2025 paragraph 445 prohibits calling
    *before* 08:00 and *after* 19:00, so 08:00:00 and 19:00:00 are themselves
    inside the window.

    ``ctx.now`` is converted into ``config.timezone`` before it is compared, so
    the decision is the same whichever zone the caller's instant happens to carry
    and the host machine's zone never takes part.
    """
    earliest = _parse_time(rule.parameters.get("earliest_local_time"), "earliest_local_time", rule)
    latest = _parse_time(rule.parameters.get("latest_local_time"), "latest_local_time", rule)

    if rule.parameters.get("express_authorisation_overrides") and ctx.compliance.borrower_authorised_out_of_hours:
        return _passed(
            rule,
            reason="Borrower expressly authorised contact outside the prescribed hours.",
        )

    local = ctx.now.astimezone(config.tzinfo).time()
    if earliest <= local <= latest:
        return _passed(rule)

    return CheckResult(
        rule_id=rule.rule_id,
        status=CheckStatus.VIOLATION,
        violation=_violation(
            rule,
            f"Contact at {local.isoformat()} ({config.timezone}) is outside the permitted "
            f"window {earliest.isoformat()}-{latest.isoformat()}.",
            blocks=True,
            conduct=ProhibitedConduct.CONTACT_OUTSIDE_PERMITTED_HOURS,
        ),
        required_actions=(RequiredAction.END_CALL,),
        prohibited_conduct=(ProhibitedConduct.CONTACT_OUTSIDE_PERMITTED_HOURS,),
    )


@register("persistent_calling")
def persistent_calling(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """Persistent-calling check.

    RBI prohibits "persistently calling" but states no number, so no threshold is
    invented here. Without both a bank-set threshold and a contact count from the
    backend, the check reports NOT_EVALUABLE.
    """
    threshold = rule.parameters.get("max_contacts_per_day") or config.max_contacts_per_day
    if threshold is None:
        return _not_evaluable(
            rule,
            "No persistent-calling threshold configured. RBI states no number; the bank must set one.",
        )
    attempts = ctx.compliance.contact_attempts_today
    if attempts is None:
        return _not_evaluable(rule, "Backend did not supply contact_attempts_today.")
    if attempts >= threshold:
        return CheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.VIOLATION,
            violation=_violation(
                rule,
                f"{attempts} recovery contacts already made today; bank threshold is {threshold}.",
                blocks=True,
                conduct=ProhibitedConduct.PERSISTENT_CALLING,
            ),
            required_actions=(RequiredAction.END_CALL,),
            prohibited_conduct=(ProhibitedConduct.PERSISTENT_CALLING,),
        )
    return _passed(rule, conduct=rule.prohibited_action)


@register("prohibited_conduct")
def prohibited_conduct(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """Publish the conduct categories the agent's output must not express.

    Nothing is decided here: a conduct prohibition is about what is *said*, and
    no text exists yet at decision time. The categories flow into the decision
    and :mod:`app.services.validation` enforces them against the draft response.
    """
    return _passed(
        rule,
        conduct=rule.prohibited_action,
        reason="Prohibition enforced against generated text by response validation.",
    )


@register("agent_identification")
def agent_identification(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """The call must not be anonymous once it becomes substantive."""
    if ctx.state.agent_identified:
        return _passed(rule, conduct=rule.prohibited_action)
    if ctx.state.current_stage in SUBSTANTIVE_STAGES:
        return CheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.VIOLATION,
            violation=_violation(
                rule,
                f"Reached stage {ctx.state.current_stage.value} without identifying the bank and the agent.",
                blocks=True,
                conduct=ProhibitedConduct.ANONYMOUS_CALL,
            ),
            required_actions=(RequiredAction.IDENTIFY_BANK_AND_AGENT,),
            prohibited_conduct=(ProhibitedConduct.ANONYMOUS_CALL,),
        )
    return _passed(
        rule,
        required=(RequiredAction.IDENTIFY_BANK_AND_AGENT,),
        conduct=rule.prohibited_action,
    )


@register("recording_disclosure")
def recording_disclosure(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """The customer must be told the conversation is being recorded."""
    if ctx.state.recording_disclosed:
        return _passed(rule)
    if ctx.state.current_stage in SUBSTANTIVE_STAGES:
        return CheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.VIOLATION,
            violation=_violation(
                rule,
                f"Reached stage {ctx.state.current_stage.value} without intimating that the call is recorded.",
                blocks=True,
            ),
            required_actions=(RequiredAction.DISCLOSE_CALL_RECORDING,),
        )
    return _passed(rule, required=(RequiredAction.DISCLOSE_CALL_RECORDING,))


@register("confidentiality_disclosure")
def confidentiality_disclosure(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """No account information before the borrower's identity is established."""
    if ctx.state.wrong_person:
        return CheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.VIOLATION,
            violation=_violation(
                rule,
                "Person on the call is not the borrower; account information must not be disclosed.",
                blocks=True,
                conduct=ProhibitedConduct.UNAUTHORISED_DISCLOSURE,
            ),
            required_actions=(RequiredAction.TERMINATE_COLLECTION_DISCUSSION, RequiredAction.END_CALL),
            prohibited_conduct=(
                ProhibitedConduct.UNAUTHORISED_DISCLOSURE,
                ProhibitedConduct.PRIVACY_INTRUSION_THIRD_PARTY,
            ),
        )
    if not ctx.state.identity_verified and ctx.state.current_stage in SUBSTANTIVE_STAGES:
        return CheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.VIOLATION,
            violation=_violation(
                rule,
                f"Reached stage {ctx.state.current_stage.value} without verifying the borrower's identity.",
                blocks=True,
                conduct=ProhibitedConduct.UNAUTHORISED_DISCLOSURE,
            ),
            required_actions=(RequiredAction.VERIFY_IDENTITY_BEFORE_DISCLOSURE,),
            prohibited_conduct=(ProhibitedConduct.UNAUTHORISED_DISCLOSURE,),
        )
    required = () if ctx.state.identity_verified else (RequiredAction.VERIFY_IDENTITY_BEFORE_DISCLOSURE,)
    return _passed(rule, required=required, conduct=rule.prohibited_action)


@register("discuss_only_with_borrower")
def discuss_only_with_borrower(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """Wrong-person handling."""
    if ctx.state.wrong_person:
        return CheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.VIOLATION,
            violation=_violation(
                rule,
                "Dues may be discussed only with the borrower or guarantor.",
                blocks=True,
                conduct=ProhibitedConduct.UNAUTHORISED_DISCLOSURE,
            ),
            required_actions=(RequiredAction.TERMINATE_COLLECTION_DISCUSSION, RequiredAction.END_CALL),
            prohibited_conduct=(ProhibitedConduct.UNAUTHORISED_DISCLOSURE,),
        )
    return _passed(rule, conduct=rule.prohibited_action)


@register("grievance_pending_hold")
def grievance_pending_hold(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """Hold recovery while a lodged grievance is undisposed.

    The frivolous/vexatious exception in paragraph 442(6) is honoured only when
    the backend asserts it, because the text requires the bank to be convinced
    "with appropriate proof". The sub judice limb says "utmost caution", not a
    prohibition, so it escalates to a human rather than blocking.
    """
    pending = ctx.compliance.grievance_pending
    if pending is None:
        return _not_evaluable(rule, "Backend did not supply grievance_pending.")

    escalate_for_sub_judice = (RequiredAction.ESCALATE_TO_HUMAN,) if ctx.compliance.sub_judice else ()

    if pending and not ctx.compliance.grievance_held_frivolous:
        return CheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.VIOLATION,
            violation=_violation(
                rule,
                "A grievance lodged by this borrower has not been finally disposed of.",
                blocks=True,
            ),
            required_actions=(
                RequiredAction.TERMINATE_COLLECTION_DISCUSSION,
                RequiredAction.ESCALATE_TO_HUMAN,
            ),
        )
    if pending:
        return _passed(
            rule,
            required=escalate_for_sub_judice,
            reason="Grievance pending but the bank holds proof of frivolous/vexatious complaints.",
        )
    return _passed(rule, required=escalate_for_sub_judice)


@register("dispute_handling")
def dispute_handling(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """Route an in-call dispute to the grievance mechanism."""
    if ctx.state.dispute:
        return _passed(
            rule,
            required=(
                RequiredAction.RECORD_DISPUTE,
                RequiredAction.PROVIDE_GRIEVANCE_MECHANISM_DETAILS,
            ),
        )
    return _passed(rule)


@register("agency_details_disclosed")
def agency_details_disclosed(ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig) -> CheckResult:
    """Borrower must have been told which agency holds the case.

    The source text obliges the bank to inform the borrower when the case is
    forwarded. It does not say contact is prohibited if that was missed, so this
    records a violation and escalates rather than blocking the call.
    """
    shared = ctx.compliance.recovery_agency_details_shared
    if shared is None:
        return _not_evaluable(rule, "Backend did not supply recovery_agency_details_shared.")
    if shared:
        return _passed(rule)
    return CheckResult(
        rule_id=rule.rule_id,
        status=CheckStatus.VIOLATION,
        violation=_violation(
            rule,
            "Borrower was not informed of the recovery agency's details for this case.",
            blocks=False,
        ),
        required_actions=(RequiredAction.ESCALATE_TO_HUMAN,),
    )


@register("digital_lending_particulars_sent")
def digital_lending_particulars_sent(
    ctx: PolicyContext, rule: RegulatoryRule, config: PolicyConfig
) -> CheckResult:
    """Digital lending: particulars must reach the borrower before contact.

    The text is explicit that communication happens "before the recovery agent
    contacts the borrower", so contact without it blocks.
    """
    sent = ctx.compliance.digital_lending_particulars_sent
    if sent is None:
        return _not_evaluable(rule, "Backend did not supply digital_lending_particulars_sent.")
    if sent:
        return _passed(rule)
    return CheckResult(
        rule_id=rule.rule_id,
        status=CheckStatus.VIOLATION,
        violation=_violation(
            rule,
            "Recovery agent particulars were not communicated to the borrower before contact.",
            blocks=True,
        ),
        required_actions=(RequiredAction.END_CALL, RequiredAction.ESCALATE_TO_HUMAN),
    )
