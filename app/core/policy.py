"""The policy engine.

Deterministic, and structured in, structured out. Given the same
:class:`~app.models.policy.PolicyContext` and the same rule file it returns the
same :class:`~app.models.policy.PolicyDecision`, with no model call and no
randomness.

What it does *not* do: write sentences. A decision says what is allowed, what
must happen and what must never be said; turning that into Hindi, English,
Marathi or Hinglish is the LLM's job, and the result comes back through
:mod:`app.services.validation` before it reaches TTS.
"""

from __future__ import annotations

from datetime import date

from app.core.checks import CheckFn, PolicyConfig, get_check, known_check_ids
from app.core.rules import RegulatoryRule, RuleSet
from app.models.enums import (
    CheckStatus,
    DpdStage,
    ProhibitedConduct,
    RequiredAction,
    ToneLevel,
)
from app.models.policy import CheckResult, PolicyContext, PolicyDecision, PolicyViolation

# --- business banding (NOT regulatory) -------------------------------------
# RBI defines no DPD buckets. These exist to select call strategy and tone, and
# are named after the stages the product targets: 5, 30 and 90 DPD.
DPD_STAGE_LOWER_BOUNDS: tuple[tuple[int, DpdStage], ...] = (
    (90, DpdStage.DPD_90),
    (30, DpdStage.DPD_30),
    (1, DpdStage.DPD_5),
    (0, DpdStage.CURRENT),
)

TONE_BY_STAGE: dict[DpdStage, ToneLevel] = {
    DpdStage.CURRENT: ToneLevel.NEUTRAL,
    DpdStage.DPD_5: ToneLevel.NEUTRAL,
    DpdStage.DPD_30: ToneLevel.FIRM,
    DpdStage.DPD_90: ToneLevel.FORMAL_FIRM,
}


class PolicyConfigurationError(RuntimeError):
    """A rule names a check the code does not implement, or vice versa."""


def dpd_stage_for(dpd: int) -> DpdStage:
    """Band a days-past-due value. Business banding, not a regulatory concept."""
    for lower, stage in DPD_STAGE_LOWER_BOUNDS:
        if dpd >= lower:
            return stage
    return DpdStage.CURRENT


def tone_for(stage: DpdStage | None) -> ToneLevel:
    """Tone escalates with DPD but never authorises prohibited conduct."""
    if stage is None:
        return ToneLevel.NEUTRAL
    return TONE_BY_STAGE[stage]


class PolicyEngine:
    """Evaluates the encoded rule set against one turn.

    Construction fails loudly if the rule file references a ``check_id`` that is
    not implemented: a rule that silently never runs is worse than a startup
    error.
    """

    def __init__(self, rule_set: RuleSet, config: PolicyConfig) -> None:
        missing = sorted(
            {r.check_id for r in rule_set.rules if r.check_id} - known_check_ids()
        )
        if missing:
            raise PolicyConfigurationError(f"rule file references unimplemented checks: {missing}")
        self._rules = rule_set
        self._config = config

    @property
    def rule_set(self) -> RuleSet:
        return self._rules

    @property
    def config(self) -> PolicyConfig:
        return self._config

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        if ctx.now.tzinfo is None or ctx.now.utcoffset() is None:
            raise ValueError(
                "PolicyContext.now must be timezone-aware; the calling-hour rules "
                "cannot be decided from a naive instant"
            )

        day: date = ctx.now.date()
        product = ctx.account.product_type if ctx.account else None
        applicable = self._rules.applicable(day, product)

        evaluated: list[str] = []
        not_evaluable: list[str] = []
        unenforced: list[str] = []
        violations: list[PolicyViolation] = []
        required: set[RequiredAction] = set()
        prohibited: set[ProhibitedConduct] = set()

        for rule in applicable:
            if rule.check_id is None:
                unenforced.append(rule.rule_id)
                continue
            check: CheckFn | None = get_check(rule.check_id)
            if check is None:  # pragma: no cover - prevented by the constructor
                raise PolicyConfigurationError(f"{rule.rule_id}: no check named {rule.check_id!r}")
            result = self._run(check, ctx, rule)
            if result.status is CheckStatus.NOT_EVALUABLE:
                not_evaluable.append(rule.rule_id)
            else:
                evaluated.append(rule.rule_id)
            if result.violation is not None:
                violations.append(result.violation)
            required.update(result.required_actions)
            prohibited.update(result.prohibited_conduct)

        allowed = not any(v.blocks_collection for v in violations)
        escalate = ctx.state.escalation_required or RequiredAction.ESCALATE_TO_HUMAN in required
        if escalate:
            required.add(RequiredAction.ESCALATE_TO_HUMAN)

        stage = dpd_stage_for(ctx.account.dpd) if ctx.account else None

        return PolicyDecision(
            allowed=allowed,
            escalate=escalate,
            tone=tone_for(stage),
            dpd_stage=stage,
            violations=tuple(violations),
            required_actions=tuple(sorted(required, key=lambda a: a.value)),
            prohibited_conduct=tuple(sorted(prohibited, key=lambda c: c.value)),
            rule_ids=tuple(evaluated),
            not_evaluable_rule_ids=tuple(not_evaluable),
            unenforced_rule_ids=tuple(unenforced),
            evaluated_at=ctx.now,
            policy_version=self._config.policy_version,
            rules_version=self._rules.rules_version,
        )

    def _run(self, check: CheckFn, ctx: PolicyContext, rule: RegulatoryRule) -> CheckResult:
        return check(ctx, rule, self._config)
