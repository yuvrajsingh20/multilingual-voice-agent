"""Regulatory rule schema and loader.

The rule file (``data/regulatory/rules/recovery_rules.json``) is the audit
artifact: it is what a compliance reviewer reads. This module only gives it a
type and a few invariants.

Design decisions
----------------
*Traceability* - every rule carries the corpus document id, the paragraph label
as the document itself numbers it, the local text file, and a ``quote`` that must
be a verbatim substring of that file. ``tests/test_regulatory_rules.py`` asserts
the substring property, so a rule cannot drift away from its source silently.

*Honesty about enforceability* - ``enforcement`` says whether a live call can
decide the rule at all. ``conversation_deterministic`` rules must carry a
``check_id``; ``organisational_obligation`` rules must not, because no call-time
check can settle them. ``requires_external_data`` rules may carry one, and that
check reports ``not_evaluable`` when the backend did not supply its input.
Rules the engine cannot decide are still loaded and still named in the decision
output, so they stay visible rather than being silently dropped.

*Time* - ``effective_from`` / ``effective_until`` let the future-effective 2026
amendment sit in the same file as the rules it will replace, without either
layer being enforced on the wrong date.

Only :mod:`json` is used to read the file. No YAML, no pickle, no eval.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import (
    Enforcement,
    ProductType,
    ProhibitedConduct,
    RequiredAction,
    RuleStatus,
    Severity,
)

SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})


class RuleSourceError(ValueError):
    """The rule file is missing, malformed, or violates a schema invariant."""


class RuleSource(BaseModel):
    """Pointer back into the local corpus. Enough to re-verify by hand."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1, description="id in data/regulatory/catalog.json")
    document_title: str = Field(min_length=1)
    paragraph: str = Field(min_length=1, description="Label as the document numbers it, e.g. '445' or '454Y(4)'.")
    text_file: str = Field(min_length=1, description="Repo-relative path under data/regulatory/text/.")
    quote: str = Field(min_length=1, description="Verbatim substring of text_file.")
    url: str | None = None


class ProvenanceRef(BaseModel):
    """An earlier instrument the current rule descends from. Never enforced."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    paragraph: str
    note: str


class RuleApplicability(BaseModel):
    """Structured scope test. Empty ``product_types`` means "all products"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str
    product_types: tuple[ProductType, ...] = ()
    excluded_product_types: tuple[ProductType, ...] = ()
    channels: tuple[str, ...] = ("voice",)

    def matches(self, product_type: ProductType | None) -> bool:
        """Whether this rule applies to ``product_type``.

        An unknown product (``None``) matches only rules that are not restricted
        to a product list, so a missing account context cannot accidentally
        activate a product-specific rule.
        """
        if product_type is not None and product_type in self.excluded_product_types:
            return False
        if not self.product_types:
            return True
        return product_type is not None and product_type in self.product_types


class RegulatoryRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: RuleSource
    provenance: tuple[ProvenanceRef, ...] = ()

    status: RuleStatus
    enforcement: Enforcement
    severity: Severity

    effective_from: date
    effective_until: date | None = None
    superseded_by: tuple[str, ...] = ()

    condition: str = Field(min_length=1, description="Plain statement of when the rule bites.")
    applies_when: RuleApplicability
    prohibited_action: tuple[ProhibitedConduct, ...] = ()
    required_action: tuple[RequiredAction, ...] = ()
    exceptions: tuple[str, ...] = ()

    check_id: str | None = Field(
        default=None,
        description="Deterministic check implementing this rule; None means recorded but not automated.",
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None

    @model_validator(mode="after")
    def _invariants(self) -> "RegulatoryRule":
        if self.effective_until is not None and self.effective_until < self.effective_from:
            raise ValueError(f"{self.rule_id}: effective_until precedes effective_from")
        if self.enforcement is Enforcement.CONVERSATION_DETERMINISTIC and self.check_id is None:
            raise ValueError(
                f"{self.rule_id}: a conversation_deterministic rule must declare a check_id"
            )
        if self.enforcement is Enforcement.ORGANISATIONAL_OBLIGATION and self.check_id is not None:
            raise ValueError(
                f"{self.rule_id}: an organisational obligation cannot be decided during a call, "
                f"so it must not declare a check_id"
            )
        if self.status is RuleStatus.HISTORICAL_SUPPORTING and self.check_id is not None:
            raise ValueError(f"{self.rule_id}: historical rules must never be enforced")
        return self

    def is_active_on(self, day: date) -> bool:
        """Whether the rule is in force on ``day``.

        Historical-supporting rules are never active: they are retained for
        lineage only.
        """
        if self.status is RuleStatus.HISTORICAL_SUPPORTING:
            return False
        if day < self.effective_from:
            return False
        if self.effective_until is not None and day > self.effective_until:
            return False
        return True


class RuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    rules_version: str = Field(description="Version of this encoded rule set, independent of the corpus.")
    generated_on: date
    corpus_catalog: str
    corpus_as_of: str
    scope: str
    notes: str
    rules: tuple[RegulatoryRule, ...]

    @model_validator(mode="after")
    def _invariants(self) -> "RuleSet":
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        seen: set[str] = set()
        for rule in self.rules:
            if rule.rule_id in seen:
                raise ValueError(f"duplicate rule_id: {rule.rule_id}")
            seen.add(rule.rule_id)
        for rule in self.rules:
            for successor in rule.superseded_by:
                if successor not in seen:
                    raise ValueError(f"{rule.rule_id}: superseded_by references unknown rule {successor}")
        return self

    def by_id(self, rule_id: str) -> RegulatoryRule:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        raise KeyError(rule_id)

    def active_on(self, day: date) -> tuple[RegulatoryRule, ...]:
        return tuple(rule for rule in self.rules if rule.is_active_on(day))

    def applicable(self, day: date, product_type: ProductType | None) -> tuple[RegulatoryRule, ...]:
        """Rules in force on ``day`` and in scope for ``product_type``."""
        return tuple(
            rule for rule in self.active_on(day) if rule.applies_when.matches(product_type)
        )


def load_rule_set(path: Path) -> RuleSet:
    """Read and validate the rule file. Raises :class:`RuleSourceError` on any problem."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuleSourceError(f"cannot read rule file {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuleSourceError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuleSourceError(f"{path} must contain a JSON object at the top level")
    try:
        return RuleSet.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError, or an invariant ValueError
        raise RuleSourceError(f"{path} failed validation: {exc}") from exc


@lru_cache(maxsize=4)
def load_rule_set_cached(path: Path) -> RuleSet:
    """Process-wide cached load. Use in request paths; use :func:`load_rule_set` in tests."""
    return load_rule_set(path)
