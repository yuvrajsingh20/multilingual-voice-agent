"""Bounded semantic decisions: the closed registry and deterministic fusion.

:mod:`.registry` fixes which decisions exist and what each may see.
:mod:`.fusion` turns a decision plus application state into what the
application does. Neither performs I/O; the provider sits behind
:class:`app.services.decision.DecisionService`.
"""

from app.services.decisions.fusion import (
    BargeInAction,
    BargeInResolution,
    DecisionSource,
    EscalationAction,
    EscalationResolution,
    IntentAction,
    IntentResolution,
    fuse_barge_in,
    fuse_escalation,
    fuse_intent,
)
from app.services.decisions.registry import (
    DECISION_REGISTRY,
    BargeInLabel,
    CustomerIntentLabel,
    DecisionSpec,
    EscalationLabel,
    build_state,
    get_spec,
)

__all__ = [
    "DECISION_REGISTRY",
    "BargeInAction",
    "BargeInLabel",
    "BargeInResolution",
    "CustomerIntentLabel",
    "DecisionSource",
    "DecisionSpec",
    "EscalationAction",
    "EscalationLabel",
    "EscalationResolution",
    "IntentAction",
    "IntentResolution",
    "build_state",
    "fuse_barge_in",
    "fuse_escalation",
    "fuse_intent",
    "get_spec",
]
