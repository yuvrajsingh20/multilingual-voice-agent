"""The decision boundary: contract, registry, configuration and isolation.

These tests pin what keeps the optional decision layer optional: it is off by
default, a disabled process never loads the provider SDK, a half-configured one
refuses to start, and nothing but a registered decision can ever be requested.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.metadata
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models.enums import Intent
from app.runtime import build_decision_service, build_runtime
from app.services.decision import (
    UNCERTAIN,
    DecisionAnswer,
    DecisionConfigurationError,
    DecisionContext,
    DecisionDisabled,
    DecisionName,
    DecisionOutcome,
    DecisionRequest,
    DecisionResult,
    DisabledDecisionService,
    confidence_bucket,
    failed_result,
    result_from_answer,
)
from app.services.decisions.fusion import HIGH_RISK_INTENTS, LOW_RISK_INTENTS
from app.services.decisions.registry import (
    DECISION_REGISTRY,
    CustomerIntentLabel,
    build_state,
    get_spec,
)
from tests.conftest import RULES_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
APP = REPO_ROOT / "app"


def make_settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        default_timezone="Asia/Kolkata",
        policy_version="test-0.1.0",
        regulatory_rules_path=RULES_PATH,
    )
    base.update(overrides)
    return Settings(**base)


def jev_settings(**overrides) -> Settings:
    base = dict(
        decision_provider="jev",
        jev_enabled=True,
        jev_api_key="test-key-123",
        jev_model="jev-1.13.0",
    )
    base.update(overrides)
    return make_settings(**base)


def _imports(path: Path) -> set[str]:
    """Every module a source file imports, at any depth in the file."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# --- disabled by default ------------------------------------------------------


def test_the_default_decision_service_is_the_disabled_one(runtime) -> None:
    assert isinstance(runtime.decisions, DisabledDecisionService)


def test_the_disabled_service_raises_instead_of_answering() -> None:
    request = DecisionRequest(
        name=DecisionName.CUSTOMER_INTENT, context=DecisionContext(utterance="hello")
    )
    with pytest.raises(DecisionDisabled):
        asyncio.run(DisabledDecisionService().decide(request))


def test_settings_default_to_disabled_with_no_credentials() -> None:
    settings = make_settings()
    assert settings.decision_provider == "disabled"
    assert settings.jev_enabled is False
    assert settings.jev_api_key is None
    assert settings.jev_model is None
    assert settings.jev_max_retries == 0


def _run_isolated(code: str, **env: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter, so ``sys.modules`` starts empty."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("DECISION_", "JEV_", "TYPESAFE_"))}
    clean.update(env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=clean,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_a_default_process_never_loads_the_sdk_or_the_adapter() -> None:
    result = _run_isolated(
        "import sys, app.main, app.orchestrator\n"
        "assert type(app.main.app.state.runtime.decisions).__name__ == 'DisabledDecisionService'\n"
        "print('typesafe_sdk' in sys.modules, 'app.services.decision_jev' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


def test_the_application_starts_even_when_the_sdk_is_not_installed() -> None:
    # sys.modules[name] = None makes any import of that name fail, as if uninstalled.
    result = _run_isolated(
        "import sys\n"
        "sys.modules['typesafe_sdk'] = None\n"
        "import app.main\n"
        "print(type(app.main.app.state.runtime.decisions).__name__)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "DisabledDecisionService"


def test_enabling_jev_without_the_sdk_installed_fails_clearly() -> None:
    result = _run_isolated(
        "import sys\n"
        "sys.modules['typesafe_sdk'] = None\n"
        "from tests.conftest import RULES_PATH\n"
        "from app.config import Settings\n"
        "from app.runtime import build_decision_service\n"
        "from app.services.decision import DecisionConfigurationError\n"
        "s = Settings(_env_file=None, decision_provider='jev', jev_enabled=True,\n"
        "             jev_api_key='k', jev_model='jev-1.13.0', regulatory_rules_path=RULES_PATH)\n"
        "try:\n"
        "    build_decision_service(s)\n"
        "except DecisionConfigurationError as exc:\n"
        "    print('refused:', exc)\n"
    )
    assert result.returncode == 0, result.stderr
    assert "refused:" in result.stdout
    assert "typesafe-sdk==0.7.1" in result.stdout


# --- configuration --------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "enabled"),
    [("jev", False), ("disabled", True)],
)
def test_the_two_switches_must_agree(provider: str, enabled: bool) -> None:
    settings = jev_settings(decision_provider=provider, jev_enabled=enabled)
    with pytest.raises(DecisionConfigurationError, match="set both or neither"):
        build_decision_service(settings)


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"jev_api_key": None}, "JEV_API_KEY"),
        ({"jev_model": None}, "JEV_MODEL"),
        ({"jev_api_key": "", "jev_model": "  "}, "JEV_API_KEY and JEV_MODEL"),
    ],
)
def test_enabled_jev_without_credentials_or_model_refuses_to_start(overrides, missing) -> None:
    with pytest.raises(DecisionConfigurationError, match=missing):
        build_decision_service(jev_settings(**overrides))


def test_a_half_configured_decision_layer_stops_the_runtime_being_built() -> None:
    with pytest.raises(DecisionConfigurationError):
        build_runtime(jev_settings(jev_enabled=False))


def test_fully_enabled_jev_builds_the_jev_adapter() -> None:
    from app.services.decision_jev import JevDecisionService

    service = build_decision_service(jev_settings())
    try:
        assert isinstance(service, JevDecisionService)
        assert service.provider == "jev"
        assert service.model == "jev-1.13.0"
    finally:
        asyncio.run(service.aclose())


def test_a_malformed_key_is_rejected_at_startup_without_echoing_it() -> None:
    secret = "bad key with spaces"
    with pytest.raises(DecisionConfigurationError) as excinfo:
        build_decision_service(jev_settings(jev_api_key=secret))
    assert secret not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None


def test_the_provider_name_is_case_insensitive_and_blank_means_disabled() -> None:
    assert make_settings(decision_provider="JEV").decision_provider == "jev"
    assert make_settings(decision_provider="  ").decision_provider == "disabled"
    with pytest.raises(ValidationError):
        make_settings(decision_provider="openai")


def test_blank_jev_values_are_unset() -> None:
    settings = make_settings(jev_api_key="", jev_model="   ", jev_base_url="")
    assert settings.jev_api_key is None
    assert settings.jev_model is None
    assert settings.jev_base_url is None


def test_the_jev_key_is_a_secret_and_never_rendered() -> None:
    settings = jev_settings(jev_api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in str(settings.model_dump())


def test_the_jev_base_url_must_be_http() -> None:
    with pytest.raises(ValidationError):
        make_settings(jev_base_url="ftp://api.typesafe.ai")
    assert make_settings(jev_base_url="https://gw.example/typesafe/").jev_base_url == (
        "https://gw.example/typesafe"
    )


@pytest.mark.parametrize(
    "field",
    ["jev_backchannel_threshold", "jev_intent_threshold", "jev_escalation_threshold"],
)
@pytest.mark.parametrize("value", [0.0, -0.1, 1.01])
def test_thresholds_must_be_probabilities(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_thresholds_are_separate_per_decision() -> None:
    settings = make_settings()
    thresholds = {
        name: getattr(settings, spec.threshold_setting) for name, spec in DECISION_REGISTRY.items()
    }
    assert len({spec.threshold_setting for spec in DECISION_REGISTRY.values()}) == 3
    # The escalation recommendation is the highest-stakes of the three.
    assert thresholds[DecisionName.HUMAN_ESCALATION] >= max(thresholds.values())


@pytest.mark.parametrize("value", [0, -1, 11])
def test_the_timeout_is_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        make_settings(jev_timeout_seconds=value)


def test_retries_are_bounded() -> None:
    with pytest.raises(ValidationError):
        make_settings(jev_max_retries=3)


def _env_example_decision_values() -> dict[str, str]:
    """The decision-layer assignments in .env.example, as settings keyword arguments."""
    values: dict[str, str] = {}
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.startswith(("DECISION_", "JEV_")) and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip().lower()] = value.split("#", 1)[0].strip()
    return values


def test_the_env_example_ships_the_layer_disabled() -> None:
    settings = make_settings(**_env_example_decision_values())
    assert settings.decision_provider == "disabled"
    assert settings.jev_enabled is False
    assert settings.jev_api_key is None
    assert isinstance(build_decision_service(settings), DisabledDecisionService)


def test_the_env_example_documents_every_decision_setting() -> None:
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for field in Settings.model_fields:
        if field.startswith(("decision_", "jev_")):
            assert f"\n{field.upper()}=" in text, field


def test_the_sdk_is_pinned_to_an_exact_version() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pins = [line.strip() for line in requirements if line.strip().startswith("typesafe-sdk")]
    assert pins == ["typesafe-sdk==0.7.1"]
    assert importlib.metadata.version("typesafe-sdk") == "0.7.1"


# --- isolation --------------------------------------------------------------------


def test_only_the_jev_adapter_imports_the_sdk() -> None:
    importers = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in APP.rglob("*.py")
        if any(name.split(".")[0] == "typesafe_sdk" for name in _imports(path))
    )
    assert importers == ["app/services/decision_jev.py"]


def test_only_the_runtime_imports_the_jev_adapter() -> None:
    importers = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in APP.rglob("*.py")
        if "app.services.decision_jev" in _imports(path)
    )
    assert importers == ["app/runtime.py"]


@pytest.mark.parametrize(
    "module",
    [
        "app/services/decision.py",
        "app/services/decisions/registry.py",
        "app/services/decisions/fusion.py",
        "app/services/decision_jev.py",
        "app/orchestrator/decisions.py",
    ],
)
def test_the_decision_layer_cannot_reach_tools_the_llm_or_the_policy_engine(module: str) -> None:
    """Structural: it cannot execute a tool, call the model or evaluate policy."""
    imported = _imports(REPO_ROOT / module)
    forbidden = {
        name
        for name in imported
        if name.startswith(("app.tools", "app.services.llm", "app.core.policy", "app.core.checks"))
    }
    assert forbidden == set()


def test_the_turn_pipeline_does_not_depend_on_the_decision_layer() -> None:
    """``process_turn`` is untouched: no decision is taken inside a turn."""
    imported = _imports(APP / "orchestrator" / "pipeline.py")
    assert not any("decision" in name for name in imported)


# --- registry -----------------------------------------------------------------------


def test_exactly_three_decisions_are_registered() -> None:
    assert set(DECISION_REGISTRY) == set(DecisionName)
    assert {name.value for name in DecisionName} == {
        "barge_in",
        "customer_intent",
        "needs_human_escalation",
    }


def test_the_registry_cannot_be_extended_at_runtime() -> None:
    with pytest.raises(TypeError):
        DECISION_REGISTRY["calling_hours"] = DECISION_REGISTRY[DecisionName.BARGE_IN]  # type: ignore[index]


@pytest.mark.parametrize("name", ["barge_in", "calling_hours", "anything", 1])
def test_a_runtime_value_cannot_name_a_decision(name) -> None:
    with pytest.raises(TypeError):
        get_spec(name)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        DecisionRequest(name=name, context=DecisionContext(utterance="hello"))


@pytest.mark.parametrize("spec", list(DECISION_REGISTRY.values()), ids=lambda s: s.key)
def test_every_label_is_described_and_uncertain_is_never_offered(spec) -> None:
    assert set(spec.criteria) == {label.value for label in spec.labels}
    assert all(description.strip() for description in spec.criteria.values())
    assert UNCERTAIN not in spec.label_values
    assert UNCERTAIN.lower() not in spec.label_values


def test_the_intent_labels_are_the_ones_specified() -> None:
    assert {label.name for label in CustomerIntentLabel} == {
        "PAYMENT_PROMISE",
        "PAYMENT_ALREADY_MADE",
        "DISPUTE",
        "REQUEST_INFORMATION",
        "REFUSAL",
        "FINANCIAL_DIFFICULTY",
        "WRONG_PERSON",
        "CALLBACK_REQUEST",
        "ESCALATION_REQUEST",
        "UNCLEAR",
    }


def test_no_decision_can_produce_identity_confirmation() -> None:
    mapped = set(LOW_RISK_INTENTS.values()) | set(HIGH_RISK_INTENTS.values())
    assert Intent.IDENTITY_CONFIRMED not in mapped


# --- contract types ------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["account_ref", "customer_ref", "customer_name", "phone", "amount_minor", "dpd", "identity_verified"],
)
def test_a_decision_context_cannot_carry_identity_or_account_data(field: str) -> None:
    with pytest.raises(ValidationError):
        DecisionContext(utterance="hello", **{field: "x"})


def test_the_state_shown_to_a_provider_is_only_what_the_decision_declares() -> None:
    context = DecisionContext(
        utterance="kal pay kar dunga",
        language="hi-en",
        agent_speaking=True,
        stage="negotiation",
        recent_customer_utterances=("haan",),
    )
    intent = build_state(get_spec(DecisionName.CUSTOMER_INTENT), context)
    barge = build_state(get_spec(DecisionName.BARGE_IN), context)
    escalation = build_state(get_spec(DecisionName.HUMAN_ESCALATION), context)

    assert set(intent) == {"customer_utterance", "language", "conversation_stage"}
    assert set(barge) == {"customer_utterance", "language", "agent_is_speaking", "recent_customer_utterances"}
    assert set(escalation) == {"customer_utterance", "language", "recent_customer_utterances"}


@pytest.mark.parametrize("utterance", ["", "   ", "x" * 1001])
def test_an_utterance_must_be_present_and_bounded(utterance: str) -> None:
    with pytest.raises(ValidationError):
        DecisionContext(utterance=utterance)


def test_at_most_three_earlier_utterances_are_accepted() -> None:
    with pytest.raises(ValidationError):
        DecisionContext(utterance="hi", recent_customer_utterances=("a", "b", "c", "d"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": math.nan},
        {"confidence": 1.5},
        {"confidence": -0.1},
        {"probabilities": {"dispute": math.inf}},
        {"probabilities": {"dispute": 1.2}},
    ],
)
def test_an_answer_that_is_not_a_probability_is_rejected(overrides) -> None:
    base = dict(
        name=DecisionName.CUSTOMER_INTENT, label="dispute", confidence=0.9, provider="test"
    )
    base.update(overrides)
    with pytest.raises(ValidationError):
        DecisionAnswer(**base)


def _answer(confidence: float) -> DecisionAnswer:
    return DecisionAnswer(
        name=DecisionName.CUSTOMER_INTENT, label="dispute", confidence=confidence, provider="test"
    )


def test_below_threshold_is_uncertain_and_uses_the_fallback() -> None:
    result = result_from_answer(_answer(0.79), threshold=0.8, latency_ms=3.0)
    assert result.outcome is DecisionOutcome.UNCERTAIN
    assert result.label is None
    assert result.top_label == "dispute"
    assert result.decision == UNCERTAIN
    assert result.fallback_used is True
    assert result.failure == "low_confidence"


def test_at_the_threshold_is_decided() -> None:
    result = result_from_answer(_answer(0.8), threshold=0.8, latency_ms=3.0)
    assert result.outcome is DecisionOutcome.DECIDED
    assert result.decision == "dispute"
    assert result.fallback_used is False


def test_a_failure_is_uncertain_with_its_category() -> None:
    result = failed_result(
        DecisionName.BARGE_IN, threshold=0.85, provider="jev", category="decision_timeout", latency_ms=500.0
    )
    assert result.outcome is DecisionOutcome.FAILED
    assert result.decision == UNCERTAIN
    assert result.fallback_used is True
    disabled = failed_result(
        DecisionName.BARGE_IN, threshold=0.85, provider="x", category="decision_disabled", latency_ms=0.0
    )
    assert disabled.outcome is DecisionOutcome.DISABLED


def test_the_measurement_fields_are_part_of_the_record() -> None:
    dumped = result_from_answer(_answer(0.93), threshold=0.8, latency_ms=12.5).model_dump()
    for field in ("name", "provider", "model", "latency_ms", "fallback_used", "confidence_bucket", "outcome"):
        assert field in dumped
    assert dumped["confidence_bucket"] == "ge_0.90"


@pytest.mark.parametrize(
    ("confidence", "bucket"),
    [(None, "none"), (0.1, "lt_0.50"), (0.5, "0.50_0.70"), (0.75, "0.70_0.80"), (0.85, "0.80_0.90"), (0.9, "ge_0.90"), (1.0, "ge_0.90")],
)
def test_confidence_buckets(confidence, bucket) -> None:
    assert confidence_bucket(confidence) == bucket


def test_a_result_is_immutable() -> None:
    result = result_from_answer(_answer(0.9), threshold=0.8, latency_ms=1.0)
    with pytest.raises(ValidationError):
        result.label = "refusal"  # type: ignore[misc]
    assert isinstance(result, DecisionResult)


# --- adversarial-review regressions -------------------------------------------------
#
# Each of these pins a defect the review of this stage found. See REPORT.md SD.14.


def test_the_coordinator_is_handed_nothing_it_could_misuse(runtime) -> None:
    """Not the tool registry, the LLM, the policy engine, the session store or the Runtime."""
    from app.orchestrator.decisions import DecisionCoordinator

    held = list(vars(DecisionCoordinator.from_runtime(runtime)).values())
    forbidden = (runtime, runtime.tools, runtime.llm, runtime.policy, runtime.sessions, runtime.validator)
    assert not any(value is thing for value in held for thing in forbidden)


def test_the_coordinator_module_does_not_import_the_runtime_at_run_time() -> None:
    tree = ast.parse((APP / "orchestrator" / "decisions.py").read_text(encoding="utf-8"))
    top_level = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "app.runtime" not in top_level  # only under TYPE_CHECKING


def test_a_missing_internal_module_is_not_misreported_as_a_missing_sdk() -> None:
    result = _run_isolated(
        "import sys\n"
        "from tests.conftest import RULES_PATH\n"
        "from app.config import Settings\n"
        "from app.runtime import build_decision_service\n"
        "sys.modules['app.services.decisions.registry'] = None\n"
        "s = Settings(_env_file=None, decision_provider='jev', jev_enabled=True,\n"
        "             jev_api_key='k', jev_model='jev-1.13.0', regulatory_rules_path=RULES_PATH)\n"
        "try:\n"
        "    build_decision_service(s)\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ModuleNotFoundError"
