"""Shared fixtures.

Every fixture is deterministic: a fixed clock, a fresh session store and the
repository's own rule file. No test depends on the wall clock or on the host
machine's timezone.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app.observability import configure_logging

from app.config import Settings
from app.core.checks import PolicyConfig
from app.core.clock import FixedClock
from app.core.policy import PolicyEngine
from app.core.rules import RuleSet, load_rule_set
from app.main import create_app
from app.models.conversation import ConversationState
from app.models.customer import AccountContext, ComplianceContext, CustomerContext
from app.models.policy import PolicyContext
from app.runtime import Runtime, build_runtime
from tests.fakes import backend_with_sample_data

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "data" / "regulatory" / "rules" / "recovery_rules.json"
IST = ZoneInfo("Asia/Kolkata")

#: A weekday morning inside every permitted calling window, before the Fourth
#: Amendment Directions, 2026 take effect.
DEFAULT_NOW = datetime(2026, 9, 24, 10, 30, tzinfo=IST)


@pytest.fixture(scope="session")
def rule_set() -> RuleSet:
    return load_rule_set(RULES_PATH)


@pytest.fixture
def policy_config() -> PolicyConfig:
    return PolicyConfig(policy_version="test-0.1.0", timezone="Asia/Kolkata")


@pytest.fixture
def engine(rule_set: RuleSet, policy_config: PolicyConfig) -> PolicyEngine:
    return PolicyEngine(rule_set, policy_config)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        default_timezone="Asia/Kolkata",
        policy_version="test-0.1.0",
        regulatory_rules_path=RULES_PATH,
    )


@pytest.fixture
def runtime(settings: Settings) -> Runtime:
    return build_runtime(
        settings,
        clock=FixedClock(DEFAULT_NOW),
        backend=backend_with_sample_data(),
    )


@pytest.fixture
def client(settings: Settings, runtime: Runtime) -> TestClient:
    return TestClient(create_app(settings, runtime))


@pytest.fixture
def log_stream():
    """Capture every log line as raw JSON text, then put logging back.

    ``configure_logging`` replaces the root handler outright, so a test that
    installs a capture without restoring it changes logging for the rest of the
    session. Install the capture *after* building an app: ``create_app`` calls
    ``configure_logging`` itself and would discard it.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    try:
        yield stream
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


@pytest.fixture
def make_context():
    """Build a :class:`PolicyContext` with sensible, overridable defaults."""

    def _make(
        *,
        now: datetime = DEFAULT_NOW,
        state: ConversationState | None = None,
        account: AccountContext | None = None,
        customer: CustomerContext | None = None,
        compliance: ComplianceContext | None = None,
    ) -> PolicyContext:
        return PolicyContext(
            state=state or ConversationState(session_id="sess-test"),
            now=now,
            account=account,
            customer=customer,
            compliance=compliance or ComplianceContext(),
        )

    return _make
