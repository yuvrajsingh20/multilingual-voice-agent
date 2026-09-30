"""Stage 3A: the Jev decision layer was removed, and nothing depends on it.

The optional decision layer - TypeSafe's Jev behind ``DecisionService``, the
``DecisionCoordinator``, its registry, fusion rules, evaluation harness and
``typesafe-sdk`` - was taken out deliberately. It was never wired into
``process_turn``, so removing it changes no turn. Nothing replaced it: the
policy engine, conversation state, identity binding, the tool registry and the
heuristic barge-in and silence turn detectors in :mod:`app.services.turn` are
what decide, and the LLM is the only model the application calls.

What is pinned
--------------
- None of the removed modules can be imported, and no Settings field, Runtime
  field or template key belongs to the layer.
- The application imports, builds, serves and shuts down with the SDK made
  unimportable, so there is no hidden runtime dependency on it.
- A deployment ``.env`` that still carries ``DECISION_PROVIDER=jev`` and
  ``JEV_*`` values starts as if they were absent: nothing reads them.
- No source, configuration or dependency file mentions Jev or TypeSafe.
- The lifespan still closes the model client on shutdown. The only test that
  drove the lifespan was one of the removed decision tests.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.runtime import Runtime, build_runtime
from app.services.llm import NotConfiguredLlmService
from tests.conftest import RULES_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent

REMOVED_MODULES = (
    "app.services.decision",
    "app.services.decision_jev",
    "app.services.decisions",
    "app.services.decisions.fusion",
    "app.services.decisions.registry",
    "app.orchestrator.decisions",
    "app.evaluation.decisions",
)

REMOVED_FILES = (
    "app/services/decision.py",
    "app/services/decision_jev.py",
    "app/services/decisions",
    "app/orchestrator/decisions.py",
    "app/evaluation/decisions.py",
    "scripts/eval_decisions.py",
    "data/eval/decision_cases.json",
)

#: Top-level packages only the Jev adapter used. ``tenacity`` came in with the SDK.
SDK_PACKAGES = ("typesafe_sdk", "tenacity")

_MENTION = re.compile(r"\bjev\b|typesafe|type_safe|decision_provider|DecisionCoordinator|DecisionService", re.I)


def _settings(**overrides) -> Settings:
    base = dict(_env_file=None, app_env="test", log_level="WARNING", regulatory_rules_path=RULES_PATH)
    base.update(overrides)
    return Settings(**base)


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_a_removed_decision_module_cannot_be_imported(module: str) -> None:
    parent = module.rpartition(".")[0]
    if parent and importlib.util.find_spec(parent) is None:
        return  # the package itself is gone, which is stronger
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize("path", REMOVED_FILES)
def test_a_removed_decision_file_is_gone(path: str) -> None:
    assert not (REPO_ROOT / path).exists()


def test_settings_carry_no_decision_layer_field() -> None:
    leftover = [
        name for name in Settings.model_fields if name.startswith(("jev_", "decision_", "typesafe_"))
    ]
    assert leftover == []


def test_the_runtime_carries_no_decision_service() -> None:
    assert [f.name for f in dataclasses.fields(Runtime) if "decision" in f.name] == []
    runtime = build_runtime(_settings())
    assert not hasattr(runtime, "decisions")


def test_the_sdk_is_not_a_requirement() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "typesafe" not in requirements
    assert "tenacity" not in requirements


def test_the_application_starts_serves_and_stops_with_the_sdk_unimportable() -> None:
    """Run in a fresh interpreter, so no earlier import can hide a dependency."""
    script = f"""
import sys

class _Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {SDK_PACKAGES!r}:
            raise ModuleNotFoundError(f"blocked: {{name}}", name=name)
        return None

sys.meta_path.insert(0, _Refuse())

from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app

settings = Settings(_env_file=None, app_env="test", log_level="WARNING",
                    regulatory_rules_path={str(RULES_PATH)!r})
with TestClient(create_app(settings)) as client:
    assert client.get("/health").status_code == 200
leaked = sorted(m for m in sys.modules if m.split(".")[0] in {SDK_PACKAGES!r})
assert leaked == [], leaked
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().endswith("ok")


def test_a_stale_dot_env_naming_jev_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator's old .env must not switch anything on, or stop the process starting."""
    for name in ("DECISION_PROVIDER", "JEV_ENABLED", "JEV_API_KEY", "JEV_MODEL", "JEV_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "DECISION_PROVIDER=jev\n"
        "JEV_ENABLED=true\n"
        "JEV_API_KEY=sk-stale-key\n"
        "JEV_MODEL=jev-1.13.0\n"
        "JEV_BASE_URL=https://api.typesafe.ai\n"
        "JEV_TIMEOUT_SECONDS=0.5\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env, regulatory_rules_path=RULES_PATH)
    runtime = build_runtime(settings)

    assert not any("jev" in name or "decision" in name for name in settings.model_dump())
    assert "sk-stale-key" not in repr(settings.model_dump())
    assert isinstance(runtime.llm, NotConfiguredLlmService)


def _source_files() -> list[Path]:
    this = Path(__file__).resolve()
    files = [
        *sorted((REPO_ROOT / "app").rglob("*.py")),
        *sorted((REPO_ROOT / "scripts").rglob("*.py")),
        *sorted((REPO_ROOT / "tests").rglob("*.py")),
    ]
    files += [
        REPO_ROOT / name
        for name in ("requirements.txt", ".env.example", "index.ts", "package.json")
        if (REPO_ROOT / name).exists()
    ]
    return [path for path in files if path.resolve() != this]


def test_no_source_configuration_or_dependency_file_mentions_the_decision_layer() -> None:
    hits = []
    for path in _source_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _MENTION.search(line):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert hits == []


def test_the_lifespan_closes_the_model_client_on_shutdown() -> None:
    class _ClosingLlm(NotConfiguredLlmService):
        closed = 0

        def close(self) -> None:
            type(self).closed += 1

    runtime = build_runtime(_settings(), llm=_ClosingLlm())
    with TestClient(create_app(runtime.settings, runtime=runtime)) as client:
        assert client.get("/health").status_code == 200
        assert _ClosingLlm.closed == 0

    assert _ClosingLlm.closed == 1
