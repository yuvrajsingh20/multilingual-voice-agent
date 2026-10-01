""".env.example loads as-is, and loading it did not cost any validation.

The template is what an operator copies to ``.env`` on a fresh checkout. Before
this was pinned, doing exactly that stopped the application from starting:
``MODEL_CONNECT_TIMEOUT_SECONDS=``, ``MODEL_MAX_OUTPUT_TOKENS=``,
``MODEL_TEMPERATURE=`` and ``MAX_RECOVERY_CALLS_PER_DAY=`` are shipped blank -
the comments beside them say blank means "unset" - but ``Settings`` tried to
parse ``""`` as a number and raised. And ``REGULATORY_RULES_PATH`` was shipped
as a *relative* path, which pydantic resolves against the process's working
directory, not the repository: from any other directory the rule file was not
found, although the code default is absolute and correct.

What is being pinned
--------------------
- The template, copied to ``.env`` in an unrelated directory and read by the
  real dotenv parser, loads. Every value it leaves blank reads as unset, the
  inline ``# ...`` comments are not part of any value, and the rules path is
  the absolute code default. ``build_runtime`` then starts on
  :class:`NotConfiguredLlmService` - no model, no network. Copying the template
  changes no setting from its code default.
- The same holds for ``Settings(_env_file=".env.example")`` directly.
- Blank-means-unset reaches only the *optional* numbers. A blank timeout or
  retry bound is still an error, a non-blank value still has to parse and pass
  its bounds, and whitespace-only is treated as blank.
- Every assignment in the template names a real setting. ``extra="ignore"``
  would otherwise swallow a typo silently, leaving the default in force while
  the operator believes they changed it.

Every test clears the process environment of anything ``Settings`` would read,
so the result depends only on the file under test. Nothing here opens a socket.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError

from app.config import Settings
from app.runtime import build_runtime

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
CODE_DEFAULT_RULES_PATH = REPO_ROOT / "data" / "regulatory" / "rules" / "recovery_rules.json"

#: The optional numbers: default ``None``, and blank in the template.
OPTIONAL_NUMBERS = (
    "model_connect_timeout_seconds",
    "model_max_output_tokens",
    "model_temperature",
    "max_recovery_calls_per_day",
)

#: Numeric settings with a real default. A blank value for any of these is an
#: operator mistake, never "use the default".
REQUIRED_NUMBERS = (
    "model_timeout_seconds",
    "model_max_retries",
    "max_active_sessions",
    "max_tool_calls_per_turn",
)

#: An uncommented ``KEY=value`` line, optionally with ``export``.
_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
#: A commented-out ``# KEY=value`` line, the template's way of documenting a
#: setting it deliberately leaves at its code default.
_COMMENTED_ASSIGNMENT = re.compile(r"^\s*#\s*([A-Z][A-Z0-9_]*)=")


# --- an environment that only the file under test can speak into -----------


@pytest.fixture
def clean_environ(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every environment variable ``Settings`` would read.

    pydantic-settings matches names case-insensitively, so anything whose
    lower-case form is a field name goes, as does anything in the model
    namespace in case a later field joins it.
    """
    fields = set(Settings.model_fields)
    for field in fields:
        monkeypatch.delenv(field.upper(), raising=False)
    for name in list(os.environ):
        lowered = name.lower()
        if lowered in fields or lowered.startswith("model_"):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def template_cwd(tmp_path: Path, clean_environ: pytest.MonkeyPatch) -> Path:
    """A fresh working directory, away from the repository, holding the template as ``.env``."""
    shutil.copyfile(ENV_EXAMPLE, tmp_path / ".env")
    clean_environ.chdir(tmp_path)
    return tmp_path


def _env_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "small.env"
    path.write_text(text, encoding="utf-8")
    return path


def _template_values() -> dict[str, str | None]:
    """The template as the real dotenv parser reads it."""
    return dict(dotenv_values(ENV_EXAMPLE, encoding="utf-8"))


def _assert_is_the_shipped_configuration(settings: Settings) -> None:
    # No model: neither half of the endpoint, and no key.
    assert settings.model_base_url is None
    assert settings.model_name is None
    assert settings.model_api_key is None
    # The optional numbers are unset, so LlmRequest's own defaults stay in force.
    assert settings.model_connect_timeout_seconds is None
    assert settings.model_max_output_tokens is None
    assert settings.model_temperature is None
    assert settings.model_reasoning_effort is None
    assert settings.max_recovery_calls_per_day is None
    # The required numbers are the documented ones.
    assert settings.model_timeout_seconds == 20.0
    assert settings.model_max_retries == 0
    # The inline "# local | dev | ..." comments are not part of the value. A
    # free-form string like APP_ENV would carry one silently, so check exactly.
    assert settings.app_env == "local"
    assert settings.log_level == "INFO"
    # The rule file is the repository's, by absolute path, whatever the cwd.
    assert settings.regulatory_rules_path.is_absolute()
    assert settings.regulatory_rules_path.is_file()
    assert settings.regulatory_rules_path == CODE_DEFAULT_RULES_PATH


# --- the template copied to .env loads --------------------------------------


def test_the_template_copied_to_dot_env_in_another_directory_loads(template_cwd: Path) -> None:
    assert Path.cwd() == template_cwd
    assert not (template_cwd / "data").exists()

    _assert_is_the_shipped_configuration(Settings())


def test_the_runtime_starts_from_the_copied_template_with_no_model(template_cwd: Path) -> None:
    from app.services.llm import NotConfiguredLlmService

    runtime = build_runtime(Settings())

    assert isinstance(runtime.llm, NotConfiguredLlmService)
    assert runtime.rule_set is not None
    assert runtime.settings.regulatory_rules_path == CODE_DEFAULT_RULES_PATH


def test_copying_the_template_changes_no_setting_from_its_code_default(template_cwd: Path) -> None:
    """The template documents the defaults; it does not quietly override one."""
    from_template = Settings().model_dump()
    from_code = Settings(_env_file=None).model_dump()

    assert from_template == from_code


def test_every_value_the_template_leaves_blank_reads_as_unset(template_cwd: Path) -> None:
    blank = sorted(key for key, value in _template_values().items() if not (value or "").strip())
    assert blank, "the template is expected to ship some settings blank"

    settings = Settings()
    for key in blank:
        field = key.lower()
        assert Settings.model_fields[field].default is None, (
            f"{key} is blank in .env.example but has a real default; blank would not mean unset"
        )
        assert getattr(settings, field) is None, key


# --- the template read directly -----------------------------------------------


def test_the_template_loads_when_named_directly_as_the_env_file(
    tmp_path: Path, clean_environ: pytest.MonkeyPatch
) -> None:
    # Read from an unrelated directory, so a relative path could not resolve by luck.
    clean_environ.chdir(tmp_path)

    _assert_is_the_shipped_configuration(Settings(_env_file=ENV_EXAMPLE))


def test_the_template_never_pins_the_rule_file_to_the_working_directory() -> None:
    """A relative REGULATORY_RULES_PATH resolves against the cwd, not the repository."""
    path = _template_values().get("REGULATORY_RULES_PATH")

    assert path is None or Path(path).is_absolute(), path


@pytest.mark.parametrize("field", OPTIONAL_NUMBERS)
def test_each_optional_number_left_blank_in_the_template_is_unset(
    field: str, tmp_path: Path, clean_environ: pytest.MonkeyPatch
) -> None:
    assert _template_values()[field.upper()] == ""

    settings = Settings(_env_file=_env_file(tmp_path, f"{field.upper()}=\n"))

    assert getattr(settings, field) is None


# --- validation is not weakened -----------------------------------------------


@pytest.mark.parametrize("source", ["dotenv", "environ"])
@pytest.mark.parametrize("field", ["model_timeout_seconds", "model_max_retries"])
def test_a_blank_timeout_or_retry_bound_is_still_rejected(
    field: str, source: str, tmp_path: Path, clean_environ: pytest.MonkeyPatch
) -> None:
    """Blank means unset only where unset is a legal value. A timeout is not optional."""
    if source == "dotenv":
        env_file: Path | None = _env_file(tmp_path, f"{field.upper()}=\n")
    else:
        clean_environ.setenv(field.upper(), "")
        env_file = None

    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=env_file)

    assert {error["loc"][0] for error in caught.value.errors()} == {field}


@pytest.mark.parametrize("field", REQUIRED_NUMBERS)
def test_no_number_with_a_real_default_accepts_a_blank(
    field: str, tmp_path: Path, clean_environ: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=_env_file(tmp_path, f"{field.upper()}=\n"))


@pytest.mark.parametrize(
    ("line", "field"),
    [
        ("MODEL_TEMPERATURE=abc", "model_temperature"),
        ("MODEL_TEMPERATURE=3", "model_temperature"),
        ("MODEL_TEMPERATURE=-0.1", "model_temperature"),
        ("MODEL_MAX_OUTPUT_TOKENS=0", "model_max_output_tokens"),
        ("MODEL_MAX_OUTPUT_TOKENS=abc", "model_max_output_tokens"),
        ("MODEL_MAX_OUTPUT_TOKENS=-5", "model_max_output_tokens"),
        ("MODEL_CONNECT_TIMEOUT_SECONDS=0", "model_connect_timeout_seconds"),
        ("MODEL_CONNECT_TIMEOUT_SECONDS=-1", "model_connect_timeout_seconds"),
        ("MODEL_CONNECT_TIMEOUT_SECONDS=soon", "model_connect_timeout_seconds"),
        ("MAX_RECOVERY_CALLS_PER_DAY=0", "max_recovery_calls_per_day"),
        ("MAX_RECOVERY_CALLS_PER_DAY=many", "max_recovery_calls_per_day"),
        ("MODEL_MAX_RETRIES=4", "model_max_retries"),
        ("MODEL_TIMEOUT_SECONDS=0", "model_timeout_seconds"),
    ],
)
def test_a_non_blank_value_still_has_to_parse_and_pass_its_bounds(
    line: str, field: str, tmp_path: Path, clean_environ: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=_env_file(tmp_path, line + "\n"))

    assert {error["loc"][0] for error in caught.value.errors()} == {field}


@pytest.mark.parametrize(
    ("value", "field"),
    [
        (0.0, "model_temperature"),
        (2.0, "model_temperature"),
        (1, "model_max_output_tokens"),
        (1, "max_recovery_calls_per_day"),
    ],
)
def test_the_bounds_themselves_are_legal_values(
    value: float, field: str, clean_environ: pytest.MonkeyPatch
) -> None:
    assert getattr(Settings(_env_file=None, **{field: value}), field) == value


def test_valid_optional_numbers_parse_from_an_env_file(
    tmp_path: Path, clean_environ: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=_env_file(
            tmp_path,
            "MODEL_TEMPERATURE=0.35\n"
            "MODEL_MAX_OUTPUT_TOKENS=160\n"
            "MODEL_CONNECT_TIMEOUT_SECONDS=2\n"
            "MAX_RECOVERY_CALLS_PER_DAY=3\n",
        )
    )

    assert settings.model_temperature == 0.35
    assert settings.model_max_output_tokens == 160
    assert isinstance(settings.model_max_output_tokens, int)
    assert settings.model_connect_timeout_seconds == 2.0
    assert settings.max_recovery_calls_per_day == 3
    assert isinstance(settings.max_recovery_calls_per_day, int)


def test_valid_optional_numbers_parse_from_the_process_environment(
    clean_environ: pytest.MonkeyPatch,
) -> None:
    clean_environ.setenv("MODEL_TEMPERATURE", "0.35")
    clean_environ.setenv("MODEL_MAX_OUTPUT_TOKENS", "160")
    clean_environ.setenv("MODEL_CONNECT_TIMEOUT_SECONDS", "2")
    clean_environ.setenv("MAX_RECOVERY_CALLS_PER_DAY", "3")

    settings = Settings(_env_file=None)

    assert settings.model_temperature == 0.35
    assert settings.model_max_output_tokens == 160
    assert settings.model_connect_timeout_seconds == 2.0
    assert settings.max_recovery_calls_per_day == 3


@pytest.mark.parametrize("field", OPTIONAL_NUMBERS)
def test_a_whitespace_only_optional_number_from_the_environment_reads_as_unset(
    field: str, clean_environ: pytest.MonkeyPatch
) -> None:
    clean_environ.setenv(field.upper(), "   ")

    assert getattr(Settings(_env_file=None), field) is None


@pytest.mark.parametrize("field", OPTIONAL_NUMBERS)
def test_a_quoted_whitespace_only_optional_number_in_an_env_file_reads_as_unset(
    field: str, tmp_path: Path, clean_environ: pytest.MonkeyPatch
) -> None:
    """Quoting keeps the spaces through the dotenv parser, so the validator must strip them."""
    assert dotenv_values(_env_file(tmp_path, f'{field.upper()}="   "\n'))[field.upper()] == "   "

    assert getattr(Settings(_env_file=tmp_path / "small.env"), field) is None


@pytest.mark.parametrize("field", OPTIONAL_NUMBERS)
def test_a_whitespace_only_optional_number_passed_directly_reads_as_unset(
    field: str, clean_environ: pytest.MonkeyPatch
) -> None:
    assert getattr(Settings(_env_file=None, **{field: "   "}), field) is None


# --- every assignment in the template is a real setting -------------------------


def _uncommented_keys() -> list[str]:
    keys = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = _ASSIGNMENT.match(line)
        if match:
            keys.append(match.group(1))
    return keys


def test_every_uncommented_assignment_in_the_template_names_a_real_setting() -> None:
    """``extra="ignore"`` would hide a typo like MODEL_TEMPRATURE= and keep the default."""
    keys = _uncommented_keys()
    assert keys, "no assignments found in .env.example"

    unknown = [key for key in keys if key.lower() not in Settings.model_fields]
    assert unknown == []
    assert all(key == key.upper() for key in keys), keys
    assert len(keys) == len(set(keys)), "a setting is assigned twice"


def test_the_dotenv_parser_sees_exactly_the_assignments_the_template_shows() -> None:
    """Every key the real parser yields is a line the template visibly sets, and vice versa."""
    assert sorted(_template_values()) == sorted(_uncommented_keys())


def test_every_setting_is_documented_in_the_template_set_or_commented_out() -> None:
    """A setting left at its default is still shown, commented out, so an operator can find it."""
    documented: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = _ASSIGNMENT.match(line) or _COMMENTED_ASSIGNMENT.match(line)
        if match:
            documented.add(match.group(1).lower())

    assert sorted(set(Settings.model_fields) - documented) == []


def test_the_commented_rules_path_example_is_the_repository_default() -> None:
    """The commented example names the same file the absolute code default does."""
    examples = [
        match.group(0)
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^\s*#\s*REGULATORY_RULES_PATH=(\S+)\s*$", line))
    ]
    assert len(examples) == 1
    example = examples[0].split("=", 1)[1].strip()

    assert (REPO_ROOT / example).resolve() == CODE_DEFAULT_RULES_PATH
    assert Settings.model_fields["regulatory_rules_path"].default == CODE_DEFAULT_RULES_PATH
    assert CODE_DEFAULT_RULES_PATH.is_absolute()
