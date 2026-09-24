"""Environment-driven application settings.

No secret has a default. ``MODEL_API_KEY`` is a ``SecretStr`` so it does not leak
through ``repr`` or the settings dump used in logs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_` is a pydantic-protected prefix; the LLM settings below opt out.
        protected_namespaces=(),
    )

    app_env: str = Field(default="local", description="local | dev | staging | prod")
    log_level: str = Field(default="INFO")

    # --- LLM boundary -----------------------------------------------------
    # Nothing is connected yet. These describe a remote OpenAI-compatible
    # endpoint so that app.services.llm can be pointed at Gemma later.
    model_base_url: str | None = Field(default=None)
    model_name: str | None = Field(default=None)
    model_api_key: SecretStr | None = Field(default=None)
    model_timeout_seconds: float = Field(default=20.0, gt=0)

    # --- Policy -----------------------------------------------------------
    default_timezone: str = Field(
        default="Asia/Kolkata",
        description="IANA zone used for every calling-hour decision. Never the host's local zone.",
    )
    policy_version: str = Field(default="0.1.0")
    regulatory_corpus_version: str = Field(
        default="2026-09-24",
        description="`as_of` of data/regulatory/catalog.json that the encoded rules were derived from.",
    )
    regulatory_rules_path: Path = Field(
        default=REPO_ROOT / "data" / "regulatory" / "rules" / "recovery_rules.json"
    )

    # Bank-configured, NOT regulatory. RBI prohibits "persistently calling" but
    # prescribes no number, so the persistent-calling check stays unevaluated
    # until the bank sets a threshold. See REPORT.md "Known Limitations".
    max_recovery_calls_per_day: int | None = Field(default=None, ge=1)

    # --- Session store ----------------------------------------------------
    max_active_sessions: int = Field(default=1000, ge=1)

    # --- Orchestrator -----------------------------------------------------
    # Tool-loop bound. Not a regulatory number and not a business rule: it is a
    # safety stop that prevents a model from driving the backend in a loop. The
    # default is 2 because the banking tools return deliberately narrow payloads
    # (see app/tools/banking.py), so the widest single question a customer can
    # ask - "what do I owe and when was it due" - needs get_outstanding_amount
    # plus get_account_status and no more. Raise it only with evidence from real
    # traffic that a turn legitimately needs more.
    max_tool_calls_per_turn: int = Field(default=2, ge=0)

    @field_validator("default_timezone")
    @classmethod
    def _timezone_must_exist(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:  # pragma: no cover - config error path
            raise ValueError(f"unknown IANA timezone: {value!r}") from exc
        return value

    @field_validator("log_level")
    @classmethod
    def _log_level_is_known(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.default_timezone)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached so the .env file is read once."""
    return Settings()
