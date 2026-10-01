"""Environment-driven application settings.

No secret has a default. ``MODEL_API_KEY`` is a ``SecretStr`` so it does not leak
through ``repr`` or the settings dump used in logs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
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
        # A validation error never echoes the value it rejected. A mistyped
        # MODEL_BASE_URL can carry a password, and a startup error goes to the
        # process's stderr - that is, to the container's logs.
        hide_input_in_errors=True,
    )

    app_env: str = Field(default="local", description="local | dev | staging | prod")
    log_level: str = Field(default="INFO")

    # --- LLM boundary -----------------------------------------------------
    # A remote OpenAI-compatible chat-completions endpoint. Provider-neutral by
    # construction: the dialect is the API, and *which* model answers is
    # `model_name`. Nothing here names Gemma, vLLM or a cloud.
    #
    # Both `model_base_url` and `model_name` are unset by default, which keeps
    # app.runtime on NotConfiguredLlmService. That default is what stops a test
    # run, a local import or a misconfigured deploy from reaching the network:
    # with neither set, no HTTP client is ever constructed. Setting exactly one
    # of the two is a deployment mistake and is rejected at startup rather than
    # falling back to some other model.
    model_base_url: str | None = Field(
        default=None, description="e.g. http://localhost:8000/v1 - include the API version."
    )
    model_name: str | None = Field(
        default=None, description="Model identifier as the serving layer exposes it."
    )
    model_api_key: SecretStr | None = Field(
        default=None, description="Sent as `Authorization: Bearer`. Never logged."
    )
    # Bounded above as well as below: past about 1e9 s the socket layer cannot
    # represent the timeout at all, and no live call waits ten minutes.
    model_timeout_seconds: float = Field(
        default=20.0,
        gt=0,
        le=600,
        description="Total budget for one model call, shared across every attempt.",
    )
    model_connect_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        le=600,
        description="Connection-establishment budget. Falls back to model_timeout_seconds.",
    )
    model_max_output_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Cap on generated tokens. None leaves LlmRequest's own default in force.",
    )
    model_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. None leaves LlmRequest's own default in force.",
    )
    # The OpenAI chat-completions `reasoning_effort` field. Unset sends nothing,
    # so the request body is exactly what it was before this setting existed.
    # "none" is how a hybrid-reasoning model is kept out of thinking mode over
    # this dialect: Ollama maps it to think=false, which Qwen3.5 needs because
    # it thinks by default and a voice turn cannot wait for it.
    model_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = Field(
        default=None,
        description="Sent as `reasoning_effort` when set. None omits the field.",
    )
    # Bounded, and off by default. A retry on a live call is spent out of the
    # customer's silence, so failing fast and ending the turn is the safer
    # default; only clearly transient failures are ever retried. See
    # app.services.llm_openai.
    model_max_retries: int = Field(default=0, ge=0, le=3)
    model_provider: str = Field(
        default="openai-compatible",
        description="Log label only. Never affects request construction.",
    )

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

    @field_validator(
        "model_base_url",
        "model_name",
        "model_api_key",
        "model_connect_timeout_seconds",
        "model_max_output_tokens",
        "model_temperature",
        "model_reasoning_effort",
        "max_recovery_calls_per_day",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat an empty environment variable as absent.

        `MODEL_API_KEY=` in a .env file is how an operator says "no key", and
        `MODEL_BASE_URL=` is how a template ships. Carrying an empty string
        forward would build an adapter with an unusable URL or send
        `Authorization: Bearer `.

        The same holds for the optional numbers, whose default is "unset":
        `MODEL_TEMPERATURE=` in .env.example means "leave LlmRequest's default
        in force", exactly as its comment says, not "parse '' as a float". Only
        fields that are optional are listed. A blank value for a field with a
        real default (a timeout, a retry bound) is still rejected, and any
        non-blank value still has to pass its numeric bounds.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("model_base_url")
    @classmethod
    def _base_url_is_http(cls, value: str | None) -> str | None:
        """Reject a base URL that is not HTTP, loudly and at startup.

        A typo here would otherwise surface as a connection failure on a live
        call, which reads like an outage rather than like a configuration error.
        """
        if value is None:
            return None
        stripped = value.strip()
        if not stripped.startswith(("http://", "https://")):
            raise ValueError("model_base_url must start with http:// or https://")
        return stripped.rstrip("/")

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
