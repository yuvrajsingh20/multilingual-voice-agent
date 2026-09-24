"""Structured logging foundation.

One JSON object per line so a log shipper can parse it without a grok pattern.
This is a logging foundation only - there is no metrics backend, no tracing and
no log sink configured.

Redaction
---------
:func:`redact` drops values whose field name is known to carry customer
identifiers. It is a last line of defence, not a substitute for not passing the
data in: callers should log identifiers (``customer_ref``) rather than
identities (``name``, ``phone``).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

REDACTED = "[redacted]"

#: Field names never emitted verbatim. Matching is on the lowercased field name
#: containing any of these fragments.
SENSITIVE_FIELD_FRAGMENTS: frozenset[str] = frozenset(
    {
        "aadhaar",
        "account_number",
        "address",
        "api_key",
        "authorization",
        "card_number",
        "customer_name",
        "dob",
        "email",
        "mobile",
        "pan",
        "password",
        "phone",
        "secret",
        "token",
        "transcript",
        "utterance",
    }
)

#: Observability fields the pipeline is expected to populate over time. Kept as
#: an explicit list so that a log record can be checked against it in tests and
#: so that gaps are visible rather than implicit.
STANDARD_EVENT_FIELDS: tuple[str, ...] = (
    "session_id",
    "turn_id",
    "timestamp",
    "language",
    "dpd_stage",
    "intent",
    "latency_ms",
    "tool_name",
    "tool_latency_ms",
    "policy_rule_ids",
    "model_latency_ms",
    "stt_latency_ms",
    "tts_latency_ms",
    "barge_in",
    "error_type",
)


def _is_sensitive(field_name: str) -> bool:
    lowered = field_name.lower()
    return any(fragment in lowered for fragment in SENSITIVE_FIELD_FRAGMENTS)


def redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with sensitive fields replaced.

    Nested dicts are redacted recursively. Lists of dicts are redacted
    element-wise; other list contents are left alone.
    """
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_sensitive(key):
            out[key] = REDACTED
        elif isinstance(value, dict):
            out[key] = redact(value)
        elif isinstance(value, list):
            out[key] = [redact(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


class JsonFormatter(logging.Formatter):
    """Render a record, plus any ``extra={"event_fields": {...}}``, as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        fields = getattr(record, "event_fields", None)
        if isinstance(fields, dict):
            payload.update(redact(fields))
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)


#: Third-party loggers held at WARNING regardless of the application's level.
#:
#: This is a redaction control, not noise reduction. The HTTP client logs one
#: INFO line per request containing the *full* request URL - which carries the
#: model endpoint's path and any credentials embedded in it
#: (``https://user:pass@host/v1/...``). That line is the library's own message
#: string, so :func:`redact` never sees it: redaction operates on structured
#: fields, and a formatted message has none. Level is therefore the only control
#: that works, and the application logs its own sanitised record for every model
#: call anyway (see :mod:`app.services.llm_openai`).
QUIET_LOGGERS: tuple[str, ...] = ("httpx", "httpx2", "httpcore", "httpcore2")


def configure_logging(level: str = "INFO", stream: Any = None) -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: repeated calls replace the handler rather than stacking them.
    """
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one structured event.

    ``fields`` are redacted before they reach the formatter.
    """
    logger.log(level, event, extra={"event": event, "event_fields": fields})


def missing_standard_fields(fields: Iterable[str]) -> tuple[str, ...]:
    """Standard observability fields not present in ``fields``.

    Used by tests and by operators to see which parts of the pipeline are not yet
    instrumented.
    """
    present = set(fields)
    return tuple(f for f in STANDARD_EVENT_FIELDS if f not in present)
