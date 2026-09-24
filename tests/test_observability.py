"""Structured logging and redaction."""

from __future__ import annotations

import io
import json
import logging

from app.observability import (
    REDACTED,
    STANDARD_EVENT_FIELDS,
    JsonFormatter,
    configure_logging,
    get_logger,
    log_event,
    missing_standard_fields,
    redact,
)


def test_sensitive_fields_are_redacted() -> None:
    out = redact({"customer_name": "A", "phone": "9876543210", "session_id": "s1"})
    assert out == {"customer_name": REDACTED, "phone": REDACTED, "session_id": "s1"}


def test_redaction_recurses_into_nested_structures() -> None:
    out = redact({"outer": {"email": "a@b.c", "dpd": 30}, "items": [{"pan": "X"}, {"turn_id": 1}]})
    assert out["outer"] == {"email": REDACTED, "dpd": 30}
    assert out["items"] == [{"pan": REDACTED}, {"turn_id": 1}]


def test_transcript_text_is_redacted() -> None:
    assert redact({"transcript": "mera naam"})["transcript"] == REDACTED
    assert redact({"user_utterance": "hello"})["user_utterance"] == REDACTED


def test_non_sensitive_observability_fields_survive() -> None:
    fields = {"session_id": "s", "turn_id": 1, "latency_ms": 12.5, "dpd_stage": "dpd_30"}
    assert redact(fields) == fields


def test_log_output_is_one_json_object_per_line() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    log_event(get_logger("test.logger"), "policy_decision", session_id="s1", turn_id=2)

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "policy_decision"
    assert payload["session_id"] == "s1"
    assert payload["turn_id"] == 2
    assert "timestamp" in payload and "level" in payload


def test_configure_logging_is_idempotent() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    configure_logging("INFO", stream=stream)
    log_event(get_logger("test.logger"), "ping")
    assert len(stream.getvalue().strip().splitlines()) == 1


def test_formatter_records_the_exception_type_not_the_message() -> None:
    logger = logging.getLogger("test.errors")
    try:
        raise ValueError("secret detail")
    except ValueError:
        record = logger.makeRecord(
            "test.errors", logging.ERROR, __file__, 1, "failed", None, exc_info=(ValueError, ValueError("x"), None)
        )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["error_type"] == "ValueError"


def test_missing_standard_fields_reports_the_gap() -> None:
    assert missing_standard_fields(STANDARD_EVENT_FIELDS) == ()
    missing = missing_standard_fields(["session_id", "turn_id"])
    assert "tts_latency_ms" in missing and "session_id" not in missing
