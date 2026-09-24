"""Tool registry: argument validation, failure modes and auditability."""

from __future__ import annotations

import json
import logging
from datetime import date

import pytest

from app.models.enums import ToolStatus
from app.models.tools import ToolRequest
from app.observability import JsonFormatter
from app.tools.banking import NullBankingBackend, build_registry
from tests.fakes import backend_with_sample_data

EXPECTED_TOOLS = {
    "get_customer_context",
    "get_account_status",
    "get_outstanding_amount",
    "get_dpd",
    "record_payment_promise",
    "create_dispute",
    "escalate_case",
}


def _request(tool: str, **arguments) -> ToolRequest:
    return ToolRequest(
        request_id="req-1", session_id="s1", turn_id=0, tool_name=tool, arguments=arguments
    )


def test_all_expected_tools_are_registered() -> None:
    assert build_registry().names == EXPECTED_TOOLS


def test_registering_the_same_name_twice_fails() -> None:
    registry = build_registry()
    from app.tools.banking import GetDpd

    with pytest.raises(ValueError, match="already registered"):
        registry.register(GetDpd(NullBankingBackend()))


def test_specs_expose_a_json_schema_per_tool() -> None:
    specs = {spec.name: spec for spec in build_registry().specs()}
    assert specs.keys() == EXPECTED_TOOLS
    schema = specs["record_payment_promise"].parameters
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"account_ref", "promise_date", "amount_minor"}


def test_unknown_tool_is_rejected() -> None:
    result = build_registry().execute(_request("definitely_not_a_tool"))
    assert result.status is ToolStatus.NOT_FOUND
    assert not result.ok


def test_missing_argument_is_rejected_before_the_backend_is_touched() -> None:
    result = build_registry(backend_with_sample_data()).execute(_request("get_dpd"))
    assert result.status is ToolStatus.INVALID_REQUEST


def test_unexpected_argument_is_rejected() -> None:
    result = build_registry(backend_with_sample_data()).execute(
        _request("get_dpd", account_ref="ACC-1", drop_table="x")
    )
    assert result.status is ToolStatus.INVALID_REQUEST


def test_wrong_argument_type_is_rejected() -> None:
    result = build_registry(backend_with_sample_data()).execute(
        _request("record_payment_promise", account_ref="ACC-1", promise_date="not-a-date", amount_minor=100)
    )
    assert result.status is ToolStatus.INVALID_REQUEST


def test_non_positive_amount_is_rejected() -> None:
    result = build_registry(backend_with_sample_data()).execute(
        _request("record_payment_promise", account_ref="ACC-1", promise_date="2026-10-05", amount_minor=0)
    )
    assert result.status is ToolStatus.INVALID_REQUEST


def test_the_default_backend_implements_nothing() -> None:
    registry = build_registry()
    for tool in sorted(EXPECTED_TOOLS):
        arguments = {"account_ref": "ACC-1"}
        if tool == "get_customer_context":
            arguments = {"customer_ref": "CUST-1"}
        elif tool == "record_payment_promise":
            arguments = {"account_ref": "ACC-1", "promise_date": "2026-10-05", "amount_minor": 100}
        elif tool in {"create_dispute", "escalate_case"}:
            arguments = {"account_ref": "ACC-1", "reason_code": "customer_request"}
        result = registry.execute(_request(tool, **arguments))
        assert result.status is ToolStatus.NOT_IMPLEMENTED, tool
        assert result.data is None


def test_a_wired_backend_returns_data() -> None:
    registry = build_registry(backend_with_sample_data())
    result = registry.execute(_request("get_outstanding_amount", account_ref="ACC-1"))
    assert result.ok
    assert result.data == {
        "currency": "INR",
        "outstanding_minor": 1_234_500,
        "minimum_due_minor": 250_000,
    }


def test_each_tool_returns_only_what_it_was_asked_for() -> None:
    """Need-to-know: the DPD tool must not hand back the whole account."""
    registry = build_registry(backend_with_sample_data())
    result = registry.execute(_request("get_dpd", account_ref="ACC-1"))
    assert result.data == {"dpd": 35}


def test_write_tools_reach_the_backend() -> None:
    backend = backend_with_sample_data()
    registry = build_registry(backend)

    promise = registry.execute(
        _request("record_payment_promise", account_ref="ACC-1", promise_date="2026-10-05", amount_minor=250_000)
    )
    dispute = registry.execute(_request("create_dispute", account_ref="ACC-1", reason_code="amount_disputed"))
    escalation = registry.execute(_request("escalate_case", account_ref="ACC-1", reason_code="customer_request"))

    assert promise.ok and dispute.ok and escalation.ok
    assert backend.promises == [("ACC-1", date(2026, 10, 5), 250_000)]
    assert backend.disputes == [("ACC-1", "amount_disputed")]
    assert backend.escalations == [("ACC-1", "customer_request")]


def test_a_backend_exception_does_not_leak_its_details() -> None:
    class ExplodingBackend(NullBankingBackend):
        def get_account(self, account_ref: str):
            raise RuntimeError("connection string postgres://user:hunter2@internal-db")

    result = build_registry(ExplodingBackend()).execute(_request("get_dpd", account_ref="ACC-1"))
    assert result.status is ToolStatus.BACKEND_ERROR
    assert result.error == "RuntimeError"
    assert "hunter2" not in (result.error or "")


def test_every_execution_is_logged_with_an_audit_trail(caplog: pytest.LogCaptureFixture) -> None:
    registry = build_registry(backend_with_sample_data())
    with caplog.at_level(logging.INFO, logger="app.tools.base"):
        registry.execute(_request("get_dpd", account_ref="ACC-1"))

    record = next(r for r in caplog.records if getattr(r, "event", None) == "tool_call")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["tool_name"] == "get_dpd"
    assert payload["status"] == "ok"
    assert payload["request_id"] == "req-1"
    assert payload["session_id"] == "s1"
    assert payload["tool_latency_ms"] >= 0


def test_argument_values_are_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    registry = build_registry(backend_with_sample_data())
    with caplog.at_level(logging.INFO, logger="app.tools.base"):
        registry.execute(_request("get_dpd", account_ref="ACC-1"))

    record = next(r for r in caplog.records if getattr(r, "event", None) == "tool_call")
    rendered = JsonFormatter().format(record)
    assert "ACC-1" not in rendered
    assert json.loads(rendered)["argument_keys"] == ["account_ref"]


def test_a_backend_error_message_never_reaches_the_log(caplog: pytest.LogCaptureFixture) -> None:
    """The backend controls this string; the audit log must carry a category only."""
    registry = build_registry(backend_with_sample_data())
    with caplog.at_level(logging.INFO, logger="app.tools.base"):
        result = registry.execute(_request("get_dpd", account_ref="ACC-SECRET-1"))

    record = next(r for r in caplog.records if getattr(r, "event", None) == "tool_call")
    rendered = JsonFormatter().format(record)
    assert "ACC-SECRET-1" not in rendered
    assert json.loads(rendered)["error_type"] == "ToolNotImplemented"
    # The descriptive message is still available to the orchestrator.
    assert "ACC-SECRET-1" in (result.error or "")
