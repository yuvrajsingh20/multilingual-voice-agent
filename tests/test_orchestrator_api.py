"""Stage 2 over HTTP.

The orchestrator is reachable at ``POST /conversation/{session_id}/turn``. These
tests drive it the way a telephony bridge would: create a session, then push one
customer turn at a time.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core.clock import FixedClock
from app.models.customer import ComplianceContext
from app.main import create_app
from app.runtime import build_runtime
from app.services.llm import ScriptedLlmService
from tests.conftest import DEFAULT_NOW
from tests.fakes import (
    backend_with_sample_data,
    sample_account,
    sample_customer,
    text_generation,
    tool_generation,
)

GROUNDED_REPLY = "Your outstanding balance is INR 12,345.00."


def client_with(settings, generations) -> TestClient:
    runtime = build_runtime(
        settings,
        clock=FixedClock(DEFAULT_NOW),
        backend=backend_with_sample_data(),
        llm=ScriptedLlmService(list(generations)),
    )
    return TestClient(create_app(settings, runtime))


def start_call(client: TestClient) -> str:
    response = client.post(
        "/conversation/session",
        json={
            "language": "en",
            "customer": sample_customer().model_dump(mode="json"),
            "account": sample_account().model_dump(mode="json"),
            "compliance": ComplianceContext(grievance_pending=False).model_dump(mode="json"),
        },
    )
    assert response.status_code == 201
    return response.json()["session_id"]


@pytest.fixture
def turn_client(settings) -> TestClient:
    return client_with(settings, [text_generation(GROUNDED_REPLY)])


def test_health_still_works(turn_client: TestClient) -> None:
    assert turn_client.get("/health").json() == {"status": "ok"}


def test_a_turn_over_http_returns_a_speakable_result(turn_client: TestClient) -> None:
    session_id = start_call(turn_client)

    response = turn_client.post(
        f"/conversation/{session_id}/turn",
        json={
            "text": "How much do I owe?",
            "language": "en",
            "intent": "query_outstanding",
            "identity_verified": True,
            "disclosed_recording": True,
            "identified_agent": True,
            "stage": "account_discussion",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "completed"
    assert body["speakable"] is True
    assert "twelve thousand three hundred forty-five rupees" in body["response_text"]
    assert body["turn_id"] == 1
    assert body["errors"] == []
    assert body["latency"]["total_ms"] > 0


def test_a_turn_for_an_unknown_session_is_a_404(turn_client: TestClient) -> None:
    response = turn_client.post(
        "/conversation/not-a-session/turn", json={"text": "hello"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "unknown session"


def test_an_unknown_field_on_a_turn_request_is_rejected(turn_client: TestClient) -> None:
    session_id = start_call(turn_client)
    response = turn_client.post(
        f"/conversation/{session_id}/turn",
        json={"text": "hello", "allow_everything": True},
    )
    assert response.status_code == 422


def test_a_policy_blocked_turn_is_reported_not_spoken(settings) -> None:
    runtime = build_runtime(
        settings,
        clock=FixedClock(DEFAULT_NOW),
        backend=backend_with_sample_data(),
        llm=ScriptedLlmService([text_generation("never generated")]),
    )
    client = TestClient(create_app(settings, runtime))
    response = client.post(
        "/conversation/session",
        json={
            "language": "en",
            "account": sample_account().model_dump(mode="json"),
            "compliance": ComplianceContext(grievance_pending=True).model_dump(mode="json"),
        },
    )
    session_id = response.json()["session_id"]

    body = client.post(
        f"/conversation/{session_id}/turn",
        json={"text": "Yes, speaking.", "identity_verified": True},
    ).json()

    assert body["outcome"] == "policy_blocked"
    assert body["speakable"] is False
    assert body["response_text"] is None
    assert body["llm_calls"] == 0
    assert "terminate_collection_discussion" in body["required_actions"]


def test_a_multi_turn_call_over_http_keeps_its_state(settings) -> None:
    client = client_with(
        settings,
        [
            text_generation("Thank you for confirming."),
            text_generation("Noted, Friday works."),
            text_generation(GROUNDED_REPLY),
        ],
    )
    session_id = start_call(client)

    base = {
        "language": "en",
        "identity_verified": True,
        "disclosed_recording": True,
        "identified_agent": True,
        "stage": "account_discussion",
    }
    first = client.post(
        f"/conversation/{session_id}/turn",
        json={**base, "text": "Yes, speaking.", "intent": "identity_confirmed"},
    ).json()
    second = client.post(
        f"/conversation/{session_id}/turn",
        json={
            **base,
            "text": "I can pay on Friday.",
            "intent": "payment_promise",
            "promise_date": "2026-09-25",
        },
    ).json()
    third = client.post(
        f"/conversation/{session_id}/turn",
        json={**base, "text": "How much do I have to pay?", "intent": "query_outstanding"},
    ).json()

    assert [first["turn_id"], second["turn_id"], third["turn_id"]] == [1, 2, 3]
    assert third["state"]["payment_promise"] is True
    assert third["state"]["promise_date"] == "2026-09-25"
    assert third["state"]["turn_count"] == 3
    assert third["outcome"] == "completed"


def test_a_tool_turn_over_http_records_the_attempt_without_its_arguments(settings) -> None:
    client = client_with(
        settings,
        [
            tool_generation("get_outstanding_amount", {"account_ref": "ACC-1"}),
            text_generation(GROUNDED_REPLY),
        ],
    )
    session_id = start_call(client)

    body = client.post(
        f"/conversation/{session_id}/turn",
        json={
            "text": "What is my balance?",
            "language": "en",
            "identity_verified": True,
            "disclosed_recording": True,
            "identified_agent": True,
            "stage": "account_discussion",
        },
    ).json()

    assert body["outcome"] == "completed"
    assert len(body["tools"]) == 1
    assert body["tools"][0]["tool_name"] == "get_outstanding_amount"
    assert body["tools"][0]["status"] == "ok"
    assert body["tools"][0]["argument_keys"] == ["account_ref"]
    # The response body carries no account or customer reference.
    dumped = json.dumps(body)
    assert "ACC-1" not in dumped
    assert "CUST-1" not in dumped


def test_a_partial_transcript_is_rejected(turn_client: TestClient) -> None:
    session_id = start_call(turn_client)
    response = turn_client.post(
        f"/conversation/{session_id}/turn",
        json={"text": "I can pay on", "is_final": False},
    )
    assert response.status_code == 400
    assert "final transcript" in response.json()["detail"]


def test_signals_on_an_audit_only_event_kind_are_rejected(turn_client: TestClient) -> None:
    """The reducer would ignore them, so a 200 would be a lie."""
    session_id = start_call(turn_client)
    response = turn_client.post(
        f"/conversation/{session_id}/event",
        json={"kind": "tool_call", "intent": "dispute", "disclosed_recording": True},
    )
    assert response.status_code == 422
    assert "do not carry state signals" in json.dumps(response.json())

    # The same kind with no signals is still accepted.
    assert (
        turn_client.post(
            f"/conversation/{session_id}/event", json={"kind": "tool_call"}
        ).status_code
        == 200
    )


def test_the_event_endpoint_still_works_alongside_the_turn_endpoint(turn_client: TestClient) -> None:
    session_id = start_call(turn_client)
    response = turn_client.post(
        f"/conversation/{session_id}/event",
        json={"kind": "user_utterance", "intent": "identity_confirmed"},
    )
    assert response.status_code == 200
    assert response.json()["state"]["identity_verified"] is True
