"""HTTP surface, including the health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.core.clock import FixedClock
from app.main import create_app
from app.runtime import build_runtime
from tests.conftest import DEFAULT_NOW, RULES_PATH

RETAIL_ACCOUNT = {
    "account_ref": "ACC-1",
    "product_type": "retail_loan",
    "outstanding_minor": 1_234_500,
    "dpd": 35,
}


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_creating_a_session_returns_state_and_a_decision(client: TestClient) -> None:
    response = client.post("/conversation/session", json={"language": "hi", "account": RETAIL_ACCOUNT})
    assert response.status_code == 201
    body = response.json()
    assert body["state"]["language"] == "hi"
    assert body["state"]["current_stage"] == "greeting"
    assert body["policy"]["allowed"] is True
    assert body["policy"]["dpd_stage"] == "dpd_30"
    assert body["policy"]["tone"] == "firm"
    assert body["event_count"] == 1


def test_a_session_can_be_created_with_no_context(client: TestClient) -> None:
    response = client.post("/conversation/session", json={})
    assert response.status_code == 201
    assert response.json()["policy"]["dpd_stage"] is None


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    assert client.post("/conversation/session", json={"lang": "hi"}).status_code == 422


def test_reading_back_a_session(client: TestClient) -> None:
    session_id = client.post("/conversation/session", json={"account": RETAIL_ACCOUNT}).json()["session_id"]
    response = client.get(f"/conversation/{session_id}")
    assert response.status_code == 200
    assert response.json()["session_id"] == session_id


def test_an_unknown_session_is_a_404(client: TestClient) -> None:
    assert client.get("/conversation/does-not-exist").status_code == 404
    assert (
        client.post("/conversation/does-not-exist/event", json={"kind": "user_utterance"}).status_code == 404
    )


def test_posting_an_event_advances_state(client: TestClient) -> None:
    session_id = client.post("/conversation/session", json={"account": RETAIL_ACCOUNT}).json()["session_id"]
    response = client.post(
        f"/conversation/{session_id}/event",
        json={"kind": "user_utterance", "intent": "identity_confirmed", "language": "hi-en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["state"]["identity_verified"] is True
    assert body["state"]["turn_count"] == 1
    assert body["event_count"] == 2


def test_a_wrong_person_event_blocks_collection_over_http(client: TestClient) -> None:
    session_id = client.post("/conversation/session", json={"account": RETAIL_ACCOUNT}).json()["session_id"]
    body = client.post(
        f"/conversation/{session_id}/event",
        json={"kind": "user_utterance", "intent": "wrong_person"},
    ).json()
    assert body["state"]["wrong_person"] is True
    assert body["policy"]["allowed"] is False
    assert "end_call" in body["policy"]["required_actions"]


def test_an_invalid_intent_is_rejected(client: TestClient) -> None:
    session_id = client.post("/conversation/session", json={}).json()["session_id"]
    response = client.post(
        f"/conversation/{session_id}/event", json={"kind": "user_utterance", "intent": "hack"}
    )
    assert response.status_code == 422


def test_an_oversized_transcript_is_rejected(client: TestClient) -> None:
    session_id = client.post("/conversation/session", json={}).json()["session_id"]
    response = client.post(
        f"/conversation/{session_id}/event",
        json={"kind": "user_utterance", "text": "x" * 5000},
    )
    assert response.status_code == 422


def test_the_decision_names_the_rules_behind_it(client: TestClient) -> None:
    body = client.post("/conversation/session", json={"account": RETAIL_ACCOUNT}).json()
    policy = body["policy"]
    assert "RBI-CB-RBC-2025-445-CALLING-HOURS" in policy["rule_ids"]
    assert policy["not_evaluable_rule_ids"]
    assert policy["unenforced_rule_ids"]
    assert policy["rules_version"]


def test_sessions_are_isolated(client: TestClient) -> None:
    first = client.post("/conversation/session", json={"account": RETAIL_ACCOUNT}).json()["session_id"]
    second = client.post("/conversation/session", json={"account": RETAIL_ACCOUNT}).json()["session_id"]
    client.post(f"/conversation/{first}/event", json={"kind": "user_utterance", "intent": "dispute"})
    assert client.get(f"/conversation/{second}").json()["state"]["dispute"] is False


def test_the_store_limit_returns_503() -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        regulatory_rules_path=RULES_PATH,
        max_active_sessions=1,
    )
    app = create_app(settings, build_runtime(settings, clock=FixedClock(DEFAULT_NOW)))
    with TestClient(app) as limited:
        assert limited.post("/conversation/session", json={}).status_code == 201
        assert limited.post("/conversation/session", json={}).status_code == 503


def test_production_refuses_caller_supplied_account_context() -> None:
    """A client must not be able to assert its own compliance inputs."""
    settings = Settings(
        _env_file=None,
        app_env="prod",
        log_level="WARNING",
        regulatory_rules_path=RULES_PATH,
    )
    app = create_app(settings, build_runtime(settings, clock=FixedClock(DEFAULT_NOW)))
    with TestClient(app) as prod:
        assert prod.post("/conversation/session", json={"account": RETAIL_ACCOUNT}).status_code == 400
        assert prod.post("/conversation/session", json={"language": "en"}).status_code == 201


def test_out_of_hours_call_is_blocked_over_http() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    settings = Settings(_env_file=None, app_env="test", log_level="WARNING", regulatory_rules_path=RULES_PATH)
    late = datetime(2026, 9, 24, 20, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    app = create_app(settings, build_runtime(settings, clock=FixedClock(late)))
    with TestClient(app) as after_hours:
        body = after_hours.post("/conversation/session", json={"account": RETAIL_ACCOUNT}).json()
        assert body["policy"]["allowed"] is False
        assert any(
            v["rule_id"] == "RBI-CB-RBC-2025-445-CALLING-HOURS" for v in body["policy"]["violations"]
        )
