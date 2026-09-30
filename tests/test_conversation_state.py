"""Conversation state and the event reducer."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError
from zoneinfo import ZoneInfo

from app.core.session import SessionLimitReached, SessionStore, apply_event, replay
from app.models.conversation import ConversationEvent, ConversationState
from app.models.enums import (
    ConversationStage,
    Emotion,
    EventKind,
    Intent,
    Language,
)

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 24, 10, 30, tzinfo=IST)


def _event(kind: EventKind = EventKind.USER_UTTERANCE, turn: int = 0, **data) -> ConversationEvent:
    return ConversationEvent(
        event_id=f"e{turn}",
        session_id="s1",
        turn_id=turn,
        occurred_at=NOW,
        kind=kind,
        language=data.pop("language", None),
        text=data.pop("text", None),
        data=data,
    )


def test_defaults_are_conservative() -> None:
    state = ConversationState(session_id="s1")
    assert state.current_stage is ConversationStage.GREETING
    assert state.identity_verified is False
    assert state.recording_disclosed is False
    assert state.agent_identified is False
    assert state.wrong_person is False
    assert state.dispute is False
    assert state.escalation_required is False


def test_state_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ConversationState(session_id="s1", dpd=30)


def test_state_does_not_carry_account_facts() -> None:
    """DPD and balances belong to the backend, not to conversation state."""
    fields = set(ConversationState.model_fields)
    assert not fields & {"dpd", "outstanding_minor", "account_ref", "customer_ref"}


def test_reducer_is_pure() -> None:
    state = ConversationState(session_id="s1")
    apply_event(state, _event(intent=Intent.DISPUTE.value))
    assert state.dispute is False


def test_user_utterance_advances_the_turn_count() -> None:
    state = ConversationState(session_id="s1")
    state = apply_event(state, _event())
    state = apply_event(state, _event(turn=1))
    assert state.turn_count == 2


def test_agent_utterance_does_not_advance_the_turn_count() -> None:
    state = apply_event(ConversationState(session_id="s1"), _event(kind=EventKind.AGENT_UTTERANCE))
    assert state.turn_count == 0


def test_identity_confirmed_sets_the_flag() -> None:
    state = apply_event(ConversationState(session_id="s1"), _event(intent=Intent.IDENTITY_CONFIRMED.value))
    assert state.identity_verified is True


def test_wrong_person_clears_identity_and_moves_to_closing() -> None:
    state = ConversationState(session_id="s1", identity_verified=True)
    state = apply_event(state, _event(intent=Intent.WRONG_PERSON.value))
    assert state.wrong_person is True
    assert state.identity_verified is False
    assert state.current_stage is ConversationStage.CLOSING


def test_a_wrong_person_call_cannot_be_steered_back_into_collection() -> None:
    state = apply_event(ConversationState(session_id="s1"), _event(intent=Intent.WRONG_PERSON.value))
    state = apply_event(state, _event(turn=1, stage=ConversationStage.NEGOTIATION.value))
    assert state.current_stage is ConversationStage.CLOSING


def test_payment_promise_records_the_date() -> None:
    state = apply_event(
        ConversationState(session_id="s1"),
        _event(intent=Intent.PAYMENT_PROMISE.value, promise_date="2026-10-05"),
    )
    assert state.payment_promise is True
    assert state.promise_date == date(2026, 10, 5)


def test_escalation_request_sets_escalation_required() -> None:
    state = apply_event(ConversationState(session_id="s1"), _event(intent=Intent.ESCALATION_REQUEST.value))
    assert state.escalation_required is True


def test_disclosure_flags_latch_on() -> None:
    state = ConversationState(session_id="s1")
    state = apply_event(state, _event(kind=EventKind.AGENT_UTTERANCE, disclosed_recording=True))
    state = apply_event(state, _event(kind=EventKind.AGENT_UTTERANCE, turn=1, identified_agent=True))
    state = apply_event(state, _event(kind=EventKind.AGENT_UTTERANCE, turn=2))
    assert state.recording_disclosed is True
    assert state.agent_identified is True


def test_language_and_emotion_are_carried_through() -> None:
    state = apply_event(
        ConversationState(session_id="s1"),
        _event(language=Language.MARATHI, emotion=Emotion.FRUSTRATED.value),
    )
    assert state.language is Language.MARATHI
    assert state.emotion is Emotion.FRUSTRATED


def test_call_ended_terminates() -> None:
    state = apply_event(ConversationState(session_id="s1"), _event(kind=EventKind.CALL_ENDED))
    assert state.current_stage is ConversationStage.TERMINATED


def test_unknown_signal_is_rejected() -> None:
    with pytest.raises(ValidationError):
        apply_event(ConversationState(session_id="s1"), _event(allow_everything=True))


def test_invalid_intent_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        apply_event(ConversationState(session_id="s1"), _event(intent="not_an_intent"))


def test_replay_reproduces_state_from_the_event_log() -> None:
    events = [
        _event(turn=0, intent=Intent.IDENTITY_CONFIRMED.value),
        _event(turn=1, kind=EventKind.AGENT_UTTERANCE, disclosed_recording=True, identified_agent=True),
        _event(turn=2, intent=Intent.PAYMENT_PROMISE.value, promise_date="2026-10-10"),
    ]
    incremental = ConversationState(session_id="s1")
    for event in events:
        incremental = apply_event(incremental, event)
    assert replay("s1", events) == incremental


# --- session store ----------------------------------------------------------


def test_store_creates_and_reads_back() -> None:
    store = SessionStore(max_sessions=10)
    session = store.create(now=NOW)
    assert store.get(session.session_id) is session
    assert len(store) == 1


def test_store_returns_none_for_an_unknown_session() -> None:
    assert SessionStore(max_sessions=10).get("nope") is None


def test_store_enforces_its_limit() -> None:
    store = SessionStore(max_sessions=1)
    store.create(now=NOW)
    with pytest.raises(SessionLimitReached):
        store.create(now=NOW)


def test_deleting_a_session_also_forgets_its_write_lock() -> None:
    """Nothing calls delete() today, but its own bookkeeping must still be complete.

    write_lock() creates one lock per session and nothing else ever removes
    it; delete() is the only place a session's resources are meant to be
    reclaimed, so it must reclaim this one too, or a deployment that starts
    calling delete() would leak a lock per deleted session for the process's
    lifetime.
    """
    store = SessionStore(max_sessions=10)
    session = store.create(now=NOW)
    lock = store.write_lock(session.session_id)  # noqa: F841 - creates the entry
    assert session.session_id in store._write_locks

    assert store.delete(session.session_id) is True
    assert store.get(session.session_id) is None
    assert session.session_id not in store._write_locks

    assert store.delete("nope") is False


def test_appending_an_event_advances_state_and_the_log() -> None:
    store = SessionStore(max_sessions=10)
    session = store.create(now=NOW)
    event = store.next_event(session, kind=EventKind.USER_UTTERANCE, now=NOW, data={"intent": "dispute"})
    updated = store.append_event(session.session_id, event)
    assert updated is not None
    assert updated.state.dispute is True
    assert len(updated.events) == 1
    assert updated.events[0].turn_id == 0


def test_turn_ids_increase_with_the_event_log() -> None:
    store = SessionStore(max_sessions=10)
    session = store.create(now=NOW)
    for expected in range(3):
        event = store.next_event(session, kind=EventKind.USER_UTTERANCE, now=NOW)
        session = store.append_event(session.session_id, event)
        assert event.turn_id == expected
