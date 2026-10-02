"""Replay of earlier turns in the model request, and reply-language coverage."""

from __future__ import annotations

from datetime import datetime, timezone

from app.models.conversation import ConversationEvent
from app.models.enums import EventKind, Language
from app.orchestrator.prompt import MAX_REPLAYED_EXCHANGES, replayed_turns

NOW = datetime(2026, 9, 30, 5, 30, tzinfo=timezone.utc)


def _event(index: int, kind: EventKind, text: str | None = None) -> ConversationEvent:
    return ConversationEvent(
        event_id=f"e{index}", session_id="s", turn_id=index, occurred_at=NOW, kind=kind, text=text
    )


def _log(*items: tuple[EventKind, str | None]) -> list[ConversationEvent]:
    return [_event(i, kind, text) for i, (kind, text) in enumerate(items)]


def test_first_turn_replays_nothing() -> None:
    events = _log((EventKind.SESSION_STARTED, None), (EventKind.USER_UTTERANCE, "hello"))
    assert replayed_turns(events) == ()


def test_earlier_exchanges_are_replayed_and_the_current_turn_is_not() -> None:
    events = _log(
        (EventKind.SESSION_STARTED, None),
        (EventKind.USER_UTTERANCE, "kitna baaki hai?"),
        (EventKind.TOOL_CALL, None),
        (EventKind.AGENT_UTTERANCE, "INR 12,345.00 baaki hai."),
        (EventKind.USER_UTTERANCE, "main kal dunga"),
        (EventKind.ACCOUNT_CONTEXT_LOADED, None),
    )
    messages = replayed_turns(events)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "kitna baaki hai?"),
        ("assistant", "INR 12,345.00 baaki hai."),
    ]


def test_a_turn_that_was_never_spoken_leaves_only_the_customer_line() -> None:
    events = _log(
        (EventKind.USER_UTTERANCE, "first"),
        (EventKind.USER_UTTERANCE, "second"),
        (EventKind.USER_UTTERANCE, "current"),
    )
    assert [m.content for m in replayed_turns(events)] == ["first", "second"]


def test_replay_is_bounded() -> None:
    items: list[tuple[EventKind, str | None]] = []
    for n in range(MAX_REPLAYED_EXCHANGES + 3):
        items += [(EventKind.USER_UTTERANCE, f"u{n}"), (EventKind.AGENT_UTTERANCE, f"a{n}")]
    items.append((EventKind.USER_UTTERANCE, "current"))
    messages = replayed_turns(_log(*items))
    assert len(messages) == 2 * MAX_REPLAYED_EXCHANGES
    assert messages[-1].content == f"a{MAX_REPLAYED_EXCHANGES + 2}"
    assert replayed_turns(_log(*items), limit=0) == ()


def test_every_language_has_a_reply_instruction() -> None:
    from app.orchestrator import prompt

    assert set(prompt._LANGUAGE_INSTRUCTIONS) == set(Language)
