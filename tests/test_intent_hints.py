"""Keyword intent hints: the provisional stand-in for upstream NLU."""

from __future__ import annotations

import pytest

from app.models.enums import Emotion, Intent
from app.services.intent import intent_hints


@pytest.mark.parametrize(
    "text",
    [
        "i have paid and tell to my father",
        "I already paid last week",
        "maine to de diya tha",
        "paisa bhar diya hai maine",
        "ye mera loan nahi hai",
        "mi paise bharle aahet",
        "he loan majha nahi",
        "मैंने भर दिया है",
        "मी पैसे भरले",
    ],
)
def test_paid_and_not_mine_claims_are_disputes(text: str) -> None:
    signals = intent_hints(text)
    assert signals is not None
    assert signals.intent is Intent.DISPUTE


@pytest.mark.parametrize(
    "text",
    ["mere papa hospital mein hain", "my father passed away", "aai aajari aahe", "पिताजी का निधन हो गया"],
)
def test_distress_escalates_and_is_marked_distressed(text: str) -> None:
    signals = intent_hints(text)
    assert signals is not None
    assert signals.intent is Intent.ESCALATION_REQUEST
    assert signals.emotion is Emotion.DISTRESSED


def test_asking_for_a_manager_escalates() -> None:
    signals = intent_hints("mujhe aapke manager se baat karni hai")
    assert signals is not None
    assert signals.intent is Intent.ESCALATION_REQUEST
    assert signals.emotion is None


def test_dispute_wins_over_escalation() -> None:
    signals = intent_hints("I already paid, get me your manager")
    assert signals is not None
    assert signals.intent is Intent.DISPUTE


@pytest.mark.parametrize(
    "text",
    [
        "70 din se nahi diye hai",
        "kal tak payment karunga",
        "how much do I owe?",
        "humanity",
        "unpaid hai abhi",
    ],
)
def test_ordinary_turns_carry_no_hint(text: str) -> None:
    assert intent_hints(text) is None
