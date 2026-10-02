"""Keyword intent hints: a provisional stand-in for upstream NLU.

The orchestrator only records a dispute or an escalation when an upstream signal
says the customer asked for one (see ``_WRITE_TOOL_PRECONDITIONS`` in the
pipeline). No NLU model is integrated, so without this a typed or transcribed
"I have already paid" can never reach ``create_dispute``.

This is phrase matching, not understanding. The phrase lists were written for
English, Hinglish, Hindi, Marathi and Marathi-English without native review or
measured call data, and they will miss paraphrases. A phrase is matched only as
whole words, and a dispute is checked before an escalation, because a dispute
halts recovery and must not be downgraded to a hand-off.
"""

from __future__ import annotations

import re

from app.core.session import EventSignals
from app.models.enums import Emotion, Intent

#: The challenge counts "already paid" as a dispute type, so it is here too.
_DISPUTE_PHRASES: tuple[str, ...] = (
    # English
    "already paid", "i have paid", "i paid", "paid already", "paid it", "paid the amount",
    "not my loan", "not mine", "wrong amount", "amount is wrong", "i dispute", "dispute",
    "never took", "never taken",
    # Hinglish
    "de diya", "de diye", "de chuka", "de chuki", "bhar diya", "bhar diye", "jama kar diya",
    "jama kar diye", "pay kar diya", "payment kar diya", "chuka diya", "mera loan nahi",
    "mera nahi hai", "maine loan nahi liya", "galat amount", "amount galat",
    # Marathi-English
    "bharle", "bharla", "bharun dile", "bharun takle", "bharun zale", "dile aahe",
    "majha nahi", "maza nahi", "majhe nahi", "maze nahi", "majha loan nahi",
    # Devanagari
    "भर दिया", "दे दिया", "जमा कर दिया", "मेरा नहीं", "भरले", "भरला", "माझे नाही", "माझं नाही",
)

_ESCALATION_PHRASES: tuple[str, ...] = (
    "manager", "supervisor", "senior", "human", "real person", "insaan", "adhikari",
    "मैनेजर", "अधिकारी",
)

_DISTRESS_PHRASES: tuple[str, ...] = (
    "died", "passed away", "death", "hospital", "accident", "icu", "cancer",
    "guzar gaye", "guzar gayi", "mar gaye", "mar gayi", "nidhan", "bimar", "beemar",
    "vaarle", "varle", "dawakhana", "aajari",
    "गुजर गए", "मर गए", "निधन", "अस्पताल", "बीमार", "वारले", "दवाखाना", "आजारी",
)


def _pattern(phrases: tuple[str, ...]) -> re.Pattern[str]:
    # (?<!\w) / (?!\w) rather than \b: \b does not hold after Devanagari vowel signs.
    alternatives = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


_DISPUTE = _pattern(_DISPUTE_PHRASES)
_ESCALATION = _pattern(_ESCALATION_PHRASES)
_DISTRESS = _pattern(_DISTRESS_PHRASES)


def intent_hints(text: str) -> EventSignals | None:
    """Signals for ``text``, or ``None`` when no phrase matches."""
    if _DISPUTE.search(text):
        return EventSignals(intent=Intent.DISPUTE)
    if _DISTRESS.search(text):
        return EventSignals(intent=Intent.ESCALATION_REQUEST, emotion=Emotion.DISTRESSED)
    if _ESCALATION.search(text):
        return EventSignals(intent=Intent.ESCALATION_REQUEST)
    return None
