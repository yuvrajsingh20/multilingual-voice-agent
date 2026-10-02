"""TTS normalisation boundary.

No TTS provider is integrated. This layer turns written text into text a
synthesiser can speak correctly, which is a separate problem from synthesis
itself.

Honest scope
------------
Span *detection* (currency, numbers, dates, phone numbers, account references,
abbreviations) is language-independent and implemented. Span *rendering* is
implemented in English words, and used for English and for the two code-mixed
languages, Hinglish and Marathi-English, where amounts and dates are commonly
spoken in English ("twelve thousand rupees", "5 October"). Pure Hindi and
Marathi rendering is NOT solved: correct spoken forms need a pronunciation
lexicon and native review, so for those languages the detected spans are
returned unrendered and ``fully_normalized`` is ``False``. Pretending otherwise
would produce a voice agent that misreads amounts to customers.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import Language, SpanKind

#: Languages whose spans are rendered as English words. The code-mixed languages
#: are here because their speakers say amounts and dates in English; that is a
#: convention of the register, not a Hindi or Marathi rendering.
RENDERABLE_LANGUAGES: frozenset[Language] = frozenset(
    {Language.ENGLISH, Language.HINGLISH, Language.MARATHI_ENGLISH}
)

_UNITS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _two_digits(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    tens, unit = divmod(n, 10)
    return _TENS[tens] if unit == 0 else f"{_TENS[tens]}-{_UNITS[unit]}"


def _three_digits(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(f"{_UNITS[hundreds]} hundred")
    if rest:
        parts.append(_two_digits(rest))
    return " ".join(parts)


def indian_number_to_words(value: int) -> str:
    """Render a non-negative integer using the Indian numbering system.

    Crore and lakh are used because that is how amounts are spoken to customers
    in India; "one lakh twenty thousand" rather than "one hundred twenty thousand".
    """
    if value < 0:
        raise ValueError("indian_number_to_words expects a non-negative integer")
    if value == 0:
        return _UNITS[0]

    crore, rest = divmod(value, 10_000_000)
    lakh, rest = divmod(rest, 100_000)
    thousand, rest = divmod(rest, 1_000)

    parts: list[str] = []
    if crore:
        parts.append(f"{indian_number_to_words(crore)} crore")
    if lakh:
        parts.append(f"{_two_digits(lakh)} lakh")
    if thousand:
        parts.append(f"{_two_digits(thousand)} thousand")
    if rest:
        parts.append(_three_digits(rest))
    return " ".join(parts)


class TextSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SpanKind
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    raw: str
    spoken: str | None = Field(default=None, description="None when this language cannot be rendered.")


class NormalizedSpeechText(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original: str
    text: str
    language: Language | None = None
    spans: tuple[TextSpan, ...] = ()
    fully_normalized: bool = False
    unrendered_kinds: tuple[SpanKind, ...] = ()


class TtsNormalizer(Protocol):
    def normalize(self, text: str, language: Language | None) -> NormalizedSpeechText: ...


# Patterns are tried in this order; the first match at a position wins, so more
# specific patterns (currency, phone) precede the generic number pattern.
_PATTERNS: tuple[tuple[SpanKind, re.Pattern[str]], ...] = (
    (SpanKind.CURRENCY, re.compile(r"(?:₹|Rs\.?|INR)\s*\d[\d,]*(?:\.\d{1,2})?", re.IGNORECASE)),
    (SpanKind.DATE, re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b")),
    (SpanKind.PHONE, re.compile(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b")),
    (SpanKind.ACCOUNT_REFERENCE, re.compile(r"\b[Xx]{2,}\d{2,6}\b|\b\d{11,18}\b")),
    (SpanKind.ABBREVIATION, re.compile(r"\b(?:EMI|NEFT|IMPS|UPI|IFSC|NPA|DPD|KYC|OTP)\b")),
    (SpanKind.NUMBER, re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")),
)


def detect_spans(text: str) -> tuple[TextSpan, ...]:
    """Find non-overlapping spans, earliest first, most specific pattern first."""
    claimed: list[tuple[int, int]] = []
    found: list[TextSpan] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < c_end and c_start < end for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            found.append(TextSpan(kind=kind, start=start, end=end, raw=match.group()))
    return tuple(sorted(found, key=lambda s: s.start))


def _digits_only(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def _render_english(span: TextSpan) -> str | None:
    raw = span.raw
    if span.kind is SpanKind.CURRENCY:
        # Match the amount at the end of the span so a "Rs." prefix cannot be
        # mistaken for a decimal point.
        amount = re.search(r"\d[\d,]*(?:\.\d{1,2})?$", raw)
        if amount is None:
            return None
        rupees_text, _, paise_text = amount.group().replace(",", "").partition(".")
        rupees = int(rupees_text or "0")
        spoken = f"{indian_number_to_words(rupees)} rupees"
        paise = int((paise_text + "00")[:2]) if paise_text else 0
        if paise:
            spoken += f" and {indian_number_to_words(paise)} paise"
        return spoken
    if span.kind is SpanKind.DATE:
        try:
            if "-" in raw:
                parsed = date.fromisoformat(raw)
            else:
                day, month, year = raw.split("/")
                parsed = date(int(year), int(month), int(day))
        except ValueError:
            return None
        return f"{parsed.day} {_MONTHS[parsed.month - 1]} {parsed.year}"
    if span.kind in (SpanKind.PHONE, SpanKind.ACCOUNT_REFERENCE):
        # Digit-by-digit is the only safe reading for an identifier.
        return " ".join(_digits_only(raw)) or None
    if span.kind is SpanKind.ABBREVIATION:
        return " ".join(raw)
    if span.kind is SpanKind.NUMBER:
        cleaned = raw.replace(",", "")
        if "." in cleaned:
            return None
        return indian_number_to_words(int(cleaned))
    return None


class SpanDetectingTtsNormalizer:
    """Detects spans in any language; renders them for English and the code-mixed languages."""

    def normalize(self, text: str, language: Language | None) -> NormalizedSpeechText:
        spans = detect_spans(text)
        renderable = language in RENDERABLE_LANGUAGES

        rendered: list[TextSpan] = []
        for span in spans:
            spoken = _render_english(span) if renderable else None
            rendered.append(span.model_copy(update={"spoken": spoken}))

        if renderable:
            out: list[str] = []
            cursor = 0
            for span in rendered:
                out.append(text[cursor : span.start])
                out.append(span.spoken if span.spoken is not None else span.raw)
                cursor = span.end
            out.append(text[cursor:])
            spoken_text = "".join(out)
        else:
            spoken_text = text

        unrendered = tuple(
            sorted({s.kind for s in rendered if s.spoken is None}, key=lambda k: k.value)
        )
        return NormalizedSpeechText(
            original=text,
            text=spoken_text,
            language=language,
            spans=tuple(rendered),
            fully_normalized=renderable and not unrendered,
            unrendered_kinds=unrendered,
        )
