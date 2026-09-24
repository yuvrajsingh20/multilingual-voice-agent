"""TTS normalisation: span detection everywhere, rendering in English only."""

from __future__ import annotations

import pytest

from app.models.enums import Language, SpanKind
from app.services.tts import (
    RENDERABLE_LANGUAGES,
    SpanDetectingTtsNormalizer,
    detect_spans,
    indian_number_to_words,
)


@pytest.fixture
def normalizer() -> SpanDetectingTtsNormalizer:
    return SpanDetectingTtsNormalizer()


@pytest.mark.parametrize(
    ("value", "words"),
    [
        (0, "zero"),
        (7, "seven"),
        (15, "fifteen"),
        (40, "forty"),
        (45, "forty-five"),
        (100, "one hundred"),
        (345, "three hundred forty-five"),
        (1_000, "one thousand"),
        (12_345, "twelve thousand three hundred forty-five"),
        (100_000, "one lakh"),
        (1_250_000, "twelve lakh fifty thousand"),
        (10_000_000, "one crore"),
        (12_345_678, "one crore twenty-three lakh forty-five thousand six hundred seventy-eight"),
    ],
)
def test_indian_numbering_is_used(value: int, words: str) -> None:
    assert indian_number_to_words(value) == words


def test_negative_numbers_are_rejected() -> None:
    with pytest.raises(ValueError):
        indian_number_to_words(-1)


def test_spans_are_detected_and_do_not_overlap() -> None:
    spans = detect_spans("Pay Rs. 12,345.50 by 2026-10-05 or call 9876543210 about your EMI.")
    kinds = [span.kind for span in spans]
    assert kinds == [SpanKind.CURRENCY, SpanKind.DATE, SpanKind.PHONE, SpanKind.ABBREVIATION]
    for earlier, later in zip(spans, spans[1:]):
        assert earlier.end <= later.start


def test_currency_prefix_is_not_read_as_a_decimal_point(normalizer) -> None:
    result = normalizer.normalize("Rs. 12,345.50 is due.", Language.ENGLISH)
    assert "twelve thousand three hundred forty-five rupees and fifty paise" in result.text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("₹500", "five hundred rupees"),
        ("INR 1,00,000", "one lakh rupees"),
        ("Rs 7", "seven rupees"),
        ("Rs. 1,234.05", "one thousand two hundred thirty-four rupees and five paise"),
    ],
)
def test_currency_rendering(normalizer, text: str, expected: str) -> None:
    assert normalizer.normalize(text, Language.ENGLISH).text == expected


def test_dates_are_spoken_not_spelled(normalizer) -> None:
    assert normalizer.normalize("due 2026-10-05", Language.ENGLISH).text == "due 5 October 2026"
    assert normalizer.normalize("due 05/10/2026", Language.ENGLISH).text == "due 5 October 2026"


def test_identifiers_are_read_digit_by_digit(normalizer) -> None:
    """Reading a phone number as a quantity would be unintelligible."""
    result = normalizer.normalize("call 9876543210", Language.ENGLISH)
    assert result.text == "call 9 8 7 6 5 4 3 2 1 0"


def test_masked_account_reference_is_read_digit_by_digit(normalizer) -> None:
    result = normalizer.normalize("account XXXX4321", Language.ENGLISH)
    assert result.text == "account 4 3 2 1"


def test_abbreviations_are_spelled_out(normalizer) -> None:
    assert normalizer.normalize("your EMI", Language.ENGLISH).text == "your E M I"


def test_english_text_with_no_spans_is_fully_normalized(normalizer) -> None:
    result = normalizer.normalize("When can you pay?", Language.ENGLISH)
    assert result.fully_normalized is True
    assert result.spans == ()


@pytest.mark.parametrize("language", [Language.HINDI, Language.MARATHI, Language.HINGLISH])
def test_unsupported_languages_are_detected_but_not_rendered(normalizer, language: Language) -> None:
    """Pronunciation in these languages is not solved; the text must not be mangled."""
    source = "Aapka EMI Rs. 12,345 bakaya hai"
    result = normalizer.normalize(source, language)
    assert result.text == source
    assert result.fully_normalized is False
    assert result.spans, "spans must still be detected so the gap is visible"
    assert all(span.spoken is None for span in result.spans)
    assert SpanKind.CURRENCY in result.unrendered_kinds


def test_only_english_is_claimed_as_renderable() -> None:
    assert RENDERABLE_LANGUAGES == frozenset({Language.ENGLISH})


def test_unknown_language_is_not_rendered(normalizer) -> None:
    result = normalizer.normalize("Rs. 500 due", None)
    assert result.text == "Rs. 500 due"
    assert result.fully_normalized is False


def test_the_original_is_always_preserved(normalizer) -> None:
    source = "Rs. 500 due on 2026-10-05"
    assert normalizer.normalize(source, Language.ENGLISH).original == source


def test_a_decimal_number_without_currency_is_left_alone(normalizer) -> None:
    """Rendering an unlabelled decimal is not solved, so it is flagged rather than guessed."""
    result = normalizer.normalize("the rate is 8.5 percent", Language.ENGLISH)
    assert "8.5" in result.text
    assert result.fully_normalized is False
