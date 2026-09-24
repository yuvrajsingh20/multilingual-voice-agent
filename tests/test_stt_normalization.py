"""Transcript normalisation boundary.

There is deliberately very little behaviour to test here, and these tests pin
that down: the default normaliser must not invent corrections, and the gaps must
stay visible.
"""

from __future__ import annotations

import pytest

from app.models.enums import Language
from app.services.stt import (
    UNIMPLEMENTED_NORMALISATIONS,
    PassthroughTranscriptNormalizer,
    TranscriptSegment,
    WhitespaceTranscriptNormalizer,
)


def _segment(text: str, **kwargs) -> TranscriptSegment:
    return TranscriptSegment(text=text, **kwargs)


def test_passthrough_changes_nothing() -> None:
    result = PassthroughTranscriptNormalizer().normalize(_segment("mera  naam"))
    assert result.normalized_text == "mera  naam"
    assert result.applied == ()


def test_whitespace_normaliser_collapses_and_trims() -> None:
    result = WhitespaceTranscriptNormalizer().normalize(_segment("  haan   ji  "))
    assert result.normalized_text == "haan ji"
    assert result.applied == ("whitespace_collapse",)


def test_no_change_means_no_applied_transformation() -> None:
    result = WhitespaceTranscriptNormalizer().normalize(_segment("haan ji"))
    assert result.applied == ()


@pytest.mark.parametrize("language", [Language.HINDI, Language.MARATHI, Language.HINGLISH, Language.ENGLISH])
def test_no_language_specific_rewriting_happens(language: Language) -> None:
    """Code-switching and Hinglish repair are not implemented; text must survive intact."""
    source = "mera EMI 12345 rupaye hai na"
    result = WhitespaceTranscriptNormalizer().normalize(_segment(source, language=language))
    assert result.normalized_text == source
    assert result.language is language


def test_devanagari_is_untouched() -> None:
    source = "मेरा खाता नंबर क्या है"
    assert WhitespaceTranscriptNormalizer().normalize(_segment(source)).normalized_text == source


def test_the_original_is_always_kept() -> None:
    result = WhitespaceTranscriptNormalizer().normalize(_segment("  haan  "))
    assert result.original_text == "  haan  "


def test_unimplemented_normalisations_are_reported(  ) -> None:
    result = WhitespaceTranscriptNormalizer().normalize(_segment("haan"))
    assert set(result.unhandled) == set(UNIMPLEMENTED_NORMALISATIONS)
    assert "hinglish_romanisation" in result.unhandled
    assert "spoken_number_parsing" in result.unhandled


def test_partial_and_final_hypotheses_are_distinguished() -> None:
    partial = _segment("mera", is_final=False)
    final = _segment("mera naam", is_final=True)
    assert partial.is_final is False
    assert final.is_final is True


def test_confidence_is_bounded() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _segment("x", confidence=1.5)
