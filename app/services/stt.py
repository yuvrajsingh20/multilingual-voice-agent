"""Transcript normalisation boundary.

There is no STT engine here and none is simulated. Streaming recognition is an
external service; this module defines what the pipeline does with what comes
back.

What is deliberately NOT implemented
------------------------------------
Hindi/English code-switching repair, Hinglish romanisation, Marathi handling,
name correction, domain-term correction and spoken-number parsing. Each of those
needs a lexicon or a model built from real call data. Hard-coding guesses now
would put unverified corrections into an audit trail, so the default normaliser
does only what is safe without evidence.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import Language

_WHITESPACE = re.compile(r"\s+")


class TranscriptSegment(BaseModel):
    """One STT hypothesis. ``is_final`` distinguishes partials from final results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    is_final: bool = False
    start_ms: int = Field(default=0, ge=0)
    end_ms: int = Field(default=0, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    language: Language | None = None


class NormalizedTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_text: str
    normalized_text: str
    language: Language | None = None
    applied: tuple[str, ...] = Field(
        default=(), description="Names of the transformations actually applied."
    )
    unhandled: tuple[str, ...] = Field(
        default=(), description="Known normalisation needs this implementation does not cover."
    )


#: Normalisations this layer is expected to grow, listed so the gap is explicit.
UNIMPLEMENTED_NORMALISATIONS: tuple[str, ...] = (
    "code_switching_hi_en",
    "hinglish_romanisation",
    "marathi_lexicon",
    "person_name_correction",
    "banking_domain_terms",
    "spoken_number_parsing",
    "spoken_date_parsing",
)


class TranscriptNormalizer(Protocol):
    def normalize(self, segment: TranscriptSegment) -> NormalizedTranscript: ...


class PassthroughTranscriptNormalizer:
    """Returns the transcript unchanged. The honest default."""

    def normalize(self, segment: TranscriptSegment) -> NormalizedTranscript:
        return NormalizedTranscript(
            original_text=segment.text,
            normalized_text=segment.text,
            language=segment.language,
            applied=(),
            unhandled=UNIMPLEMENTED_NORMALISATIONS,
        )


class WhitespaceTranscriptNormalizer:
    """Collapses runs of whitespace and trims the ends. Nothing else.

    This is the only correction that is safe across Hindi, English, Marathi and
    Hinglish without evidence from real traffic.
    """

    def normalize(self, segment: TranscriptSegment) -> NormalizedTranscript:
        collapsed = _WHITESPACE.sub(" ", segment.text).strip()
        applied = ("whitespace_collapse",) if collapsed != segment.text else ()
        return NormalizedTranscript(
            original_text=segment.text,
            normalized_text=collapsed,
            language=segment.language,
            applied=applied,
            unhandled=UNIMPLEMENTED_NORMALISATIONS,
        )
