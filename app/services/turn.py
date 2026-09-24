"""Turn detection and barge-in.

No ML model is implemented or simulated here. What is implemented is the part
that must be deterministic anyway: the state machine, and a transparent
heuristic for telling a backchannel apart from a real interruption.

The distinction that matters: while the agent is speaking, "haan" or "hmm" is the
customer following along and TTS should continue; an actual sentence means the
customer wants the floor and TTS must stop.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import BargeInClass, Language, TurnState

#: Provisional backchannel vocabulary. Derived from the languages in scope, NOT
#: from measured call data - it needs replacing with tokens observed in real
#: traffic before anyone relies on it.
BACKCHANNEL_TOKENS: dict[Language, frozenset[str]] = {
    Language.ENGLISH: frozenset({"hmm", "hm", "mhm", "ok", "okay", "yeah", "yes", "right", "uh huh"}),
    Language.HINDI: frozenset({"haan", "han", "haa", "hmm", "hm", "ji", "achha", "acha", "theek", "theek hai"}),
    Language.HINGLISH: frozenset(
        {"haan", "han", "hmm", "hm", "ji", "achha", "acha", "ok", "okay", "theek hai", "right"}
    ),
    Language.MARATHI: frozenset({"ho", "hoy", "bara", "bare", "hmm", "hm", "theek", "ok", "okay"}),
}

#: Used when the language is unknown: the union, so an unknown-language "hmm"
#: does not needlessly cut the agent off.
_ANY_BACKCHANNEL: frozenset[str] = frozenset().union(*BACKCHANNEL_TOKENS.values())

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


class InvalidTurnTransition(RuntimeError):
    """A transition the machine does not allow was requested."""


class VadResult(BaseModel):
    """One voice-activity decision from the audio layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    is_speech: bool
    speech_probability: float = Field(ge=0.0, le=1.0)
    timestamp_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)


class BargeInSignal(BaseModel):
    """Customer audio detected while the agent is speaking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vad: VadResult
    partial_transcript: str | None = None
    language: Language | None = None


class BargeInDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    classification: BargeInClass
    stop_tts: bool
    reason: str


class BargeInClassifier(Protocol):
    def classify(self, signal: BargeInSignal) -> BargeInDecision: ...


class HeuristicBargeInClassifier:
    """Transparent rules, no model.

    ``backchannel_max_ms`` bounds how long an acknowledgement can be; anything
    longer is treated as the customer taking the floor even if the words look
    like a backchannel. ``min_speech_ms`` filters out clicks and coughs.
    """

    def __init__(
        self,
        *,
        min_speech_ms: int = 200,
        backchannel_max_ms: int = 800,
        min_speech_probability: float = 0.5,
    ) -> None:
        self.min_speech_ms = min_speech_ms
        self.backchannel_max_ms = backchannel_max_ms
        self.min_speech_probability = min_speech_probability

    def classify(self, signal: BargeInSignal) -> BargeInDecision:
        vad = signal.vad
        if not vad.is_speech or vad.speech_probability < self.min_speech_probability:
            return BargeInDecision(
                classification=BargeInClass.NOISE, stop_tts=False, reason="No speech detected."
            )
        if vad.duration_ms < self.min_speech_ms:
            return BargeInDecision(
                classification=BargeInClass.NOISE,
                stop_tts=False,
                reason=f"Speech shorter than {self.min_speech_ms} ms.",
            )

        text = (signal.partial_transcript or "").strip()
        if not text:
            if vad.duration_ms > self.backchannel_max_ms:
                return BargeInDecision(
                    classification=BargeInClass.INTERRUPTION,
                    stop_tts=True,
                    reason="Sustained speech with no transcript yet.",
                )
            return BargeInDecision(
                classification=BargeInClass.UNKNOWN,
                stop_tts=False,
                reason="Short speech with no transcript yet; waiting.",
            )

        normalised = _NON_WORD.sub(" ", text.lower())
        normalised = " ".join(normalised.split())
        vocabulary = BACKCHANNEL_TOKENS.get(signal.language, _ANY_BACKCHANNEL) if signal.language else _ANY_BACKCHANNEL

        if normalised in vocabulary and vad.duration_ms <= self.backchannel_max_ms:
            return BargeInDecision(
                classification=BargeInClass.BACKCHANNEL,
                stop_tts=False,
                reason=f"Recognised backchannel {normalised!r} within {self.backchannel_max_ms} ms.",
            )
        return BargeInDecision(
            classification=BargeInClass.INTERRUPTION,
            stop_tts=True,
            reason="Customer produced substantive speech.",
        )


class TurnCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_complete: bool
    reason: str


class TurnDetector(Protocol):
    def is_turn_complete(self, *, trailing_silence_ms: int, is_final: bool, text: str) -> TurnCompletion: ...


class SilenceTurnDetector:
    """End-of-turn on trailing silence. Deterministic and the current default."""

    def __init__(self, *, silence_threshold_ms: int = 700) -> None:
        self.silence_threshold_ms = silence_threshold_ms

    def is_turn_complete(self, *, trailing_silence_ms: int, is_final: bool, text: str) -> TurnCompletion:
        if not is_final:
            return TurnCompletion(is_complete=False, reason="STT hypothesis is not final.")
        if trailing_silence_ms >= self.silence_threshold_ms:
            return TurnCompletion(
                is_complete=True, reason=f"Silence of {trailing_silence_ms} ms met the threshold."
            )
        return TurnCompletion(is_complete=False, reason="Trailing silence below threshold.")


class SemanticTurnDetector(Protocol):
    """Interface for a future semantic end-of-turn model.

    NOT IMPLEMENTED. Deciding that "mera account number hai nau aath saat..." is
    mid-thought rather than finished needs a trained model; a punctuation or
    keyword heuristic would cut customers off mid-sentence, especially in
    code-switched speech.
    """

    def is_turn_complete(self, *, text: str, language: Language | None) -> TurnCompletion: ...


class CompositeTurnDetector:
    """Silence detection, optionally confirmed by a semantic detector.

    With no semantic detector supplied this behaves exactly like
    :class:`SilenceTurnDetector`. The composition point exists so that adding a
    model later does not mean rewriting the pipeline.
    """

    def __init__(
        self,
        silence: SilenceTurnDetector,
        semantic: SemanticTurnDetector | None = None,
        *,
        language: Language | None = None,
    ) -> None:
        self.silence = silence
        self.semantic = semantic
        self.language = language

    def is_turn_complete(self, *, trailing_silence_ms: int, is_final: bool, text: str) -> TurnCompletion:
        by_silence = self.silence.is_turn_complete(
            trailing_silence_ms=trailing_silence_ms, is_final=is_final, text=text
        )
        if not by_silence.is_complete or self.semantic is None:
            return by_silence
        by_meaning = self.semantic.is_turn_complete(text=text, language=self.language)
        if by_meaning.is_complete:
            return TurnCompletion(is_complete=True, reason=f"{by_silence.reason} {by_meaning.reason}")
        return TurnCompletion(is_complete=False, reason=f"Silence met, but: {by_meaning.reason}")


class TurnStateMachine:
    """Explicit turn-taking states.

    Invalid transitions raise rather than being ignored, so a pipeline bug shows
    up as an error instead of as a call where both parties talk at once.
    """

    _ALLOWED: dict[TurnState, frozenset[TurnState]] = {
        TurnState.IDLE: frozenset({TurnState.LISTENING, TurnState.AGENT_SPEAKING, TurnState.ENDED}),
        TurnState.LISTENING: frozenset({TurnState.USER_SPEAKING, TurnState.AGENT_SPEAKING, TurnState.ENDED}),
        TurnState.USER_SPEAKING: frozenset({TurnState.PROCESSING, TurnState.LISTENING, TurnState.ENDED}),
        TurnState.PROCESSING: frozenset({TurnState.AGENT_SPEAKING, TurnState.LISTENING, TurnState.ENDED}),
        TurnState.AGENT_SPEAKING: frozenset({TurnState.LISTENING, TurnState.USER_SPEAKING, TurnState.ENDED}),
        TurnState.ENDED: frozenset(),
    }

    def __init__(self, state: TurnState = TurnState.IDLE) -> None:
        self._state = state

    @property
    def state(self) -> TurnState:
        return self._state

    def _to(self, target: TurnState) -> TurnState:
        if target not in self._ALLOWED[self._state]:
            raise InvalidTurnTransition(f"{self._state.value} -> {target.value}")
        self._state = target
        return self._state

    def start_listening(self) -> TurnState:
        return self._to(TurnState.LISTENING)

    def user_started_speaking(self) -> TurnState:
        return self._to(TurnState.USER_SPEAKING)

    def user_turn_complete(self) -> TurnState:
        return self._to(TurnState.PROCESSING)

    def agent_started_speaking(self) -> TurnState:
        return self._to(TurnState.AGENT_SPEAKING)

    def agent_finished_speaking(self) -> TurnState:
        return self._to(TurnState.LISTENING)

    def end_call(self) -> TurnState:
        return self._to(TurnState.ENDED)

    def on_customer_audio(self, decision: BargeInDecision) -> TurnState:
        """Apply a barge-in decision.

        A backchannel leaves the agent speaking. A real interruption stops TTS and
        hands the floor back. Outside AGENT_SPEAKING the decision changes nothing.
        """
        if self._state is not TurnState.AGENT_SPEAKING:
            return self._state
        if decision.stop_tts:
            return self._to(TurnState.USER_SPEAKING)
        return self._state
