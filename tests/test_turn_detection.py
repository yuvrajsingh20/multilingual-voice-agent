"""Turn taking and barge-in.

The behaviour under test is the one that decides whether the agent keeps talking
over the customer or hands back the floor.
"""

from __future__ import annotations

import pytest

from app.models.enums import BargeInClass, Language, TurnState
from app.services.turn import (
    BargeInSignal,
    HeuristicBargeInClassifier,
    InvalidTurnTransition,
    SilenceTurnDetector,
    CompositeTurnDetector,
    TurnCompletion,
    TurnStateMachine,
    VadResult,
)


def _signal(
    text: str | None,
    duration_ms: int = 400,
    *,
    is_speech: bool = True,
    probability: float = 0.9,
    language: Language | None = Language.HINGLISH,
) -> BargeInSignal:
    return BargeInSignal(
        vad=VadResult(
            is_speech=is_speech,
            speech_probability=probability,
            timestamp_ms=1000,
            duration_ms=duration_ms,
        ),
        partial_transcript=text,
        language=language,
    )


@pytest.fixture
def classifier() -> HeuristicBargeInClassifier:
    return HeuristicBargeInClassifier()


# --- backchannels keep the agent speaking -----------------------------------


@pytest.mark.parametrize("token", ["haan", "hmm", "okay", "ji", "achha", "theek hai"])
def test_backchannels_do_not_stop_tts(classifier, token: str) -> None:
    decision = classifier.classify(_signal(token, duration_ms=350))
    assert decision.classification is BargeInClass.BACKCHANNEL
    assert decision.stop_tts is False


def test_backchannel_matching_ignores_case_and_punctuation(classifier) -> None:
    assert classifier.classify(_signal("Haan!", duration_ms=300)).classification is BargeInClass.BACKCHANNEL


def test_marathi_backchannel_is_recognised(classifier) -> None:
    decision = classifier.classify(_signal("ho", duration_ms=300, language=Language.MARATHI))
    assert decision.classification is BargeInClass.BACKCHANNEL


def test_a_long_backchannel_word_is_treated_as_an_interruption(classifier) -> None:
    """Past the backchannel window the customer is taking the floor, whatever the word."""
    decision = classifier.classify(_signal("haan", duration_ms=2000))
    assert decision.classification is BargeInClass.INTERRUPTION
    assert decision.stop_tts is True


# --- real interruptions stop TTS --------------------------------------------


def test_a_sentence_stops_tts(classifier) -> None:
    decision = classifier.classify(_signal("ruko main abhi baat nahi kar sakta", duration_ms=1200))
    assert decision.classification is BargeInClass.INTERRUPTION
    assert decision.stop_tts is True


def test_sustained_speech_with_no_transcript_stops_tts(classifier) -> None:
    decision = classifier.classify(_signal(None, duration_ms=1500))
    assert decision.classification is BargeInClass.INTERRUPTION
    assert decision.stop_tts is True


def test_short_speech_with_no_transcript_waits(classifier) -> None:
    decision = classifier.classify(_signal(None, duration_ms=300))
    assert decision.classification is BargeInClass.UNKNOWN
    assert decision.stop_tts is False


# --- noise ------------------------------------------------------------------


def test_non_speech_is_noise(classifier) -> None:
    decision = classifier.classify(_signal("haan", is_speech=False))
    assert decision.classification is BargeInClass.NOISE
    assert decision.stop_tts is False


def test_low_confidence_speech_is_noise(classifier) -> None:
    assert classifier.classify(_signal("haan", probability=0.2)).classification is BargeInClass.NOISE


def test_a_click_is_too_short_to_count(classifier) -> None:
    assert classifier.classify(_signal("x", duration_ms=50)).classification is BargeInClass.NOISE


# --- state machine ----------------------------------------------------------


def test_a_normal_turn_cycle() -> None:
    machine = TurnStateMachine()
    assert machine.start_listening() is TurnState.LISTENING
    assert machine.user_started_speaking() is TurnState.USER_SPEAKING
    assert machine.user_turn_complete() is TurnState.PROCESSING
    assert machine.agent_started_speaking() is TurnState.AGENT_SPEAKING
    assert machine.agent_finished_speaking() is TurnState.LISTENING


def test_a_backchannel_leaves_the_agent_speaking(classifier) -> None:
    machine = TurnStateMachine(TurnState.AGENT_SPEAKING)
    machine.on_customer_audio(classifier.classify(_signal("hmm", duration_ms=300)))
    assert machine.state is TurnState.AGENT_SPEAKING


def test_an_interruption_hands_the_floor_back(classifier) -> None:
    machine = TurnStateMachine(TurnState.AGENT_SPEAKING)
    machine.on_customer_audio(classifier.classify(_signal("ek minute suniye", duration_ms=1100)))
    assert machine.state is TurnState.USER_SPEAKING


def test_customer_audio_outside_agent_speech_changes_nothing(classifier) -> None:
    machine = TurnStateMachine(TurnState.PROCESSING)
    machine.on_customer_audio(classifier.classify(_signal("ek minute suniye", duration_ms=1100)))
    assert machine.state is TurnState.PROCESSING


def test_an_invalid_transition_raises() -> None:
    machine = TurnStateMachine(TurnState.IDLE)
    with pytest.raises(InvalidTurnTransition):
        machine.user_turn_complete()


def test_an_ended_call_accepts_nothing_further() -> None:
    machine = TurnStateMachine(TurnState.LISTENING)
    machine.end_call()
    with pytest.raises(InvalidTurnTransition):
        machine.start_listening()


# --- turn detection ---------------------------------------------------------


def test_a_partial_hypothesis_never_ends_a_turn() -> None:
    detector = SilenceTurnDetector(silence_threshold_ms=700)
    assert not detector.is_turn_complete(trailing_silence_ms=5000, is_final=False, text="haan").is_complete


def test_silence_below_the_threshold_does_not_end_a_turn() -> None:
    detector = SilenceTurnDetector(silence_threshold_ms=700)
    assert not detector.is_turn_complete(trailing_silence_ms=699, is_final=True, text="haan").is_complete


def test_silence_at_the_threshold_ends_the_turn() -> None:
    detector = SilenceTurnDetector(silence_threshold_ms=700)
    assert detector.is_turn_complete(trailing_silence_ms=700, is_final=True, text="haan").is_complete


def test_composite_without_a_semantic_detector_matches_silence_only() -> None:
    composite = CompositeTurnDetector(SilenceTurnDetector(silence_threshold_ms=700))
    assert composite.is_turn_complete(trailing_silence_ms=800, is_final=True, text="ok").is_complete


def test_a_semantic_detector_can_veto_a_silence_based_end_of_turn() -> None:
    class MidThoughtDetector:
        def is_turn_complete(self, *, text: str, language: Language | None) -> TurnCompletion:
            return TurnCompletion(is_complete=False, reason="Utterance looks unfinished.")

    composite = CompositeTurnDetector(SilenceTurnDetector(silence_threshold_ms=700), MidThoughtDetector())
    result = composite.is_turn_complete(trailing_silence_ms=900, is_final=True, text="mera account number hai nau")
    assert result.is_complete is False
    assert "unfinished" in result.reason
