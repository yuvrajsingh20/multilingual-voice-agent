"""Routing, fallback, thresholds, fusion and measurement in the decision coordinator.

The provider here is :class:`ScriptedDecisionService`. It answers what the test
tells it to, so these tests pin what the *application* does with an answer - or
with the lack of one. They say nothing about whether Jev would give that answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.core.session import apply_event
from app.models.conversation import ConversationEvent, ConversationState
from app.models.enums import BargeInClass, EventKind, Intent, Language, ToneLevel, TurnState
from app.models.policy import PolicyDecision
from app.orchestrator import DecisionCoordinator
from app.runtime import build_runtime
from app.services.decision import (
    DecisionAnswer,
    DecisionAuthenticationFailed,
    DecisionConnectionFailed,
    DecisionError,
    DecisionMalformedResponse,
    DecisionName,
    DecisionOutcome,
    DecisionRateLimited,
    DecisionTimeout,
    DecisionUpstreamError,
    DisabledDecisionService,
    ScriptedDecisionService,
)
from app.services.decisions.fusion import (
    HALT_PHRASES,
    HIGH_RISK_INTENTS,
    LOW_RISK_INTENTS,
    BargeInAction,
    DecisionSource,
    EscalationAction,
    IntentAction,
    contains_halt_phrase,
)
from app.services.decisions.registry import CustomerIntentLabel
from app.services.turn import BargeInSignal, HeuristicBargeInClassifier, TurnStateMachine, VadResult
from tests.conftest import DEFAULT_NOW, RULES_PATH


def make_settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        default_timezone="Asia/Kolkata",
        policy_version="test-0.1.0",
        regulatory_rules_path=RULES_PATH,
    )
    base.update(overrides)
    return Settings(**base)


def coordinator(service, **settings) -> DecisionCoordinator:
    return DecisionCoordinator.from_runtime(build_runtime(make_settings(**settings), decisions=service))


def scripted(*items, delay: float = 0.0) -> ScriptedDecisionService:
    return ScriptedDecisionService(list(items), delay_seconds=delay)


def answer(name: DecisionName, label: str, confidence: float = 0.95) -> DecisionAnswer:
    return DecisionAnswer(
        name=name, label=label, confidence=confidence, provider="scripted", model="test-model"
    )


def signal(
    text: str | None,
    duration_ms: int = 400,
    *,
    is_speech: bool = True,
    probability: float = 0.9,
    language: Language | None = Language.ENGLISH,
) -> BargeInSignal:
    return BargeInSignal(
        vad=VadResult(
            is_speech=is_speech, speech_probability=probability, timestamp_ms=1000, duration_ms=duration_ms
        ),
        partial_transcript=text,
        language=language,
    )


def run(coro):
    return asyncio.run(coro)


BARGE = DecisionName.BARGE_IN
INTENT = DecisionName.CUSTOMER_INTENT
ESCALATION = DecisionName.HUMAN_ESCALATION


# --- disabled: behaviour is exactly what it was -------------------------------------


class CountingDisabled(DisabledDecisionService):
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, request):  # type: ignore[override]
        self.calls += 1
        return await super().decide(request)


BARGE_IN_MATRIX = [
    signal("hmm"),
    signal("yeah"),
    signal("uh huh"),
    signal("right"),
    signal("Yeah, I don't agree with that"),
    signal("stop"),
    signal("wait"),
    signal("no"),
    signal("one second"),
    signal("haan ji bilkul", language=Language.HINGLISH),
    signal("okay", duration_ms=1500),
    signal(None, duration_ms=300),
    signal(None, duration_ms=1200),
    signal("hmm", duration_ms=100),
    signal("hmm", is_speech=False),
    signal("hmm", probability=0.2),
]


@pytest.mark.parametrize("sig", BARGE_IN_MATRIX)
@pytest.mark.parametrize("agent_speaking", [True, False])
def test_disabled_barge_in_is_exactly_the_heuristic(sig: BargeInSignal, agent_speaking: bool) -> None:
    service = CountingDisabled()
    resolution = run(coordinator(service).resolve_barge_in(sig, agent_speaking=agent_speaking))

    assert resolution.decision == HeuristicBargeInClassifier().classify(sig)
    assert resolution.source is DecisionSource.HEURISTIC
    assert resolution.semantic is None
    assert service.calls == 0


def test_disabled_intent_and_escalation_take_the_existing_path() -> None:
    service = CountingDisabled()
    co = coordinator(service)
    intent = run(co.resolve_intent("I will pay on Friday"))
    escalation = run(
        co.resolve_escalation("I want a manager", state=ConversationState(session_id="s"), policy=None)
    )

    assert intent.action is IntentAction.EXISTING_PATH and intent.signal_intent is None
    assert escalation.action is EscalationAction.NO_CHANGE and not escalation.authoritative
    assert service.calls == 0
    assert co.enabled is False


def test_disabled_logs_no_decision_records(log_stream) -> None:
    co = coordinator(DisabledDecisionService())
    run(co.resolve_intent("hello"))
    run(co.resolve_barge_in(signal("haan ji bilkul"), agent_speaking=True))
    assert '"event": "decision"' not in log_stream.getvalue()


# --- barge-in: backchannel vs interruption ------------------------------------------


@pytest.mark.parametrize("text", ["hmm", "yeah", "uh-huh", "right"])
def test_a_semantic_backchannel_keeps_the_agent_speaking(text: str) -> None:
    service = scripted(answer(BARGE, "backchannel"))
    resolution = run(coordinator(service).resolve_barge_in(signal(text), agent_speaking=True))

    assert resolution.action is BargeInAction.CONTINUE_TTS
    assert resolution.decision.stop_tts is False
    assert resolution.source is DecisionSource.SEMANTIC
    assert service.requests[0].context.agent_speaking is True


@pytest.mark.parametrize("text", ["yeah", "right", "hmm"])
def test_a_backchannel_word_is_not_assumed_to_be_a_backchannel(text: str) -> None:
    """The heuristic calls these backchannels on vocabulary alone; context can overrule it."""
    assert HeuristicBargeInClassifier().classify(signal(text)).classification is BargeInClass.BACKCHANNEL

    service = scripted(answer(BARGE, "interruption"))
    resolution = run(
        coordinator(service).resolve_barge_in(
            signal(text), agent_speaking=True, recent_customer_utterances=("that is not what I owe",)
        )
    )
    assert resolution.action is BargeInAction.STOP_TTS
    assert resolution.decision.classification is BargeInClass.INTERRUPTION
    assert service.requests[0].context.recent_customer_utterances == ("that is not what I owe",)


def test_an_acknowledgement_followed_by_an_objection_stops_the_agent() -> None:
    sig = signal("Yeah, I don't agree with that")
    service = scripted(answer(BARGE, "interruption"))
    resolution = run(coordinator(service).resolve_barge_in(sig, agent_speaking=True))
    assert resolution.action is BargeInAction.STOP_TTS
    # And without a provider, the heuristic already stops for it.
    fallback = run(coordinator(DisabledDecisionService()).resolve_barge_in(sig, agent_speaking=True))
    assert fallback.action is BargeInAction.STOP_TTS


@pytest.mark.parametrize("text", ["no", "No, that's wrong", "arre nahi"])
def test_a_semantic_interruption_stops_the_agent(text: str) -> None:
    service = scripted(answer(BARGE, "interruption"))
    resolution = run(coordinator(service).resolve_barge_in(signal(text), agent_speaking=True))
    assert resolution.action is BargeInAction.STOP_TTS
    assert resolution.decision.stop_tts is True
    assert len(service.requests) == 1


@pytest.mark.parametrize("text", ["stop", "wait", "one second", "Wait, stop.", "ruko", "ek minute", "थांबा"])
def test_an_explicit_request_to_stop_is_never_left_to_the_provider(text: str) -> None:
    """Even a confident 'backchannel' cannot keep the agent talking over 'stop'."""
    service = scripted(answer(BARGE, "backchannel", 0.99))
    resolution = run(coordinator(service).resolve_barge_in(signal(text), agent_speaking=True))

    assert resolution.action is BargeInAction.STOP_TTS
    assert resolution.source is DecisionSource.HEURISTIC
    assert "explicit request to stop" in resolution.reason
    assert service.requests == []


def test_halt_phrases_match_whole_words_only() -> None:
    assert contains_halt_phrase("please WAIT!")
    assert contains_halt_phrase("hold on a moment")
    assert not contains_halt_phrase("stopwatch")
    assert not contains_halt_phrase("basically fine")
    assert all(contains_halt_phrase(phrase) for phrase in HALT_PHRASES)


def test_a_semantic_continuation_hands_the_floor_back() -> None:
    service = scripted(answer(BARGE, "continuation"))
    resolution = run(
        coordinator(service).resolve_barge_in(
            signal("so what I was saying"),
            agent_speaking=True,
            recent_customer_utterances=("I lost my job last month and",),
        )
    )
    assert resolution.action is BargeInAction.STOP_TTS
    assert "continuation" in resolution.reason


def test_a_multiword_backchannel_the_heuristic_misses_can_keep_the_agent_speaking() -> None:
    sig = signal("haan ji bilkul", language=Language.HINGLISH)
    assert HeuristicBargeInClassifier().classify(sig).stop_tts is True

    resolution = run(coordinator(scripted(answer(BARGE, "backchannel"))).resolve_barge_in(sig, agent_speaking=True))
    assert resolution.action is BargeInAction.CONTINUE_TTS


def test_sustained_speech_is_a_hard_interruption_the_provider_is_not_asked_about() -> None:
    service = scripted(answer(BARGE, "backchannel", 0.99))
    resolution = run(
        coordinator(service).resolve_barge_in(signal("haan haan theek hai", duration_ms=1500), agent_speaking=True)
    )
    assert resolution.action is BargeInAction.STOP_TTS
    assert "hard interruption signal" in resolution.reason
    assert service.requests == []


@pytest.mark.parametrize(
    ("sig", "agent_speaking", "expected"),
    [
        (signal("hmm", is_speech=False), True, BargeInAction.CONTINUE_TTS),
        (signal("hmm", probability=0.1), True, BargeInAction.CONTINUE_TTS),
        (signal("hmm", duration_ms=50), True, BargeInAction.CONTINUE_TTS),
        (signal(None, duration_ms=300), True, BargeInAction.WAIT),
        (signal("   ", duration_ms=300), True, BargeInAction.WAIT),
        (signal("yeah"), False, BargeInAction.CONTINUE_TTS),
    ],
    ids=["no-speech", "low-vad-probability", "too-short", "no-transcript", "blank-transcript", "agent-silent"],
)
def test_audio_the_acoustic_layer_settles_never_reaches_the_provider(sig, agent_speaking, expected) -> None:
    service = scripted(answer(BARGE, "interruption", 0.99))
    resolution = run(coordinator(service).resolve_barge_in(sig, agent_speaking=agent_speaking))
    assert resolution.action is expected
    assert resolution.source is DecisionSource.HEURISTIC
    assert service.requests == []


def test_a_resolution_drives_the_existing_turn_state_machine() -> None:
    stop = run(coordinator(scripted(answer(BARGE, "interruption"))).resolve_barge_in(signal("no"), agent_speaking=True))
    keep = run(coordinator(scripted(answer(BARGE, "backchannel"))).resolve_barge_in(signal("hmm"), agent_speaking=True))

    machine = TurnStateMachine(TurnState.AGENT_SPEAKING)
    assert machine.on_customer_audio(keep.decision) is TurnState.AGENT_SPEAKING
    assert machine.on_customer_audio(stop.decision) is TurnState.USER_SPEAKING


# --- fallback on every failure ------------------------------------------------------


FAILURES = [
    DecisionTimeout(),
    DecisionConnectionFailed(),
    DecisionAuthenticationFailed(401),
    DecisionRateLimited(),
    DecisionUpstreamError(503),
    DecisionMalformedResponse(),
    DecisionError(),
]


@pytest.mark.parametrize("failure", FAILURES, ids=lambda e: e.category)
def test_every_failure_falls_back_to_the_heuristic(failure: DecisionError) -> None:
    sig = signal("haan ji bilkul", language=Language.HINGLISH)
    resolution = run(coordinator(scripted(failure)).resolve_barge_in(sig, agent_speaking=True))

    assert resolution.decision == HeuristicBargeInClassifier().classify(sig)
    assert resolution.source is DecisionSource.HEURISTIC
    assert resolution.semantic is not None
    assert resolution.semantic.outcome is DecisionOutcome.FAILED
    assert resolution.semantic.fallback_used is True
    assert resolution.semantic.failure == failure.category


@pytest.mark.parametrize("failure", FAILURES, ids=lambda e: e.category)
def test_every_failure_leaves_intent_and_escalation_on_the_existing_path(failure: DecisionError) -> None:
    co = coordinator(scripted(failure, failure))
    intent = run(co.resolve_intent("I will pay on Friday"))
    escalation = run(co.resolve_escalation("manager please", state=ConversationState(session_id="s"), policy=None))
    assert intent.action is IntentAction.EXISTING_PATH and intent.signal_intent is None
    assert escalation.action is EscalationAction.NO_CHANGE and not escalation.authoritative


def test_a_provider_that_raises_anything_at_all_cannot_break_the_caller() -> None:
    class Exploding:
        provider = "exploding"

        async def decide(self, request):
            raise RuntimeError(f"provider exploded while reading {request.context.utterance}")

    resolution = run(coordinator(Exploding()).resolve_intent("I will pay on Friday"))
    assert resolution.action is IntentAction.EXISTING_PATH
    assert resolution.semantic.failure == "decision_failed"


def test_the_coordinator_enforces_the_deadline_on_a_provider_that_does_not() -> None:
    service = scripted(answer(BARGE, "backchannel"), delay=5.0)
    co = coordinator(service, jev_timeout_seconds=0.05)
    sig = signal("haan ji bilkul")

    started = time.perf_counter()
    resolution = run(co.resolve_barge_in(sig, agent_speaking=True))
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert resolution.semantic.failure == "decision_timeout"
    assert resolution.decision == HeuristicBargeInClassifier().classify(sig)


def test_an_answer_to_a_different_question_is_malformed() -> None:
    service = scripted(answer(INTENT, "dispute"))
    resolution = run(coordinator(service).resolve_barge_in(signal("haan ji bilkul"), agent_speaking=True))
    assert resolution.semantic.failure == "decision_malformed_response"
    assert resolution.source is DecisionSource.HEURISTIC


def test_a_label_the_registry_does_not_offer_is_malformed_whatever_the_provider() -> None:
    service = scripted(answer(INTENT, "approve_waiver", 0.99))
    resolution = run(coordinator(service).resolve_intent("please waive my charges"))
    assert resolution.semantic.failure == "decision_malformed_response"
    assert resolution.action is IntentAction.EXISTING_PATH


# --- thresholds -----------------------------------------------------------------------


def test_below_the_threshold_is_uncertain_and_falls_back() -> None:
    sig = signal("haan ji bilkul", language=Language.HINGLISH)
    resolution = run(coordinator(scripted(answer(BARGE, "backchannel", 0.84))).resolve_barge_in(sig, agent_speaking=True))
    assert resolution.semantic.outcome is DecisionOutcome.UNCERTAIN
    assert resolution.semantic.decision == "UNCERTAIN"
    assert resolution.semantic.top_label == "backchannel"
    assert resolution.action is BargeInAction.STOP_TTS  # the heuristic's answer
    assert resolution.source is DecisionSource.HEURISTIC


def test_each_decision_has_its_own_threshold() -> None:
    """The same 0.82 is enough for an intent, not for keeping TTS on or recommending a human."""
    co = coordinator(
        scripted(answer(BARGE, "backchannel", 0.82), answer(INTENT, "refusal", 0.82), answer(ESCALATION, "yes", 0.82))
    )
    barge = run(co.resolve_barge_in(signal("haan ji bilkul"), agent_speaking=True))
    intent = run(co.resolve_intent("I won't pay"))
    escalation = run(co.resolve_escalation("get me a manager", state=ConversationState(session_id="s"), policy=None))

    assert barge.semantic.outcome is DecisionOutcome.UNCERTAIN
    assert intent.semantic.outcome is DecisionOutcome.DECIDED
    assert escalation.semantic.outcome is DecisionOutcome.UNCERTAIN


def test_thresholds_come_from_configuration() -> None:
    co = coordinator(scripted(answer(INTENT, "refusal", 0.82)), jev_intent_threshold=0.9)
    assert co.threshold(INTENT) == 0.9
    assert run(co.resolve_intent("I won't pay")).action is IntentAction.EXISTING_PATH


# --- customer intent ---------------------------------------------------------------


def test_an_already_made_payment_is_classified_and_left_to_the_application() -> None:
    resolution = run(
        coordinator(scripted(answer(INTENT, "payment_already_made"))).resolve_intent("I already paid this yesterday.")
    )
    assert resolution.label is CustomerIntentLabel.PAYMENT_ALREADY_MADE
    assert resolution.semantic.confidence == pytest.approx(0.95)
    assert resolution.action is IntentAction.EXISTING_PATH
    assert resolution.application_intent is None
    assert resolution.signal_intent is None


@pytest.mark.parametrize("label", list(HIGH_RISK_INTENTS))
def test_a_high_risk_intent_is_returned_for_confirmation_and_never_applied(label: CustomerIntentLabel) -> None:
    resolution = run(coordinator(scripted(answer(INTENT, label.value, 0.99))).resolve_intent("..."))
    assert resolution.action is IntentAction.CONFIRM_FIRST
    assert resolution.application_intent is HIGH_RISK_INTENTS[label]
    assert resolution.signal_intent is None


@pytest.mark.parametrize("label", list(LOW_RISK_INTENTS))
def test_a_low_risk_intent_may_be_applied(label: CustomerIntentLabel) -> None:
    resolution = run(coordinator(scripted(answer(INTENT, label.value))).resolve_intent("..."))
    assert resolution.action is IntentAction.APPLY_SIGNAL
    assert resolution.signal_intent is LOW_RISK_INTENTS[label]


@pytest.mark.parametrize(
    "label",
    [label for label in CustomerIntentLabel if label not in HIGH_RISK_INTENTS and label not in LOW_RISK_INTENTS],
)
def test_an_intent_with_no_application_meaning_takes_the_existing_path(label: CustomerIntentLabel) -> None:
    resolution = run(coordinator(scripted(answer(INTENT, label.value))).resolve_intent("..."))
    assert resolution.action is IntentAction.EXISTING_PATH
    assert resolution.label is label
    assert resolution.signal_intent is None


def test_an_upstream_intent_is_never_second_guessed_or_sent_anywhere() -> None:
    service = scripted(answer(INTENT, "dispute", 0.99))
    resolution = run(coordinator(service).resolve_intent("haan kal", caller_intent=Intent.PAYMENT_PROMISE))
    assert resolution.action is IntentAction.EXISTING_PATH
    assert resolution.source is DecisionSource.UPSTREAM
    assert service.requests == []


@pytest.mark.parametrize("utterance", ["", "   ", "x" * 5000])
def test_an_utterance_that_is_empty_or_too_long_is_not_sent(utterance: str) -> None:
    service = scripted(answer(INTENT, "dispute"))
    resolution = run(coordinator(service).resolve_intent(utterance))
    assert resolution.action is IntentAction.EXISTING_PATH
    assert service.requests == []


def test_the_risk_tiers_cover_every_label_exactly_once() -> None:
    low, high = set(LOW_RISK_INTENTS), set(HIGH_RISK_INTENTS)
    assert not low & high
    assert low | high <= set(CustomerIntentLabel)


def _reduce(intent: Intent) -> tuple[ConversationState, ConversationState]:
    before = ConversationState(session_id="s", identity_verified=True)
    event = ConversationEvent(
        event_id="e1",
        session_id="s",
        turn_id=1,
        occurred_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
        kind=EventKind.USER_UTTERANCE,
        data={"intent": intent.value},
    )
    return before, apply_event(before, event)


def _changed(before: ConversationState, after: ConversationState) -> set[str]:
    a, b = before.model_dump(), after.model_dump()
    return {key for key in a if a[key] != b[key]} - {"turn_count"}


@pytest.mark.parametrize("intent", list(LOW_RISK_INTENTS.values()))
def test_a_low_risk_intent_changes_nothing_but_the_recorded_intent(intent: Intent) -> None:
    """Pins the tiering to the real reducer, so a reducer change cannot silently raise the stakes."""
    assert _changed(*_reduce(intent)) == {"intent"}


@pytest.mark.parametrize("intent", list(HIGH_RISK_INTENTS.values()))
def test_every_high_risk_intent_really_does_change_gating_state(intent: Intent) -> None:
    assert _changed(*_reduce(intent)) - {"intent"}


# --- escalation ----------------------------------------------------------------------


def _policy(escalate: bool) -> PolicyDecision:
    return PolicyDecision(
        allowed=True,
        escalate=escalate,
        tone=ToneLevel.NEUTRAL,
        dpd_stage=None,
        evaluated_at=DEFAULT_NOW,
        policy_version="test",
        rules_version="test",
    )


def test_a_policy_escalation_stands_whatever_the_provider_would_say() -> None:
    service = scripted(answer(ESCALATION, "no", 0.99))
    resolution = run(
        coordinator(service).resolve_escalation(
            "all good", state=ConversationState(session_id="s"), policy=_policy(escalate=True)
        )
    )
    assert resolution.action is EscalationAction.ESCALATE
    assert resolution.authoritative is True
    assert resolution.source is DecisionSource.POLICY
    assert service.requests == []


def test_a_state_escalation_stands_too() -> None:
    service = scripted(answer(ESCALATION, "no", 0.99))
    state = ConversationState(session_id="s", escalation_required=True)
    resolution = run(coordinator(service).resolve_escalation("fine", state=state, policy=None))
    assert resolution.action is EscalationAction.ESCALATE and resolution.authoritative
    assert service.requests == []


def test_a_semantic_yes_is_only_a_recommendation() -> None:
    state = ConversationState(session_id="s")
    policy = _policy(escalate=False)
    resolution = run(
        coordinator(scripted(answer(ESCALATION, "yes", 0.97))).resolve_escalation(
            "my father passed away last week", state=state, policy=policy
        )
    )
    assert resolution.action is EscalationAction.RECOMMEND_REVIEW
    assert resolution.authoritative is False
    assert state.escalation_required is False and policy.escalate is False


def test_a_semantic_no_changes_nothing() -> None:
    resolution = run(
        coordinator(scripted(answer(ESCALATION, "no"))).resolve_escalation(
            "okay tell me the amount", state=ConversationState(session_id="s"), policy=None
        )
    )
    assert resolution.action is EscalationAction.NO_CHANGE and not resolution.authoritative


def test_an_uncertain_yes_changes_nothing() -> None:
    resolution = run(
        coordinator(scripted(answer(ESCALATION, "yes", 0.6))).resolve_escalation(
            "hmm I don't know", state=ConversationState(session_id="s"), policy=None
        )
    )
    assert resolution.action is EscalationAction.NO_CHANGE
    assert resolution.semantic.outcome is DecisionOutcome.UNCERTAIN


# --- measurement ------------------------------------------------------------------------


def _decision_records(stream) -> list[dict]:
    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return [record for record in records if record.get("event") == "decision"]


def test_every_decision_attempt_is_measured_without_the_customers_words(log_stream) -> None:
    co = coordinator(
        scripted(answer(INTENT, "payment_already_made", 0.95), DecisionTimeout(), answer(ESCALATION, "yes", 0.5))
    )
    words = "I already paid, my PAN is ABCDE1234F and my number is 9876543210"
    run(co.resolve_intent(words, session_id="sess-42"))
    run(co.resolve_barge_in(signal("haan ji bilkul"), agent_speaking=True, session_id="sess-42"))
    run(co.resolve_escalation(words, state=ConversationState(session_id="s"), policy=None, session_id="sess-42"))

    records = _decision_records(log_stream)
    assert [r["decision_name"] for r in records] == ["customer_intent", "barge_in", "needs_human_escalation"]
    assert [r["outcome"] for r in records] == ["decided", "failed", "uncertain"]
    assert [r["fallback_used"] for r in records] == [False, True, True]
    assert [r["fallback_reason"] for r in records] == [None, "decision_timeout", "low_confidence"]
    assert [r["confidence_bucket"] for r in records] == ["ge_0.90", "none", "0.50_0.70"]
    assert records[1]["level"] == "WARNING"
    for record in records:
        assert record["session_id"] == "sess-42"
        assert record["provider"] == "scripted"
        assert record["latency_ms"] >= 0
    output = log_stream.getvalue()
    for fragment in ("already paid", "ABCDE1234F", "9876543210", "haan ji", "PAN"):
        assert fragment not in output


def test_routing_skips_are_not_measured_as_decisions(log_stream) -> None:
    co = coordinator(scripted())
    run(co.resolve_barge_in(signal("hmm", is_speech=False), agent_speaking=True))
    run(co.resolve_intent("x", caller_intent=Intent.REFUSAL))
    assert _decision_records(log_stream) == []


# --- concurrency -------------------------------------------------------------------------


def test_concurrent_decisions_run_side_by_side() -> None:
    service = scripted(*(answer(INTENT, "refusal") for _ in range(20)), delay=0.05)
    co = coordinator(service, jev_timeout_seconds=2.0)

    async def many():
        return await asyncio.gather(*(co.resolve_intent(f"no {n}") for n in range(20)))

    started = time.perf_counter()
    results = run(many())
    # Serially this is 1.0 s. Side by side it is one delay plus overhead; the
    # bound is loose for slow CI but far below what a per-call block allows.
    assert time.perf_counter() - started < 0.4
    assert all(r.action is IntentAction.APPLY_SIGNAL for r in results)


def _max_loop_lag_ms(co: DecisionCoordinator) -> float:
    """Largest gap between 5 ms ticks while one intent decision is pending."""
    gaps: list[float] = []

    async def ticker(done: asyncio.Event):
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append((now - last) * 1000.0)
            last = now

    async def both():
        done = asyncio.Event()
        tick = asyncio.ensure_future(ticker(done))
        await asyncio.sleep(0)  # let the ticker take its first timestamp
        await co.resolve_intent("no")
        done.set()
        await tick

    run(both())
    return max(gaps)


class BlockingService:
    """Breaks the async contract: blocks the event loop while it 'decides'."""

    provider = "blocking"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    async def decide(self, request):
        time.sleep(self.seconds)
        return answer(INTENT, "refusal")


def test_the_coordinator_does_not_block_the_event_loop() -> None:
    """While a decision waits on its provider, other work on the loop keeps running."""
    co = coordinator(scripted(answer(INTENT, "refusal"), delay=0.2), jev_timeout_seconds=2.0)
    # Normal lag is about 6 ms (5 ms ticks); 25 ms catches a real block.
    assert _max_loop_lag_ms(co) < 25


def test_the_loop_lag_measurement_does_detect_a_blocked_loop() -> None:
    """The check above can fail: a provider that blocks shows up as lag."""
    co = coordinator(BlockingService(0.15), jev_timeout_seconds=2.0)
    assert _max_loop_lag_ms(co) >= 120


def test_an_answer_that_arrives_after_the_deadline_is_discarded() -> None:
    """A provider that blocks the loop answers late; its answer is stale."""
    resolution = run(coordinator(BlockingService(0.3), jev_timeout_seconds=0.1).resolve_intent("no"))
    assert resolution.semantic.outcome is DecisionOutcome.FAILED
    assert resolution.semantic.failure == "decision_timeout"
    assert resolution.action is IntentAction.EXISTING_PATH


# --- adversarial-review regressions -------------------------------------------------
#
# Each of these pins a defect the review of this stage found. See REPORT.md SD.14.


#: Kept separate from HALT_PHRASES on purpose: a gap in the list must show up as a
#: failure here, not be hidden by a test that checks the list against itself.
STOP_VARIANTS = [
    "ruk jao", "ruk ja", "ruk jaiye", "arre ruko na", "thehro", "ek minit", "hang on",
    "रुक जाओ", "रुक जाइए", "ठहरो", "थांब", "थांबा जरा", "एक सेकंड", "एक मिनिट", "एक मिनट",
    "रुको‌", "रु‍को",  # zero-width non-joiner / joiner inside the word
    "Stop!", "WAIT...", "hold on, hold on",
]


@pytest.mark.parametrize("text", STOP_VARIANTS)
def test_common_ways_to_say_stop_are_hard_signals(text: str) -> None:
    assert contains_halt_phrase(text)
    service = scripted(answer(BARGE, "backchannel", 0.99))
    resolution = run(coordinator(service).resolve_barge_in(signal(text, 600), agent_speaking=True))
    assert resolution.action is BargeInAction.STOP_TTS
    assert service.requests == []


@pytest.mark.parametrize("text", ["haan ji", "achha theek hai", "okay go on", "busy", "trucking", "रुकावट नहीं"])
def test_ordinary_words_are_not_mistaken_for_stop(text: str) -> None:
    assert not contains_halt_phrase(text)


@pytest.mark.parametrize("duration_ms", [250, 400, 700])  # past the 200 ms acoustic gate
def test_the_provider_wait_is_charged_to_the_acknowledgement_window(duration_ms: int) -> None:
    """Speech already heard plus the provider wait never exceeds the hard bound.

    The agent keeps talking while a provider is consulted, so a provider that
    never answers must not stretch the window: it gets only what is left.
    """
    service = scripted(answer(BARGE, "backchannel"), delay=10.0)
    co = coordinator(service, jev_timeout_seconds=5.0)

    started = time.perf_counter()
    resolution = run(co.resolve_barge_in(signal("haan ji bilkul", duration_ms), agent_speaking=True))
    waited_ms = (time.perf_counter() - started) * 1000.0

    assert duration_ms + waited_ms <= 800 + 60  # scheduling slack only
    assert resolution.semantic.failure == "decision_timeout"
    assert resolution.decision == HeuristicBargeInClassifier().classify(signal("haan ji bilkul", duration_ms))


def test_no_provider_call_when_the_window_is_all_but_spent() -> None:
    service = scripted(answer(BARGE, "backchannel"))
    resolution = run(coordinator(service).resolve_barge_in(signal("haan ji bilkul", 780), agent_speaking=True))
    assert service.requests == []
    assert resolution.source is DecisionSource.HEURISTIC


@pytest.mark.parametrize(
    "returned",
    [None, {"label": "backchannel"}, "backchannel",
     DecisionAnswer.model_construct(name=BARGE, label="backchannel", confidence=float("nan"), provider="x"),
     DecisionAnswer.model_construct(name=BARGE, label="backchannel", confidence=None, provider="x"),
     DecisionAnswer.model_construct(name=BARGE, label="backchannel", confidence="0.99", provider="x"),
     DecisionAnswer.model_construct(name=BARGE, label="backchannel", provider="x"),
     DecisionAnswer.model_construct(name=BARGE, label=["backchannel"], confidence=0.99, provider="x"),
     DecisionAnswer.model_construct(label="backchannel", confidence=0.99, provider="x"),
     # Lax validation would turn True into a *decided* 1.0; strict re-validation refuses it.
     DecisionAnswer.model_construct(name=BARGE, label="backchannel", confidence=True, provider="x")],
    ids=["none", "dict", "str", "unvalidated-nan", "unvalidated-none-confidence",
         "unvalidated-str-confidence", "unvalidated-missing-confidence", "unvalidated-list-label",
         "unvalidated-missing-name", "unvalidated-bool-confidence"],
)
def test_a_provider_that_returns_garbage_falls_back_instead_of_raising(returned) -> None:
    class Garbage:
        provider = "garbage"

        async def decide(self, request):
            return returned

    sig = signal("haan ji bilkul")
    resolution = run(coordinator(Garbage()).resolve_barge_in(sig, agent_speaking=True))
    assert resolution.decision == HeuristicBargeInClassifier().classify(sig)
    assert resolution.semantic.failure == "decision_malformed_response"


@pytest.mark.parametrize("outcome", [DecisionOutcome.UNCERTAIN, DecisionOutcome.FAILED, DecisionOutcome.DISABLED])
def test_a_result_that_is_not_decided_cannot_carry_a_label(outcome: DecisionOutcome) -> None:
    from pydantic import ValidationError

    from app.services.decision import DecisionResult

    with pytest.raises(ValidationError):
        DecisionResult(name=BARGE, outcome=outcome, label="backchannel", threshold=0.85, provider="x")


def test_fusion_ignores_a_label_on_a_result_that_is_not_decided() -> None:
    """Belt and braces behind the validator: even a hand-built contradictory
    result (which the validator makes impossible to construct normally) gives
    fusion nothing to act on."""
    from app.services.decision import DecisionResult
    from app.services.decisions.fusion import _usable

    for outcome in (DecisionOutcome.UNCERTAIN, DecisionOutcome.FAILED, DecisionOutcome.DISABLED):
        bogus = DecisionResult.model_construct(
            name=BARGE, outcome=outcome, label="backchannel", threshold=0.85, provider="x"
        )
        assert _usable(bogus, BARGE) is None
    decided = DecisionResult(
        name=BARGE, outcome=DecisionOutcome.DECIDED, label="backchannel", threshold=0.85, provider="x"
    )
    assert _usable(decided, BARGE) == "backchannel"
    assert _usable(decided, INTENT) is None


def test_a_loop_stalled_by_other_work_also_discards_a_well_behaved_answer(log_stream) -> None:
    """The discard is about the moment passing, not about blaming the provider.

    The record carries the deadline that applied, so a timeout on a small
    barge-in budget is distinguishable from one on the configured deadline.
    """
    service = scripted(answer(INTENT, "refusal"), delay=0.02)  # awaits; never blocks
    co = coordinator(service, jev_timeout_seconds=0.1)

    async def stall_while_pending():
        async def hog():
            await asyncio.sleep(0.005)
            time.sleep(0.15)  # someone else's work blocks the loop

        hog_task = asyncio.ensure_future(hog())
        resolution = await co.resolve_intent("no", session_id="s-1")
        await hog_task
        return resolution

    resolution = run(stall_while_pending())
    assert resolution.semantic.failure == "decision_timeout"
    (record,) = _decision_records(log_stream)
    assert record["deadline_ms"] == 100.0


def test_the_record_carries_the_barge_in_window_budget(log_stream) -> None:
    co = coordinator(scripted(answer(BARGE, "backchannel")), jev_timeout_seconds=0.5)
    run(co.resolve_barge_in(signal("haan ji bilkul", 700), agent_speaking=True))
    (record,) = _decision_records(log_stream)
    assert record["deadline_ms"] == pytest.approx(100.0)
