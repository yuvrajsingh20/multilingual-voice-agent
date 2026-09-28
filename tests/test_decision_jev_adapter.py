"""The Jev adapter, driven through the real ``typesafe-sdk`` over a mock transport.

Nothing here reaches a network and nothing needs a TypeSafe key. The SDK is the
pinned official one; only its transport is replaced, so request building,
response parsing and the SDK's own exception mapping all run for real. What is
*not* exercised is Jev itself: every answer below was written by hand, so these
tests prove the plumbing, not the model's accuracy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time

import httpx2
import pytest

from app.services.decision import (
    DecisionAuthenticationFailed,
    DecisionConfigurationError,
    DecisionConnectionFailed,
    DecisionContext,
    DecisionError,
    DecisionMalformedResponse,
    DecisionName,
    DecisionRateLimited,
    DecisionRequest,
    DecisionRequestRejected,
    DecisionTimeout,
    DecisionUpstreamError,
)
from app.services.decision_jev import DEFAULT_BASE_URL, SDK_LOGGER, JevDecisionService
from app.services.decisions.registry import DECISION_REGISTRY, get_spec

KEY = "test-key-5f2c9a"
UTTERANCE = "I already paid this yesterday, my PAN is ABCDE1234F"
TOP_LABEL = {
    DecisionName.BARGE_IN: "interruption",
    DecisionName.CUSTOMER_INTENT: "payment_already_made",
    DecisionName.HUMAN_ESCALATION: "no",
}


def _service(handler, **kwargs) -> JevDecisionService:
    options = dict(api_key=KEY, model="jev-1.13.0", timeout_seconds=0.5)
    options.update(kwargs)
    return JevDecisionService(transport=httpx2.MockTransport(handler), **options)


def _request(name: DecisionName = DecisionName.CUSTOMER_INTENT, **context) -> DecisionRequest:
    fields = dict(utterance=UTTERANCE)
    fields.update(context)
    return DecisionRequest(name=name, context=DecisionContext(**fields))


def _body(
    name: DecisionName = DecisionName.CUSTOMER_INTENT,
    *,
    choice: str | None = None,
    confidence: float = 0.93,
    probabilities: dict | None = None,
    model: str = "jev-1.13.0",
) -> dict:
    choice = choice or TOP_LABEL[name]
    return {
        "model": model,
        "usage": {"input_tokens": 57, "output_tokens": 1},
        "answers": {
            name.value: {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probabilities if probabilities is not None else {choice: 0.95},
            }
        },
    }


def _respond(body=None, *, status: int = 200, raw: bytes | None = None, seen: list | None = None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        if raw is not None:
            return httpx2.Response(status, content=raw, headers={"content-type": "application/json"})
        return httpx2.Response(status, json=body if body is not None else _body())

    return handler


def decide(service: JevDecisionService, request: DecisionRequest | None = None):
    async def _run():
        try:
            return await service.decide(request or _request())
        finally:
            await service.aclose()

    return asyncio.run(_run())


# --- success -------------------------------------------------------------------


def test_a_valid_answer_is_parsed_into_the_provider_neutral_type() -> None:
    answer = decide(_service(_respond(_body(probabilities={"payment_already_made": 0.95, "dispute": 0.05}))))
    assert answer.name is DecisionName.CUSTOMER_INTENT
    assert answer.label == "payment_already_made"
    assert answer.confidence == pytest.approx(0.93)
    assert answer.probabilities == {"payment_already_made": 0.95, "dispute": 0.05}
    assert answer.provider == "jev"
    assert answer.model == "jev-1.13.0"
    assert answer.input_tokens == 57
    assert answer.latency_ms > 0


@pytest.mark.parametrize("name", list(DecisionName))
def test_every_registered_decision_round_trips(name: DecisionName) -> None:
    answer = decide(_service(_respond(_body(name))), _request(name))
    assert answer.name is name
    assert answer.label == TOP_LABEL[name]


def test_a_provider_controlled_model_name_is_recorded_only_if_it_looks_like_one() -> None:
    answer = decide(_service(_respond(_body(model="jev\n<script>alert(1)</script>"))))
    assert answer.model is None


# --- the request is the documented contract --------------------------------------


def test_the_request_is_one_choice_question_to_the_systemone_endpoint() -> None:
    seen: list[httpx2.Request] = []
    decide(_service(_respond(seen=seen)))

    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == f"{DEFAULT_BASE_URL}/v1/systemone"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert set(body) == {"state", "model", "questions"}
    assert body["model"] == "jev-1.13.0"
    spec = get_spec(DecisionName.CUSTOMER_INTENT)
    assert body["questions"] == {
        spec.key: {"type": "choice", "instructions": spec.instructions, "criteria": dict(spec.criteria)}
    }


@pytest.mark.parametrize("name", list(DecisionName))
def test_only_the_declared_context_reaches_the_provider(name: DecisionName) -> None:
    seen: list[httpx2.Request] = []
    request = _request(
        name,
        language="mr",
        agent_speaking=True,
        stage="negotiation",
        recent_customer_utterances=("mala vel dya",),
    )
    decide(_service(_respond(_body(name), seen=seen)), request)

    state = json.loads(seen[0].content)["state"]
    allowed = {
        "utterance": "customer_utterance",
        "language": "language",
        "agent_speaking": "agent_is_speaking",
        "stage": "conversation_stage",
        "recent_customer_utterances": "recent_customer_utterances",
    }
    assert set(state) == {allowed[field] for field in DECISION_REGISTRY[name].context_fields}


def test_the_configured_base_url_is_what_is_dialled() -> None:
    seen: list[httpx2.Request] = []
    decide(_service(_respond(seen=seen), base_url="https://ai-gateway.example/typesafe/"))
    assert str(seen[0].url) == "https://ai-gateway.example/typesafe/v1/systemone"


def test_the_sdks_own_environment_variables_are_never_consulted(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key-should-not-be-used")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://env.example")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-latest")
    seen: list[httpx2.Request] = []
    decide(_service(_respond(seen=seen)))

    (request,) = seen
    assert request.url.host == "api.typesafe.ai"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content)["model"] == "jev-1.13.0"


# --- timeouts and transport failures ---------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [httpx2.ReadTimeout("slow"), httpx2.ConnectTimeout("slow"), httpx2.PoolTimeout("slow")],
    ids=lambda e: type(e).__name__,
)
def test_an_sdk_timeout_becomes_a_decision_timeout(exc: Exception) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exc

    with pytest.raises(DecisionTimeout):
        decide(_service(handler))


def test_the_whole_call_is_bounded_by_the_deadline() -> None:
    """A server that answers too slowly is cut off at the deadline, not after it."""

    async def handler(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(5)
        return httpx2.Response(200, json=_body())

    started = time.perf_counter()
    with pytest.raises(DecisionTimeout):
        decide(_service(handler, timeout_seconds=0.1))
    assert time.perf_counter() - started < 1.5


@pytest.mark.parametrize(
    "exc",
    [httpx2.ConnectError("refused"), httpx2.ReadError("reset"), httpx2.RemoteProtocolError("bad")],
    ids=lambda e: type(e).__name__,
)
def test_a_transport_failure_is_a_connection_failure_not_uncertainty(exc: Exception) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exc

    with pytest.raises(DecisionConnectionFailed):
        decide(_service(handler))


# --- HTTP errors -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, DecisionAuthenticationFailed),
        (403, DecisionAuthenticationFailed),
        (429, DecisionRateLimited),
        (400, DecisionRequestRejected),
        (404, DecisionRequestRejected),
        (422, DecisionRequestRejected),
        (500, DecisionUpstreamError),
        (502, DecisionUpstreamError),
        (503, DecisionUpstreamError),
        (504, DecisionUpstreamError),
    ],
)
def test_every_http_error_is_classified(status: int, error: type[DecisionError]) -> None:
    with pytest.raises(error) as excinfo:
        decide(_service(_respond({"error": "nope"}, status=status)))
    if hasattr(excinfo.value, "status_code"):
        assert excinfo.value.status_code == status


def test_an_error_never_carries_the_provider_body_the_key_or_the_customer_text() -> None:
    """The provider's message can echo the request. None of it survives the boundary."""
    echo = {"error": f"invalid key {KEY}; request was: {UTTERANCE}"}
    with pytest.raises(DecisionAuthenticationFailed) as excinfo:
        decide(_service(_respond(echo, status=401)))

    error = excinfo.value
    chain = []
    seen_ids = set()
    node: BaseException | None = error
    while node is not None and id(node) not in seen_ids:
        seen_ids.add(id(node))
        chain.append(node)
        node = node.__cause__ or node.__context__
    assert chain == [error], "no SDK exception may be chained"
    for rendered in (str(error), repr(error), error.detail, str(error.args)):
        assert KEY not in rendered
        assert "ABCDE1234F" not in rendered
        assert "already paid" not in rendered


# --- malformed answers -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [b"<html>gateway error</html>", b"", b"[]", b'{"answers": {}}'],
    ids=["html", "empty", "array", "no-model-no-usage"],
)
def test_a_body_that_is_not_the_documented_shape_is_malformed(raw: bytes) -> None:
    with pytest.raises(DecisionMalformedResponse):
        decide(_service(_respond(raw=raw)))


def _nan_body() -> bytes:
    return (
        b'{"model":"jev-1.13.0","usage":{"input_tokens":1,"output_tokens":1},"answers":'
        b'{"customer_intent":{"type":"choice","choice":"dispute","confidence":NaN,'
        b'"probabilities":{"dispute":0.9}}}}'
    )


@pytest.mark.parametrize(
    "body",
    [
        # The SDK accepts every one of these. The adapter must not.
        _body(choice="approve_waiver"),
        _body(confidence=1.5),
        _body(confidence=-0.2),
        _body(probabilities={"dispute": 0.7, "approve_waiver": 0.3}),
        _body(probabilities={"dispute": 1.7}),
        {"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": {}},
        {
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"some_other_question": {"type": "noul", "noul": 0.9}},
        },
        {
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"customer_intent": {"type": "noul", "noul": 0.9}},
        },
    ],
    ids=[
        "label-not-offered",
        "confidence-above-one",
        "confidence-negative",
        "probability-for-unoffered-label",
        "probability-above-one",
        "no-answers",
        "answer-to-another-question",
        "wrong-answer-type",
    ],
)
def test_an_answer_the_sdk_accepts_but_is_unusable_is_malformed(body: dict) -> None:
    with pytest.raises(DecisionMalformedResponse):
        decide(_service(_respond(body)))


def test_a_nan_confidence_is_malformed() -> None:
    with pytest.raises(DecisionMalformedResponse):
        decide(_service(_respond(raw=_nan_body())))


def test_a_confidence_at_the_bounds_is_accepted() -> None:
    assert decide(_service(_respond(_body(confidence=0.0)))).confidence == 0.0
    assert decide(_service(_respond(_body(confidence=1.0)))).confidence == 1.0


# --- retries ----------------------------------------------------------------------


def _sequence(*responses: httpx2.Response | Exception, seen: list):
    queue = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handler


def test_nothing_is_retried_by_default() -> None:
    seen: list = []
    handler = _sequence(httpx2.Response(503, json={}), httpx2.Response(200, json=_body()), seen=seen)
    with pytest.raises(DecisionUpstreamError):
        decide(_service(handler))
    assert len(seen) == 1


def test_a_configured_retry_recovers_a_transient_failure_inside_the_deadline() -> None:
    seen: list = []
    handler = _sequence(httpx2.Response(503, json={}), httpx2.Response(200, json=_body()), seen=seen)
    answer = decide(_service(handler, max_retries=1, timeout_seconds=1.0))
    assert answer.label == "payment_already_made"
    assert len(seen) == 2


def test_a_timeout_is_never_retried_even_when_retries_are_on() -> None:
    seen: list = []
    handler = _sequence(httpx2.ReadTimeout("slow"), httpx2.Response(200, json=_body()), seen=seen)
    with pytest.raises(DecisionTimeout):
        decide(_service(handler, max_retries=2))
    assert len(seen) == 1


@pytest.mark.parametrize("status", [400, 401, 422, 501])
def test_a_non_transient_status_is_never_retried(status: int) -> None:
    seen: list = []
    handler = _sequence(httpx2.Response(status, json={}), httpx2.Response(200, json=_body()), seen=seen)
    with pytest.raises(DecisionError):
        decide(_service(handler, max_retries=2))
    assert len(seen) == 1


def test_retries_stop_at_the_configured_count() -> None:
    seen: list = []
    handler = _sequence(*(httpx2.Response(503, json={}) for _ in range(5)), seen=seen)
    with pytest.raises(DecisionUpstreamError):
        decide(_service(handler, max_retries=2, timeout_seconds=2.0))
    assert len(seen) == 3


# --- construction -----------------------------------------------------------------


@pytest.mark.parametrize("key", ["", "   ", "has space", "tab\tkey", "ключ"])
def test_an_unusable_key_is_a_configuration_error_that_does_not_echo_it(key: str) -> None:
    with pytest.raises(DecisionConfigurationError) as excinfo:
        JevDecisionService(api_key=key, model="jev-1.13.0", timeout_seconds=0.5)
    if key.strip():
        assert key not in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs",
    [{"model": ""}, {"model": "  "}, {"timeout_seconds": 0}, {"timeout_seconds": math.inf}, {"max_retries": -1}],
)
def test_an_unusable_setting_is_a_configuration_error(kwargs) -> None:
    options = dict(api_key=KEY, model="jev-1.13.0", timeout_seconds=0.5)
    options.update(kwargs)
    with pytest.raises(DecisionConfigurationError):
        JevDecisionService(**options)


# --- logging -----------------------------------------------------------------------


def test_the_sdk_logger_is_held_at_warning_even_if_debug_was_requested() -> None:
    logger = logging.getLogger(SDK_LOGGER)
    saved = logger.level
    try:
        logger.setLevel(logging.DEBUG)  # what TYPESAFE_LOG_LEVEL=debug does at SDK import
        service = _service(_respond())
        assert logger.level == logging.WARNING
        asyncio.run(service.aclose())
    finally:
        logger.setLevel(saved)


def test_no_log_line_carries_the_customer_text_or_the_key(log_stream) -> None:
    logger = logging.getLogger(SDK_LOGGER)
    saved = logger.level
    try:
        logger.setLevel(logging.DEBUG)
        decide(_service(_respond()))
        with pytest.raises(DecisionAuthenticationFailed):
            decide(_service(_respond({"error": UTTERANCE}, status=401)))
    finally:
        logger.setLevel(saved)
    output = log_stream.getvalue()
    assert "already paid" not in output
    assert "ABCDE1234F" not in output
    assert KEY not in output


# --- concurrency -------------------------------------------------------------------


def test_concurrent_decisions_share_one_client_and_actually_overlap() -> None:
    in_flight = 0
    peak = 0
    requests: list[httpx2.Request] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal in_flight, peak
        requests.append(request)
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx2.Response(200, json=_body())

    service = _service(handler, timeout_seconds=2.0)
    client = service._client  # noqa: SLF001 - identity check only

    async def run():
        try:
            return await asyncio.gather(*(service.decide(_request(utterance=f"turn {n}")) for n in range(25)))
        finally:
            await service.aclose()

    started = time.perf_counter()
    answers = asyncio.run(run())
    elapsed = time.perf_counter() - started

    assert len(answers) == 25 and all(a.label == "payment_already_made" for a in answers)
    assert len(requests) == 25
    assert service._client is client  # noqa: SLF001
    assert peak > 1, "decisions were serialised; the adapter blocked"
    assert elapsed < 25 * 0.05, "25 decisions took as long as running them one after another"


# --- adversarial-review regressions -------------------------------------------------
#
# Each of these pins a defect the review of this stage found. See REPORT.md SD.14.


def test_a_failed_connect_is_the_only_transport_error_retried() -> None:
    seen: list = []
    handler = _sequence(httpx2.ConnectError("refused"), httpx2.Response(200, json=_body()), seen=seen)
    assert decide(_service(handler, max_retries=1, timeout_seconds=1.0)).label == "payment_already_made"
    assert len(seen) == 2


@pytest.mark.parametrize(
    "exc",
    [httpx2.ReadError("reset"), httpx2.WriteError("broken"), httpx2.RemoteProtocolError("bad")],
    ids=lambda e: type(e).__name__,
)
def test_an_error_after_the_request_may_have_been_delivered_is_never_retried(exc: Exception) -> None:
    """Re-sending would send the customer's words to the provider a second time."""
    seen: list = []
    handler = _sequence(exc, httpx2.Response(200, json=_body()), seen=seen)
    with pytest.raises(DecisionConnectionFailed):
        decide(_service(handler, max_retries=2, timeout_seconds=1.0))
    assert len(seen) == 1


def _echoing_unknown_answer() -> dict:
    """A valid answer plus one of a type the SDK does not know, echoing the request."""
    body = _body()
    body["answers"][f"echo {UTTERANCE}"] = {"type": f"x-{UTTERANCE}"}
    return body


def test_provider_text_in_the_sdks_own_warning_never_reaches_the_logs(log_stream) -> None:
    from app.observability import configure_logging

    service = _service(_respond(_echoing_unknown_answer()))
    # Re-running configure_logging resets the SDK logger's level; the drop
    # filter must survive that.
    configure_logging("DEBUG", stream=log_stream)
    answer = decide(service)

    assert answer.label == "payment_already_made"
    output = log_stream.getvalue()
    assert "ABCDE1234F" not in output
    assert "already paid" not in output
    assert '"logger": "typesafe_sdk"' not in output


@pytest.mark.parametrize("model", ["jev-1.13.0\n", "9876543210", "jev 1", "", "a" * 81])
def test_a_reported_model_that_is_not_an_identifier_is_not_recorded(model: str) -> None:
    assert decide(_service(_respond(_body(model=model)))).model is None


@pytest.mark.parametrize("model", ["jev-1.13.0", "typesafe-ai/jev", "~typesafe/jev-latest"])
def test_a_real_model_identifier_is_recorded(model: str) -> None:
    assert decide(_service(_respond(_body(model=model)))).model == model


def test_identifiers_in_the_customers_words_are_masked_before_they_leave() -> None:
    seen: list[httpx2.Request] = []
    words = (
        "PAN ABCDE1234F, phone 98765 43210, Aadhaar 2345-6789-0123, account 50100234567891, "
        "mail me at borrower@example.com; I will pay 5000 on 12/10/2026"
    )
    decide(
        _service(_respond(_body(DecisionName.HUMAN_ESCALATION), seen=seen)),
        _request(DecisionName.HUMAN_ESCALATION, utterance=words, recent_customer_utterances=("my number is 9876543210",)),
    )
    sent = seen[0].content.decode()
    for identifier in ("ABCDE1234F", "98765 43210", "2345-6789-0123", "50100234567891", "borrower@example.com", "9876543210"):
        assert identifier not in sent
    state = json.loads(sent)["state"]
    assert "<id>" in state["customer_utterance"] and "<email>" in state["customer_utterance"]
    assert "5000" in state["customer_utterance"] and "12/10/2026" in state["customer_utterance"]
    assert state["recent_customer_utterances"] == ["my number is <number>"]


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("call me on 9 8 7 6 5 4 3 2 1 0", "call me on <number>"),
        ("+91 98765 43210", "+<number>"),  # the country code goes too
        ("card 4111-1111-1111-1111", "card <number>"),
        ("I will pay on 12 10 2026", "I will pay on 12 10 2026"),
        ("on 5-10-2026 I pay 25000", "on 5-10-2026 I pay 25000"),
        ("I will pay on 2026-10-12", "I will pay on 2026-10-12"),
        ("amount 1,25,000 by 12/10/2026", "amount 1,25,000 by 12/10/2026"),
        # A 2-2-4 grouping that is not a real date is not exempt.
        ("account 98 76 2019", "account <number>"),
        ("31 02 2026", "<number>"),
        # Readings with commas, periods or double spaces.
        ("9, 8, 7, 6, 5, 4, 3, 2, 1, 0", "<number>"),
        ("98765  43210", "<number>"),
        ("98.765.432.10", "<number>"),
        # Documented over-masking: information lost in the safe direction.
        ("I will pay on 12102026", "I will pay on <number>"),
        ("pay 5000 5000", "pay <number>"),
    ],
)
def test_masking_keeps_dates_and_amounts_but_not_identifiers(text: str, masked: str) -> None:
    from app.services.decisions.registry import mask_identifiers

    assert mask_identifiers(text) == masked


@pytest.mark.parametrize(
    "text",
    ["1" * 1000, "a." * 499 + "@", "a" * 1000, ("1 , " * 7 + "x") * 60, "ABCDE1234" * 110],
    ids=["digits", "email-like", "word", "near-miss-runs", "pan-like"],
)
def test_masking_is_fast_on_adversarial_input(text: str) -> None:
    from app.services.decisions.registry import mask_identifiers

    started = time.perf_counter()
    for _ in range(10):
        mask_identifiers(text[:1000])
    assert (time.perf_counter() - started) / 10 < 0.02  # 20 ms per call is already generous


def test_the_real_adapter_does_not_block_the_event_loop() -> None:
    """Loop lag measured through the actual SDK, with a server that takes 200 ms."""
    from app.orchestrator.decisions import DecisionCoordinator
    from app.runtime import build_runtime
    from tests.conftest import RULES_PATH
    from app.config import Settings

    async def handler(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0.2)
        return httpx2.Response(200, json=_body())

    service = _service(handler, timeout_seconds=2.0)
    settings = Settings(_env_file=None, app_env="test", regulatory_rules_path=RULES_PATH, jev_timeout_seconds=2.0)
    co = DecisionCoordinator.from_runtime(build_runtime(settings, decisions=service))
    gaps: list[float] = []

    async def ticker(done: asyncio.Event) -> None:
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append((now - last) * 1000.0)
            last = now

    async def both():
        done = asyncio.Event()
        tick = asyncio.ensure_future(ticker(done))
        await asyncio.sleep(0)
        try:
            resolution = await co.resolve_intent("I already paid")
        finally:
            done.set()
            await tick
            await service.aclose()
        return resolution

    resolution = asyncio.run(both())
    assert resolution.label is not None  # the answer arrived
    assert len(gaps) > 20  # the ticker really ran while the call was pending
    # Normal lag here is about 6 ms (5 ms ticks); 25 ms catches a 30 ms block.
    assert max(gaps) < 25
