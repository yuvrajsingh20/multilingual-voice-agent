"""``reasoning_effort``: how thinking mode is kept off over the chat-completions dialect.

Track 1 requires thinking disabled for every run. Ollama 0.35.0 rejects the
Modelfile form (``PARAMETER think false``), so the switch travels on the request
as ``"reasoning_effort": "none"``. These tests pin that the field is sent
exactly when configured, never otherwise - an unconfigured deployment's body
is unchanged - and that a value outside the dialect is refused at startup.

Offline: every request goes to an in-process transport.
"""

from __future__ import annotations

import json

import httpx2
import pytest
from pydantic import ValidationError

from app.runtime import build_llm_service
from app.services.llm import LlmMessage, LlmRequest
from app.services.llm_openai import OpenAiCompatibleLlmService
from tests.fakes import openai_text_completion
from tests.test_orchestrator import make_settings

BASE_URL = "http://model.invalid:11434/v1"


def _request() -> LlmRequest:
    return LlmRequest(messages=(LlmMessage(role="user", content="namaste"),))


def _capture(**kwargs) -> dict:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        seen.append(request)
        return httpx2.Response(200, json=openai_text_completion("ok"))

    service = OpenAiCompatibleLlmService(
        base_url=BASE_URL,
        model="qwen-voice-4b",
        client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        **kwargs,
    )
    service.generate(_request())
    return json.loads(seen[0].content)


def test_unset_reasoning_effort_leaves_the_body_unchanged() -> None:
    assert "reasoning_effort" not in _capture()


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high"])
def test_a_configured_reasoning_effort_is_sent(effort: str) -> None:
    assert _capture(reasoning_effort=effort)["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", ["off", "", "NONE", "false"])
def test_a_reasoning_effort_outside_the_dialect_is_refused(effort: str) -> None:
    with pytest.raises(ValueError):
        OpenAiCompatibleLlmService(base_url=BASE_URL, model="m", reasoning_effort=effort)


def test_the_setting_reaches_the_adapter() -> None:
    service = build_llm_service(
        make_settings(
            model_base_url=BASE_URL, model_name="qwen-voice-4b", model_reasoning_effort="none"
        )
    )
    try:
        assert service._reasoning_effort == "none"  # noqa: SLF001
    finally:
        service.close()


def test_a_blank_setting_reads_as_unset() -> None:
    assert make_settings(model_reasoning_effort="").model_reasoning_effort is None


def test_an_unknown_setting_value_is_rejected_at_startup() -> None:
    with pytest.raises(ValidationError):
        make_settings(model_reasoning_effort="off")
