"""The model boundary.

These tests pin the two properties that keep the model from becoming the source
of truth: it must be swappable, and it must fail loudly when absent.
"""

from __future__ import annotations

import pytest

from app.services.llm import (
    LlmGeneration,
    LlmMessage,
    LlmNotConfigured,
    LlmRequest,
    LlmToolCall,
    NotConfiguredLlmService,
    ScriptedLlmService,
)
from app.tools.banking import build_registry


def _request() -> LlmRequest:
    return LlmRequest(messages=(LlmMessage(role="user", content="kitna bakaya hai?"),))


def test_an_unconfigured_model_fails_loudly(runtime) -> None:
    with pytest.raises(LlmNotConfigured):
        runtime.llm.generate(_request())


def test_the_default_service_is_the_unconfigured_one(runtime) -> None:
    assert isinstance(runtime.llm, NotConfiguredLlmService)


def test_any_service_satisfying_the_interface_can_be_swapped_in(settings) -> None:
    from app.runtime import build_runtime

    scripted = ScriptedLlmService([LlmGeneration(text="Namaste.", model="stand-in")])
    runtime = build_runtime(settings, llm=scripted)
    assert runtime.llm.generate(_request()).text == "Namaste."


def test_tool_specs_are_offered_to_the_model_as_json_schema() -> None:
    specs = build_registry().specs()
    assert specs
    for spec in specs:
        assert spec.parameters["type"] == "object"
        assert spec.description


def test_a_tool_call_the_model_invents_is_not_executable() -> None:
    """The model requests; the registry decides. An unknown name goes nowhere."""
    from app.models.tools import ToolRequest

    call = LlmToolCall(call_id="c1", tool_name="wire_transfer", arguments={"amount": 1})
    result = build_registry().execute(
        ToolRequest(
            request_id="r1",
            session_id="s1",
            turn_id=0,
            tool_name=call.tool_name,
            arguments=call.arguments,
        )
    )
    assert result.status.value == "not_found"


def test_generation_carries_latency_for_observability() -> None:
    generation = LlmGeneration(text="ok", latency_ms=42.0)
    assert generation.latency_ms == 42.0
