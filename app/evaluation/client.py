"""The evaluation's model client: one streamed chat-completions call, recorded raw.

Why not :class:`app.services.llm_openai.OpenAiCompatibleLlmService`? That
adapter is the production boundary and is strict on purpose: it raises on a
tool call whose arguments are not JSON, on a reply that is a tool call written
out as text, and on a cut-off generation. Those are exactly the failures PS-3
has to count and classify, so the evaluation needs the model's output *before*
anything refuses it. This client parses nothing. It records what came back -
text, raw argument strings, finish reason, usage, any reasoning text - and the
scorers decide what it means. The application's own path to the same models is
exercised separately, in tests/live/test_live_ollama_app.py.

Every request carries the settings that make a run reproducible and valid for
Track 1: ``reasoning_effort: "none"`` (thinking off; Ollama maps it to
think=false), ``temperature: 0`` and a fixed ``seed``. Whether any reasoning
text arrived anyway is recorded per call, so "thinking was disabled" is checked
on every response, not asserted once.

Streaming is used for one reason: time to first token. It is measured at
concurrency 1 on a single-stream backend (Ollama), which the challenge itself
says cannot express behaviour under load; it is reported as exactly that.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx2


@dataclass
class RawToolCall:
    name: str | None
    arguments: str  # exactly as streamed; not parsed here
    call_id: str | None = None


@dataclass
class CallRecord:
    """One model call, as observed. Serialised verbatim into the results file."""

    model: str
    reported_model: str | None = None
    content: str = ""
    reasoning: str = ""
    tool_calls: list[RawToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    error: str | None = None

    @property
    def thinking_leaked(self) -> bool:
        return bool(self.reasoning.strip())

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["thinking_leaked"] = self.thinking_leaked
        return data


@dataclass(frozen=True)
class ModelEndpoint:
    """Where a model lives and how it is asked. Identical settings for every model."""

    model: str
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key_env: str | None = None
    reasoning_effort: str | None = "none"
    temperature: float = 0.0
    seed: int = 42
    timeout_seconds: float = 600.0


class ChatClient:
    def __init__(self, endpoint: ModelEndpoint) -> None:
        self.endpoint = endpoint
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key_env:
            key = os.environ.get(endpoint.api_key_env)
            if not key:
                raise RuntimeError(f"{endpoint.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        self._headers = headers
        self._client = httpx2.Client(timeout=httpx2.Timeout(endpoint.timeout_seconds))

    def close(self) -> None:
        self._client.close()

    def payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "temperature": self.endpoint.temperature,
            "seed": self.endpoint.seed,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.endpoint.reasoning_effort is not None:
            body["reasoning_effort"] = self.endpoint.reasoning_effort
        if tools:
            body["tools"] = tools
        return body

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 256,
        extra: dict[str, Any] | None = None,
    ) -> CallRecord:
        record = CallRecord(model=self.endpoint.model)
        body = self.payload(messages, tools=tools, max_tokens=max_tokens)
        if extra:
            body.update(extra)
        calls: dict[int, dict[str, Any]] = {}
        started = time.perf_counter()
        try:
            with self._client.stream(
                "POST",
                f"{self.endpoint.base_url.rstrip('/')}/chat/completions",
                json=body,
                headers=self._headers,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    record.error = f"http_{response.status_code}"
                    return record
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    self._absorb(chunk, record, calls, started)
        except httpx2.TimeoutException:
            record.error = "timeout"
        except httpx2.RequestError as exc:
            record.error = f"transport:{type(exc).__name__}"
        except json.JSONDecodeError:
            record.error = "stream_not_json"
        finally:
            record.total_ms = round((time.perf_counter() - started) * 1000.0, 1)
        record.tool_calls = [
            RawToolCall(name=c.get("name"), arguments=c.get("arguments", ""), call_id=c.get("id"))
            for _, c in sorted(calls.items())
        ]
        return record

    @staticmethod
    def _absorb(
        chunk: dict[str, Any],
        record: CallRecord,
        calls: dict[int, dict[str, Any]],
        started: float,
    ) -> None:
        if chunk.get("model"):
            record.reported_model = chunk["model"]
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            record.prompt_tokens = usage.get("prompt_tokens")
            record.completion_tokens = usage.get("completion_tokens")
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            produced = False
            if delta.get("content"):
                record.content += delta["content"]
                produced = True
            for key in ("reasoning", "reasoning_content"):
                if delta.get(key):
                    record.reasoning += delta[key]
                    produced = True
            for position, tc in enumerate(delta.get("tool_calls") or []):
                index = tc.get("index", position)
                slot = calls.setdefault(index, {"arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                function = tc.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    slot["arguments"] += arguments
                elif isinstance(arguments, dict):
                    slot["arguments"] += json.dumps(arguments, ensure_ascii=False)
                produced = True
            if produced and record.ttft_ms is None:
                record.ttft_ms = round((time.perf_counter() - started) * 1000.0, 1)
            if choice.get("finish_reason"):
                record.finish_reason = choice["finish_reason"]


def ollama_version(base_url: str = "http://127.0.0.1:11434") -> str | None:
    try:
        return httpx2.get(f"{base_url}/api/version", timeout=5).json().get("version")
    except Exception:  # noqa: BLE001 - informational only
        return None


def ollama_model_digest(model: str, base_url: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    """Digest, quantisation and parameter size of a local Ollama model, for the run manifest."""
    try:
        tags = httpx2.get(f"{base_url}/api/tags", timeout=5).json().get("models", [])
    except Exception:  # noqa: BLE001 - informational only
        return {}
    wanted = model if ":" in model else f"{model}:latest"
    for entry in tags:
        if entry.get("name") == wanted:
            details = entry.get("details") or {}
            return {
                "name": entry.get("name"),
                "digest": entry.get("digest"),
                "size_bytes": entry.get("size"),
                "quantization": details.get("quantization_level"),
                "parameter_size": details.get("parameter_size"),
                "family": details.get("family"),
            }
    return {}
