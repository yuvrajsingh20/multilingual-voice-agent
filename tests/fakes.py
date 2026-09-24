"""Test doubles.

Fake customer and account data lives here, in the test package, so that no
fabricated banking data ships in ``app/``.

The doubles at the bottom of this module exist to drive the orchestrator's
failure branches. None of them simulates *working* production behaviour: each
one only fails, in the one way a real component could fail.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic import BaseModel

from app.core.session import SessionStore
from app.models.conversation import ConversationEvent
from app.models.customer import AccountContext, ComplianceContext, CustomerContext
from app.models.enums import EventKind, Language, ProductType
from app.models.policy import PolicyContext
from app.services.llm import LlmGeneration, LlmRequest, LlmToolCall
from app.services.stt import NormalizedTranscript, TranscriptSegment
from app.tools.base import ToolNotImplemented


class InMemoryBankingBackend:
    """A :class:`~app.tools.banking.BankingBackend` backed by dictionaries."""

    def __init__(
        self,
        customers: dict[str, CustomerContext] | None = None,
        accounts: dict[str, AccountContext] | None = None,
        compliance: dict[str, ComplianceContext] | None = None,
    ) -> None:
        self.customers = customers or {}
        self.accounts = accounts or {}
        self.compliance = compliance or {}
        self.promises: list[tuple[str, date, int]] = []
        self.disputes: list[tuple[str, str]] = []
        self.escalations: list[tuple[str, str]] = []

    def get_customer(self, customer_ref: str) -> CustomerContext:
        try:
            return self.customers[customer_ref]
        except KeyError as exc:
            raise ToolNotImplemented(f"unknown customer_ref {customer_ref}") from exc

    def get_account(self, account_ref: str) -> AccountContext:
        try:
            return self.accounts[account_ref]
        except KeyError as exc:
            raise ToolNotImplemented(f"unknown account_ref {account_ref}") from exc

    def get_compliance(self, account_ref: str) -> ComplianceContext:
        return self.compliance.get(account_ref, ComplianceContext())

    def record_payment_promise(self, account_ref: str, promise_date: date, amount_minor: int) -> str:
        self.promises.append((account_ref, promise_date, amount_minor))
        return f"PROMISE-{len(self.promises)}"

    def create_dispute(self, account_ref: str, reason_code: str) -> str:
        self.disputes.append((account_ref, reason_code))
        return f"DISPUTE-{len(self.disputes)}"

    def escalate_case(self, account_ref: str, reason_code: str) -> str:
        self.escalations.append((account_ref, reason_code))
        return f"ESC-{len(self.escalations)}"


def sample_customer(customer_ref: str = "CUST-1") -> CustomerContext:
    return CustomerContext(
        customer_ref=customer_ref,
        display_name="Test Borrower",
        preferred_language=Language.HINGLISH,
    )


def sample_account(
    account_ref: str = "ACC-1",
    *,
    product_type: ProductType = ProductType.RETAIL_LOAN,
    dpd: int = 35,
    outstanding_minor: int = 1_234_500,
) -> AccountContext:
    return AccountContext(
        account_ref=account_ref,
        product_type=product_type,
        outstanding_minor=outstanding_minor,
        minimum_due_minor=250_000,
        dpd=dpd,
        due_date=date(2026, 8, 20),
        last_payment_date=date(2026, 7, 18),
        last_payment_minor=500_000,
    )


def backend_with_sample_data() -> InMemoryBankingBackend:
    return InMemoryBankingBackend(
        customers={"CUST-1": sample_customer()},
        accounts={"ACC-1": sample_account()},
        compliance={"ACC-1": ComplianceContext(grievance_pending=False)},
    )


# --- orchestrator failure doubles ------------------------------------------
#
# Each of these fails and does nothing else. They exist so that a failure branch
# the orchestrator must handle can be reached deterministically.


class BrokenTranscriptNormalizer:
    """A transcript normaliser whose backing lexicon blew up."""

    def normalize(self, segment: TranscriptSegment) -> NormalizedTranscript:
        raise RuntimeError("normaliser lexicon unavailable")


class BrokenTtsNormalizer:
    """A TTS normaliser that cannot render anything."""

    def normalize(self, text: str, language: Language | None):
        raise RuntimeError("pronunciation table unavailable")


class BrokenValidator:
    """A response validator that cannot decide. Nothing may be spoken after this."""

    def validate(self, draft: object, decision: object, grounding: object):
        raise RuntimeError("validator rule index unavailable")


class FailingLlmService:
    """A configured model endpoint that errors. Not the same as 'not configured'."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or TimeoutError("model endpoint timed out")
        self.requests: list[LlmRequest] = []

    def generate(self, request: LlmRequest) -> LlmGeneration:
        self.requests.append(request)
        raise self._exc


class BrokenPolicyEngine:
    """A policy engine that raises. ``exc`` selects which failure."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def evaluate(self, ctx: PolicyContext):
        self.calls += 1
        raise self._exc


class _EventProbe(BaseModel):
    """Only used to manufacture a genuine pydantic ``ValidationError``."""

    required_field: int


class RejectingSessionStore(SessionStore):
    """A store whose reducer rejects one event kind.

    Reaching ``TurnErrorCategory.INVALID_CONVERSATION_EVENT`` through the typed
    API is not possible - ``EventSignals`` is validated when it is constructed -
    so this forces the defensive branch that exists for the day an event shape
    changes underneath the reducer.
    """

    def __init__(self, max_sessions: int, reject_kind: EventKind) -> None:
        super().__init__(max_sessions)
        self.reject_kind = reject_kind

    def append_event(self, session_id: str, event: ConversationEvent):
        if event.kind is self.reject_kind:
            _EventProbe.model_validate({})  # raises ValidationError
        return super().append_event(session_id, event)


def text_generation(text: str) -> LlmGeneration:
    """A model turn that is words only."""
    return LlmGeneration(text=text, model="scripted")


def tool_generation(tool_name: str, arguments: dict | None = None, *, call_id: str = "call-1") -> LlmGeneration:
    """A model turn that requests one tool."""
    return LlmGeneration(
        text=None,
        tool_calls=(LlmToolCall(call_id=call_id, tool_name=tool_name, arguments=arguments or {}),),
        model="scripted",
    )


# --- OpenAI-compatible model server doubles --------------------------------
#
# Nothing here talks to a real model, to a cloud or to the internet. The bodies
# are hand-written examples of the OpenAI chat-completions wire format, and the
# server below binds to loopback on an ephemeral port.


def openai_text_completion(
    text: str,
    *,
    model: str = "test-model",
    finish_reason: str = "stop",
    usage: dict | None = None,
) -> dict:
    """A normal assistant reply, in OpenAI chat-completions form."""
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def openai_tool_completion(
    *calls: tuple[str, dict],
    model: str = "test-model",
    content: str | None = None,
    finish_reason: str = "tool_calls",
    call_ids: tuple[str, ...] | None = None,
) -> dict:
    """An assistant reply that requests one or more tools.

    ``arguments`` is serialised as a JSON *string*, which is what the OpenAI wire
    format specifies and what vLLM emits.
    """
    ids = call_ids or tuple(f"call_{i}" for i in range(len(calls)))
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                        for call_id, (name, arguments) in zip(ids, calls)
                    ],
                },
                "finish_reason": finish_reason,
            }
        ],
    }


@dataclass
class RecordedRequest:
    """What :class:`FakeOpenAiServer` actually received on the wire."""

    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> object:
        return json.loads(self.body.decode("utf-8"))


class FakeOpenAiServer:
    """A real HTTP server on loopback that speaks the chat-completions dialect.

    ``httpx2.MockTransport`` is enough for parsing tests and is what most of the
    suite uses. This exists for the handful of properties a transport stub cannot
    demonstrate, because they are properties of an actual socket: that the
    configured base URL is the one dialled, that the ``Authorization`` header
    leaves the process, and that a slow server trips the configured timeout.

    Binds to 127.0.0.1 on an ephemeral port. Nothing leaves the machine.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        body: object = None,
        raw_body: bytes | None = None,
        delay_seconds: float = 0.0,
        content_type: str = "application/json",
    ) -> None:
        self.status = status
        self.body = body
        self.raw_body = raw_body
        self.delay_seconds = delay_seconds
        self.content_type = content_type
        self.requests: list[RecordedRequest] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:  # pragma: no cover - misuse
            raise RuntimeError("server is not running")
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self) -> "FakeOpenAiServer":
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
                length = int(self.headers.get("Content-Length") or 0)
                payload = self.rfile.read(length) if length else b""
                outer.requests.append(
                    RecordedRequest(
                        method="POST",
                        path=self.path,
                        headers={k.lower(): v for k, v in self.headers.items()},
                        body=payload,
                    )
                )
                if outer.delay_seconds:
                    time.sleep(outer.delay_seconds)
                if outer.raw_body is not None:
                    encoded = outer.raw_body
                else:
                    encoded = json.dumps(outer.body or {}).encode("utf-8")
                self.send_response(outer.status)
                self.send_header("Content-Type", outer.content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args: object) -> None:
                """Silence the default stderr access log; the suite is not a web server."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def unused_loopback_url() -> str:
    """A base URL on loopback with nothing listening behind it.

    Bound and released, so the port was free at the moment it was chosen. Used to
    provoke a genuine connection failure rather than a simulated one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"
