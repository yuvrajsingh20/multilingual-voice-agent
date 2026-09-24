"""Test doubles.

Fake customer and account data lives here, in the test package, so that no
fabricated banking data ships in ``app/``.

The doubles at the bottom of this module exist to drive the orchestrator's
failure branches. None of them simulates *working* production behaviour: each
one only fails, in the one way a real component could fail.
"""

from __future__ import annotations

from datetime import date

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
