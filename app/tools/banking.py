"""Backend tool stubs.

No banking system is connected and no customer data is fabricated. Every tool
delegates to a :class:`BankingBackend`; the default implementation raises
:class:`~app.tools.base.ToolNotImplemented`, so an unwired deployment fails
loudly instead of inventing an outstanding amount.

Each tool returns the narrowest payload that answers its question. That is
deliberate: RBI (Commercial Banks - Managing Risks in Outsourcing) Directions,
2025 paragraph 24 puts service-provider access to customer information on a
need-to-know basis, so ``get_outstanding_amount`` returns an amount rather than
the whole account record.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.models.customer import AccountContext, ComplianceContext, CustomerContext
from app.tools.base import Tool, ToolNotImplemented, ToolRegistry


class BankingBackend(Protocol):
    """What the bank's systems must provide. Implemented by nothing yet."""

    def get_customer(self, customer_ref: str) -> CustomerContext: ...
    def get_account(self, account_ref: str) -> AccountContext: ...
    def get_compliance(self, account_ref: str) -> ComplianceContext: ...
    def record_payment_promise(self, account_ref: str, promise_date: date, amount_minor: int) -> str: ...
    def create_dispute(self, account_ref: str, reason_code: str) -> str: ...
    def escalate_case(self, account_ref: str, reason_code: str) -> str: ...


class NullBankingBackend:
    """Default backend. Every call raises, by design."""

    _MESSAGE = "No banking backend is configured; this foundation does not connect to one."

    def get_customer(self, customer_ref: str) -> CustomerContext:
        raise ToolNotImplemented(self._MESSAGE)

    def get_account(self, account_ref: str) -> AccountContext:
        raise ToolNotImplemented(self._MESSAGE)

    def get_compliance(self, account_ref: str) -> ComplianceContext:
        raise ToolNotImplemented(self._MESSAGE)

    def record_payment_promise(self, account_ref: str, promise_date: date, amount_minor: int) -> str:
        raise ToolNotImplemented(self._MESSAGE)

    def create_dispute(self, account_ref: str, reason_code: str) -> str:
        raise ToolNotImplemented(self._MESSAGE)

    def escalate_case(self, account_ref: str, reason_code: str) -> str:
        raise ToolNotImplemented(self._MESSAGE)


# --- argument models -------------------------------------------------------


class CustomerRefArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_ref: str = Field(min_length=1, max_length=64)


class AccountRefArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_ref: str = Field(min_length=1, max_length=64)


class PaymentPromiseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_ref: str = Field(min_length=1, max_length=64)
    promise_date: date
    amount_minor: int = Field(gt=0, description="Promised amount in paise.")


class CaseActionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_ref: str = Field(min_length=1, max_length=64)
    reason_code: str = Field(min_length=1, max_length=64)


# --- tools -----------------------------------------------------------------


class _BackendTool(Tool):
    def __init__(self, backend: BankingBackend) -> None:
        self.backend = backend


class GetCustomerContext(_BackendTool):
    name = "get_customer_context"
    description = "Look up the customer's preferred language and display name by opaque customer reference."
    args_model = CustomerRefArgs

    def run(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, CustomerRefArgs)
        customer = self.backend.get_customer(args.customer_ref)
        return customer.model_dump(mode="json")


class GetAccountStatus(_BackendTool):
    name = "get_account_status"
    description = "Return the account's product type, currency, days past due and due date."
    args_model = AccountRefArgs

    def run(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, AccountRefArgs)
        account = self.backend.get_account(args.account_ref)
        return {
            "account_ref": account.account_ref,
            "product_type": account.product_type.value,
            "currency": account.currency,
            "dpd": account.dpd,
            "due_date": account.due_date.isoformat() if account.due_date else None,
        }


class GetOutstandingAmount(_BackendTool):
    name = "get_outstanding_amount"
    description = "Return only the outstanding and minimum due amounts, in minor units (paise)."
    args_model = AccountRefArgs

    def run(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, AccountRefArgs)
        account = self.backend.get_account(args.account_ref)
        return {
            "currency": account.currency,
            "outstanding_minor": account.outstanding_minor,
            "minimum_due_minor": account.minimum_due_minor,
        }


class GetDpd(_BackendTool):
    name = "get_dpd"
    description = "Return only the number of days the account is past due."
    args_model = AccountRefArgs

    def run(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, AccountRefArgs)
        account = self.backend.get_account(args.account_ref)
        return {"dpd": account.dpd}


class RecordPaymentPromise(_BackendTool):
    name = "record_payment_promise"
    description = "Record a promise to pay a stated amount by a stated date. Returns the backend reference."
    args_model = PaymentPromiseArgs

    def run(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, PaymentPromiseArgs)
        reference = self.backend.record_payment_promise(
            args.account_ref, args.promise_date, args.amount_minor
        )
        return {"promise_ref": reference}


class CreateDispute(_BackendTool):
    name = "create_dispute"
    description = "Raise a dispute against the dues on this account. Returns the dispute reference."
    args_model = CaseActionArgs

    def run(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, CaseActionArgs)
        return {"dispute_ref": self.backend.create_dispute(args.account_ref, args.reason_code)}


class EscalateCase(_BackendTool):
    name = "escalate_case"
    description = "Hand the case to a human recovery officer. Returns the escalation reference."
    args_model = CaseActionArgs

    def run(self, args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, CaseActionArgs)
        return {"escalation_ref": self.backend.escalate_case(args.account_ref, args.reason_code)}


TOOL_CLASSES: tuple[type[_BackendTool], ...] = (
    GetCustomerContext,
    GetAccountStatus,
    GetOutstandingAmount,
    GetDpd,
    RecordPaymentPromise,
    CreateDispute,
    EscalateCase,
)


def build_registry(backend: BankingBackend | None = None) -> ToolRegistry:
    """Register every banking tool against ``backend`` (default: the null backend)."""
    resolved = backend or NullBankingBackend()
    registry = ToolRegistry()
    for tool_class in TOOL_CLASSES:
        registry.register(tool_class(resolved))
    return registry
