"""Customer, account and compliance context supplied by the backend.

The backend/database is the source of truth for everything in this module. The
LLM may read these values; it may never author them. Nothing here is persisted by
this application.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import Language, ProductType


class CustomerContext(BaseModel):
    """Minimum identity a collection call needs.

    ``display_name`` is personal data: it exists because the agent must address
    the customer, and it is redacted by :func:`app.observability.redact` before
    logging. Full contact details are deliberately absent - the telephony layer
    dials, the agent does not need the number.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    customer_ref: str = Field(min_length=1, description="Opaque backend identifier, not a PAN/Aadhaar.")
    display_name: str | None = Field(default=None, description="Personal data. Never log.")
    preferred_language: Language | None = None


class AccountContext(BaseModel):
    """Loan/account facts. Amounts are integer minor units (paise) - never float."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    account_ref: str = Field(min_length=1, description="Opaque backend identifier, not the account number.")
    product_type: ProductType
    currency: str = Field(default="INR", min_length=3, max_length=3)
    outstanding_minor: int = Field(ge=0, description="Total outstanding in paise.")
    minimum_due_minor: int | None = Field(default=None, ge=0)
    dpd: int = Field(ge=0, description="Days past due as computed by the backend.")
    due_date: date | None = None
    last_payment_date: date | None = None
    last_payment_minor: int | None = Field(default=None, ge=0)


class ComplianceContext(BaseModel):
    """Externally-supplied facts that regulatory rules depend on.

    Every field here corresponds to a rule whose ``enforcement`` is
    ``requires_external_data``. ``None`` means "the backend did not tell us", and
    the policy engine reports the corresponding rule as *not evaluable* rather
    than assuming compliance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    grievance_pending: bool | None = Field(
        default=None,
        description="A grievance/complaint lodged by this borrower is not yet finally disposed of.",
    )
    grievance_held_frivolous: bool = Field(
        default=False,
        description="Bank holds documented proof of continuous frivolous/vexatious complaints.",
    )
    sub_judice: bool = Field(default=False, description="Subject matter of the dues is sub judice.")
    recovery_agency_details_shared: bool | None = Field(
        default=None,
        description="Borrower was informed of the recovery agency's details for this case.",
    )
    digital_lending_particulars_sent: bool | None = Field(
        default=None,
        description="Digital-lending only: agent particulars sent by email/SMS before contact.",
    )
    contact_attempts_today: int | None = Field(
        default=None, ge=0, description="Recovery contacts already made to this borrower today."
    )
    borrower_authorised_out_of_hours: bool = Field(
        default=False,
        description="Borrower expressly requested/authorised contact outside the permitted window.",
    )
