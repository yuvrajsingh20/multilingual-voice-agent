from pydantic import BaseModel
from typing import Optional


class ConversationState(BaseModel):
    language: Optional[str] = None
    dpd: Optional[int] = None
    intent: Optional[str] = None
    emotion: Optional[str] = None

    payment_promise: bool = False
    promise_date: Optional[str] = None

    dispute: bool = False
    wrong_person: bool = False
    escalation_required: bool = False

    current_stage: str = "greeting"
