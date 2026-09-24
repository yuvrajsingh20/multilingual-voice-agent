"""Closed vocabularies shared by the conversation, policy and tool layers.

Every enum here is part of a machine contract: policy decisions, tool results and
log records reference these values, so they must stay stable and explicit.

Provenance note
---------------
Values whose names start with a regulatory concept (prohibited conduct, required
actions) exist because a rule in ``data/regulatory/rules/recovery_rules.json``
references them. Values that encode *business* choices (tone, DPD buckets,
conversation stages) are marked as such and are NOT derived from any RBI text.
"""

from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    """Languages the agent is scoped to support."""

    HINDI = "hi"
    ENGLISH = "en"
    MARATHI = "mr"
    HINGLISH = "hi-en"


class ConversationStage(str, Enum):
    """Coarse position in the call. Business construct, not regulatory."""

    GREETING = "greeting"
    IDENTITY_VERIFICATION = "identity_verification"
    DISCLOSURE = "disclosure"
    ACCOUNT_DISCUSSION = "account_discussion"
    NEGOTIATION = "negotiation"
    RESOLUTION = "resolution"
    CLOSING = "closing"
    TERMINATED = "terminated"


class Intent(str, Enum):
    """Customer intent as classified upstream (NLU/LLM).

    Intent is an *input signal* to the policy engine. It is never itself a
    compliance decision.
    """

    IDENTITY_CONFIRMED = "identity_confirmed"
    WRONG_PERSON = "wrong_person"
    QUERY_OUTSTANDING = "query_outstanding"
    QUERY_DPD = "query_dpd"
    PAYMENT_INTENT = "payment_intent"
    PAYMENT_PROMISE = "payment_promise"
    DISPUTE = "dispute"
    REFUSAL = "refusal"
    CALLBACK_REQUEST = "callback_request"
    ESCALATION_REQUEST = "escalation_request"
    ABUSIVE = "abusive"
    UNKNOWN = "unknown"


class Emotion(str, Enum):
    """Coarse customer affect. Provisional; not yet validated against data."""

    NEUTRAL = "neutral"
    CALM = "calm"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    DISTRESSED = "distressed"


class DpdStage(str, Enum):
    """Days-past-due bucket.

    BUSINESS construct. RBI does not define these buckets; they exist only to
    select tone and call strategy. Thresholds live in ``app.core.dpd``.
    """

    CURRENT = "current"
    DPD_5 = "dpd_5"
    DPD_30 = "dpd_30"
    DPD_90 = "dpd_90"


class ToneLevel(str, Enum):
    """Permitted firmness. BUSINESS construct.

    Firmness never authorises anything the policy engine prohibits: a FORMAL_FIRM
    tone is still subject to every prohibited-conduct category.
    """

    NEUTRAL = "neutral"
    FIRM = "firm"
    FORMAL_FIRM = "formal_firm"


class ProductType(str, Enum):
    """Loan product class. Determines which regulatory rules apply.

    ``MICROFINANCE`` matters because RBC 2025 paragraph 445 expressly excludes
    microfinance loans and paragraph 410(2) states a different calling window.
    ``DIGITAL_LENDING`` matters because the Digital Lending Directions, 2025 apply
    only to digital lending products.
    """

    RETAIL_LOAN = "retail_loan"
    CREDIT_CARD = "credit_card"
    MICROFINANCE = "microfinance"
    DIGITAL_LENDING = "digital_lending"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RuleStatus(str, Enum):
    """Regulatory status of an encoded rule, mirroring the corpus catalog layers."""

    CURRENT_OPERATIVE = "current_operative"
    FUTURE_EFFECTIVE = "future_effective"
    CONDITIONAL = "conditional"
    HISTORICAL_SUPPORTING = "historical_supporting"


class Enforcement(str, Enum):
    """How far a rule can be enforced automatically.

    This is the honesty field of the rule schema. Only
    ``CONVERSATION_DETERMINISTIC`` rules are decided by the policy engine from
    call-time inputs alone.
    """

    CONVERSATION_DETERMINISTIC = "conversation_deterministic"
    REQUIRES_EXTERNAL_DATA = "requires_external_data"
    ORGANISATIONAL_OBLIGATION = "organisational_obligation"


class ProhibitedConduct(str, Enum):
    """Conduct categories a response must never express.

    Each value is referenced by at least one encoded regulatory rule.
    """

    INTIMIDATION = "intimidation"
    VERBAL_HARASSMENT = "verbal_harassment"
    PHYSICAL_HARASSMENT = "physical_harassment"
    ABUSIVE_LANGUAGE = "abusive_language"
    PUBLIC_HUMILIATION = "public_humiliation"
    PRIVACY_INTRUSION_THIRD_PARTY = "privacy_intrusion_third_party"
    INAPPROPRIATE_MESSAGING = "inappropriate_messaging"
    SOCIAL_MEDIA_EXPOSURE = "social_media_exposure"
    THREATENING_CALL = "threatening_call"
    ANONYMOUS_CALL = "anonymous_call"
    VIOLENCE_THREAT = "violence_threat"
    PERSISTENT_CALLING = "persistent_calling"
    CONTACT_OUTSIDE_PERMITTED_HOURS = "contact_outside_permitted_hours"
    FALSE_OR_MISLEADING_REPRESENTATION = "false_or_misleading_representation"
    UNAUTHORISED_DISCLOSURE = "unauthorised_disclosure"


class RequiredAction(str, Enum):
    """Actions the orchestrator must perform or must have performed."""

    DISCLOSE_CALL_RECORDING = "disclose_call_recording"
    IDENTIFY_BANK_AND_AGENT = "identify_bank_and_agent"
    VERIFY_IDENTITY_BEFORE_DISCLOSURE = "verify_identity_before_disclosure"
    TERMINATE_COLLECTION_DISCUSSION = "terminate_collection_discussion"
    END_CALL = "end_call"
    PROVIDE_GRIEVANCE_MECHANISM_DETAILS = "provide_grievance_mechanism_details"
    RECORD_DISPUTE = "record_dispute"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    CONFIRM_PAYMENT_PROMISE_DETAILS = "confirm_payment_promise_details"
    DEFER_TO_REQUESTED_CALLBACK = "defer_to_requested_callback"


class CheckStatus(str, Enum):
    """Outcome of one deterministic rule check."""

    PASS = "pass"
    VIOLATION = "violation"
    NOT_EVALUABLE = "not_evaluable"


class EventKind(str, Enum):
    """Conversation event types. Events are the append-only audit trail."""

    SESSION_STARTED = "session_started"
    USER_UTTERANCE = "user_utterance"
    AGENT_UTTERANCE = "agent_utterance"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    POLICY_DECISION = "policy_decision"
    BARGE_IN = "barge_in"
    ACCOUNT_CONTEXT_LOADED = "account_context_loaded"
    CALL_ENDED = "call_ended"


class ToolStatus(str, Enum):
    OK = "ok"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    NOT_IMPLEMENTED = "not_implemented"
    BACKEND_ERROR = "backend_error"


class TurnState(str, Enum):
    """States of the turn-taking machine."""

    IDLE = "idle"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    PROCESSING = "processing"
    AGENT_SPEAKING = "agent_speaking"
    ENDED = "ended"


class BargeInClass(str, Enum):
    """Classification of customer audio detected while the agent is speaking."""

    NOISE = "noise"
    BACKCHANNEL = "backchannel"
    INTERRUPTION = "interruption"
    UNKNOWN = "unknown"


class SpanKind(str, Enum):
    """Text spans TTS normalisation must treat specially."""

    CURRENCY = "currency"
    NUMBER = "number"
    DATE = "date"
    PHONE = "phone"
    ACCOUNT_REFERENCE = "account_reference"
    ABBREVIATION = "abbreviation"
