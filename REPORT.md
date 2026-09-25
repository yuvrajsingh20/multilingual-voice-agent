# Multilingual Debt Voice Agent — Foundation Report

Date: 2026-09-24
Python 3.10.12 · pydantic 2.13.5 · FastAPI 0.141.1 · pytest 9.1.1

---

# 1. Executive Summary

This change turns the repository from a skeleton (an empty config module, a
`/health` endpoint and a placeholder policy function) into a working
deterministic foundation for a commercial-bank debt-recovery voice agent.

What is **implemented and tested**:

- A regulatory rule schema and a rule file containing **49 rules** encoded from
  the local RBI corpus. Every rule's `quote` is a verbatim substring of the
  cited local text file, and a test enforces that.
- A **deterministic policy engine** with 11 registered checks. Same inputs →
  same structured decision. No model call, no randomness, no natural-language
  generation.
- Domain models, a pure conversation-state reducer, an in-memory session store,
  a tool registry with argument validation and audit logging, a response
  validator, and structured JSON logging with field redaction.
- Interface boundaries for the LLM, STT transcript normalisation, TTS
  normalisation, turn detection and barge-in.
- 249 tests, all passing.

What is **interface-only or stubbed** (nothing here is connected):
telephony, STT, TTS, the Gemma model endpoint, and the banking backend. The
default banking backend raises rather than returning invented data.

No claim is made that this system is compliant, secure or ready for production.
It has not been reviewed by a compliance function, has not been load tested, and
has no authentication. Section 12 and 14 list what is missing.

---

# 2. Architecture

```
audio in ─► [STT]* ─► transcript normalisation ─► turn detection ─► ConversationState
                                                                        │
              backend facts (AccountContext, ComplianceContext) ────────┤
                                                                        ▼
                                                              ►  POLICY ENGINE  ◄  recovery_rules.json
                                                                        │ PolicyDecision (structured)
                                                                        ▼
                                                                  [LLM / Gemma]*  ── tool requests ─► ToolRegistry ─► [banking backend]*
                                                                        │ DraftResponse
                                                                        ▼
                                                              RESPONSE VALIDATION
                                                                        │ (blocked drafts never proceed)
                                                                        ▼
                                                              TTS normalisation ─► [TTS]* ─► audio out

while the agent speaks:  customer audio ─► barge-in classifier ─► backchannel: keep speaking
                                                                └► interruption: stop TTS, hand back the floor

structured logging wraps every stage        * = interface only, not connected
```

Boundaries that are enforced by code, not by convention:

| Boundary | Enforced how |
| --- | --- |
| The LLM cannot decide compliance | The policy engine reads only `ConversationState`, backend context and the rule file. No LLM call exists anywhere in `app/core/`. |
| The LLM cannot bypass policy | The validator (`app/services/validation.py`) takes the `PolicyDecision` and blocks a draft that violates it. |
| The LLM cannot execute tools | It emits `LlmToolCall`; only `ToolRegistry.execute` runs anything, after validating arguments against the tool's own schema. |
| The LLM cannot invent figures | Every currency amount and date in a draft must appear in `GroundingFacts`, which is built from backend results. |
| The backend is the source of truth | `ConversationState` carries no account data — not even `dpd`. |
| Timezone is configuration | `PolicyConfig.timezone` (default `Asia/Kolkata`); the engine converts whatever instant it is handed. The host's zone never participates. |
| Nothing is a hidden global | Everything is built in `app/runtime.py` and passed explicitly. |

---

# 3. Files Created

| File | Purpose |
| --- | --- |
| `app/models/enums.py` | Closed vocabularies shared by policy, tools and logs. Marks which values are regulatory and which are business choices. |
| `app/models/conversation.py` | `ConversationState` (small, mutable) and `ConversationEvent` (append-only audit record). |
| `app/models/customer.py` | `CustomerContext`, `AccountContext`, `ComplianceContext` — backend-supplied facts. |
| `app/models/policy.py` | `PolicyContext`, `PolicyDecision`, `PolicyViolation`, `CheckResult`, `RuleCitation`. |
| `app/models/tools.py` | `ToolRequest` / `ToolResult` envelope. |
| `app/core/rules.py` | Rule schema (pydantic) + loader with invariants. JSON only; no YAML/pickle/eval. |
| `app/core/checks.py` | The 11 deterministic checks and `PolicyConfig`. |
| `app/core/clock.py` | `Clock` protocol, `SystemClock`, `FixedClock`. Makes "now" explicit and testable. |
| `app/core/session.py` | Pure state reducer, event replay, thread-safe in-memory `SessionStore`. |
| `app/observability.py` | JSON log formatter, `redact()`, `log_event()`, the standard observability field list. |
| `app/runtime.py` | Composition root. |
| `app/services/llm.py` | Provider-neutral model interface + `NotConfiguredLlmService` + `ScriptedLlmService`. |
| `app/services/stt.py` | Transcript-normalisation interface; passthrough and whitespace implementations. |
| `app/services/tts.py` | Span detection (all languages) + English rendering, Indian numbering system. |
| `app/services/turn.py` | VAD result, turn detectors, barge-in classifier, turn state machine. |
| `app/services/validation.py` | Response validator: prohibited conduct, grounding, promises, tool requests, confirmations. |
| `app/tools/base.py` | `Tool` ABC and `ToolRegistry` (validate → execute → audit log). |
| `app/tools/banking.py` | `BankingBackend` protocol, `NullBankingBackend`, 7 tool stubs. |
| `app/api/routes.py` | `/health`, session create, session read, event post. |
| `scripts/build_recovery_rules.py` | Regenerates the rule file by slicing quotes out of the corpus; fails if an anchor stops matching. |
| `tests/conftest.py`, `tests/fakes.py` | Deterministic fixtures; all fake banking data lives here, never in `app/`. |
| `tests/test_*.py` (13 files) | See section 11. |

---

# 4. Files Modified

| File | Change |
| --- | --- |
| `app/config.py` | Was empty. Now `Settings` (pydantic-settings) with app/log/model/policy/timezone/session settings, `SecretStr` API key, validators for timezone and log level. |
| `app/main.py` | Health route replaced by `create_app()`: configures logging, builds the runtime onto `app.state`, includes the router. |
| `app/core/policy.py` | Placeholder `evaluate_policy()` replaced by `PolicyEngine`, DPD banding and tone mapping. |
| `app/models/__init__.py`, `app/tools/__init__.py`, `app/api/__init__.py`, `app/core/__init__.py`, `app/services/__init__.py` | Were empty; now export the package surface / carry a module docstring. |
| `data/regulatory/rules/recovery_rules.json` | Was `{"schema_version":"1.0","rules":[]}`. Now 49 encoded rules. |
| `requirements.txt` | Added `httpx2` (see section 13); grouped runtime vs test. |
| `.env.example` | Was empty. Now documents every setting, with no credentials. |

**Removed:** `app/core/state.py` — `ConversationState` moved to `app/models/conversation.py` so that `app/core/` holds engine logic and `app/models/` holds the domain. See section 7 for the other changes to that model.

Not touched: `data/regulatory/catalog.json`, `sources.json`, `README.md`, the corpus files, `regulatory_corpus/`, `regulatory/`, `scripts/ingest_rbi.py`, `.gitignore`.

---

# 5. Regulatory Implementation

Source: the local corpus only (`data/regulatory/text/`). Nothing was re-downloaded.
49 rules: **27 current operative, 1 conditional, 18 future effective, 3 historical/supporting.**

Every rule records `document_id`, the paragraph label as the document numbers it,
the local text file, a verbatim `quote`, and the source URL.
`tests/test_regulatory_rules.py::test_every_quote_is_verbatim_in_its_source_file`
re-reads each file and asserts the quote is a literal substring.

`enforcement` is the honesty field:

- `conversation_deterministic` (30) — decidable from call-time inputs; must carry a `check_id`.
- `requires_external_data` (7) — needs a fact the backend must supply; reported as *not evaluable* when absent.
- `organisational_obligation` (12) — a bank/process duty a live call cannot settle; must **not** carry a `check_id`.

## CURRENT OPERATIVE (27)

| Rule ID | Source | Para | Enforcement | Sev | Check |
| --- | --- | --- | --- | --- | --- |
| `RBI-CB-RBC-2025-343-NO-UNDUE-HARASSMENT` | rbi_cb_rbc_2025 | 343 | deterministic | critical | prohibited_conduct |
| `RBI-CB-RBC-2025-410-1-MFI-NO-THREATENING-LANGUAGE` | rbi_cb_rbc_2025 | 410(1) | deterministic | critical | prohibited_conduct |
| `RBI-CB-RBC-2025-410-2-MFI-CALLING-HOURS` | rbi_cb_rbc_2025 | 410(2) | deterministic | critical | calling_hours |
| `RBI-CB-RBC-2025-410-3-MFI-NO-HARASSING-THIRD-PARTIES` | rbi_cb_rbc_2025 | 410(3) | deterministic | critical | prohibited_conduct |
| `RBI-CB-RBC-2025-410-4-MFI-NO-PUBLISHING-NAMES` | rbi_cb_rbc_2025 | 410(4) | deterministic | high | prohibited_conduct |
| `RBI-CB-RBC-2025-410-5-MFI-NO-VIOLENCE` | rbi_cb_rbc_2025 | 410(5) | deterministic | critical | prohibited_conduct |
| `RBI-CB-RBC-2025-410-6-MFI-NO-MISLEADING` | rbi_cb_rbc_2025 | 410(6) | deterministic | critical | prohibited_conduct |
| `RBI-CB-RBC-2025-442-2-AGENT-DUE-DILIGENCE` | rbi_cb_rbc_2025 | 442(2) | organisational | high | — |
| `RBI-CB-RBC-2025-442-3-AGENCY-DETAILS-TO-BORROWER` | rbi_cb_rbc_2025 | 442(3) | external data | high | agency_details_disclosed |
| `RBI-CB-RBC-2025-442-4-CALL-RECORDING-INTIMATION` | rbi_cb_rbc_2025 | 442(4) | deterministic | high | recording_disclosure |
| `RBI-CB-RBC-2025-442-6-GRIEVANCE-PENDING-HOLD` | rbi_cb_rbc_2025 | 442(6) | external data | critical | grievance_pending_hold |
| `RBI-CB-RBC-2025-442-7-GRIEVANCE-MECHANISM` | rbi_cb_rbc_2025 | 442(7) | deterministic | medium | dispute_handling |
| `RBI-CB-RBC-2025-443-NO-UNCIVILISED-CONDUCT-INCENTIVES` | rbi_cb_rbc_2025 | 443 | organisational | high | — |
| `RBI-CB-RBC-2025-445-CALLING-HOURS` | rbi_cb_rbc_2025 | 445 | deterministic | critical | calling_hours |
| `RBI-CB-RBC-2025-445-NO-PERSISTENT-CALLING` | rbi_cb_rbc_2025 | 445 | external data | high | persistent_calling |
| `RBI-CB-RBC-2025-445-NO-INTIMIDATION-OR-HARASSMENT` | rbi_cb_rbc_2025 | 445 | deterministic | critical | prohibited_conduct |
| `RBI-CB-RBC-2025-445-NO-PUBLIC-HUMILIATION` | rbi_cb_rbc_2025 | 445 | deterministic | critical | prohibited_conduct |
| `RBI-CB-RBC-2025-445-NO-PRIVACY-INTRUSION` | rbi_cb_rbc_2025 | 445 | deterministic | critical | confidentiality_disclosure |
| `RBI-CB-RBC-2025-445-NO-INAPPROPRIATE-MESSAGES` | rbi_cb_rbc_2025 | 445 | deterministic | high | prohibited_conduct |
| `RBI-CB-RBC-2025-445-NO-THREATENING-CALLS` | rbi_cb_rbc_2025 | 445 | deterministic | critical | prohibited_conduct |
| `RBI-CB-RBC-2025-445-NO-ANONYMOUS-CALLS` | rbi_cb_rbc_2025 | 445 | deterministic | critical | agent_identification |
| `RBI-CB-RBC-2025-445-NO-FALSE-REPRESENTATION` | rbi_cb_rbc_2025 | 445 | deterministic | critical | prohibited_conduct |
| `RBI-CB-RBC-2025-446-CUSTOMER-CONFIDENTIALITY` | rbi_cb_rbc_2025 | 446 | deterministic | critical | confidentiality_disclosure |
| `RBI-CB-RBC-2025-447-AGENT-TRAINING` | rbi_cb_rbc_2025 | 447 | organisational | medium | — |
| `RBI-CB-RBC-2025-452-BANK-RESPONSIBLE-FOR-AGENTS` | rbi_cb_rbc_2025 | 452 | organisational | critical | — |
| `RBI-CB-OUTSRC-2025-17-RESPONSIBLE-FOR-AGENTS-AND-CONFIDENTIALITY` | rbi_cb_outsourcing_2025 | 17 | organisational | critical | — |
| `RBI-CB-OUTSRC-2025-24-NEED-TO-KNOW` | rbi_cb_outsourcing_2025 | 24 | deterministic | high | confidentiality_disclosure |

Applicability: paragraph 445 rules carry `excluded_product_types: [microfinance]`,
because paragraph 445 states "This direction shall not be applicable to
microfinance loans". Paragraph 410 rules carry `product_types: [microfinance]`.
Both sets carry `effective_until: 2026-12-31` (see below) except paragraph 343,
which sits in the Fair Practices Code section and is not deleted.

## CONDITIONAL (1)

| Rule ID | Source | Para | Condition |
| --- | --- | --- | --- |
| `RBI-DL-2025-8-V-AGENT-PARTICULARS-BEFORE-CONTACT` | rbi_digital_lending_2025 | 8(v) | Applies only when `product_type == digital_lending`. Requires external data (`digital_lending_particulars_sent`); blocks contact when false. |

## FUTURE EFFECTIVE (18) — all `effective_from: 2027-01-01`

The Fourth Amendment Directions, 2026 state "These Directions shall come into
effect from January 1, 2027" and that "the paragraphs 408 to 416 and 442 to 454
shall be deleted". The rule set models both halves: the 2025 rules in those
ranges carry `effective_until: 2026-12-31` and `superseded_by`, and the 2026
rules carry `effective_from: 2027-01-01`. Neither layer is enforced on the wrong
date, and `tests/test_calling_hours.py` asserts the handover.

`454H`, `454I`, `454L`, `454O`, `454P`, `454Q`, `454Y(1)`, `454Y(4)`, `454Y(6)`,
`454Z(1)`–`454Z(8)`, `454AA`.

Two consequences worth flagging to a compliance reviewer:

1. **Microfinance calling window changes.** Today microfinance is 09:00–18:00
   (paragraph 410(2)) and is excluded from paragraph 445. From 2027-01-01
   paragraph 454Y(4) states a single 08:00–19:00 window with no microfinance
   carve-out, and paragraphs 408–416 are deleted. The encoded rules reflect this.
2. **The grievance hold is not carried forward.** Paragraph 442(6) (do not
   forward cases to recovery agencies while a grievance is undisposed) is deleted
   and new Section L contains no equivalent. `RBI-CB-RBC-2025-442-6-GRIEVANCE-PENDING-HOLD`
   therefore has no `superseded_by`. Whether the bank retains this control after
   2027-01-01 as internal policy **requires regulatory review**.

## HISTORICAL / SUPPORTING (3) — never enforced

| Rule ID | Source | Para | Role |
| --- | --- | --- | --- |
| `RBI-2022-108-2-HARASSMENT-AND-CALLING-HOURS` | rbi_recovery_agents_2022 | 2 | Origin of the harassment + 08:00/19:00 language now in RBC 2025 paragraph 445. |
| `RBI-2008-RECOVERY-AGENTS-X-TRAINING-LINEAGE` | rbi_recovery_agents_banks_2008 | (x) | Ancestor of paragraph 447. |
| `RBI-2006-OUTSOURCING-5-7-3-HARASSMENT-LINEAGE` | rbi_outsourcing_banks_2006 | 5.7.3 | Earliest harassment prohibition in this corpus. |

`RegulatoryRule.is_active_on()` returns `False` for these on every date, the
schema forbids them having a `check_id`, and a test asserts both.

## Operationalisations, stated explicitly

These are places where a rule's text was turned into a call-time action. Each is
recorded in the rule's `notes`, and none adds an obligation the text does not
contain:

- **Agent identification** is derived from the prohibition on *anonymous* calls
  (paragraph 445). RBI prescribes no script or wording for a telephone call.
- **Recording disclosure before substantive discussion.** Paragraph 442(4) says
  "reasonable precaution such as intimating the customer"; it does not say when.
  The earliest defensible point was chosen.
- **Grievance-mechanism details on an in-call dispute** operationalises
  paragraph 442(7), which obliges the bank to have the mechanism and furnish its
  details.
- **Abusive language** is mapped under paragraph 445 ("harassment of any kind,
  either verbal") and paragraph 343 ("undue harassment", introduced by "viz.").
  Paragraph 454Z(1) names it expressly from 2027.
- **Sub judice escalates rather than blocks**, because paragraph 442(6) says
  "utmost caution", not a prohibition.
- **Missing agency details escalates rather than blocks**, because paragraph
  442(3) obliges the bank to inform the borrower but does not say contact is
  prohibited if that was missed.

## Deliberately not encoded

- No persistent-calling threshold. RBI prohibits "persistently calling" and
  states no number; `MAX_RECOVERY_CALLS_PER_DAY` is unset by default and the
  check reports *not evaluable* rather than guessing.
- No timezone is stated in the source text. `Asia/Kolkata` is a configuration
  default, documented as such.
- Paragraph 442(3)'s notice / authorisation letter / identity card and paragraph
  454X's identity-card display are written for in-person visits; they are not
  asserted as call-time rules.
- Whether IIBF training (paragraphs 447, 448, 454I) applies to an automated agent
  rather than a person **requires regulatory review** and is recorded as such.

---

# 6. Policy Engine

`PolicyEngine.evaluate(PolicyContext) -> PolicyDecision`. Given the same context
and rule file it returns the same decision; there is no clock read, no I/O and no
model call inside it.

**How it runs:** select rules active on `now.date()` and in scope for the account's
`product_type` → for each, dispatch to its `check_id` → aggregate. Construction
fails with `PolicyConfigurationError` if the rule file names a check that is not
implemented, so a rule cannot silently never run.

**Deterministic (11 checks):** `calling_hours`, `persistent_calling`,
`prohibited_conduct`, `agent_identification`, `recording_disclosure`,
`confidentiality_disclosure`, `discuss_only_with_borrower`,
`grievance_pending_hold`, `dispute_handling`, `agency_details_disclosed`,
`digital_lending_particulars_sent`.

**The decision output:**

```python
PolicyDecision(
    allowed, escalate, tone, dpd_stage,
    violations,            # each with rule_id, severity, citation, blocks_collection
    required_actions,      # enum
    prohibited_conduct,    # enum — what the response must not express
    rule_ids,              # rules actually evaluated
    not_evaluable_rule_ids,# active rules whose backend input was missing
    unenforced_rule_ids,   # active rules recorded for audit, no automated check
    evaluated_at, policy_version, rules_version,
)
```

This differs from the shape in the brief in two ways, both deliberate: the
conceptual `actions` field was dropped as redundant with `required_actions`, and
`not_evaluable_rule_ids` / `unenforced_rule_ids` were added. A decision that
silently ignored a rule would not be auditable; a test asserts that every
applicable rule appears in exactly one of the three lists.

**Outside the engine, on purpose:**

- All natural language. The engine never writes a sentence.
- Detecting prohibited conduct *in text*. At decision time no text exists; the
  engine publishes the prohibited categories and `app/services/validation.py`
  enforces them against the draft.
- Anything needing data the backend has not supplied — reported, not assumed.
- Organisational obligations (12 rules) — visible in `unenforced_rule_ids`.
- DPD banding and tone: **business** rules, marked as such in code, not derived
  from any RBI text.

---

# 7. Conversation State

```python
class ConversationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    language: Language | None = None
    intent: Intent | None = None
    emotion: Emotion | None = None

    payment_promise: bool = False
    promise_date: date | None = None

    dispute: bool = False
    wrong_person: bool = False
    escalation_required: bool = False

    current_stage: ConversationStage = ConversationStage.GREETING

    identity_verified: bool = False      # required by paragraph 446 / 454Y(1)
    recording_disclosed: bool = False    # required by paragraph 442(4) / 454P
    agent_identified: bool = False       # required by paragraph 445 (anonymous calls)

    turn_count: int = 0
```

Changes from the previous model:

- **`dpd` removed.** Days-past-due is account data and is read from
  `AccountContext`. Duplicating it in mutable conversation state would let the
  call drift away from the system of record, which the brief names as the source
  of truth. A test asserts the state model carries no account fields.
- **Three disclosure booleans added**, each because an encoded rule depends on it.
- `language`, `intent`, `emotion` are now typed enums rather than free strings.
- `session_id` added so a state object is self-identifying in logs.

Separation, as required:

| Concern | Where |
| --- | --- |
| State | `ConversationState` — small, mutable, replayable |
| Events / transcript | `ConversationEvent` — append-only, carries `text` |
| Customer & account data | `CustomerContext`, `AccountContext` (backend) |
| Compliance inputs | `ComplianceContext` (backend) |
| Policy decisions | `PolicyDecision` — never stored on state |

`apply_event(state, event)` is pure; `replay(session_id, events)` rebuilds state
from the log, and a test asserts replay equals incremental application.

---

# 8. Tool Interfaces

All seven are registered and validate arguments. None is connected to a banking
system. With the default `NullBankingBackend` every one returns
`ToolStatus.NOT_IMPLEMENTED` — no data is fabricated.

| Tool | Arguments | Status |
| --- | --- | --- |
| `get_customer_context` | `customer_ref` | interface + stub |
| `get_account_status` | `account_ref` | interface + stub |
| `get_outstanding_amount` | `account_ref` | interface + stub |
| `get_dpd` | `account_ref` | interface + stub |
| `record_payment_promise` | `account_ref`, `promise_date`, `amount_minor` | interface + stub |
| `create_dispute` | `account_ref`, `reason_code` | interface + stub |
| `escalate_case` | `account_ref`, `reason_code` | interface + stub |

Each returns the narrowest payload that answers its question — `get_dpd` returns
`{"dpd": n}`, not the account record — following the need-to-know requirement in
the Outsourcing Directions paragraph 24.

Auditability: every execution emits one structured `tool_call` event with
`request_id`, `session_id`, `turn_id`, `tool_name`, `tool_latency_ms`, `status`,
`argument_keys` and `error_type`. Argument **values** are never logged, and a
backend exception's message is never logged (only its type). A test asserts both;
the second test was added after an earlier version leaked an account reference
into the log through an exception message.

Wiring a real backend means implementing the `BankingBackend` protocol and
passing it to `build_runtime(backend=...)`. `tests/fakes.py` shows this working
end to end.

---

# 9. LLM Boundary

Nothing is connected. No provider SDK is installed, and no module under `app/`
makes a network call.

`app/services/llm.py` defines `LlmService` — one method,
`generate(LlmRequest) -> LlmGeneration` — plus provider-neutral
`LlmMessage`, `LlmToolSpec`, `LlmToolCall` types. The default
`NotConfiguredLlmService` raises `LlmNotConfigured` rather than degrading
quietly.

To connect Gemma later, with no redesign:

1. Write one adapter class with a `generate` method that calls the remote
   OpenAI-compatible endpoint using `MODEL_BASE_URL`, `MODEL_NAME`,
   `MODEL_API_KEY` from `Settings`.
2. Pass it to `build_runtime(llm=...)`.

Nothing else changes, because:

- `ToolRegistry.specs()` already emits JSON Schema per tool for the model.
- The model returns `LlmToolCall`s; only the registry executes anything, after
  validating arguments.
- The draft goes to `ResponseValidator.validate(draft, decision, grounding)`
  before TTS; a blocked draft does not proceed.
- Prompt content is not where the rules live — `recovery_rules.json` is.

A test asserts the unconfigured service raises, and that a substitute satisfying
the interface can be swapped in.

---

# 10. STT / TTS / Turn Detection

## Transcript normalisation — interface only

`TranscriptNormalizer` protocol, with `PassthroughTranscriptNormalizer` and
`WhitespaceTranscriptNormalizer` (collapse runs of whitespace, trim — the only
correction that is safe across all four languages without evidence).

**Not implemented, and reported as such** by every result's `unhandled` field:
Hindi/English code-switching, Hinglish romanisation, Marathi lexicon, person-name
correction, banking domain terms, spoken-number parsing, spoken-date parsing. No
correction dictionary was hard-coded, because each of those needs a lexicon built
from real call data.

## TTS normalisation — partially implemented

- **Span detection works for all languages**: currency, number, date, phone,
  account reference, abbreviation, non-overlapping.
- **Rendering is implemented for English only**, using the Indian numbering
  system (lakh/crore), with identifiers read digit-by-digit and abbreviations
  spelled out.
- **Hindi, Marathi and Hinglish are not rendered.** Spans are still detected, but
  `spoken` is `None`, `fully_normalized` is `False` and the text is returned
  unchanged. Correct spoken forms need a pronunciation lexicon and native review;
  pretending otherwise would produce an agent that misreads amounts to customers.
  Tests assert this rather than papering over it.
- No TTS provider is integrated.

## Turn detection and barge-in — deterministic parts implemented, no model

- `VadResult` — the audio layer's output shape (no VAD model here).
- `SilenceTurnDetector` — deterministic, threshold-based; a non-final hypothesis
  never ends a turn.
- `SemanticTurnDetector` — **protocol only, no implementation**. A punctuation or
  keyword heuristic would cut customers off mid-sentence in code-switched speech.
  `CompositeTurnDetector` is the composition point for when a model exists.
- `HeuristicBargeInClassifier` — transparent rules, no ML:
  - not speech / low probability / under 200 ms → `NOISE`, keep speaking;
  - a recognised backchannel ("haan", "hmm", "okay", "ji", "achha", "ho", …)
    within 800 ms → `BACKCHANNEL`, **keep speaking**;
  - substantive speech, or a backchannel token that runs past the window, or
    sustained speech with no transcript → `INTERRUPTION`, **stop TTS**;
  - short speech with no transcript yet → `UNKNOWN`, wait.
  - The backchannel vocabulary is **provisional** — derived from the languages in
    scope, not from measured call data, and marked so in code.
- `TurnStateMachine` — explicit states with an allowed-transition table; an
  invalid transition raises rather than being ignored.

---

# 11. Testing

```
249 passed, 0 failed, 0 skipped, 0 errors   (pytest 9.1.1, 0.37s)
```

| File | Tests | Covers |
| --- | --- | --- |
| `test_tts_normalization.py` | 32 | Indian numbering, currency/date/identifier rendering, unsupported languages |
| `test_policy_engine.py` | 31 | Determinism, decision completeness, DPD banding, disclosures, persistent calling |
| `test_turn_detection.py` | 26 | Backchannel vs interruption, state machine, turn detection |
| `test_calling_hours.py` | 24 | Both windows' boundaries, timezone, 2027 handover |
| `test_validation.py` | 23 | Prohibited content, grounding, promises, tool requests, confirmations |
| `test_conversation_state.py` | 22 | Reducer purity, flows, replay, session store |
| `test_regulatory_rules.py` | 21 | Verbatim quotes, catalog cross-check, schema invariants, date scoping |
| `test_tools.py` | 16 | Argument validation, failure modes, audit log, leak prevention |
| `test_api.py` | 15 | Health, session lifecycle, input validation, limits |
| `test_flows.py` | 13 | Wrong person, dispute, grievance, escalation |
| `test_stt_normalization.py` | 12 | No invented corrections; gaps reported |
| `test_observability.py` | 8 | Redaction, JSON output |
| `test_llm_boundary.py` | 6 | Unconfigured model fails loudly; service is swappable |

Every test asserts real behaviour; none asserts a constant.

Test cases worth naming:

- **08:00 / 19:00 boundaries.** 07:59:59 blocked, 08:00:00 allowed, 19:00:00
  allowed, 19:00:01 blocked — the endpoints are inside the window because
  paragraph 445 prohibits calling *before* 08:00 and *after* 19:00.
- **Microfinance boundaries.** 08:30 is allowed for a retail loan and blocked for
  a microfinance loan; 09:00 and 18:00 allowed, 18:00:01 blocked.
- **Timezone is configuration.** 14:00 UTC is blocked under `Asia/Kolkata`
  (19:30 local) and allowed under `UTC` — same instant, different configuration.
  A naive datetime is rejected outright.
- **2027 handover.** On 2026-12-31 paragraph 445 is evaluated and 454Y(4) is not;
  on 2027-01-01 the reverse. The express-authorisation exception works only from
  2027, because paragraph 445 states no such exception.
- **Verbatim quotes.** All 49 quotes are re-read from their source files.
- **No silent rules.** Every applicable rule appears in exactly one of
  evaluated / not-evaluable / unenforced.
- **No invented threshold.** With `MAX_RECOVERY_CALLS_PER_DAY` unset, 50 contacts
  in a day is *not evaluable*, not a violation.
- **Grounding.** "You owe Rs. 99,999.00" is blocked; "Rs. 12,345.00" (the backend
  figure) passes.
- **No leaks.** A backend exception carrying an account reference does not reach
  the log.

Not tested, because it does not exist: telephony, STT, TTS, model inference,
banking integration, concurrency under load, persistence.

---

# 12. Security Review

This section describes what the code does. **It is not an assurance that the
system is secure.** No penetration test, threat model or external review has been
performed.

## Implemented

| Control | Where |
| --- | --- |
| No secrets in Git | `.env` is gitignored and absent; `.env.example` contains no values; a grep for hardcoded key/secret/password/token patterns finds none. |
| Secret handling | `MODEL_API_KEY` is `SecretStr`, so it does not appear in `repr()` or a settings dump. |
| Input validation | Every API body and tool argument model sets `extra="forbid"` with typed, bounded fields; transcript text is capped at 4000 characters; unknown enum values are rejected (422). |
| No unsafe deserialization | The rule file is read with `json` only. No YAML, pickle, marshal, `eval`, `exec` or `__import__` anywhere in `app/`. |
| No arbitrary tool execution | The model cannot execute anything. `ToolRegistry.execute` resolves the name against a fixed registry and validates arguments before any backend call. |
| Model output is constrained | `ResponseValidator` blocks prohibited conduct, ungrounded amounts and dates, unauthorised concessions, unknown tool requests and missing mandatory disclosures. |
| Log hygiene | `redact()` drops values under names matching name/phone/email/PAN/Aadhaar/account/transcript/token/secret patterns, recursively. Tool argument values and backend exception messages are never logged. |
| Error hygiene | A backend exception is returned as its type name; no stack trace or connection string reaches the caller or the log. |
| Resource bound | `MAX_ACTIVE_SESSIONS` caps the in-memory store; exceeding it returns 503 rather than growing unbounded. |
| Concurrency | `SessionStore` is lock-guarded; FastAPI runs sync endpoints on a thread pool. |
| Production guard | Caller-supplied customer/account/compliance context is rejected when `APP_ENV=prod`, so a client cannot assert its own compliance inputs. |

## Known gaps (present in this code)

- **No authentication or authorisation.** Anyone who can reach the API can create
  and read any session by id. Session ids are `uuid4` and unguessable, but that
  is not an access control.
- **No rate limiting** and no request-size limit beyond the field caps.
- **No TLS configuration**; termination is assumed to be upstream.
- **Outside production, callers supply account context.** That is how the policy
  engine is exercised without a backend, and it is exactly why the prod guard
  exists — but it means a non-prod deployment trusts its clients.
- **`ToolResult.error` may carry backend-supplied text.** It is kept out of logs
  but returned to the orchestrator; it must never be spoken to a customer or
  logged verbatim.
- **Redaction is name-based.** A sensitive value logged under an innocuous field
  name will pass through.
- **Sessions are in memory.** No encryption at rest is relevant yet, and nothing
  survives a restart.

## Not implemented (production controls)

Authentication/authorisation, secret management integration, audit-log
persistence and tamper evidence, PII retention and deletion policy, DPDP Act
obligations, call-recording storage and the six-month retention that paragraph
454P will require from 2027, encryption at rest and in transit, network policy,
dependency and container scanning, and an incident-response path.

---

# 13. Dependencies

One dependency was added.

| Package | Why |
| --- | --- |
| `httpx2` | **Test-only.** Starlette's `TestClient` needs an HTTP client library; this Starlette version emits a deprecation warning for `httpx` and expects `httpx2`. Without it the API cannot be tested through real routing and serialization. Nothing in `app/` imports it. |

Unchanged: `fastapi`, `pydantic`, `pydantic-settings`, `python-dotenv`,
`uvicorn[standard]`, `pytest`. `zoneinfo`, `json`, `re`, `threading`, `uuid`,
`logging` and `datetime` are standard library.

No HTTP client, model SDK, database driver, cache, queue, NLP library or
container runtime was added.

---

# 14. Known Limitations

**Regulatory**

1. The encoded rules have not been reviewed by a compliance or legal function.
   They reflect one engineer's reading of the supplied corpus. **Requires
   regulatory review before any live use.**
2. The corpus catalog records that the 2022 circular's source changed on
   re-fetch (`source_changed_review_required`). That review has not been done.
3. Whether IIBF training and agent-identification requirements written for human
   agents apply to an automated agent is unresolved.
4. No persistent-calling threshold exists, so that rule is inert by default.
5. The corpus covers RBI only. DPDP Act obligations, TRAI/TCCCPR commercial
   communication rules (referenced by paragraph 454AB) and state law are not
   represented. The repository's separate `regulatory_corpus/` tree contains DPDP
   documents that were **not** used here, because the brief scoped this work to
   `data/regulatory/`.

**Engineering**

6. No telephony, STT, TTS, model or banking backend is connected. End-to-end
   latency is unknown and untested.
7. Hindi, Marathi and Hinglish TTS rendering is not implemented.
8. The prohibited-conduct lexicon is small, English and romanised-Hindi only, and
   will miss paraphrases. It has a high false-negative rate by construction. The
   grounding checks are the strong part of validation; the lexical checks are not
   a content-safety system.
9. The backchannel vocabulary and the 200 ms / 800 ms barge-in thresholds are
   provisional and not tuned against real audio.
10. Semantic turn detection does not exist.
11. Sessions are in-memory and single-process; nothing persists.
12. Emotion is a typed enum with no detector behind it.
13. `ProductType` is single-valued, so a loan that is both digital lending and a
    retail loan must be classified as one of them. If both rule sets must apply
    at once, this needs to become a set.
14. The event `data` payload is validated against a fixed signal model; adding a
    new upstream signal means changing `EventSignals`.
15. Not load tested, not soak tested, no performance budget.

---

# 15. Next Phase

**Recommended next phase: connect Gemma behind the existing `LlmService`
interface and build the turn orchestrator that joins state → policy → model →
tools → validation.**

Why this and not something else: the deterministic layers exist and are tested,
but nothing drives them yet. Until one component calls the policy engine, asks
the model for a draft, runs the draft through the validator and loops, the
boundaries are untested as a system. This phase also proves the LLM boundary
holds — that a blocked draft really cannot reach output — which is the claim the
whole design rests on.

It is the right next step ahead of telephony, STT or TTS because it needs no
audio, no GPU and no vendor: a remote OpenAI-compatible endpoint and a text
harness are enough, and text-mode conversations can be evaluated against the
policy engine before any real-time constraint is added.

Concretely, that phase would cover: one `LlmService` adapter for a remote
OpenAI-compatible endpoint; a prompt builder that takes the `PolicyDecision` as
structured input rather than restating rules in prose; a turn loop with the
tool-call cycle; a retry path for a blocked draft; and text-mode conversation
tests over the 5 / 30 / 90 DPD stages in all four languages.

Not started. No work on it has been done.

---

# Stage 2 — Conversation Orchestrator

Date: 2026-09-24 · 321 tests passing (249 from Stage 1, 72 new)

Stage 2 adds the component that was missing at the end of Stage 1: something
that actually drives the deterministic layers. Nothing new is connected —
telephony, STT, TTS, Gemma and the banking backend are still absent, and the
default backend still raises rather than returning invented data.

## S2.1 What was built

One deterministic turn pipeline: raw transcript in, a structured `TurnResult`
out. The orchestrator owns the order of operations and the decision to stop. It
owns no policy, no tool execution, no validation and no language.

```
                  ┌─────────────────────────────────────────┐
  transcript ──►  │ 1  normalise            (TranscriptNormalizer)
                  │ 2  → ConversationEvent
                  │ 3  apply_event          (pure reducer)
                  │ 4  load backend context (from the Session)
                  │ 5  POLICY ▸ PRE_LLM     (PolicyEngine)
                  └───────────────┬─────────────────────────┘
                                  │
                   not allowed ◄──┴──► allowed
                        │                 │
             policy_blocked,              ▼
             no model call        ┌───────────────────────────┐
             required_actions     │ 6  LlmRequest  (LlmService)
             returned             │ 7  tool calls?            │
                                  └──────┬────────────┬───────┘
                                     yes │            │ no
                                         ▼            │
                              ┌──────────────────┐    │
                              │ bind identity    │    │
                              │ args             │    │
                              │ ToolRegistry     │    │
                              │ → BankingBackend │    │
                              │ absorb result    │    │
                              │ POLICY ▸ POST_TOOL    │
                              └──────┬───────────┘    │
                                     └────────────────┤
                                                      ▼
                                       ┌──────────────────────────┐
                                       │ 8  POLICY ▸ PRE_TTS      │
                                       │ 9  ResponseValidator     │
                                       │ 10 TTS normalisation     │
                                       │ 11 AGENT_UTTERANCE event │
                                       └────────────┬─────────────┘
                                                    ▼
                                               TurnResult
```

**The model never decides anything.** It is asked for two things — words, and
structured tool requests. Whether collection is lawful, whether a tool may run,
whether a violation is acceptable and whether a sentence may be spoken are all
settled outside it, by the policy engine, the tool registry and the response
validator.

## S2.2 Files

**Created**

| File | Lines | What |
| --- | --- | --- |
| `app/orchestrator/__init__.py` | 47 | Package exports |
| `app/orchestrator/pipeline.py` | 1111 | `ConversationOrchestrator` — the turn pipeline |
| `app/orchestrator/prompt.py` | 154 | Builds the `LlmRequest` from a decision plus grounding facts |
| `app/orchestrator/result.py` | 257 | `TurnResult` and its enums |
| `tests/test_orchestrator.py` | 1302 | 62 tests |
| `tests/test_orchestrator_api.py` | 250 | 10 tests |

**Modified**

| File | Change | Why |
| --- | --- | --- |
| `app/core/session.py` | Added `SIGNAL_BEARING_KINDS`; `apply_event` parses `data` as signals only for those kinds. Added `SessionStore.set_account` | See S2.3 |
| `app/services/validation.py` | Added `UNGROUNDED_NUMBER` and `GroundingFacts.numbers`; bare figures are now grounding-checked | See S2.13 |
| `app/config.py` | Added `max_tool_calls_per_turn` (default 2) | Tool-loop bound |
| `app/api/routes.py` | Added `TurnRequest` and `POST /conversation/{id}/turn`; `EventRequest` now rejects signals on audit-only event kinds | A caller surface for the orchestrator |
| `app/main.py` | Builds one `ConversationOrchestrator` onto `app.state` | Tool specs are read once, not per turn |
| `.env.example` | Documented `MAX_TOOL_CALLS_PER_TURN` | |
| `tests/fakes.py` | Added failure doubles | Reaching each failure branch |
| `tests/conftest.py` | Added a restoring `log_stream` fixture | `configure_logging` replaces the root handler |

Stage 1's models, policy engine, checks, rule file, tool registry, validator and
normalisers were **not** rewritten.

### Why `TurnResult` is not in `app/models`

It was, briefly, and it created a real import cycle: `app.models` →
`app.services.validation` → `app.core.rules` → `app.models`. `import
app.core.rules` failed outright. A turn result aggregates the policy engine's
output *and* the validator's, so it sits above both. `app/orchestrator/` is the
top layer: it imports everything and nothing imports it.

## S2.3 The three Stage 1 changes, and why each was necessary

`apply_event` ran every event's `data` through `EventSignals`, which is
`extra="forbid"`. That meant a `TOOL_CALL` event carrying `{"tool_name": ...,
"status": ...}` raised `ValidationError` — and because `SessionStore.append_event`
assigns state *before* appending, the event was dropped entirely. The audit trail
could not record a tool call at all.

The fix is a kind whitelist:

```python
SIGNAL_BEARING_KINDS = {SESSION_STARTED, USER_UTTERANCE, AGENT_UTTERANCE,
                        BARGE_IN, ACCOUNT_CONTEXT_LOADED, CALL_ENDED}
```

`TOOL_CALL`, `TOOL_RESULT` and `POLICY_DECISION` now carry opaque audit metadata.
Keying on the event *kind* rather than on the shape of `data` preserves the
strictness that matters: a signal-bearing event with an unknown key is still
rejected.

Blast radius was measured, not assumed: the full Stage 1 suite was re-run with
the patched reducer under four different whitelist choices. Excluding those three
kinds changes **zero** existing tests (249 passed). `AGENT_UTTERANCE` must stay
signal-bearing — `test_disclosure_flags_latch_on` puts `disclosed_recording` on
it.

One untested behaviour does change: posting signal fields on a `tool_call` event
over HTTP no longer applies them to state. No test covered that, and it was
nonsense behaviour. `EventRequest` now rejects it with a 422 rather than
accepting it and quietly doing nothing.

**Two further changes came out of the adversarial review (S2.13):**

`SessionStore.set_account` was added because a tool-refreshed account context was
being thrown away at the end of the turn, so the next turn re-decided policy on
data the previous turn had already disproved. There was no existing way to write
it back.

`ResponseValidator` gained an `UNGROUNDED_NUMBER` check and `GroundingFacts`
gained a `numbers` field, because grounding only covered `CURRENCY` and `DATE`
spans: a figure written as a bare number — "you owe 99,999 rupees", "you are 180
days past due" — was never checked at all. That is the component whose job this
is, so the check belongs there rather than being duplicated in the orchestrator.
The change is additive and the full Stage 1 suite passes unchanged.

## S2.4 Turn lifecycle, exactly

1. **Normalise** — `TranscriptNormalizer.normalize(segment)`. A raise ends the
   turn before any state changes.
2. **Event** — a `USER_UTTERANCE` carrying the normalised text and the upstream
   signals (`intent`, `emotion`, `promise_date`, …).
3. **State** — through `SessionStore.append_event` → `apply_event`. The
   orchestrator never assigns to `ConversationState`.
4. **Context** — `AccountContext` / `ComplianceContext` are read from the
   `Session`. If no account is loaded, that is recorded as a non-fatal
   `MISSING_BACKEND_CONTEXT` error and grounding stays empty.
5. **Policy ▸ PRE_LLM**.
6. **If not allowed** → stop. No model call. Return `policy_blocked` plus the
   decision's `required_actions`.
7. **Generate** — system prompt (constraints + facts) + the customer's turn +
   the registry's tool specs.
8. **Tools, if requested** — see S2.5. Then **Policy ▸ POST_TOOL**, and back to 7
   with the tool results in the message history.
9. **Policy ▸ PRE_TTS** — against the state the turn actually ended in.
10. **Validate** — `ResponseValidator`. Gated on `blocked`, not on `valid`:
    a non-blocking issue (an unmet *reported* required action) makes `valid`
    false but is not a reason to stay silent.
11. **TTS normalise** — and an `AGENT_UTTERANCE` event.

**Turn identity.** `ConversationEvent.turn_id` is an event index
(`len(session.events)`), so the five events of one turn get five different
values. `TurnResult.turn_id` is instead `state.turn_count` after the customer's
utterance — the only counter that advances exactly once per customer turn.

## S2.5 Tool execution

```
LlmToolCall → bind identity args → ToolRequest → ToolRegistry → BankingBackend
```

The orchestrator calls no backend method:
`grep -rEn "\bbackend\.[a-z_]+\(" app/orchestrator/` returns nothing, and
`BankingBackend` is not imported anywhere in the package. The only path to the
bank's systems is `ToolRegistry.execute`.

**Identity arguments are bound, not accepted.** `account_ref` and `customer_ref`
are overwritten with the session's own values before dispatch, using the argument
names each tool publishes in its spec. The model is never told an account
reference and so cannot address another borrower's account — not because it is
asked not to, but because it has no way to. If the session has no such context,
the call is refused and never reaches the registry.

**Writes are gated on what was actually said.** `record_payment_promise`,
`create_dispute` and `escalate_case` change the case file and cannot be undone by
a later turn. Each requires the corresponding conversation state to already be
true — `payment_promise`, `dispute`, `escalation_required` — which arrives as an
upstream NLU signal from the customer's own words. A model cannot manufacture the
event it is reporting. The promised *date* is likewise bound from
`state.promise_date`, not taken from the model, and the call is refused if no
date was captured. A write must also be the only tool call in its round, so the
decision immediately before it is the freshest one available.

**Binding fails closed.** If a tool is registered after the orchestrator read the
registry's specs, its identity arguments cannot be bound, so it is refused rather
than dispatched with whatever the model supplied.

**Failure handling**

| Situation | `ToolStatus` | Recorded as | Turn |
| --- | --- | --- | --- |
| Unknown tool name | `NOT_FOUND` | `INVALID_TOOL_REQUEST` | fails |
| Bad arguments | `INVALID_REQUEST` | `INVALID_TOOL_REQUEST` | fails |
| Refused before dispatch | — (`dispatched=False`) | `INVALID_TOOL_REQUEST` | fails |
| No backend behind the tool | `NOT_IMPLEMENTED` | `TOOL_BACKEND_NOT_IMPLEMENTED` | **continues** |
| Backend raised | `BACKEND_ERROR` | `TOOL_EXECUTION_FAILED` | fails |

`NOT_IMPLEMENTED` is the one that continues, because unavailability is a fact the
agent can honestly act on. The model is told `"<tool> unavailable
(not_implemented). Do not state this fact; say you will check."` — a status, never
the backend's own message, which carries the account reference.

**Loop bound.** `MAX_TOOL_CALLS_PER_TURN`, default 2. Not a business number: the
banking tools return deliberately narrow payloads, so the widest single question —
"what do I owe and when was it due" — needs `get_outstanding_amount` plus
`get_account_status`. On exceeding it the turn stops with `TOOL_LIMIT_EXCEEDED`
and produces no answer.

## S2.6 Policy checkpoints

| Checkpoint | When | Why it is not redundant |
| --- | --- | --- |
| `PRE_LLM` | Before generation | A prohibited call must not produce a compliance-sensitive sentence at all |
| `POST_TOOL` | After each tool round | A tool reads the system of record and can change the answer |
| `PRE_TTS` | Before validation and speech | The decision must match the state the turn ended in |

Every decision is kept, in order, on `TurnResult.policy_evaluations`.

`POST_TOOL` earns its place. The tested case: a session's cached context says
retail loan and collection is allowed; `get_account_status` reads the system of
record, which says **digital lending**; the pre-contact particulars for digital
lending were never sent, so `RBI-DL-2025-8-V-AGENT-PARTICULARS-BEFORE-CONTACT`
produces a blocking violation. A decision taken before the tool would have been
wrong. Note also that with no account loaded a product-scoped rule is absent from
the decision entirely — not merely "not evaluable" — so an empty
`not_evaluable_rule_ids` must never be read as "nothing to worry about".

## S2.7 Grounding

`GroundingFacts` is built from the loaded `AccountContext` and from tool results.
The model is never a source.

- A tool may update only the `AccountContext` fields listed for it in
  `_ACCOUNT_PROJECTION`. Anything else a backend returns is ignored.
- **Nothing is constructed from nothing.** If no `AccountContext` was loaded, a
  partial tool payload cannot build one; the orchestrator returns `None`
  unchanged rather than inventing the fields the payload lacks.
- `allow_concession_offer` is always false. No tool in this foundation authorises
  a waiver, settlement or discount, so none may be offered.
- After a successful `record_payment_promise`, the promised **date** becomes a
  grounded fact — and only because the orchestrator, not the model, supplied it.
  The promised **amount** is deliberately not grounded: no signal carries it, so
  it is whatever the model wrote, and admitting it would let the model mint a
  figure and then state it as the balance.
- Facts are **named** in the prompt (`outstanding`, `minimum_due`,
  `last_payment_amount`, `due_date`, `days_past_due`). An unlabelled list of
  figures invites the model to present the last payment as the balance, and
  grounding is set membership — it would not catch that, because the number is
  genuine.

`TurnResult` records grounding as provenance and a count —
`grounding_sources=(account_context, tool_result)`, `grounded_fact_count=5` —
never as values.

## S2.8 Error handling

Thirteen categories on `TurnErrorCategory`, covering all ten cases the brief
required. Each records a *category* plus an application-authored detail; a
backend message or a provider exception string is never copied in.

| # | Case | Category | Safety-critical |
| --- | --- | --- | --- |
| 1 | Transcript normalisation failed | `TRANSCRIPT_NORMALIZATION_FAILED` | yes |
| 2 | Invalid conversation event | `INVALID_CONVERSATION_EVENT` | yes |
| 3 | Missing backend context | `MISSING_BACKEND_CONTEXT` | no |
| 4 | Policy misconfigured | `POLICY_CONFIGURATION_ERROR` | yes |
| 4b | Policy undecidable | `POLICY_EVALUATION_FAILED` | yes |
| 5 | No model configured | `LLM_NOT_CONFIGURED` | yes |
| 5b | Model errored | `LLM_FAILED` | yes |
| 6 | Invalid tool request | `INVALID_TOOL_REQUEST` | yes |
| 7 | Tool execution failed | `TOOL_EXECUTION_FAILED` | yes |
| 8 | Backend not implemented | `TOOL_BACKEND_NOT_IMPLEMENTED` | no |
| — | Tool budget exhausted | `TOOL_LIMIT_EXCEEDED` | yes |
| 9 | Response validation failed/blocked | `RESPONSE_VALIDATION_FAILED` | yes |
| 10 | TTS normalisation failed | `TTS_NORMALIZATION_FAILED` | yes |

Two conditions are raised rather than returned, because they are caller errors
rather than turn outcomes: `UnknownSession` (no such session) and `IncompleteTurn`
(a non-final transcript). Deciding that a turn is over belongs upstream, in
`app/services/turn.py`; running the pipeline on half a sentence would advance
state and could commit a backend write. The HTTP layer maps them to 404 and 400.

`speakable` is the single question a caller must ask before playing audio. It is
true only for `COMPLETED`.

**An unrenderable span is a failure, not a cosmetic gap.** TTS rendering is
implemented for English only. An amount or date returned unrendered would be read
out wrongly by a synthesiser — a false representation of an account fact — so the
turn stops. Text with no such spans stays speakable in every language, so this
does not silently disable Hindi, Marathi or Hinglish calls.

## S2.9 Observability and latency

One `turn` record per turn. `missing_standard_fields()` on the emitted record
returns `()`.

Logged: `session_id`, `turn_id`, `language`, `intent`, `dpd_stage`, `outcome`,
`speakable`, `policy_allowed`, `policy_escalate`, `policy_rule_ids`,
`policy_violation_rule_ids`, `policy_checkpoints`, `tool_count`, `tool_name`,
`tool_statuses`, `tool_latency_ms`, `model_latency_ms`, `llm_calls`,
`stt_latency_ms`, `tts_latency_ms`, `validation_valid`, `validation_blocked`,
`validation_codes`, `tts_fully_normalized`, `tts_unrendered_kinds`,
`latency_ms`, `barge_in`, `error_type`, `error_categories`.

**Not logged, on any path:** the transcript, the draft, the spoken text, account
and customer references, any amount, and every backend message. `redact()` would
not have saved us here — it matches field-name fragments and does not cover
`account_ref`, `customer_ref`, `outstanding_minor` or `display_name`, and it does
not recurse into pydantic models at all, so a model passed to `log_event` would be
stringified in full with zero redaction. The orchestrator therefore passes only
hand-picked scalars and lists of strings. A test greps the raw captured log for
`ACC-1`, `CUST-1`, `Test Borrower`, `1234500` and the transcript.

Nine stages are measured separately on `TurnLatency`. Stages that did not run
stay at zero, so a zero reads as "did not happen" rather than "was instant";
`total_ms` covers the whole turn, so it is not the sum of the parts. Nothing has
been optimised — these exist so that the next phase optimises something real.

## S2.10 Known limitations

1. **`claimed_actions` is asserted, not verified.** The validator blocks a
   response when a required in-turn action (disclose recording, identify the
   bank) is not claimed. The orchestrator cannot verify that a Hindi sentence
   actually performs the action, so it does not pretend to: the claim is an input
   from the caller. Nothing checks it is true. A language-aware check is needed.
2. **A promised amount cannot be confirmed back.** Because the amount is
   model-authored (see S2.7), the agent cannot say "noted, INR 2,500 on 5
   October" — the figure is not grounded. `CONFIRM_PAYMENT_PROMISE_DETAILS` is a
   non-blocking required action, so the turn still completes, but the
   confirmation is degraded. Closing this properly needs an NLU signal carrying
   the promised amount, and a role-tagged grounding model.
3. **Grounding has no roles.** `GroundingFacts` is a flat set of permitted
   amounts and dates. Naming the facts in the prompt (S2.7) makes the right
   answer obvious to the model, but it is guidance, not a control: a draft that
   states the *last payment* as the *outstanding balance* still passes, because
   the number really is a backend fact. Fixing this means giving grounding a
   role per fact and having the validator check the role, which is a change to
   the Stage 1 validator contract.
4. **The bare-number check is blunt.** Any figure that is not a known fact now
   blocks the response (S2.13). That correctly stops fabricated balances and
   invented DPD, but it also blocks legitimate incidental numbers — "I will call
   you in 2 days" — and numbers written as words ("ninety-nine thousand rupees")
   produce no span at all and are still unchecked. The first is a usability cost
   taken deliberately; the second is a real remaining hole.
5. **`ComplianceContext` cannot be refreshed mid-call.** No tool exposes
   `BankingBackend.get_compliance`, so compliance facts enter only at session
   creation. `POST_TOOL` re-evaluation therefore sees new *account* data but never
   new *compliance* data.
6. **Conversation history is one turn deep.** Earlier turns are not replayed to
   the model — memory is `ConversationState` plus backend facts, by design, so the
   transcript is not accumulated. Whether that is enough for natural multi-turn
   dialogue is untested against real conversations.
7. **`process_turn` is long** (~350 lines). The order of operations is the
   artifact here, so it is written as one readable sequence rather than split
   across helpers, but each exit builds its own result and the latency
   bookkeeping is repeated.
8. **No retry path for a blocked draft.** A blocked response ends the turn. A
   production agent would want one bounded re-generation with the validation issue
   fed back.
9. **The tool budget of 2 is a judgement, not a measurement.** It is configurable
   and tested; it has not been checked against real call traffic.
10. **A write still commits before the turn's final policy check.** Keeping a
    write alone in its round means the decision immediately before it is fresh,
    but `PRE_TTS` runs afterwards and could still block a turn whose write has
    already landed. Nothing is rolled back; there is no compensating action.
11. **Single-threaded assumptions.** `SessionStore` is lock-guarded, but a turn
    reads the session, works, then appends — two concurrent turns on one session
    would interleave. Telephony gives one turn at a time per call; nothing enforces
    it.
12. **`NOT_IMPLEMENTED` is ambiguous.** The backend returns it both for "no
    backend is wired" and for "this account does not exist". The orchestrator
    treats both as benign unavailability, so a genuinely missing account reads as
    an unwired system.

## S2.11 What remains unimplemented

Unchanged from Stage 1: telephony, STT, TTS synthesis, the Gemma endpoint, the
banking backend, authentication, persistence, GPU/cloud infrastructure. Also
absent: barge-in is wired into no pipeline (`barge_in` is logged as `null`), there
is no streaming or partial-transcript handling, and no evaluation harness.

**No claim is made that this is production ready.** It is a deterministic
orchestration layer with fake collaborators, tested against fake collaborators.

## S2.12 Verification performed

- `.venv/bin/python -m pytest -q` → **321 passed**.
- Import checks: `app.main`, `app.orchestrator`, `app.api.routes`,
  `app.core.rules`, `app.core.session`, `app.models`, `app.runtime` — all import
  cleanly, in any order.
- `GET /health` → `{"status": "ok"}`.
- End-to-end fake turn, multi-turn fake conversation, policy-blocked turn,
  tool-call turn and final-response validation are each covered by tests over
  both the Python API and HTTP.

### Test coverage of the required cases

| Required case | Test |
| --- | --- |
| 1. Normal transcript → successful turn | `test_normal_transcript_produces_a_speakable_turn` |
| 2. Policy-blocked → LLM not called | `test_policy_blocked_turn_never_calls_the_model` |
| 3. No tool → response validation | `test_llm_with_no_tool_request_goes_straight_to_validation` |
| 4. Valid tool → registry executes | `test_valid_tool_request_is_executed_through_the_registry` |
| 5. Invalid tool → blocked | `test_a_tool_request_the_registry_does_not_know_is_blocked`, `..._with_bad_arguments_...` |
| 6. Tool returns NOT_IMPLEMENTED | `test_not_implemented_backend_does_not_fabricate_a_fact` |
| 7. Tool changes context → policy re-evaluates | `test_policy_is_re_evaluated_after_a_tool_changes_the_account_context` |
| 8. Response violates policy → TTS blocked | `test_a_response_that_violates_policy_never_reaches_tts` |
| 9. Grounding missing → fabrication blocked | `test_with_no_account_context_no_amount_or_date_may_be_stated` |
| 10. Multiple turns preserve state | `test_a_session_carries_state_across_turns` |
| 11. Tool-loop limit | `test_the_tool_loop_stops_at_the_configured_limit` |
| 12. LLM unavailable | `test_an_unconfigured_model_fails_the_turn_loudly` |
| 13. TTS normalisation failure | `test_tts_normalization_failure_stops_the_turn` |
| 14. Sensitive values absent from logs | `test_no_sensitive_value_reaches_the_log` |
| 15. TurnResult does not expose sensitive data | `test_the_turn_result_does_not_carry_account_or_customer_references` |


## S2.13 Adversarial review, and what it changed

The implementation was reviewed by independent agents working five lenses
(control flow, sensitive-data escape, the tool boundary, fabrication/grounding,
and spec/test honesty), with every high-severity finding put to two verifiers
prompted to *refute* it. 27 findings were raised; 8 went to verification and all
8 survived. Everything below was demonstrated by running the real orchestrator,
not by reading it.

**Fixed, with a regression test each:**

| Finding | What was wrong | Fix |
| --- | --- | --- |
| Promise laundering (critical) | `_absorb` copied the model's own `amount_minor` and `promise_date` into `GroundingFacts`. Because grounding has no roles, a model could record a promise of INR 98,765.43 and then state that as the outstanding balance — validated, spoken, `speakable=True`. | The amount is no longer grounded at all; the date is bound from `state.promise_date`. Writes require corroborating state. |
| Raw payloads in the prompt (critical) | `_tool_message` serialised the whole tool payload, handing the model `outstanding_minor=1234500` (paise — a 100× misstatement waiting to be read out) plus `account_ref`, `customer_ref` and `display_name`, contradicting this module's own documented invariant. | Tool results now report an outcome only. Facts reach the model through the FACTS block, correctly formatted. |
| Bare numbers bypassed grounding (critical) | The validator checked only `CURRENCY` and `DATE` spans, so "your outstanding balance is 99,999 rupees and you are 180 days past due" passed clean and was spoken. The pipeline docstring claimed the opposite. | Added `ValidationCode.UNGROUNDED_NUMBER` and `GroundingFacts.numbers`; DPD is now a grounded fact. Zero Stage 1 tests changed. |
| Binding failed open (high) | An unknown tool name skipped identity binding entirely. Harmless today, but a tool registered after construction would have been dispatched with the model's own `account_ref`. | Refused instead, when the registry knows a tool the orchestrator's snapshot does not. |
| Refreshed context was discarded (high) | A tool read the system of record, the orchestrator used it for the turn, and then threw it away — so the next turn re-decided policy on data this turn had already disproved. | `SessionStore.set_account` persists the correction, with an `ACCOUNT_CONTEXT_LOADED` event. |
| Partial transcripts ran the pipeline (medium) | `TranscriptSegment.is_final` was plumbed through and never read; a partial hypothesis advanced state and could commit a write. | `IncompleteTurn`, mapped to HTTP 400. |
| Model-authored argument keys in the audit record (medium) | `argument_keys` recorded whatever key names the model invented. | Only keys the tool's schema declares. |
| A weak security test (high) | `test_the_model_cannot_choose_which_account_a_tool_reads` passed with the binding removed. | Rewritten against a second borrower with a distinct balance, and mutation-checked: deleting the binding now fails it. |
| A tautological assertion (medium) | `assert date(2026,10,5) in {date(2026,10,5)}` — a literal compared with itself. | Replaced by three tests that exercise the promise path for real. |
| Silent signal drop over HTTP (medium) | Posting `intent` on a `tool_call` event returned 200 and did nothing. | `EventRequest` now rejects it with 422. |

**Raised and deliberately not fixed**, because each is a Stage 1 contract change
rather than a Stage 2 defect: grounding has no roles (limitation 3), and figures
written as words produce no span to check (limitation 4).

The three exploit scenarios the reviewers demonstrated — stating a laundered
promise amount as the balance, echoing raw paise as rupees, and fabricating a
balance and DPD as bare numbers — were each re-run after the fixes. All three now
end `response_blocked`, `speakable=False`, with the offending figure named in the
validation issues, while the legitimate turn still completes.

Stage 3 has not been started.


# Stage 3A — Real LLM Service Boundary + OpenAI-Compatible HTTP Client

**REAL GEMMA MODEL IS NOT CONNECTED YET.** No model weights were downloaded, no
model server was started, and no inference has ever run against this code.

**GCP/vLLM deployment is intentionally deferred to the next step.** Nothing was
deployed, no CUDA or NVIDIA driver was installed, no vLLM was installed, and no
cloud resource was created. Every test in this repository runs offline against an
in-process transport stub or a `http.server` bound to `127.0.0.1`.

## S3A.1 Objective

Make the Stage 2 orchestrator able to talk to a remotely hosted, OpenAI-compatible
model server — without the orchestrator learning that it is doing so. Stage 2
already depended on an `LlmService` protocol with a single `generate(LlmRequest)
-> LlmGeneration` method and a `NotConfiguredLlmService` that failed loudly. Stage
3A supplies the one implementation that was missing, and nothing else: no new
abstraction, no change to prompt construction, no change to policy, no change to
tool execution.

## S3A.2 Architecture

```
ConversationOrchestrator          owns order, policy checkpoints, identity binding
        |
        v
LlmService (Protocol)             app/services/llm.py — contract + error taxonomy, no I/O
        |
        v
OpenAiCompatibleLlmService        app/services/llm_openai.py — the only module that opens a socket
        |
        v
POST {MODEL_BASE_URL}/chat/completions
        |
        v
any OpenAI-compatible server      vLLM / llama.cpp / TGI / OpenAI — a deployment choice, not a code one
```

and, separately, the path a tool request takes — which the adapter is not on:

```
model  ->  tool_calls JSON  ->  LlmToolCall  ->  ConversationOrchestrator
                                                       |  binds account_ref/customer_ref
                                                       |  checks write preconditions
                                                       v
                                                  ToolRegistry  ->  banking backend
```

The orchestrator does not know whether the model is Gemma, where it runs, what
serves it, that it is reached over HTTP, or that it is authenticated. It imports
`LlmService`, `LlmMessage`, `LlmToolCall` and `LlmError` — no HTTP type, no
provider name, no URL. `app/services/llm.py` itself imports no HTTP library.

### Why the adapter is a translator and nothing else

`app/services/llm_openai.py` contains no prompt text, no policy, no grounding, no
conversation state and no tool execution. It converts an `LlmRequest` to JSON and
JSON back to an `LlmGeneration`. A tool call arrives as data and leaves as data;
what makes that claim true rather than aspirational is that the adapter has no
reference to `ToolRegistry`, to any backend, or to the session — it cannot execute
anything even by mistake. `test_the_adapter_never_executes_a_tool` puts a
tripwire on `ToolRegistry.execute` and asserts it is never reached.

## S3A.3 Files added

| File | What it is |
| --- | --- |
| `app/services/llm_openai.py` | The OpenAI-compatible HTTP adapter: request translation, response parsing, error mapping, bounded retries, latency measurement, structured logging. |
| `tests/test_llm_http_adapter.py` | 99 tests over the adapter in isolation. |
| `tests/test_llm_http_integration.py` | 38 tests driving the real orchestrator through an HTTP endpoint. |

## S3A.4 Files modified

| File | Change |
| --- | --- |
| `app/services/llm.py` | Added the `LlmError` taxonomy (`LlmNotConfigured` now subclasses it), `LlmConfigurationError`, and `LlmUsage` + `LlmGeneration.usage`. No existing field or signature changed. |
| `app/config.py` | Added seven `MODEL_*` settings and two validators (blank-means-unset; base URL must be HTTP). |
| `app/runtime.py` | Added `build_llm_service(settings)`; `build_runtime` now uses it instead of hard-coding `NotConfiguredLlmService()`. |
| `app/orchestrator/result.py` | Added six `TurnErrorCategory` members whose string values equal the boundary's `LlmError.category` values. |
| `app/orchestrator/pipeline.py` | One `except LlmError` clause between the existing `LlmNotConfigured` and generic handlers, plus `_llm_error_category`. Control flow is unchanged: every boundary failure still ends the turn with no speakable text. |
| `app/observability.py` | Added `QUIET_LOGGERS`, applied in `configure_logging`. A redaction control — see S3A.9. |
| `app/main.py` | Added a lifespan that closes the model connection pool on shutdown. |
| `tests/fakes.py` | Added OpenAI-compatible response builders, `FakeOpenAiServer` (a real loopback server), and `unused_loopback_url`. |
| `.env.example`, `requirements.txt` | Documented below. |

Not touched: `data/regulatory/`, `app/core/`, `app/models/`, `app/tools/`,
`app/api/`, `app/orchestrator/prompt.py`, `app/services/validation.py`,
`app/services/stt.py`, `app/services/tts.py`, `app/services/turn.py`. No existing
test was weakened, loosened or deleted.

## S3A.5 Configuration

All environment-driven, all documented in `.env.example`. The `MODEL_*` prefix is
the convention Stage 1 already established, so no new naming scheme was invented.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_BASE_URL` | *unset* | OpenAI-compatible base URL including the API version, e.g. `http://localhost:8000/v1`. Must be `http://` or `https://`. |
| `MODEL_NAME` | *unset* | The model identifier exactly as the serving layer exposes it. |
| `MODEL_API_KEY` | *unset* | Sent as `Authorization: Bearer`. A `SecretStr`. Blank sends no header at all. |
| `MODEL_TIMEOUT_SECONDS` | `20` | Total budget for one model call, shared across every attempt and the waits between them. |
| `MODEL_CONNECT_TIMEOUT_SECONDS` | *unset* | Connection-establishment budget; falls back to `MODEL_TIMEOUT_SECONDS`. |
| `MODEL_MAX_OUTPUT_TOKENS` | *unset* | Cap on generated tokens; unset leaves `LlmRequest`'s own default (512). |
| `MODEL_TEMPERATURE` | *unset* | Sampling temperature; unset leaves `LlmRequest`'s own default (0.2). |
| `MODEL_MAX_RETRIES` | `0` | Bounded retries, off by default. Max 3. |
| `MODEL_PROVIDER` | `openai-compatible` | Log label only; never affects request construction. |

### The default cannot make a network request

With both `MODEL_BASE_URL` and `MODEL_NAME` unset — the shipped default and what
`tests/conftest.py` builds — `build_llm_service` returns `NotConfiguredLlmService`
and **no HTTP client object is ever constructed**. That is the control, not a
convention: an unconfigured process has nothing to make a request with.

Setting exactly one of the two raises `LlmConfigurationError` while the
application is being built, so the process fails to start rather than failing on a
customer's call. Nothing falls back to another model, another URL or a stub. A
blank `MODEL_BASE_URL=` in a shipped `.env` template reads as unset rather than as
a half-configuration.

## S3A.6 HTTP library

No dependency was added. `httpx2` (2.13.1) was already in `requirements.txt` — it
is the HTTP client this Starlette version expects for `TestClient` — and it
provides everything the adapter needs: per-phase timeouts (`connect`/`read`),
typed transport exceptions, and `MockTransport` for a pluggable test transport. It
moved out of the "test-only" section of `requirements.txt` and is now documented
as a runtime dependency. No provider SDK is installed: the chat-completions
dialect is spoken directly, which is what keeps the deployment choice out of the
code.

## S3A.7 Request / response mapping

**Request.** `LlmRequest` → `POST {base}/chat/completions`:

```json
{ "model": "<MODEL_NAME>",
  "messages": [{"role": "system", "content": "..."},
               {"role": "user", "content": "..."},
               {"role": "tool", "content": "...", "tool_call_id": "..."}],
  "max_tokens": 512, "temperature": 0.2, "stream": false,
  "tools": [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {…}}}] }
```

`tool_call_id` appears only on messages that carry one. `tools` is omitted
entirely when none are offered. `parameters` is the registry's own
`model_json_schema()` passed through **unaltered** — rewriting it would offer the
model a looser contract than the registry enforces. `stream` is always `false`;
streaming is out of scope for this stage.

Prompt construction remains entirely `app/orchestrator/prompt.py`'s. The adapter
appends nothing, edits nothing and authors nothing;
`test_the_prompt_the_orchestrator_built_is_what_goes_on_the_wire` pins that, and
`test_the_model_is_still_never_told_an_account_or_customer_reference` re-checks
the Stage 2 invariant at the byte level now that a real serialiser is involved.

`max_tokens` and `temperature` come from configuration **unless** the caller set
them explicitly on the `LlmRequest` (detected via pydantic's `model_fields_set`,
which distinguishes "the caller chose 0.2" from "0.2 is the field default"). The
request wins when it spoke.

**Response.** `choices[0].message` → `LlmGeneration`:

| Wire | Internal |
| --- | --- |
| `message.content` | `text` (whitespace-only becomes `None`) |
| `message.tool_calls[]` | `tool_calls: tuple[LlmToolCall, ...]` |
| `choices[0].finish_reason` | `finish_reason` |
| `model` | `model`, falling back to the configured name |
| `usage` | `usage: LlmUsage \| None` |
| measured | `latency_ms` |

Parsing is strict. Every shape the adapter does not recognise raises rather than
being coerced into something plausible: a body the application had to guess at is
a body it cannot claim the model produced.

## S3A.8 Tool-call handling

`message.tool_calls[].function.arguments` is a JSON **string** on the wire.
`_parse_arguments` decodes it and requires a JSON object. It also accepts an
already-decoded object, because several OpenAI-compatible servers emit one; that
is tolerance for two documented wire forms, not a guess about intent.

Everything else raises `LlmInvalidToolCall`: a broken JSON string, valid JSON that
is not an object (`[1,2,3]`, `"ACC-1"`, `42`), a missing or empty tool name, a
`tool_calls` value that is not an array, a call with no `function` object, or an
`id` of the wrong type. **No tool call is ever repaired** — a repaired tool call
is one the application invented, and the registry would then validate arguments no
model actually asked for.

A missing `id` is the one tolerated omission: it becomes `""`, and the
orchestrator's existing `request_id = call.call_id or uuid4().hex` mints one.

The adapter does not execute, dispatch, queue or forward tool calls. It returns
them. Stage 2's controls are untouched and still run afterwards: identity binding
overwrites `account_ref`/`customer_ref` from the session, write tools require
corroborating conversation state, a write must be alone in its round, and
`MAX_TOOL_CALLS_PER_TURN` bounds the loop.
`test_the_model_still_cannot_choose_which_account_a_tool_reads_over_http` runs a
model that names `ACC-VICTIM` over HTTP and shows this session's own account is
what gets read.

## S3A.9 Timeout, error and retry behaviour

Every failure is a typed subclass of `LlmError` with a stable `category` string.
Those strings are deliberately equal to the corresponding `TurnErrorCategory`
values, so `pipeline.py` maps a failure by lookup rather than by a table that
could drift out of date.

| Condition | Exception | Turn error category |
| --- | --- | --- |
| Nothing configured | `LlmNotConfigured` | `llm_not_configured` |
| Half-configured | `LlmConfigurationError` | raised at startup, not at turn time |
| Read/connect/pool timeout | `LlmTimeout` | `llm_timeout` |
| Connect, DNS, TLS, proxy, protocol error | `LlmConnectionFailed` | `llm_connection_failed` |
| Any status ≥ 400 | `LlmUpstreamError` (status only) | `llm_upstream_error` |
| Body not JSON, or wrong shape | `LlmMalformedResponse` | `llm_malformed_response` |
| No text and no tool call | `LlmEmptyResponse` | `llm_empty_response` |
| Unreadable tool call | `LlmInvalidToolCall` | `llm_invalid_tool_call` |

An empty model response is a failure, not an empty turn: an agent that says
nothing on a live call is a defect, and an empty draft must not be handed to
validation as though the model had chosen silence.

**Retries are bounded and off by default.** A retry on a live call is spent out of
the customer's silence, so the default is to fail fast and end the turn
(`MODEL_MAX_RETRIES=0`). When enabled, only clearly transient failures are
retried: a connection that never opened (`ConnectError`, `ProxyError`, connect
or pool timeout), and 408/429/500/502/503/504. A read error, a write error, a
protocol error, a 400/401/403/404/409/422, a malformed body and a **read**
timeout are not retried. The constructor rejects a non-positive timeout and a
`max_retries` outside 0..3, the same cap as configuration.

**One deadline, shared.** `MODEL_TIMEOUT_SECONDS` is the budget for the whole
call. Each attempt's HTTP timeout is the time still left, and it is set on the
request itself, so an injected client cannot disable it. Before another attempt
the adapter waits: `Retry-After` when the server sent a positive delay,
otherwise a short exponential backoff (50 ms, 100 ms, 200 ms, capped at 1 s).
If that wait does not fit in the time still left, the call fails with the error
already in hand. It does not start a fresh 20 s clock per attempt.

**Latency** is measured at the boundary with `time.perf_counter()` around the
whole call, carried on `LlmGeneration.latency_ms`, logged as `model_latency_ms`,
and accumulated by the orchestrator into `TurnLatency.llm_ms`. Observability only
— nothing here is optimised, and no streaming was added.

## S3A.10 Security and logging

The adapter emits one `llm_call` record per attempt through the existing
`app.observability.log_event`. No second logging system was introduced.

Logged: `provider`, `model`, `endpoint` (scheme + host + port only), `request_id`
(a per-call correlation id), `attempt`, `outcome` (`ok` / `retrying` / `failed`),
`http_status`, `error_type`, `error_category`, `model_latency_ms`,
`finish_reason`, `tool_call_count`, `usage_prompt`, `usage_completion`,
`usage_total`.

Never logged: the API key, the `Authorization` header, the prompt, the FACTS
block, the draft, the transcript, the response body, the request path, any
customer name, account reference or amount.

Three details worth stating explicitly:

1. **Credentials in the base URL.** A base URL may legitimately carry userinfo
   (`https://user:pass@host/v1`). `_endpoint_label` keeps only scheme, host and
   port, so the password never reaches a log line.
2. **`usage_*` rather than `*_tokens`.** `redact()` blanks any field whose name
   contains `token` — correct for credentials, and it would otherwise blank the
   token counts too. The names avoid the fragment rather than weakening the
   redaction list.
3. **`QUIET_LOGGERS`.** This was found by a test, not by inspection. `httpx2` logs
   one INFO line per request containing the **full** request URL — including any
   embedded credentials. That line is the library's own message string, so
   `redact()` never sees it: redaction operates on structured fields and a
   formatted message has none. Level is the only control that works, so
   `configure_logging` now holds `httpx`/`httpx2`/`httpcore`/`httpcore2` at
   WARNING. The application logs its own sanitised record for every model call
   regardless.

Upstream response bodies are discarded at the boundary. `LlmUpstreamError` retains
the status code and nothing else; there are tests asserting that an API key or an
organisation name embedded in a 401 body appears in neither the exception nor the
log.

## S3A.11 Test coverage

Every test is offline. Most drive the adapter through `httpx2.MockTransport` in
this process; a handful use `FakeOpenAiServer`, a real `http.server` bound to
`127.0.0.1` on an ephemeral port, for the properties a transport stub cannot
demonstrate — that the configured URL is the one dialled, that the
`Authorization` header leaves the process, and that a slow server really does trip
the configured timeout.

### The cases Stage 3A required

| Required case | Test |
| --- | --- |
| 1. Normal text response | `test_a_normal_assistant_reply_becomes_a_generation` |
| 2. Tool-call response | `test_a_tool_call_response_is_parsed_into_the_projects_tool_call_type` |
| 3. Multiple tool calls | `test_multiple_tool_calls_are_all_preserved_in_order` |
| 4. Malformed JSON | `test_a_body_that_is_not_json_is_a_malformed_response`, `test_over_a_real_socket_a_non_json_body_is_malformed` |
| 5. Missing response fields | `test_a_body_missing_expected_fields_is_a_malformed_response` (8 shapes) |
| 6. HTTP 400 | `test_every_http_error_status_becomes_a_typed_upstream_error[400]` |
| 7. HTTP 401 / 403 | same, `[401]` and `[403]` |
| 8. HTTP 404 | same, `[404]` |
| 9. HTTP 429 | same, `[429]` |
| 10. HTTP 500 / 502 / 503 | same, `[500]`, `[502]`, `[503]`, `[504]`; plus `test_over_a_real_socket_a_500_is_an_upstream_error` |
| 11. Timeout | `test_a_timeout_becomes_an_llm_timeout` (3 kinds), `test_a_slow_server_trips_the_configured_timeout` (real socket) |
| 12. Connection failure | `test_a_transport_failure_becomes_a_connection_failure`, `test_a_real_closed_port_is_a_connection_failure` |
| 13. Empty model response | `test_a_model_that_says_nothing_is_a_failure_not_an_empty_turn` (`None`, `""`, whitespace) |
| 14. API key sent, never logged | `test_the_configured_api_key_is_sent_as_a_bearer_token`, `test_the_api_key_never_appears_in_a_log_line`, `test_credentials_embedded_in_the_base_url_are_not_logged`, `test_the_repr_does_not_carry_the_api_key` |
| 15. Model name sent | `test_the_configured_model_name_is_what_is_sent`, `test_over_a_real_socket_the_url_key_and_model_all_arrive` |
| 16. Base URL respected | `test_the_configured_base_url_is_what_is_dialled`, `test_a_trailing_slash_on_the_base_url_does_not_double_up` |
| 17. Request timeout respected | `test_the_configured_timeout_is_applied_to_the_client`, `test_the_connect_timeout_falls_back_to_the_request_timeout`, `test_a_slow_server_trips_the_configured_timeout` |

### The orchestrator integration cases

| Required case | Test |
| --- | --- |
| Orchestrator → LlmService → mock endpoint → orchestrator continues | `test_a_turn_completes_with_the_model_answering_over_http`, `test_a_full_turn_over_a_real_loopback_server` |
| Tool call → orchestrator → ToolRegistry executes | `test_a_tool_call_arriving_over_http_is_executed_through_the_registry` |
| Policy checkpoints remain active | `test_the_policy_checkpoints_are_all_still_evaluated_around_an_http_tool_call` |
| The adapter never executes the tool | `test_the_adapter_never_executes_a_tool`, `test_the_http_adapter_is_not_what_reached_the_banking_backend` |

### Beyond the required list

Identity binding still defeats a model that names another account over HTTP; a
policy-blocked turn makes **no HTTP request at all**; an ungrounded figure arriving
over HTTP is still blocked before TTS; tool results are fed back as outcomes and
never as payloads (no paise, no `account_ref`); every boundary failure maps to its
own turn error category with `speakable=False`; upstream error detail reaches the
operator but never the customer; retries are bounded, never applied to 4xx, never
to read timeouts, never to malformed bodies, and each attempt is logged so a
degrading endpoint is visible; configuration reaches the wire intact; a
half-configuration fails loudly; and an injected HTTP client is not closed by the
service that did not create it.

## S3A.12 Verification performed

- Baseline before any Stage 3A change: `.venv/bin/python -m pytest -q` → **321 passed**.
- After Stage 3A: `.venv/bin/python -m pytest -q` → **458 passed** (137 new; 99 adapter, 38 integration).
- All 321 Stage 1 and Stage 2 tests pass **unchanged**. No existing test was modified.

## S3A.13 Known limitations

1. **No real model has ever answered.** Every response this code has parsed was
   written by hand in `tests/fakes.py`. A real server's quirks — how vLLM emits a
   Gemma tool call, whether it returns `usage`, what it does on a truncated
   generation — are unverified. The first real connection is expected to find
   something.
2. **No streaming.** `stream` is always `false`. A voice agent ultimately wants
   token streaming to cut time-to-first-audio; that is a later stage and would
   change the `LlmService` contract, not just the adapter.
3. **Synchronous.** `generate` blocks. Under FastAPI a sync path runs in a
   threadpool, which is adequate here but is not the shape a high-concurrency
   telephony deployment will want.
4. **One connection pool, no health checking.** There is no circuit breaker, no
   endpoint failover and no warm-up. A degrading endpoint is visible in the logs
   and fails turns; nothing reacts to it automatically.
5. **httpx phase timeouts are capped, not a single socket timer.** The shared
   deadline caps each phase (connect, read, write, pool) at the time still left
   and refuses another attempt once that time cannot hold the backoff. A single
   attempt can still spend up to one phase-timeout per phase. The default of
   zero retries means a normal call makes one attempt inside that budget.
6. **`finish_reason` is recorded, not acted on.** A generation truncated by
   `max_tokens` arrives as ordinary text with `finish_reason="length"`. Validation
   and grounding still apply, so nothing unbacked can be spoken, but the turn is
   not retried or flagged.
7. **No authentication scheme but bearer tokens.** mTLS, cloud IAM signing and
   header-based API keys are not supported.
8. **Tool-call streaming deltas are not handled**, since streaming is off. A
   server that only emits tool calls in streaming mode would not work.
9. Everything listed under Stage 1 and Stage 2 limitations still stands: no
   telephony, no STT, no TTS synthesis, no banking backend, no authentication, no
   persistence, no evaluation harness.

**No claim is made that this is production ready, and no claim is made that Gemma
works against a real model.** What Stage 3A delivers is a tested, typed, offline
boundary that a remote OpenAI-compatible Gemma server can be pointed at by
configuration alone.
