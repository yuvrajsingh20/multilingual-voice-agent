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

---

# Stage 3A — Optional Decision Layer

> **Jev is an optional semantic decision provider. It is not the regulatory
> policy authority, authentication layer, banking source of truth, or tool
> authorization layer.**

**NO LIVE JEV CALL HAS BEEN MADE.** No TypeSafe key was available, and nothing
was sent to TypeSafe or to any gateway. Every Jev answer this code has parsed
was written by hand. The official SDK ran for real in the tests, against an
in-process transport. Jev's accuracy on this domain, and its latency from this
deployment, are **unmeasured**. No GCP resource, GPU, vLLM, Gemma or other model
infrastructure was touched.

**Disabled by default.** A fresh clone behaves exactly as before: the SDK is
never imported, no client is built, and no request can leave the process.

## SD.1 Why the layer exists

Some decisions in a collections call are small, closed and semantic. Is "haan ji
bilkul" said over the agent an acknowledgement or a bid for the floor? Is "I
already paid this yesterday" a payment claim? Does the customer want a human?
Deterministic rules handle these badly: the barge-in heuristic misses any
backchannel outside its vocabulary. A general LLM is the wrong tool too. It is
slow, costly and generative for what is a pick-one-label question.

Jev (TypeSafe's "System One" model) answers exactly that shape: named labels in,
one label plus a calibrated confidence out, and no generated text. This stage
adds it **behind a provider-neutral boundary**, for **three registered
decisions**, with an **explicit fallback** for every way it can fail or be
unsure. Whether it is worth using is a question for measurement (SD.9). This
stage builds the means to answer it; it does not claim the answer.

## SD.2 Where it sits

```
Regulatory rules ─► Policy engine ─► Conversation state ─► Orchestrator (sync turn pipeline, UNCHANGED)
                                                                ▲
                                   upstream signals (EventSignals) │  ← an applied low-risk intent
                                                                │     enters here, as any NLU signal does
Audio ─► VAD / acoustic ─► STT ─► DecisionCoordinator (async)  ─┘      app/orchestrator/decisions.py
                                    │  routing: is a decision needed at all?
                                    │  deadline · threshold · fallback · one log record per decision
                                    ▼
                               DecisionService (Protocol)       app/services/decision.py — contract, no I/O
                                    │
                                    ▼
                               JevDecisionService               app/services/decision_jev.py — the only SDK importer
                                    │
                                    ▼
                               POST {JEV_BASE_URL}/v1/systemone   (typesafe-sdk==0.7.1, AsyncTypeSafeClient)
                                    │
                                    ▼
                               fusion (pure)                    app/services/decisions/fusion.py
                                    │  + heuristic result, hard signals, policy, state
                                    ▼
                    CONTINUE_TTS / STOP_TTS / WAIT · APPLY_SIGNAL / CONFIRM_FIRST / EXISTING_PATH ·
                    ESCALATE / RECOMMEND_REVIEW / NO_CHANGE
```

The coordinator depends on `DecisionService` and never on Jev. Four structural
tests pin the boundary:

- only `decision_jev.py` imports `typesafe_sdk`;
- within `app/`, only `runtime.py` imports `decision_jev.py`. The offline
  measurement script also imports it, for `--mock`;
- no decision module directly imports `app.tools`, `app.services.llm`,
  `app.core.policy` or `app.core.checks`;
- `pipeline.py` imports nothing decision-related.

Direct imports are not the whole story. Python lets any module import anything,
and `app.orchestrator`'s package `__init__` loads the pipeline anyway. What
matters is what the coordinator *holds*. It is built with
`DecisionCoordinator.from_runtime(runtime)`, which hands it exactly three
things: the decision service, the settings and the barge-in classifier. It never
holds the `Runtime`, the tool registry, the LLM, the policy engine or the
session store. A test checks the built instance for all six, and the module
imports `Runtime` only under `TYPE_CHECKING`.

### The architectural conflict, reported rather than forced

`ConversationOrchestrator.process_turn` **does not call the decision layer**, and
this is deliberate. Wiring it in would have been the "forced" integration the
task warned against, for three reasons found in the code:

1. **The turn pipeline is synchronous.** So are `LlmService.generate` and the
   FastAPI routes. The decision service is async, as the task required, because
   its real caller is the live audio loop. Bridging async into `process_turn`
   would need a per-call event loop, which defeats connection pooling, or a tie
   to FastAPI's threadpool.
2. **No response path in this codebase avoids the LLM.** Every allowed turn
   calls the model. A Jev call inside the turn would add its latency to every
   turn and remove no model call. That is exactly the "Jev → LLM on every
   utterance" pattern the task ruled out.
3. **Intent here is an upstream signal that gates writes.** The reducer turns
   `PAYMENT_PROMISE`, `DISPUTE` and `ESCALATION_REQUEST` into flags that unlock
   `record_payment_promise`, `create_dispute` and `escalate_case`, and
   `WRONG_PERSON` clears identity verification. The only intents that are safe
   to apply without confirmation (`REFUSAL`, `CALLBACK_REQUEST`) change nothing
   that policy reads. Inside the turn, Jev would either be unsafe or inert.

So the layer is integrated at the seams that exist:

- the `Runtime` composition root (`runtime.decisions`, default disabled);
- the app lifespan, which closes its connection pool;
- an async `DecisionCoordinator` for the future audio loop;
- an applied low-risk intent, which reaches `process_turn` through the
  existing `signals=` parameter exactly as upstream NLU does. Tests exercise
  this path end to end.

Putting a decision *inside* the turn needs an async turn path and a non-LLM
response path. Both are architecture changes, and both are out of scope.

## SD.3 Decisions delegated to Jev

Exactly three, fixed in `app/services/decisions/registry.py`. `DecisionName` is
a closed enum, and `DecisionRequest.name` is `strict`, so even the string
`"barge_in"` is refused. The registry is a read-only mapping, validated at
import. Each entry fixes the labels, the literal question wording, the context
fields the decision may see, and its threshold setting. Each is put to Jev as one
`Choice` question.

| Decision | Labels | Context sent | Routing: provider **not** called when |
| --- | --- | --- | --- |
| `barge_in` | `backchannel`, `interruption`, `continuation` | utterance, language, agent speaking, ≤3 earlier customer utterances | agent silent · heuristic says NOISE (acoustic gate) · no transcript · speech > 800 ms (hard signal) · an explicit stop phrase (hard signal) · less than 50 ms of the 800 ms window left |
| `customer_intent` | the ten specified: `payment_promise`, `payment_already_made`, `dispute`, `request_information`, `refusal`, `financial_difficulty`, `wrong_person`, `callback_request`, `escalation_request`, `unclear` | utterance, language, stage | upstream NLU already supplied an intent · utterance empty or > 1000 chars |
| `needs_human_escalation` | `yes`, `no` | utterance, language, ≤3 earlier customer utterances | policy or session state already requires escalation (authoritative) |

`UNCERTAIN` is never offered to the provider as an option. It is what the
coordinator reports when there is no confident answer.

**Fusion** (`fusion.py`, pure functions) decides what each answer may change:

- **Barge-in.** The existing `HeuristicBargeInClassifier` always runs first and
  is the fallback. A decided `backchannel` → `CONTINUE_TTS`; `interruption` or
  `continuation` → `STOP_TTS`. The heuristic's own `UNKNOWN` → `WAIT`. The
  result carries a `BargeInDecision`, so it drives the existing
  `TurnStateMachine` unchanged. Jev can never override the acoustic gate or a
  hard signal.

  **The wait counts against the window.** The agent keeps speaking while the
  provider is consulted. So the provider gets only what is left of the 800 ms
  window after the speech already heard, and never more than
  `JEV_TIMEOUT_SECONDS`. A signal at 700 ms gets at most 100 ms. Speech so far
  plus the wait therefore never exceeds the window. A later answer is discarded
  as a timeout, and the heuristic's decision applies.

  The caller contract is in the docstring: keep feeding partials while a call
  is pending, and act on the newest resolution. Any `STOP_TTS` stops the agent.
  Before the adversarial review (SD.14) the wait was *added* to the window: up
  to 800 ms plus the full deadline.
- **Intent**, tiered against the real reducer:

  | Tier | Labels | Result |
  | --- | --- | --- |
  | Low risk | `refusal`, `callback_request` | `APPLY_SIGNAL`: may be passed to the reducer. Sets `ConversationState.intent` and nothing else. |
  | High risk | `payment_promise`, `dispute`, `escalation_request`, `wrong_person` | `CONFIRM_FIRST`: returned with the candidate `Intent`, never applied. |
  | No application meaning | `payment_already_made`, `request_information`, `financial_difficulty`, `unclear` | `EXISTING_PATH`: reported for the caller. |

  Tests apply each intent through `apply_event` and assert the tiering matches
  what the reducer really changes. A reducer change that raised the stakes of a
  "low-risk" intent would fail CI.
- **Escalation.** A policy or state escalation → `ESCALATE` (authoritative),
  whatever Jev says. A decided `yes` → `RECOMMEND_REVIEW`
  (`authoritative=False`); the application's escalation path must confirm it.
  Anything else → `NO_CHANGE`.

## SD.4 Decisions that remain deterministic

Not registered, never sent to Jev, not duplicated:

- **Policy rules**: calling hours, persistent calling, prohibited conduct,
  recording disclosure, agent identification, grievance-pending hold, dispute
  handling, agency details. All stay in the policy engine.
- **Identity and access**: identity verification and account/customer binding.
  `_bind` in the orchestrator is untouched, and `DecisionContext` has no
  identity field.
- **Tools**: tool authorisation, argument validation and write preconditions.
- **Audio**: VAD, the acoustic gate and end-of-turn detection
  (`SilenceTurnDetector` is untouched; Jev is never a turn detector).
- **Hard interruption signals**: speech longer than 800 ms, and an explicit
  request to stop. `HALT_PHRASES` covers English, romanised Hindi/Hinglish and
  Marathi, and Devanagari: "stop", "wait", "hang on", "ruk" (which covers "ruk
  jao"), "ruko", "thehro", "ek minute", "bas", "thamb", "रुक", "ठहरो", "थांब",
  "एक सेकंड", "एक मिनिट" and others. Matching is on whole words after NFC
  normalisation, with punctuation, symbols and zero-width joiners removed. The
  list was added after test design showed that a confident wrong `backchannel`
  on "stop" would otherwise keep the agent talking over a customer who asked it
  to stop. The review then found common forms it missed, and they were added
  (SD.14). The list only removes decisions from Jev; it never changes what the
  heuristic does.

## SD.5 Decisions that remain with the LLM

All language:

- every response the agent speaks;
- open-ended questions ("Can you explain why I received this notice?");
- anything labelled `request_information`, `payment_already_made`,
  `financial_difficulty` or `unclear`. These need words, or the system of
  record, not a label.

The decision layer never calls the LLM and never generates text. It does not
decide whether the LLM is called. In the current pipeline every allowed turn
still calls it, as before.

## SD.6 Fallback behaviour

Every row ends in the pre-existing behaviour, and each is tested
(`tests/test_decision_layer.py`, `tests/test_decision_jev_adapter.py`).

| Condition | Recorded as | Barge-in | Intent | Escalation |
| --- | --- | --- | --- | --- |
| Provider disabled (default) | no call, no log record | heuristic, exactly | `EXISTING_PATH` | policy/state only |
| Timeout (adapter `wait_for` and coordinator deadline) | `decision_timeout` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| Connection failure | `decision_connection_failed` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| 401 / 403 | `decision_authentication_failed` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| 429 | `decision_rate_limited` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| 400 / 404 / 422 | `decision_request_rejected` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| Other HTTP error | `decision_upstream_error` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| Malformed answer (see below) | `decision_malformed_response` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| Confidence below threshold | `UNCERTAIN` / `low_confidence` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| Provider raises anything else | `decision_failed` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| Provider *returns* something that is not a valid answer (None, a dict, or a `DecisionAnswer` that fails strict re-validation) | `decision_malformed_response` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| Answer measured after the deadline, because the event loop stalled while it was pending (the provider blocked it, or other work did) | `decision_timeout`, answer discarded; the record carries `deadline_ms` | heuristic | `EXISTING_PATH` | `NO_CHANGE` |
| Misconfiguration (switches disagree, key/model missing, SDK absent, bad key) | startup error | process does not start | | |

Transport errors are exceptions, classified by type and status. Uncertainty is
not an error: it is an answer below its threshold, reported as `UNCERTAIN` with
the provider's leaning kept in `top_label` for evaluation only.

**Malformed** covers what the SDK lets through, and all of it was verified
against 0.7.1:

- a label that was not offered;
- a NaN, negative or >1 confidence;
- probabilities for unknown labels, or outside [0, 1];
- a missing answer, or an answer of the wrong type;
- a body that is not JSON.

The SDK validates only shape. The coordinator re-checks the label against the
registry whatever the provider claims.

**Retries** are off by default (`JEV_MAX_RETRIES=0`, max 2). When on, two
kinds of failure are retried, inside the same deadline:

- a connection that never opened (`httpx2.ConnectError`), where nothing was
  sent;
- 408/429/500/502/503/504, where the server received the request and reported
  a transient failure. A retry here **sends the customer's words again**, which
  is one reason retries are off by default.

A read, write or protocol error is not retried, because whether the request
was delivered is unknown. A timeout is not retried either: it has already
spent the budget. The SDK's own default (2
retries, 30 s budget, every 5xx and every transport error) is overridden. Tests
pin both sides. The first version retried every transport error, contradicting
this paragraph; the review caught it (SD.14).

## SD.7 Confidence thresholds

| Setting | Default | Gates | Why this default |
| --- | --- | --- | --- |
| `JEV_BACKCHANNEL_THRESHOLD` | 0.85 | keeping the agent talking, or stopping it, on a semantic call | Wrongly continuing means talking over the customer. |
| `JEV_INTENT_THRESHOLD` | 0.80 | classifying intent | High-risk intents are confirm-first anyway, so a wrong label cannot act alone. |
| `JEV_ESCALATION_THRESHOLD` | 0.90 | recommending a human | The highest-stakes recommendation. |

**These are provisional, chosen conservatively, not derived from data.** Jev's
own docs say thresholds depend on domain and model version. They mean nothing
for an alias like `jev-latest`, because the answering model can change under it.
Every decision records the model that answered.

To choose real thresholds, label real calls in the shape of
`data/eval/decision_cases.json` and run `python scripts/eval_decisions.py --live`.
It prints, per decision, coverage and accuracy at 0.50–0.95, failure counts, and
latency percentiles. The percentiles are truncated at the deadline, so raise
`JEV_TIMEOUT_SECONDS` for the measurement run.

The shipped 44 cases are **synthetic** and hand-written, as a template and smoke
set, not a benchmark. Every label appears at least once, and every decision has
cases in each of the four languages, but most labels have only one or two
cases. The set is mostly English (25 English, 10 Hinglish, 5 Marathi, 4
Hindi).

## SD.8 Security boundaries

Each item below is pinned by tests in `tests/test_decision_safety.py` and
`tests/test_decision_boundary.py`, which give the provider every possible answer
at 0.999 confidence.

- **Identity.** Session customer A. The utterance and the model both name
  account B. Whatever the intent, and even if the application confirms and
  applies a high-risk one, the backend is asked only about A (20 cases).
  `DecisionContext` has no structured field for an account ref, customer ref,
  name, phone, amount, DPD or identity flag, and rejects one if given.
  (Free-text identifiers are covered under "Data leaving the process".) No
  decision can produce `IDENTITY_CONFIRMED`.
- **Tools.** No decision module imports the tool registry, the LLM service or
  the policy engine. The coordinator is handed none of them (SD.2). With every
  answer, the backend is touched zero times and the LLM is called zero times.
- **Policy.** With every intent answer, a turn outside calling hours is still
  `POLICY_BLOCKED`, and the LLM is not called. The policy decision is identical
  before and after every decision. A policy escalation stands against a
  confident `no`.
- **State.** The coordinator writes nothing: session state and event count are
  unchanged after every decision.
- **Writes.** A confident `payment_promise`, `dispute` or `escalation_request`
  cannot unlock `record_payment_promise`, `create_dispute` or `escalate_case`.
  The orchestrator refuses the write, and the backend records nothing.
- **Secrets.** `JEV_API_KEY` is a `SecretStr`. The SDK validates its format at
  construction; the adapter reports a bad key without echoing it. It is passed
  explicitly, so the SDK's `TYPESAFE_*` environment variables are never read.
- **Provider text never crosses the boundary.** SDK exceptions carry the
  response body, and their `str` embeds the provider's message, which can echo
  the customer's words. They are classified by type and status and dropped from
  the chain (`__cause__` and `__context__` are both cleared). The SDK response
  object, which holds the raw HTTP body, never leaves the adapter. A reported
  model name is recorded only if it is a short identifier containing a letter
  (full match). A bare number, or anything with spaces or a newline, is
  dropped.
- **Logs.** The SDK logs request URLs at INFO and full bodies at DEBUG, and
  `TYPESAFE_LOG_LEVEL=debug` turns that on at import. Its one WARNING line
  ("Ignoring answer %r with unrecognized type %r") quotes provider-controlled
  text. So the adapter drops **every** SDK record with a logging filter, which
  survives `configure_logging` being re-run, and `typesafe_sdk` is also in
  `QUIET_LOGGERS`. The first version only held the level at WARNING, and the
  review showed that line leaking echoed customer text (SD.14). The
  coordinator logs one record per decision: name, provider, model, outcome,
  label category, confidence bucket, threshold, latency, `fallback_used`,
  fallback reason and session id. It never logs text. Tests capture every log
  line at DEBUG and assert that no utterance, PAN, phone number or key appears.
- **Data leaving the process.** When enabled, the customer's utterance, and for
  some decisions up to three earlier utterances, the language and the stage,
  are sent to TypeSafe or the configured gateway. That is personal data
  transferred to a third-party processor. The DPDP Act 2023 and the RBI
  outsourcing directions in this repo's corpus apply, and **no data-processing
  assessment has been done**. Two measures limit what is sent:
  - the registry's per-decision field allowlist, which sends no stage for
    barge-in and no earlier utterances for intent;
  - `mask_identifiers`, which runs on free text before anything leaves the
    process.
    - **Masked:** PANs (`<id>`), email addresses (`<email>`), and runs of 8 or
      more digits with at most two separator characters (space, comma, period
      or hyphen) between neighbours (`<number>`). That covers phone, Aadhaar,
      account and card numbers, including ones read out digit by digit.
    - **Kept:** shorter numbers, which covers amounts under one crore, and a
      run that is exactly a *valid* d-m-yyyy or yyyy-mm-dd date ("12 10 2026",
      "2026-10-12").
    - **Over-masked, in the safe direction:** compact dates ("12102026"),
      amounts of 8 or more digits, and neighbouring numbers that together reach
      8 digits ("5000 5000").

  Masking is best-effort. It misses names, numbers spoken as words, digits
  separated by anything else or by wider gaps, identifiers shorter than 8
  digits, and an identifier that happens to form a valid date in 2-2-4
  grouping. On a 1000-character adversarial input it runs in under 0.1 ms. Utterances from the identity-verification stage are not
  specially excluded; the caller should not send them. Adversarial content is a
  documented Jev
  weakness ("jaggedness", item 6). That is one more reason it decides nothing
  safety-relevant.

## SD.9 Latency measurements

**Measured.** Local overhead only, no network. Intel i3-1115G4, 4 CPUs, Python
3.10.12, 2,000 sequential calls per cell, logging disabled. Re-measured after
both rounds of review fixes (masking, the halt-phrase normaliser, strict
re-validation).

| Path | barge_in p50 / p95 | customer_intent p50 / p95 | escalation p50 / p95 |
| --- | --- | --- | --- |
| Coordinator, provider disabled (heuristic / policy only) | 0.022 / 0.027 ms | 0.005 / 0.008 ms | 0.007 / 0.010 ms |
| Coordinator + fusion, instant stub provider | 0.068 / 0.092 ms | 0.046 / 0.075 ms | 0.048 / 0.070 ms |
| Coordinator + real Jev adapter + real SDK, instant in-process transport | 0.392 / 0.569 ms | 0.356 / 0.482 ms | 0.352 / 0.454 ms |

One isolated outlier of about 15 ms appeared in the last row's
`customer_intent` max, in all three runs. It is consistent with a garbage-collection
pause, but was not investigated. The layer costs about 0.35–0.4 ms before the
network. `scripts/eval_decisions.py
--mock` reproduces the last row through the adapter alone.

**Not measured:**

- Jev's real round-trip from this deployment;
- the gateway overhead;
- behaviour under TypeSafe's rate limits (1,200 req/min currently, "adjusting
  dynamically");
- any LLM, since none is connected.

TypeSafe's launch post says "End-to-end response time is 70ms-500ms"
(typesafe.ai/blog/introducing-system-one-models-and-jev). Its docs say "most
queries complete in about 100 ms". Both figures are the vendor's and have not
been verified here. `JEV_TIMEOUT_SECONDS=0.5` is set at that upper bound and
is also unmeasured.

**Does Jev reduce end-to-end latency? Not shown, and on the evidence here, not
for these decisions today.**

- **Barge-in.** The existing heuristic decides in about 0.02 ms. Any network
  call is orders of magnitude slower. Jev can only be justified here by
  *accuracy*: it catches backchannels the vocabulary misses, and interruptions
  hidden in backchannel words. That accuracy is unevaluated.
- **Intent and escalation.** This pipeline never asked an LLM for them, so no
  LLM call is removed. Jev replaces upstream NLU, which is outside this repo.
  There is no baseline to compare against.

## SD.10 Files

**Created**

| File | What it is |
| --- | --- |
| `app/services/decision.py` | Provider-neutral contract: `DecisionName`, `DecisionContext`, `DecisionRequest`, `DecisionAnswer`, `DecisionResult`, error taxonomy, `DecisionService` protocol, `DisabledDecisionService`, `ScriptedDecisionService`. No I/O. |
| `app/services/decisions/__init__.py` | Package exports. |
| `app/services/decisions/registry.py` | The closed registry: labels, question wording, context allowlist, threshold keys; `build_state` and `mask_identifiers`; import-time consistency check. |
| `app/services/decisions/fusion.py` | Pure fusion: barge-in, intent tiers, escalation; hard signals; `HALT_PHRASES` and its normaliser. |
| `app/services/decision_jev.py` | The Jev adapter; the only `typesafe_sdk` importer. |
| `app/orchestrator/decisions.py` | `DecisionCoordinator` (built with `from_runtime`): routing, deadline and barge-in window budget, threshold, fallback, measurement, fusion. |
| `app/evaluation/decisions.py` | Provider-agnostic evaluation: threshold sweep, failures, latency percentiles. |
| `scripts/eval_decisions.py` | CLI: `--mock` (offline overhead) or `--live` (configured provider). |
| `data/eval/decision_cases.json` | 44 synthetic labelled cases: every label, and every decision in all four languages. |
| `tests/test_decision_boundary.py` | 86 tests: defaults, config, isolation, registry, contract types. |
| `tests/test_decision_jev_adapter.py` | 98 tests: real SDK over a mock transport, identifier masking, and event-loop lag through the real adapter. |
| `tests/test_decision_layer.py` | 163 tests: routing, fusion, fallback, thresholds, deadline, window budget, metrics, concurrency, event-loop lag. |
| `tests/test_decision_safety.py` | 65 tests: identity, tools, policy, state, writes. |
| `tests/test_decision_evaluation.py` | 16 tests: harness arithmetic, case coverage, the script (including masked cases), lifespan close. |

**Modified**

| File | Change |
| --- | --- |
| `app/config.py` | Ten settings: `DECISION_PROVIDER`, `JEV_ENABLED`, `JEV_API_KEY`, `JEV_MODEL`, `JEV_BASE_URL`, `JEV_TIMEOUT_SECONDS`, `JEV_MAX_RETRIES` and three thresholds. Validators: blank-is-unset, base URL must be HTTP, provider case-insensitive. |
| `app/runtime.py` | `build_decision_service(settings)`; `Runtime.decisions` (defaulted, so every existing construction is unchanged); `build_runtime(..., decisions=)`. |
| `app/main.py` | The lifespan also `await`s the decision service's `aclose()`. |
| `app/observability.py` | `typesafe_sdk` added to `QUIET_LOGGERS` (the adapter additionally drops every SDK record). |
| `app/orchestrator/__init__.py` | Exports `DecisionCoordinator`. |
| `requirements.txt` | `typesafe-sdk==0.7.1`. |
| `.env.example` | Decision-layer section, disabled. |
| `REPORT.md` | This section. |

**Not touched:**

- `app/services/llm_openai.py`, `app/services/llm.py`, and the LLM
  retry/deadline logic;
- `app/orchestrator/pipeline.py`, `prompt.py` and `result.py`;
- `app/core/` (policy engine, checks, rules, session reducer);
- `app/models/`, `app/tools/` (registry, banking interfaces) and `app/api/`;
- `app/services/turn.py`, `stt.py`, `tts.py` and `validation.py`;
- `data/regulatory/` and `regulatory_corpus/`;
- `tests/fakes.py`, `tests/conftest.py`, and every pre-existing test.

## SD.11 Dependency

`typesafe-sdk==0.7.1`, pinned exactly. It is TypeSafe's official Python SDK,
released 2026-09-21. The wheel sha256 is `9d04eee1…43e9d2ad`, verified against
PyPI before installation. It requires `httpx2>=2.0.0` and `pydantic>=2.12.0`,
both already present, plus `tenacity>=9.0.0`, which is new (9.1.4 installed).
Installing it upgraded nothing.

It is always installed and imported only when enabled. A subprocess test blocks
the import entirely and shows the application still starts. Enabling Jev without
it fails at startup and names the requirement.

## SD.12 Tests

| | Collected | Passed |
| --- | --- | --- |
| Before (commit `e594f05`) | 473 | 473 |
| After | 901 | 901 |
| New | 428 | 428 |

The 428 include 87 regression tests added across two rounds of adversarial
review. The fixes were mutation-checked, with the results in SD.14.

No existing test was modified, skipped or deleted, and nothing in the new tests
is skipped. Coverage of the required cases:

| Required | Where |
| --- | --- |
| Disabled → zero calls, behaviour unchanged | `test_disabled_barge_in_is_exactly_the_heuristic` (32 signal × speaking cases), `test_disabled_intent_and_escalation_take_the_existing_path`, `test_a_default_process_never_loads_the_sdk_or_the_adapter` |
| Success parsed | `test_a_valid_answer_is_parsed_into_the_provider_neutral_type`, `test_every_registered_decision_round_trips` |
| Timeout → fallback | `test_an_sdk_timeout_becomes_a_decision_timeout`, `test_the_whole_call_is_bounded_by_the_deadline`, `test_the_coordinator_enforces_the_deadline_on_a_provider_that_does_not`, `test_the_provider_wait_is_charged_to_the_acknowledgement_window`, `test_an_answer_that_arrives_after_the_deadline_is_discarded` |
| Transport failure → fallback | `test_a_transport_failure_is_a_connection_failure_not_uncertainty`, `test_every_failure_falls_back_to_the_heuristic` |
| Malformed → reject → fallback | `test_an_answer_the_sdk_accepts_but_is_unusable_is_malformed` (8), `test_a_nan_confidence_is_malformed`, `test_a_label_the_registry_does_not_offer_is_malformed_whatever_the_provider` |
| Low confidence → UNCERTAIN → fallback | `test_below_the_threshold_is_uncertain_and_falls_back`, `test_each_decision_has_its_own_threshold` |
| Backchannel ("hmm", "yeah", "uh-huh", "right") vs context | `test_a_semantic_backchannel_keeps_the_agent_speaking`, `test_a_backchannel_word_is_not_assumed_to_be_a_backchannel`, `test_an_acknowledgement_followed_by_an_objection_stops_the_agent` |
| Interruption ("stop", "wait", "no", "one second") | `test_an_explicit_request_to_stop_is_never_left_to_the_provider`, `test_common_ways_to_say_stop_are_hard_signals` (20 variants), `test_a_semantic_interruption_stops_the_agent` |
| Intent | `test_an_already_made_payment_is_classified_and_left_to_the_application` and the tier tests |
| Escalation | the five escalation tests in `test_decision_layer.py` |
| Identity A ≠ B | `test_no_decision_can_make_a_tool_read_another_customers_account` (20) |

**Passing tests do not make this safe to rely on.** They prove the plumbing,
routing, fallbacks and boundaries. They cannot show that Jev gives good answers
on real Hindi, Marathi or Hinglish calls.

## SD.13 Known limitations

1. **Jev has never answered.** Accuracy and latency are unmeasured (SD.9).
2. **Not in the turn path** (SD.2). No production caller exists yet; the audio
   loop that would call the coordinator is not built.
3. **Thresholds are provisional**, and the eval set is synthetic (SD.7).
4. **Language.** TypeSafe says English is Jev's primary language, and others
   are "handled but not equally well". Three of the four languages here are
   non-English, and code-mixed Hinglish is untested.
5. **Adversarial input.** Jev documents that state text can steer it. That is
   bounded here by design, since nothing it decides is safety-relevant. A
   customer could still, for example, talk the barge-in decision into keeping
   the agent speaking. The bound: speech heard plus the provider wait stays
   within 800 ms, plus however long the caller takes to deliver the next
   partial, whose duration then trips the hard signal. That assumes the caller
   follows the contract in SD.3. A caller that stops feeding partials while a
   call is pending loses the bound.
6. **`HALT_PHRASES` is provisional**, like the backchannel vocabulary it sits
   beside. It is derived from the languages, not from calls.
7. **Escalation `yes` bundles several judgments** into one question: explicit
   request, distress, repeated misunderstanding. Jev's docs advise one judgment
   per question. Decomposing it into separate questions is a follow-up that
   needs evaluation data first.
8. **One question per call.** Jev evaluates many questions in one request more
   cheaply than separate calls. Intent and escalation for the same utterance
   are two calls today; batching them would need a `decide_many` contract.
9. **Deadline granularity.** The adapter and the coordinator both enforce
   `JEV_TIMEOUT_SECONDS`, and barge-in also enforces its window budget. The
   SDK's own timeout is per HTTP phase. A cancelled request is abandoned, not
   drained. A provider that *blocks* the event loop defeats every async
   deadline. Its late answer is discarded, but the stall has already happened.
   Loop-lag tests pin both the coordinator and the real Jev adapter (through
   the SDK) at under 25 ms of lag, against about 6 ms normally. A stall caused
   by *other* work on the loop also discards a pending answer as a timeout.
   That is safe, but the record then points at the provider, so read it
   together with `deadline_ms`.
10. **Model aliases are accepted.** `jev-latest` and gateway IDs such as
    `typesafe-ai/jev` move under you. Pin a versioned ID once thresholds are
    tuned; each record logs the answering model.
11. **No circuit breaker.** A degraded provider costs up to the deadline on
    every eligible decision until someone disables it.
12. **One event loop per adapter instance.** Pooled connections belong to the
    loop that opened them. Use and close a `JevDecisionService` on one loop;
    uvicorn's single loop and the lifespan do. A per-call `asyncio.run` breaks
    the instance, not just its pooling.
13. **Identifier masking is best-effort** (SD.8). It misses names, numbers
    spoken as words, oddly separated digits and short identifiers. It
    over-masks some dates and large amounts.
14. **Accepted without argmax check.** The adapter does not verify that the
    chosen label is the most probable one in the answer's own probabilities.
    The review raised this; its verifier rejected it (SD.14).
15. **Pre-existing, found incidentally, not changed:** `.env.example` cannot be
    loaded verbatim as `.env`. Its blank `MODEL_CONNECT_TIMEOUT_SECONDS=`,
    `MODEL_MAX_OUTPUT_TOKENS=`, `MODEL_TEMPERATURE=` and
    `MAX_RECOVERY_CALLS_PER_DAY=` fail numeric validation. The new decision
    keys are all loadable.

## SD.14 Adversarial review, and what it changed

After the first version passed (814/814), five independent reviewers each
examined the change through a different lens:

- safety and authority;
- SDK-contract fidelity;
- async, fallback and configuration;
- privacy and logging;
- documentation honesty and test vacuity.

Each finding then went to a skeptic prompted to *refute* it, with reproductions
run against the real code. That produced 23 findings. 12 were confirmed. None
was confirmed critical or high: 11 were rated low, and one medium, the vacuous
test. The other 11 were rejected.

**Confirmed, and fixed**

| # | Finding | Fix | Pinned by |
| --- | --- | --- | --- |
| 1 | The barge-in provider wait was *added* to the acknowledgement window. The agent could talk over up to 800 ms plus the full deadline, including a "stop" said during the wait. | The wait is charged to the window, `min(deadline, 800 ms − speech so far)`; nothing is attempted with under 50 ms left; the caller contract is documented. | `test_the_provider_wait_is_charged_to_the_acknowledgement_window`, `test_no_provider_call_when_the_window_is_all_but_spent` |
| 2 | `HALT_PHRASES` missed "ruk jao", "रुक जाओ", "थांब", "एक सेकंड", "एक मिनिट" and "ठहरो"; zero-width joiners broke matching. | Stems and both scripts added; NFC normalisation; format characters stripped; phrases compared in the same normal form. | `test_common_ways_to_say_stop_are_hard_signals`, which checks a variant list kept apart from the list itself |
| 3 | Retries re-sent after read, write and protocol errors, when the utterance may already have been delivered. The docs said only failed connects were retried. | Code changed to match the docs: `api_connection_error=False` plus a predicate that allows only `httpx2.ConnectError`. | `test_an_error_after_the_request_may_have_been_delivered_is_never_retried`, `test_a_failed_connect_is_the_only_transport_error_retried` |
| 4 | The SDK's one WARNING line put provider-controlled text, including an echoed utterance, into the application log. | A filter drops every SDK record; unlike a level, it survives `configure_logging`. | `test_provider_text_in_the_sdks_own_warning_never_reaches_the_logs` |
| 5 | `test_the_coordinator_does_not_block_the_event_loop` could not fail. | It now measures loop lag while a decision is pending. A companion test shows a blocking provider is detected. The side-by-side bound is tightened. | `test_the_loop_lag_measurement_does_detect_a_blocked_loop` |
| 6 | "No decision module can reach the registry" held only for direct imports; the coordinator was handed the whole `Runtime`. | Built via `from_runtime` with only the service, settings and classifier; `Runtime` is imported only for typing. | `test_the_coordinator_is_handed_nothing_it_could_misuse` |
| 7 | The report said the eval cases covered every label "in the four languages". | 10 non-English cases added; the wording is corrected; per-language coverage per decision is tested. | `test_every_decision_has_cases_in_every_language_in_scope` |

The seven rows cover all 12 confirmed findings, because several were reported
more than once:

- row 3 three times: two reports of the behaviour and one of the wording;
- row 4 three times;
- row 5 twice.

The "800 ms" bound in SD.13 was part of row 1's finding.

**Rejected by the verifiers, but hardened anyway because it was cheap**

- Fusion now acts only on `DECIDED` results. `DecisionResult` now enforces
  "label if and only if decided", so a contradictory result cannot be built.
- A provider that *returns* garbage now falls back instead of raising. The
  first round covered None and a dict. The second round (below) found an
  unvalidated `DecisionAnswer` still escaping, and closed that too.
- An answer measured after its deadline is discarded.
- The reported model name must fully match an identifier containing a letter.
- Only SDK-related missing modules are reported as "install typesafe-sdk"; any
  other missing module re-raises with its traceback.
- Identifier-shaped spans in free text are masked before egress.

**Rejected and left as is**

- **No argmax check on answers.** The verifier showed that confidence is
  derived from the distribution's shape, not `probabilities[choice]`, so the
  proposed check would reject legitimate answers. Listed in SD.13.
- **A closed client raises `RuntimeError`, not a `DecisionError`.** The
  coordinator already records it as `decision_failed`. It only happens after
  shutdown.
- **The eval percentiles exclude timeouts.** The timeout count is reported
  alongside, and the deadline can be raised for measurement. A docstring now
  says so.
- **Cross-event-loop reuse fails.** No code path does it. It is documented in
  the adapter and in SD.13.
- **The vendor latency figure had no source.** The source was found and is now
  cited in SD.9.

**Second round.** A smaller pass, with two reviewers and one skeptic, checked
the fixes and this report. It confirmed 12 more findings, 8 distinct once
duplicates are merged. One was medium: a `DecisionAnswer` built without
validation (`model_construct`) and holding a wrong-typed or missing field
still raised out of the coordinator. None was a safety or authority hole. All
were fixed:

- answers are strictly re-validated behind a catch-all, and `confidence=True`,
  which lax validation turns into a decided 1.0, is refused;
- the date exemption requires a real calendar date, so "98 76 2019" is masked;
- masking separators were widened, and the masking claims restated to match
  the code exactly (SD.8);
- the stated cause of late-answer discards was corrected: any loop stall causes
  one, not only a blocking provider. The record now carries `deadline_ms`;
- the eval `--mock` run now keys cases by the masked utterance;
- the retry rationale was corrected: retried statuses do re-send the
  customer's words;
- this section's accounting was corrected;
- a loop-lag test now runs through the real adapter. The earlier one allowed a
  30 ms block.

**Mutation check, final state.** Each code fix from both rounds was reverted in
a scratch copy of the repository, one at a time, and its regression test was
run. 24 mutations were tried and 22 were killed. These include the two ways of
reintroducing a blocking call that the original vacuous test let through, and a
30 ms block in the real adapter.

The two survivors are each a *redundant* defence layer in the coordinator's
answer check: the `isinstance` guard and the catch-all. Removing either alone
changes nothing, because another layer covers it. Removing each together with
the layer that masks it (isinstance with the catch-all, strict re-validation
with the catch-all) is killed. Two changes are pinned but were not
mutation-checked: the eval-case additions, pinned by a coverage test, and the
masking regex's anchoring, a performance change pinned by a generous timing
bound.

## SD.15 Stage 3B

`REPORT.md` does not define Stage 3B. This section takes it to be the step the
Stage 3A banner defers: connecting a real model, meaning the GCP/vLLM Gemma
deployment. **It remains exactly as blocked or deferred as before.** This change
neither depends on it nor unblocks it:

- no model infrastructure was touched;
- `LlmService` and its adapter are unchanged;
- the decision layer runs with or without a model.

**No claim is made that this is production ready, that Jev improves latency, or
that Jev is more accurate than the existing heuristic or upstream NLU.** What
this stage delivers is an isolated, disabled-by-default, tested boundary.
Through it, a bounded semantic provider can be switched on by configuration,
measured with the included harness, and bounded by a deterministic fallback
in every failure mode the tests exercise.
