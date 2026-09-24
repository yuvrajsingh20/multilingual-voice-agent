"""Regenerate data/regulatory/rules/recovery_rules.json from the local corpus.

Run from the repository root::

    python scripts/build_recovery_rules.py

Why this script exists
----------------------
Every rule's ``quote`` is *sliced* out of the local corpus text file rather than
retyped, so a quote cannot silently drift from its source. Each rule names a
start anchor and an optional end anchor - ASCII-safe fragments chosen to avoid
the curly quotes and spaced slashes the RBI documents use - and the script
extracts the text between them. If an anchor stops matching (because a document
was re-ingested and changed), the script exits non-zero and names the rule
instead of writing a wrong quote.

This script does NOT fetch anything. It reads only files already present under
``data/regulatory/text/``. ``tests/test_regulatory_rules.py`` independently
re-verifies that every quote in the generated file is verbatim.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

DOCS = {
    "rbi_cb_rbc_2025": {
        "title": "Reserve Bank of India (Commercial Banks – Responsible Business Conduct) Directions, 2025",
        "file": "data/regulatory/text/rbi_cb_rbc_2025.txt",
        "url": "https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=13140",
        "from": "2025-11-28",
    },
    "rbi_cb_outsourcing_2025": {
        "title": "Reserve Bank of India (Commercial Banks – Managing Risks in Outsourcing) Directions, 2025",
        "file": "data/regulatory/text/rbi_cb_outsourcing_2025.txt",
        "url": "https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=13139&Mode=0",
        "from": "2025-11-28",
    },
    "rbi_digital_lending_2025": {
        "title": "Reserve Bank of India (Digital Lending) Directions, 2025",
        "file": "data/regulatory/text/rbi_digital_lending_2025.txt",
        "url": "https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=12848&Mode=0",
        "from": "2025-05-08",
    },
    "rbi_cb_rbc_4th_amd_2026": {
        "title": "Reserve Bank of India (Commercial Banks – Responsible Business Conduct) Fourth Amendment Directions, 2026",
        "file": "data/regulatory/text/rbi_cb_rbc_4th_amd_2026.txt",
        "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13665&Mode=0",
        "from": "2027-01-01",
    },
    "rbi_recovery_agents_2022": {
        "title": "Outsourcing of Financial Services - Responsibilities of regulated entities employing Recovery Agents",
        "file": "data/regulatory/text/rbi_recovery_agents_2022.txt",
        "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12378&Mode=0",
        "from": "2022-08-12",
    },
    "rbi_recovery_agents_banks_2008": {
        "title": "Recovery Agents Engaged by Banks",
        "file": "data/regulatory/text/rbi_recovery_agents_banks_2008.txt",
        "url": "https://www.rbi.org.in/scripts/BS_CircularIndexDisplay.aspx?Id=4141",
        "from": "2008-04-24",
    },
    "rbi_outsourcing_banks_2006": {
        "title": "Guidelines on Managing Risks and Code of Conduct in Outsourcing of Financial Services by banks",
        "file": "data/regulatory/text/rbi_outsourcing_banks_2006.txt",
        "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=3148",
        "from": "2006-11-03",
    },
}

_TEXT = {k: (ROOT / v["file"]).read_text(encoding="utf-8") for k, v in DOCS.items()}

_failures: list[str] = []


def slice_quote(doc_id: str, start: str, end: str | None, rule_id: str) -> str:
    text = _TEXT[doc_id]
    i = text.find(start)
    if i < 0:
        _failures.append(f"{rule_id}: start anchor not found in {doc_id}: {start!r}")
        return start
    if end is None:
        return start
    j = text.find(end, i)
    if j < 0:
        _failures.append(f"{rule_id}: end anchor not found in {doc_id}: {end!r}")
        return start
    return text[i : j + len(end)]


# Supersession: the Fourth Amendment Directions, 2026 delete RBC 2025
# paragraphs 408-416 and 442-454 with effect from 2027-01-01.
DELETED_2027 = "2026-12-31"

RULES: list[dict] = []


def rule(**kw) -> None:
    RULES.append(kw)


# --------------------------------------------------------------------------
# RBC 2025 - Chapter VIII Section A (Fair Practices Code). NOT deleted in 2027.
# --------------------------------------------------------------------------
rule(
    rule_id="RBI-CB-RBC-2025-343-NO-UNDUE-HARASSMENT",
    title="No undue harassment in recovery of loans",
    doc="rbi_cb_rbc_2025",
    paragraph="343",
    anchor=("In the matter of recovery of loans, the bank shall not resort to undue harassment",
            "use of muscle power for recovery of loans, etc."),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Always, during any loan recovery contact.",
    applies_when={"description": "All loan recovery contacts by the bank or its agents."},
    prohibited_action=["verbal_harassment", "abusive_language", "intimidation", "persistent_calling"],
    check_id="prohibited_conduct",
    notes=(
        "Sits in the Fair Practices Code section, which the Fourth Amendment Directions, 2026 "
        "do not delete, so this rule has no end date. 'Undue harassment' is read as covering "
        "abusive language; the paragraph gives examples ('viz.') rather than an exhaustive list."
    ),
)

# --------------------------------------------------------------------------
# RBC 2025 - Section I.4 microfinance recovery (paragraphs 408-416, deleted 2027)
# --------------------------------------------------------------------------
MFI_SCOPE = {
    "description": "Microfinance loans only. Paragraph 445 expressly excludes microfinance, and paragraph 410 states a separate microfinance regime.",
    "product_types": ["microfinance"],
}

rule(
    rule_id="RBI-CB-RBC-2025-410-1-MFI-NO-THREATENING-LANGUAGE",
    title="Microfinance: threatening or abusive language is a harsh method",
    doc="rbi_cb_rbc_2025",
    paragraph="410(1)",
    anchor=("(1) Use of threatening or abusive language", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Microfinance recovery contact.",
    applies_when=MFI_SCOPE,
    prohibited_action=["abusive_language", "threatening_call", "intimidation"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-1-NO-MINATORY-OR-ABUSIVE-LANGUAGE"],
)

rule(
    rule_id="RBI-CB-RBC-2025-410-2-MFI-CALLING-HOURS",
    title="Microfinance: no calls before 09:00 or after 18:00",
    doc="rbi_cb_rbc_2025",
    paragraph="410(2)",
    anchor=("(2) Persistently calling the borrower and / or calling the borrower before 9:00 a.m. and after 6:00 p.m.", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="A microfinance recovery contact is attempted outside 09:00-18:00 borrower local time.",
    applies_when=MFI_SCOPE,
    prohibited_action=["contact_outside_permitted_hours", "persistent_calling"],
    required_action=["end_call"],
    check_id="calling_hours",
    parameters={
        "earliest_local_time": "09:00:00",
        "latest_local_time": "18:00:00",
        "boundary_semantics": "inclusive_endpoints",
    },
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Y-4-CONTACT-HOURS"],
    notes=(
        "The microfinance window (09:00-18:00) is narrower than the general window in paragraph 445 "
        "(08:00-19:00). From 2027-01-01 paragraph 454Y(4) states a single 08:00-19:00 window with no "
        "microfinance carve-out, so this narrower window ends with paragraphs 408-416."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-410-3-MFI-NO-HARASSING-THIRD-PARTIES",
    title="Microfinance: harassing relatives, friends or co-workers is a harsh method",
    doc="rbi_cb_rbc_2025",
    paragraph="410(3)",
    anchor=("(3) Harassing relatives, friends, or co-workers of the borrower", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Microfinance recovery contact.",
    applies_when=MFI_SCOPE,
    prohibited_action=["privacy_intrusion_third_party", "verbal_harassment"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-6-NO-INTIMIDATION-OR-HARASSMENT"],
)

rule(
    rule_id="RBI-CB-RBC-2025-410-4-MFI-NO-PUBLISHING-NAMES",
    title="Microfinance: publishing borrower names is a harsh method",
    doc="rbi_cb_rbc_2025",
    paragraph="410(4)",
    anchor=("(4) Publishing the name of borrowers", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="high",
    condition="Microfinance recovery contact.",
    applies_when=MFI_SCOPE,
    prohibited_action=["public_humiliation", "unauthorised_disclosure"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-2-NO-SOCIAL-MEDIA-EXPOSURE"],
)

rule(
    rule_id="RBI-CB-RBC-2025-410-5-MFI-NO-VIOLENCE",
    title="Microfinance: use or threat of violence is a harsh method",
    doc="rbi_cb_rbc_2025",
    paragraph="410(5)",
    anchor=("(5) Use or threat of use of violence or other similar means to harm the borrower",
            "assets / reputation"),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Microfinance recovery contact.",
    applies_when=MFI_SCOPE,
    prohibited_action=["violence_threat", "physical_harassment"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-7-NO-VIOLENCE"],
)

rule(
    rule_id="RBI-CB-RBC-2025-410-6-MFI-NO-MISLEADING",
    title="Microfinance: misleading the borrower about the debt is a harsh method",
    doc="rbi_cb_rbc_2025",
    paragraph="410(6)",
    anchor=("(6) Misleading the borrower about the extent of the debt or the consequences of non-repayment.", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Microfinance recovery contact.",
    applies_when=MFI_SCOPE,
    prohibited_action=["false_or_misleading_representation"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-8-NO-FALSE-REPRESENTATION"],
)

# --------------------------------------------------------------------------
# RBC 2025 - Section K (paragraphs 442-454, deleted 2027)
# --------------------------------------------------------------------------
GENERAL_SCOPE_NON_MFI = {
    "description": "All loan products except microfinance; paragraph 445 states it does not apply to microfinance loans.",
    "excluded_product_types": ["microfinance"],
}
ALL_PRODUCTS = {"description": "All loan products handled by the recovery voice agent."}

rule(
    rule_id="RBI-CB-RBC-2025-442-2-AGENT-DUE-DILIGENCE",
    title="Due diligence before engaging a recovery agent",
    doc="rbi_cb_rbc_2025",
    paragraph="442(2)",
    anchor=("The bank shall have a due diligence process in place for engagement of recovery agents",
            "individuals involved in the recovery process."),
    status="current_operative",
    enforcement="organisational_obligation",
    severity="high",
    condition="Before an agent (including an automated voice agent operated under an outsourcing arrangement) is engaged.",
    applies_when=ALL_PRODUCTS,
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454H-DUE-DILIGENCE"],
    notes="Onboarding control. A live call cannot verify it, so no automated check exists.",
)

rule(
    rule_id="RBI-CB-RBC-2025-442-3-AGENCY-DETAILS-TO-BORROWER",
    title="Borrower must be told the recovery agency's details",
    doc="rbi_cb_rbc_2025",
    paragraph="442(3)",
    anchor=("the bank shall inform the borrower the details of recovery agency firms / companies while forwarding default cases to the recovery agency", None),
    status="current_operative",
    enforcement="requires_external_data",
    severity="high",
    condition="A default case has been forwarded to a recovery agency and contact is about to be made.",
    applies_when=ALL_PRODUCTS,
    required_action=["identify_bank_and_agent"],
    check_id="agency_details_disclosed",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454L-AGENCY-DETAILS-BEFORE-VISIT"],
    notes=(
        "The paragraph also requires the agent to carry a notice, authorisation letter and identity card; "
        "that part is specific to in-person visits and is not enforceable on a voice call."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-442-4-CALL-RECORDING-INTIMATION",
    title="Tell the customer the conversation is being recorded",
    doc="rbi_cb_rbc_2025",
    paragraph="442(4)",
    anchor=("The bank shall ensure that there is a tape recording of the content / text of the calls made by recovery agents to the customers",
            "intimating the customer that the conversation is being recorded, etc."),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="high",
    condition="Any recovery call, before the account or the dues are discussed.",
    applies_when=ALL_PRODUCTS,
    required_action=["disclose_call_recording"],
    check_id="recording_disclosure",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454P-CALL-RECORDING-AND-INTIMATION"],
    notes=(
        "The text says 'reasonable precaution such as intimating the customer'. The point in the call at "
        "which the disclosure must be made is a bank choice; this rule requires it before substantive "
        "collection discussion, which is the earliest defensible point."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-442-6-GRIEVANCE-PENDING-HOLD",
    title="Do not pursue recovery through agents while a grievance is undisposed",
    doc="rbi_cb_rbc_2025",
    paragraph="442(6)",
    anchor=("Where a grievance / complaint has been lodged, the bank shall not forward cases to recovery agencies till they have finally disposed of any grievance / complaint lodged by the concerned borrower.", None),
    status="current_operative",
    enforcement="requires_external_data",
    severity="critical",
    condition="A grievance or complaint lodged by this borrower has not been finally disposed of.",
    applies_when=ALL_PRODUCTS,
    exceptions=[
        "Paragraph 442(6): where the bank is convinced, with appropriate proof, that the borrower is continuously making frivolous / vexatious complaints, recovery through agents may continue.",
        "Paragraph 442(6): where the subject matter of the dues may be sub judice, the bank shall exercise utmost caution in referring the matter to recovery agencies.",
    ],
    required_action=["terminate_collection_discussion", "escalate_to_human"],
    check_id="grievance_pending_hold",
    effective_until=DELETED_2027,
    notes=(
        "The sub judice limb says 'utmost caution', not a prohibition. The check therefore escalates to a "
        "human for sub judice cases instead of asserting a ban the text does not state. "
        "No superseded_by: the Fourth Amendment Directions, 2026 delete paragraph 442(6) and the new "
        "Section L contains no equivalent hold on referring a case while a grievance is undisposed. "
        "Whether the bank keeps this control after 2027-01-01 as internal policy is a bank decision and "
        "requires regulatory review."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-442-7-GRIEVANCE-MECHANISM",
    title="A recovery grievance mechanism must exist and its details be furnished",
    doc="rbi_cb_rbc_2025",
    paragraph="442(7)",
    anchor=("The bank shall have a mechanism whereby the borrowers", "recovery process can be addressed."),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="medium",
    condition="The borrower raises a dispute or complaint about the dues or the recovery process during the call.",
    applies_when=ALL_PRODUCTS,
    required_action=["provide_grievance_mechanism_details", "record_dispute"],
    check_id="dispute_handling",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454AA-GRIEVANCE-OFFICER-DETAILS"],
    notes=(
        "The paragraph obliges the bank to have the mechanism and to furnish its details. Surfacing those "
        "details when the borrower disputes during a call is an operationalisation; RBI prescribes no wording."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-443-NO-UNCIVILISED-CONDUCT-INCENTIVES",
    title="Agent contracts must not induce uncivilised, unlawful or questionable conduct",
    doc="rbi_cb_rbc_2025",
    paragraph="443",
    anchor=("The bank shall ensure that the contracts with the recovery agents do not induce adoption of uncivilised, unlawful and questionable behaviour or recovery process.", None),
    status="current_operative",
    enforcement="organisational_obligation",
    severity="high",
    condition="When contracting with, or setting incentives for, a recovery agent.",
    applies_when=ALL_PRODUCTS,
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Q-INCENTIVES"],
    notes="Contractual control. Relevant here because the agent's objective function is a design-time choice.",
)

_P445 = ("445. The bank shall strictly ensure", "This direction shall not be applicable to microfinance loans.")
_MFI_EXC = ["Paragraph 445: 'This direction shall not be applicable to microfinance loans.'"]

rule(
    rule_id="RBI-CB-RBC-2025-445-CALLING-HOURS",
    title="No recovery calls before 08:00 or after 19:00",
    doc="rbi_cb_rbc_2025",
    paragraph="445",
    anchor=("calling the borrower before 8:00 a.m. and after 7:00 p.m. for recovery of overdue loans", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="A recovery contact is attempted outside 08:00-19:00 in the configured borrower timezone.",
    applies_when=GENERAL_SCOPE_NON_MFI,
    exceptions=_MFI_EXC,
    prohibited_action=["contact_outside_permitted_hours"],
    required_action=["end_call"],
    check_id="calling_hours",
    parameters={
        "earliest_local_time": "08:00:00",
        "latest_local_time": "19:00:00",
        "boundary_semantics": "inclusive_endpoints",
    },
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Y-4-CONTACT-HOURS"],
    notes=(
        "'before 8:00 a.m.' and 'after 7:00 p.m.' are read as excluding the endpoints themselves, so "
        "08:00:00 and 19:00:00 are inside the permitted window and 07:59:59 and 19:00:01 are not. "
        "The timezone is app configuration (default Asia/Kolkata), never the host machine's zone; RBI "
        "does not name a timezone in the text."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-445-NO-PERSISTENT-CALLING",
    title="No persistent calling of the borrower",
    doc="rbi_cb_rbc_2025",
    paragraph="445",
    anchor=("persistently calling the borrower", None),
    status="current_operative",
    enforcement="requires_external_data",
    severity="high",
    condition="Recovery contacts to this borrower exceed a threshold the bank has defined.",
    applies_when=GENERAL_SCOPE_NON_MFI,
    exceptions=_MFI_EXC,
    prohibited_action=["persistent_calling"],
    check_id="persistent_calling",
    parameters={"max_contacts_per_day": None},
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-4-NO-EXCESSIVE-OR-OUT-OF-HOURS-CONTACT"],
    notes=(
        "RBI states no number. The check stays unevaluated until the bank sets "
        "MAX_RECOVERY_CALLS_PER_DAY and the backend supplies the day's contact count. No threshold is "
        "invented here."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-445-NO-INTIMIDATION-OR-HARASSMENT",
    title="No intimidation or harassment, verbal or physical",
    doc="rbi_cb_rbc_2025",
    paragraph="445",
    anchor=("do not resort to intimidation or harassment of any kind, either verbal or physical, against any person in their debt collection efforts", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Always, for every utterance the agent produces.",
    applies_when=GENERAL_SCOPE_NON_MFI,
    exceptions=_MFI_EXC,
    prohibited_action=["intimidation", "verbal_harassment", "physical_harassment", "abusive_language"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-6-NO-INTIMIDATION-OR-HARASSMENT"],
    notes=(
        "'harassment of any kind, either verbal or physical' is read as covering abusive language. "
        "From 2027-01-01 paragraph 454Z(1) names 'minatory or abusive language' expressly."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-445-NO-PUBLIC-HUMILIATION",
    title="No acts intended to humiliate publicly",
    doc="rbi_cb_rbc_2025",
    paragraph="445",
    anchor=("including acts intended to humiliate publicly", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Always, for every utterance the agent produces.",
    applies_when=GENERAL_SCOPE_NON_MFI,
    exceptions=_MFI_EXC,
    prohibited_action=["public_humiliation"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-6-NO-INTIMIDATION-OR-HARASSMENT"],
)

rule(
    rule_id="RBI-CB-RBC-2025-445-NO-PRIVACY-INTRUSION",
    title="No intrusion into the privacy of family, referees or friends",
    doc="rbi_cb_rbc_2025",
    paragraph="445",
    anchor=("intrude upon the privacy of the debtors", "family members, referees and friends"),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="A person other than the borrower or guarantor is on the call, or third parties are discussed.",
    applies_when=GENERAL_SCOPE_NON_MFI,
    exceptions=_MFI_EXC,
    prohibited_action=["privacy_intrusion_third_party", "unauthorised_disclosure"],
    required_action=["verify_identity_before_disclosure"],
    check_id="confidentiality_disclosure",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-6-NO-INTIMIDATION-OR-HARASSMENT"],
)

rule(
    rule_id="RBI-CB-RBC-2025-445-NO-INAPPROPRIATE-MESSAGES",
    title="No inappropriate messages on mobile or social media",
    doc="rbi_cb_rbc_2025",
    paragraph="445",
    anchor=("sending inappropriate messages either on mobile or through social media", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="high",
    condition="Always. On a voice call this constrains what the agent may say it will do.",
    applies_when=GENERAL_SCOPE_NON_MFI,
    exceptions=_MFI_EXC,
    prohibited_action=["inappropriate_messaging", "social_media_exposure"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-3-NO-INAPPROPRIATE-MESSAGES"],
    notes="The prohibition targets another channel; it is enforced here so the agent cannot threaten to use it.",
)

rule(
    rule_id="RBI-CB-RBC-2025-445-NO-THREATENING-CALLS",
    title="No threatening calls",
    doc="rbi_cb_rbc_2025",
    paragraph="445",
    anchor=("making threatening and / or anonymous calls", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Always, for every utterance the agent produces.",
    applies_when=GENERAL_SCOPE_NON_MFI,
    exceptions=_MFI_EXC,
    prohibited_action=["threatening_call", "violence_threat"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-5-NO-THREATENING-OR-ANONYMOUS-CALLS"],
)

rule(
    rule_id="RBI-CB-RBC-2025-445-NO-ANONYMOUS-CALLS",
    title="No anonymous calls: the bank and the agent must be identified",
    doc="rbi_cb_rbc_2025",
    paragraph="445",
    anchor=("making threatening and / or anonymous calls", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="The call has progressed past the greeting without the bank and the agent being identified.",
    applies_when=GENERAL_SCOPE_NON_MFI,
    exceptions=_MFI_EXC,
    prohibited_action=["anonymous_call"],
    required_action=["identify_bank_and_agent"],
    check_id="agent_identification",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-5-NO-THREATENING-OR-ANONYMOUS-CALLS"],
    notes=(
        "Requiring identification is the operational consequence of the prohibition on anonymous calls. "
        "RBI prescribes no script or wording for a telephone call; the bank must define that."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-445-NO-FALSE-REPRESENTATION",
    title="No false or misleading representations",
    doc="rbi_cb_rbc_2025",
    paragraph="445",
    anchor=("making false and misleading representations", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Always. Covers amounts, dates, consequences of non-payment and the agent's authority.",
    applies_when=GENERAL_SCOPE_NON_MFI,
    exceptions=_MFI_EXC,
    prohibited_action=["false_or_misleading_representation"],
    check_id="prohibited_conduct",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454Z-8-NO-FALSE-REPRESENTATION"],
    notes=(
        "Operationally this is why every figure the agent states must be grounded in a tool result; see "
        "app.services.validation."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-446-CUSTOMER-CONFIDENTIALITY",
    title="Recovery agents must observe strict customer confidentiality",
    doc="rbi_cb_rbc_2025",
    paragraph="446",
    anchor=("It is essential that the Recovery Agents refrain from action that could damage the integrity and reputation of the bank and that they observe strict customer confidentiality.", None),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Before any account-specific information is spoken on the call.",
    applies_when=ALL_PRODUCTS,
    prohibited_action=["unauthorised_disclosure"],
    required_action=["verify_identity_before_disclosure"],
    check_id="confidentiality_disclosure",
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454O-NEED-TO-KNOW"],
)

rule(
    rule_id="RBI-CB-RBC-2025-447-AGENT-TRAINING",
    title="Recovery agents must be trained on calling hours and customer privacy",
    doc="rbi_cb_rbc_2025",
    paragraph="447",
    anchor=("The bank shall ensure that the Recovery Agents are properly trained to handle with care and sensitivity",
            "privacy of customer information, etc."),
    status="current_operative",
    enforcement="organisational_obligation",
    severity="medium",
    condition="Before an agent is put in front of customers.",
    applies_when=ALL_PRODUCTS,
    effective_until=DELETED_2027,
    superseded_by=["RBI-CB-RBC-AMD4-2026-454I-TRAINING"],
    notes=(
        "Whether and how this applies to an automated agent rather than a person is not settled by the "
        "text and requires regulatory review before deployment."
    ),
)

rule(
    rule_id="RBI-CB-RBC-2025-452-BANK-RESPONSIBLE-FOR-AGENTS",
    title="The bank, as principal, is responsible for the actions of its agents",
    doc="rbi_cb_rbc_2025",
    paragraph="452",
    anchor=("The bank, as principal, is responsible for the actions of its agents.", None),
    status="current_operative",
    enforcement="organisational_obligation",
    severity="critical",
    condition="Always.",
    applies_when=ALL_PRODUCTS,
    effective_until=DELETED_2027,
    notes="The reason the policy engine exists: the bank owns whatever this system says.",
)

# --------------------------------------------------------------------------
# Outsourcing Directions 2025
# --------------------------------------------------------------------------
rule(
    rule_id="RBI-CB-OUTSRC-2025-17-RESPONSIBLE-FOR-AGENTS-AND-CONFIDENTIALITY",
    title="Outsourcing does not diminish the bank's obligations or confidentiality duty",
    doc="rbi_cb_outsourcing_2025",
    paragraph="17",
    anchor=("The bank shall, therefore, be responsible for the actions of its service provider",
            "The bank shall retain ultimate control of the outsourced activity."),
    status="current_operative",
    enforcement="organisational_obligation",
    severity="critical",
    condition="Always, where the voice agent is operated under an outsourcing arrangement.",
    applies_when=ALL_PRODUCTS,
)

rule(
    rule_id="RBI-CB-OUTSRC-2025-24-NEED-TO-KNOW",
    title="Service provider access to customer information is on a need-to-know basis",
    doc="rbi_cb_outsourcing_2025",
    paragraph="24",
    anchor=("Access to customer information by a service provider or its staff shall be on a",
            "limited to those areas where the information is required in order to perform the outsourced function."),
    status="current_operative",
    enforcement="conversation_deterministic",
    severity="high",
    condition="Any point at which the agent would disclose or request customer information.",
    applies_when=ALL_PRODUCTS,
    prohibited_action=["unauthorised_disclosure"],
    required_action=["verify_identity_before_disclosure"],
    check_id="confidentiality_disclosure",
)

# --------------------------------------------------------------------------
# Digital Lending Directions 2025 - conditional
# --------------------------------------------------------------------------
rule(
    rule_id="RBI-DL-2025-8-V-AGENT-PARTICULARS-BEFORE-CONTACT",
    title="Digital lending: recovery agent particulars must reach the borrower before contact",
    doc="rbi_digital_lending_2025",
    paragraph="8(v)",
    anchor=("In case of a loan default, when a recovery agent is assigned for recovery",
            "before the recovery agent contacts the borrower for recovery."),
    status="conditional",
    enforcement="requires_external_data",
    severity="critical",
    condition="The product is a digital lending product and a recovery agent has been assigned or changed.",
    applies_when={
        "description": "Digital lending products only.",
        "product_types": ["digital_lending"],
    },
    required_action=["terminate_collection_discussion"],
    check_id="digital_lending_particulars_sent",
)

# --------------------------------------------------------------------------
# Fourth Amendment Directions, 2026 - effective 2027-01-01
# --------------------------------------------------------------------------
AMD = "rbi_cb_rbc_4th_amd_2026"

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454H-DUE-DILIGENCE",
    title="Due diligence and ongoing antecedent verification for recovery agents",
    doc=AMD,
    paragraph="454H",
    anchor=("A bank engaging recovery agencies shall put in place a due diligence process for their engagement",
            "as specified in the bank"),
    status="future_effective",
    enforcement="organisational_obligation",
    severity="high",
    condition="Before and during engagement of a recovery agency.",
    applies_when=ALL_PRODUCTS,
)

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454I-TRAINING",
    title="Recovery agents must hold an IIBF certificate",
    doc=AMD,
    paragraph="454I",
    anchor=("A bank shall ensure that the recovery agency engages only those agents who have obtained the certificate from Indian Institute of Banking and Finance", None),
    status="future_effective",
    enforcement="organisational_obligation",
    severity="medium",
    condition="Before an agent is engaged.",
    applies_when=ALL_PRODUCTS,
    notes="Application to an automated agent is unsettled and requires regulatory review.",
)

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454L-AGENCY-DETAILS-BEFORE-VISIT",
    title="Agency details to the borrower at least one day before the first in-person visit",
    doc=AMD,
    paragraph="454L",
    anchor=("While forwarding a case to any recovery agency for recovery of loan dues through in-person visit",
            "at least one day prior to the first visit."),
    status="future_effective",
    enforcement="requires_external_data",
    severity="high",
    condition="A case is forwarded to a recovery agency for an in-person visit.",
    applies_when=ALL_PRODUCTS,
    check_id="agency_details_disclosed",
    notes=(
        "The one-day notice obligation in this paragraph is written for in-person visits, not calls. "
        "Whether it extends to telephone contact is not stated and requires regulatory review."
    ),
)

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454O-NEED-TO-KNOW",
    title="Borrower information disclosed only as far as recovery duties require",
    doc=AMD,
    paragraph="454O",
    anchor=("A bank shall ensure that the disclosure of any borrower",
            "to discharge their loan recovery related duties."),
    status="future_effective",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="Any point at which the agent would disclose customer information.",
    applies_when=ALL_PRODUCTS,
    prohibited_action=["unauthorised_disclosure"],
    required_action=["verify_identity_before_disclosure"],
    check_id="confidentiality_disclosure",
)

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454P-CALL-RECORDING-AND-INTIMATION",
    title="Record recovery calls, document time and number, and tell the borrower",
    doc=AMD,
    paragraph="454P",
    anchor=("A bank shall document the time and number of calls made by its employee / recovery agent",
            "intimating the borrower / guarantor that the conversation is being recorded, etc."),
    status="future_effective",
    enforcement="conversation_deterministic",
    severity="high",
    condition="Any recovery call, before the dues are discussed.",
    applies_when=ALL_PRODUCTS,
    required_action=["disclose_call_recording"],
    check_id="recording_disclosure",
    parameters={"record_retention_months": 6},
    notes=(
        "Adds a six-month retention obligation (longer if sub judice) and a duty to document the time and "
        "number of calls. Retention is a storage obligation, not a call-time check; it is not implemented."
    ),
)

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454Q-INCENTIVES",
    title="Recovery targets and incentives must not induce harsh practices",
    doc=AMD,
    paragraph="454Q",
    anchor=("A bank shall ensure that the recovery targets or the structure of incentives",
            "do not induce adoption of harsh recovery practices as described at paragraph 454Z below."),
    status="future_effective",
    enforcement="organisational_obligation",
    severity="high",
    condition="When setting agent targets or contractual incentives.",
    applies_when=ALL_PRODUCTS,
)

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454Y-1-DISCUSS-ONLY-WITH-BORROWER",
    title="Discuss the dues only with the borrower or guarantor",
    doc=AMD,
    paragraph="454Y(1)",
    anchor=("An employee / recovery agent shall discuss the matters related to the loan dues and collection / recovery thereof only with the borrower / guarantor, as applicable.", None),
    status="future_effective",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="The person on the call is not the borrower or guarantor, or identity is unverified.",
    applies_when=ALL_PRODUCTS,
    prohibited_action=["unauthorised_disclosure", "privacy_intrusion_third_party"],
    required_action=["terminate_collection_discussion", "verify_identity_before_disclosure"],
    check_id="discuss_only_with_borrower",
)

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454Y-4-CONTACT-HOURS",
    title="Contact only between 08:00 and 19:00 hours",
    doc=AMD,
    paragraph="454Y(4)",
    anchor=("An employee / recovery agent shall contact / visit the borrower / guarantor only between 08:00 hours and 19:00 hours.",
            "shall be honoured in normal circumstances."),
    status="future_effective",
    enforcement="conversation_deterministic",
    severity="critical",
    condition="A recovery contact is attempted outside 08:00-19:00 in the configured borrower timezone.",
    applies_when=ALL_PRODUCTS,
    exceptions=[
        "Paragraph 454Y(4): calls or visits outside the window are permitted only where the borrower / guarantor has expressly given a request or authorisation.",
        "Paragraph 454Y(4): a borrower's request to avoid contact at a particular time shall be honoured in normal circumstances.",
    ],
    prohibited_action=["contact_outside_permitted_hours"],
    required_action=["end_call"],
    check_id="calling_hours",
    parameters={
        "earliest_local_time": "08:00:00",
        "latest_local_time": "19:00:00",
        "boundary_semantics": "inclusive_endpoints",
        "express_authorisation_overrides": True,
    },
    notes=(
        "Stated as clock hours rather than 'before 8:00 a.m. / after 7:00 p.m.', and with no microfinance "
        "carve-out, so from 2027-01-01 one window covers every product. The express-authorisation exception "
        "is honoured only when the backend sets borrower_authorised_out_of_hours."
    ),
)

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454Y-6-AVOID-INAPPROPRIATE-OCCASIONS",
    title="Avoid bereavement, medical emergencies and similar occasions",
    doc=AMD,
    paragraph="454Y(6)",
    anchor=("An employee / recovery agent shall avoid inappropriate occasions such as bereavement in the family",
            "for making calls / visits to recover loan dues from a borrower / guarantor."),
    status="future_effective",
    enforcement="requires_external_data",
    severity="high",
    condition="The bank knows of a bereavement, medical emergency, calamity or marriage function.",
    applies_when=ALL_PRODUCTS,
    notes=(
        "No automated check: nothing in the current context model carries this signal, and inferring it "
        "from conversation would be a guess. The backend must supply it before this can be enforced."
    ),
)

_HARSH = [
    ("454Z-1-NO-MINATORY-OR-ABUSIVE-LANGUAGE", "454Z(1)", "(1) Use of minatory or abusive language;",
     ["abusive_language", "intimidation"], "critical", "minatory or abusive language"),
    ("454Z-2-NO-SOCIAL-MEDIA-EXPOSURE", "454Z(2)",
     "(2) Use of social media for posting video / audio recordings or personal details of the borrower / guarantor;",
     ["social_media_exposure", "public_humiliation", "unauthorised_disclosure"], "critical",
     "posting borrower recordings or personal details on social media"),
    ("454Z-3-NO-INAPPROPRIATE-MESSAGES", "454Z(3)",
     "(3) Sending inappropriate messages either on mobile or through social media;",
     ["inappropriate_messaging"], "high", "inappropriate messages on mobile or social media"),
    ("454Z-4-NO-EXCESSIVE-OR-OUT-OF-HOURS-CONTACT", "454Z(4)",
     "(4) Excessively calling / messaging to the borrower / guarantor and / or calling / messaging outside the prescribed hours;",
     ["persistent_calling", "contact_outside_permitted_hours"], "high",
     "excessive contact, or contact outside the prescribed hours"),
    ("454Z-5-NO-THREATENING-OR-ANONYMOUS-CALLS", "454Z(5)",
     "(5) Making threatening and / or anonymous calls;",
     ["threatening_call", "anonymous_call"], "critical", "threatening or anonymous calls"),
    ("454Z-6-NO-INTIMIDATION-OR-HARASSMENT", "454Z(6)",
     "(6) Intimidating or harassing the borrower / guarantor and / or his / her relatives, referees, friends, or co-workers",
     ["intimidation", "verbal_harassment", "physical_harassment", "public_humiliation", "privacy_intrusion_third_party"],
     "critical", "intimidating or harassing the borrower, relatives, referees, friends or co-workers"),
    ("454Z-7-NO-VIOLENCE", "454Z(7)",
     "(7) Use or threat of use of violence or other similar means to harm the borrower / guarantor or their family / assets / reputation;",
     ["violence_threat", "physical_harassment"], "critical", "use or threat of violence"),
    ("454Z-8-NO-FALSE-REPRESENTATION", "454Z(8)",
     "(8) Making false or misleading representations to the borrower / guarantor, especially about the extent of the debt or the consequences of non-repayment.",
     ["false_or_misleading_representation"], "critical",
     "false or misleading representations about the debt or the consequences of non-repayment"),
]

for suffix, para, anchor_start, prohibited, sev, plain in _HARSH:
    rid = f"RBI-CB-RBC-AMD4-2026-{suffix}"
    end = None
    if suffix.startswith("454Z-6"):
        end = "intruding upon their privacy;"
    checks = "persistent_calling" if suffix.startswith("454Z-4") else "prohibited_conduct"
    enf = "requires_external_data" if suffix.startswith("454Z-4") else "conversation_deterministic"
    params = {"max_contacts_per_day": None} if suffix.startswith("454Z-4") else {}
    rule(
        rule_id=rid,
        title=f"Harsh practice prohibited: {plain}",
        doc=AMD,
        paragraph=para,
        anchor=(anchor_start, end),
        status="future_effective",
        enforcement=enf,
        severity=sev,
        condition="Always, for every utterance and every contact attempt.",
        applies_when=ALL_PRODUCTS,
        prohibited_action=prohibited,
        check_id=checks,
        parameters=params,
    )

rule(
    rule_id="RBI-CB-RBC-AMD4-2026-454AA-GRIEVANCE-OFFICER-DETAILS",
    title="Recovery communications must carry grievance redressal officer details",
    doc=AMD,
    paragraph="454AA",
    anchor=("A bank shall have a dedicated mechanism for redressal of recovery related grievances.",
            "whom the borrower / guarantor can contact."),
    status="future_effective",
    enforcement="organisational_obligation",
    severity="medium",
    condition="Every recovery related communication issued by the bank.",
    applies_when=ALL_PRODUCTS,
    notes=(
        "Written for issued communications. Whether a spoken call must recite the officer's details is "
        "not stated and requires regulatory review."
    ),
)

# --------------------------------------------------------------------------
# Historical / provenance. Never enforced.
# --------------------------------------------------------------------------
rule(
    rule_id="RBI-2022-108-2-HARASSMENT-AND-CALLING-HOURS",
    title="2022 recovery agent circular: harassment and calling-hour instruction",
    doc="rbi_recovery_agents_2022",
    paragraph="2",
    anchor=("the REs shall strictly ensure that they or their agents do not resort to intimidation or harassment",
            "making false and misleading representations, etc."),
    status="historical_supporting",
    enforcement="organisational_obligation",
    severity="critical",
    condition="Provenance only. Superseded in operative effect by RBC 2025 paragraph 445 for commercial banks.",
    applies_when=ALL_PRODUCTS,
    exceptions=[
        "Paragraph 6: does not apply to microfinance loans covered by the Microfinance Loans Directions, 2022.",
    ],
    notes=(
        "Paragraph 3 states these instructions supplement and are read with existing RBI directions, which "
        "is why the 2006 and 2008 instruments are retained in this corpus. Paragraph 5 lists the regulated "
        "entities covered, including all commercial banks other than payments banks. Retained for lineage; "
        "the enforced text is RBC 2025 paragraph 445."
    ),
)

rule(
    rule_id="RBI-2008-RECOVERY-AGENTS-X-TRAINING-LINEAGE",
    title="2008 recovery agents circular: training on hours of calling and customer privacy",
    doc="rbi_recovery_agents_banks_2008",
    paragraph="(x)",
    anchor=("hours of calling, privacy of customer information etc.", None),
    status="historical_supporting",
    enforcement="organisational_obligation",
    severity="low",
    condition="Provenance only.",
    applies_when=ALL_PRODUCTS,
    notes="Ancestor of RBC 2025 paragraph 447. Text extracted from a PDF; line breaks follow the PDF layout.",
)

rule(
    rule_id="RBI-2006-OUTSOURCING-5-7-3-HARASSMENT-LINEAGE",
    title="2006 outsourcing guidelines: no intimidation or harassment in debt collection",
    doc="rbi_outsourcing_banks_2006",
    paragraph="5.7.3",
    anchor=("should not resort to intimidation or harassment of any kind", None),
    status="historical_supporting",
    enforcement="organisational_obligation",
    severity="low",
    condition="Provenance only.",
    applies_when=ALL_PRODUCTS,
    notes="Earliest text in this corpus carrying the harassment prohibition now in RBC 2025 paragraph 445.",
)


def build() -> dict:
    out = []
    for r in RULES:
        doc = DOCS[r["doc"]]
        start, end = r["anchor"]
        quote = slice_quote(r["doc"], start, end, r["rule_id"])
        entry = {
            "rule_id": r["rule_id"],
            "title": r["title"],
            "source": {
                "document_id": r["doc"],
                "document_title": doc["title"],
                "paragraph": r["paragraph"],
                "text_file": doc["file"],
                "quote": quote,
                "url": doc["url"],
            },
            "status": r["status"],
            "enforcement": r["enforcement"],
            "severity": r["severity"],
            "effective_from": r.get("effective_from", doc["from"]),
            "effective_until": r.get("effective_until"),
            "superseded_by": r.get("superseded_by", []),
            "condition": r["condition"],
            "applies_when": r["applies_when"],
            "prohibited_action": r.get("prohibited_action", []),
            "required_action": r.get("required_action", []),
            "exceptions": r.get("exceptions", []),
            "check_id": r.get("check_id"),
            "parameters": r.get("parameters", {}),
            "notes": r.get("notes"),
        }
        if r.get("provenance"):
            entry["provenance"] = r["provenance"]
        out.append(entry)
    return {
        "schema_version": "1.0",
        "rules_version": "0.1.0",
        "generated_on": "2026-09-24",
        "corpus_catalog": "data/regulatory/catalog.json",
        "corpus_as_of": "2026-09-24",
        "scope": "Commercial bank, debt/loan recovery, outbound voice recovery agent, India.",
        "notes": (
            "Every quote is a verbatim substring of the referenced local text file; "
            "tests/test_regulatory_rules.py enforces that. Rules are encoded only where the supplied corpus "
            "states them. 'enforcement' records how far each rule can be decided during a live call: "
            "organisational obligations are recorded for audit and are never evaluated by the policy engine. "
            "The Fourth Amendment Directions, 2026 delete RBC 2025 paragraphs 408-416 and 442-454 from "
            "2027-01-01; those rules therefore carry effective_until 2026-12-31 and the replacement rules "
            "carry effective_from 2027-01-01."
        ),
        "rules": out,
    }


payload = build()
if _failures:
    print("ANCHOR FAILURES:", file=sys.stderr)
    for f in _failures:
        print("  " + f, file=sys.stderr)
    sys.exit(1)

target = ROOT / "data/regulatory/rules/recovery_rules.json"
target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {len(payload['rules'])} rules to {target}")
