# Regulatory corpus

Local copy of the RBI and MeitY instruments that can govern this debt-collection voice agent. Ingested on 24 September 2026 as corpus `2026-09-24.2`.

The two entry points were `https://rbi.org.in/` and `https://मेईटी.सरकार.भारत/` (MeitY). The files themselves come from the English document pages on `www.rbi.org.in` and `www.meity.gov.in`. This is not a copy of either website.

`app/core/policy.py` is unchanged. `applicability.json` names the controls and the paragraph that supports each one. The policy engine is what will enforce them. The model does not decide whether a rule applies.

## Layout

- `sources.json` — the pinned URL list
- `catalog.json` — hash, date, lender, and status for each file
- `applicability.json` — which paragraphs govern a collection call, and from when
- `raw/` — official HTML or PDF snapshot
- `text/` — extracted text used for review
- `refresh.py` — re-downloads the pinned URLs

RBI's PDF host answered this network with a bot challenge. Where that happened, `raw/` holds the official notification or master-direction HTML page, which contains the same instrument. MeitY files are the gazette PDFs.

## Refresh

From the repository root:

```bash
python3 regulatory_corpus/refresh.py
```

If a file hash changes, the script keeps the previous snapshot as `*.before-<date>.*` and sets `refresh_review` on that catalog entry. It does not rewrite policy rules. A person has to read the diff and update `applicability.json` before any rule change.

## What is in force on 24 September 2026

Collection calls are governed by the November 2025 Responsible Business Conduct Directions. For a commercial bank that is paragraph 445, and for an NBFC paragraph 100: no harassment, and no calls before 8:00 a.m. or after 7:00 p.m. Those calling hours do not apply to microfinance loans.

On 1 January 2027 the 6 August 2026 amendments replace those paragraphs. Contact is then allowed only from 08:00 to 19:00 unless the borrower or guarantor has expressly authorised another time.

A digital-lending loan has an extra condition, in both the Digital Lending Directions and paragraph 9(6) of the Credit Facilities Directions: the lender must send the agent's particulars by email or SMS before the agent calls.

The DPDP Act's notice, consent, and security duties are not in force yet. G.S.R. 843(E) brings sections 3 to 17 into force on 13 May 2027.
