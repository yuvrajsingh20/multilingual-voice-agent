# Regulatory source inventory (commercial bank)

Bank-only pin list for a debt / loan recovery voice agent (Hindi / English / Marathi / Hinglish).

Canonical layered inventory: [`regulatory/inventory/sources.json`](../../regulatory/inventory/sources.json).

## Layers

| Layer | What goes here |
| --- | --- |
| `current_applicable` | In-force commercial-bank instruments to extract for call behavior |
| `conditional_product` | Apply only when the loan is digital lending or card dues |
| `future_effective` | Store now; enforce only from the stated date (1 Jan 2027 recovery amendment) |
| `historical_provenance` | FAQ-cited / consolidated instruments (e.g. 2022 recovery-agent circular) — audit lineage, not current rules |
| `historical_supporting` | 2003/2006/2008 background — do not put into primary rules |

## Next step

Download and ingest the pinned URLs into the local regulatory corpus, keeping current vs historical files distinct.
