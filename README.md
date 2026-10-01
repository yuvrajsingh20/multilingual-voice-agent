# Multilingual Collections Voice Agent — Predixion Open-Weight Challenge, Track 1

This repository answers Track 1 of the Predixion Open-Weight Collections Agent Challenge
(v1.0, 28 August 2026): can an open-weight model run a compliant collections call in Hindi,
Hinglish and Marathi?

It contains two things:

1. **A collections voice agent backend** (`app/`). The model only proposes words and tool calls.
   Policy, tool execution, grounding of amounts and dates, and whether a sentence may be spoken
   are decided by code, never by the model.
2. **The Track 1 evaluation harness** (`app/evaluation/`, `scripts/track1_*.py`) for PS-1, PS-2 and
   PS-3, run against Qwen3.5-4B and Qwen3.5-9B at Q4 on a CPU-only laptop.

Results, methodology and limitations are in [`TRACK1_REPORT.md`](TRACK1_REPORT.md). The design of
the agent itself is in [`REPORT.md`](REPORT.md).

## What was run, and on what

| Item | Value |
|---|---|
| Hardware | Intel Core i3-1115G4 (2 cores, 4 threads), 15 GiB RAM, no GPU |
| OS | Zorin OS 17.3 (Ubuntu 22.04 base), Linux 6.8.0-138-generic |
| Python | 3.10.12 |
| Inference server | Ollama 0.35.0, OpenAI-compatible endpoint `http://127.0.0.1:11434/v1` |
| Models | `qwen3.5:4b` and `qwen3.5:9b`, both Q4_K_M, as published by Ollama |
| Aliases | `qwen-voice-4b`, `qwen-voice-9b` (`ollama/Modelfile.*`) |
| Thinking | off on every request (`reasoning_effort: "none"`) |
| Sampling | temperature 0, seed 42 on every request |
| System prompt | the challenge baseline prompt (section 6.4), identical for every model |
| Tool schemas | the five fixed challenge schemas (section 6.3), unmodified |

Exact model digests, the Ollama version, request settings and the SHA-256 of each suite are written
into every run's `manifest.json`.

Q4 is not the precision production uses. Nothing here is presented as a production conclusion.

## Setup from scratch

```bash
git clone https://github.com/yuvrajsingh20/multilingual-voice-agent.git
cd multilingual-voice-agent

python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Ollama 0.35.0: https://ollama.com
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
ollama create qwen-voice-4b -f ollama/Modelfile.qwen-voice-4b
ollama create qwen-voice-9b -f ollama/Modelfile.qwen-voice-9b
```

### Thinking mode

The challenge says to disable thinking with `PARAMETER think false` in the Modelfile. Ollama 0.35.0
rejects that line (`Error: unknown parameter 'think'`), and qwen3.5 uses a built-in renderer, so
there is no template switch either. Thinking is therefore disabled on every request with
`"reasoning_effort": "none"`, which Ollama maps to `think=false`. Every evaluation call records
whether any reasoning text came back (`thinking_leaked`), and the probe in
`data/eval/track1/evidence/thinking_probe.py` shows the difference.

## Reproducing the Track 1 results

Run one model call at a time. Ollama serves one stream at a time on a laptop, so anything running
alongside inflates the latency numbers.

The whole queue, both models, in the order it was run:

```bash
systemd-inhibit --what=sleep:idle --why="Track 1 eval" \
  scripts/track1_queue.sh >> data/eval/track1/queue.log 2>&1
```

On the reference laptop this takes roughly 15 hours in total. Every step appends to its output file
and skips cases already done, so the script can be stopped and re-run.

Individual steps, for one model:

```bash
# PS-3 · tool calls under code-mixing (200 cases)
.venv/bin/python scripts/track1_ps3.py run   --model qwen-voice-4b
.venv/bin/python scripts/track1_ps3.py score --model qwen-voice-4b

# PS-2 · code-mix register (6 calls x 8 turns, at 5, 30 and 90 DPD)
.venv/bin/python scripts/track1_ps2.py run   --model qwen-voice-4b
.venv/bin/python scripts/track1_ps2.py score --model qwen-voice-4b

# PS-1 · guardrail gauntlet (168 adversarial turns, judged by the 9B)
.venv/bin/python scripts/track1_ps1.py run   --model qwen-voice-4b
.venv/bin/python scripts/track1_ps1.py judge --model qwen-voice-4b --judge qwen-voice-9b
.venv/bin/python scripts/track1_ps1.py score --model qwen-voice-4b --judge qwen-voice-9b

# Result tables for the report, from the saved metrics only
.venv/bin/python scripts/track1_tables.py > data/eval/track1/tables.md
```

`score` never calls a model. It re-reads `raw.jsonl`, so the scoring rules can be checked or changed
without re-running anything.

### Human validation

```bash
# PS-1: blinded sample, label it, then compare the judge with the humans
.venv/bin/python scripts/track1_ps1.py sample --models qwen-voice-4b qwen-voice-9b
.venv/bin/python scripts/track1_ps1.py label  --rater <your-name>
.venv/bin/python scripts/track1_ps1.py agree

# PS-2: blinded rating sheet; two raters fill it, then agreement
.venv/bin/python scripts/track1_ps2.py sheet --models qwen-voice-4b qwen-voice-9b
.venv/bin/python scripts/track1_ps2.py agree
```

### A hosted baseline

Every `run` and `judge` command accepts another OpenAI-compatible endpoint:

```bash
BASELINE_API_KEY=... .venv/bin/python scripts/track1_ps3.py run \
  --model <model-id> --base-url https://<host>/v1 \
  --api-key-env BASELINE_API_KEY --no-reasoning-effort
```

Challenge section 8 allows one declared hosted baseline, and no model API hosted outside India.
Whether a baseline was run is stated in `TRACK1_REPORT.md`.

## Where the data is

```
data/eval/track1/
  ps1/  suite.jsonl  taxonomy.json  judge_prompt.txt  runs/<model>/
  ps2/  calls.jsonl  rubric.json                        runs/<model>/
  ps3/  suite.jsonl                                     runs/<model>/
  evidence/   thinking probes and live application test logs
  queue.log   the full log of the measured runs
```

Each `runs/<model>/` directory holds:

| File | Contents |
|---|---|
| `manifest.json` | model digest, Ollama version, request settings, suite hash, timestamps |
| `raw.jsonl` | the model output exactly as streamed, one line per case |
| `scores.jsonl` | one verdict per case, from the scorer |
| `metrics.json` | the aggregates the report tables are rendered from |

All suites are synthetic. No real borrower data is used anywhere.

## Tests

```bash
# Deterministic suite: no network, no model
.venv/bin/python -m pytest

# The application driven by a real local model (slow; needs Ollama running)
LIVE_OLLAMA_MODEL=qwen-voice-4b .venv/bin/python -m pytest tests/live -v -s
```

## Running the API

```bash
cp .env.example .env
# For the local model, set in .env:
#   MODEL_BASE_URL=http://127.0.0.1:11434/v1
#   MODEL_NAME=qwen-voice-4b
#   MODEL_REASONING_EFFORT=none
#   MODEL_TIMEOUT_SECONDS=120
.venv/bin/uvicorn app.main:app --reload
```

With `MODEL_BASE_URL` and `MODEL_NAME` left empty, no model is connected and no HTTP client is
created. Telephony, speech-to-text, text-to-speech and the banking backend are interfaces only.

## Disclosure

The harness, the suites and the reports were written with an AI coding assistant, which materially
shaped the methodology. The Hindi, Hinglish, Marathi and Marathi-English utterances were also
written by the assistant and have not been reviewed by a native speaker.

## Licence

Released under Apache 2.0, as the challenge requires.
