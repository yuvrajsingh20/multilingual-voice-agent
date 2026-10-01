#!/usr/bin/env bash
# Runs every Track 1 measurement one after another. Ollama serves one stream at a time on this
# CPU, so any concurrent call inflates the latency numbers; start nothing else while this runs.
# Every step resumes from where it stopped, so the script can simply be re-run.
# The 4B block runs first so a complete set of 4B results exists even if the 9B block is cut short.
#
#   systemd-inhibit --what=sleep:idle --why="Track 1 eval" scripts/track1_queue.sh >> data/eval/track1/queue.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
log() { echo "[$(date -Is)] $*"; }

step() {
  log "START $*"
  "$@" || log "FAILED ($?) $*"
  log "END $*"
}

for model in qwen-voice-4b qwen-voice-9b; do
  step $PY scripts/track1_ps3.py run --model "$model"
  step $PY scripts/track1_ps3.py score --model "$model"
  step $PY scripts/track1_ps2.py run --model "$model"
  step $PY scripts/track1_ps2.py score --model "$model"
  step $PY scripts/track1_ps1.py run --model "$model"
  step $PY scripts/track1_ps1.py judge --model "$model" --judge qwen-voice-9b
  step $PY scripts/track1_ps1.py score --model "$model" --judge qwen-voice-9b
done
log "QUEUE DONE"
