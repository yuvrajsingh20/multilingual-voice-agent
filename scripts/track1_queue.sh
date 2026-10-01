#!/usr/bin/env bash
# Runs every Track 1 measurement one after another. Ollama serves one stream at a time on this
# CPU, so any concurrent call inflates the latency numbers; start nothing else while this runs.
# Every step resumes from where it stopped, so the script can simply be re-run.
#
# Models are unloaded before each switch: on 15 GiB, the 4B still resident while the 9B loads
# got llama-server OOM-killed (2026-10-01 13:24). An endpoint failure aborts the step
# (EndpointUnavailable) rather than writing errors as results.
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

unload() {
  for m in $(ollama ps | awk 'NR>1 {print $1}'); do ollama stop "$m"; done
  log "UNLOADED; $(free -m | awk '/Mem:/ {print $7 " MB available"}')"
}

measure() {
  local model=$1
  unload
  step $PY scripts/track1_ps3.py run --model "$model"
  step $PY scripts/track1_ps3.py score --model "$model"
  step $PY scripts/track1_ps2.py run --model "$model"
  step $PY scripts/track1_ps2.py score --model "$model"
  step $PY scripts/track1_ps1.py run --model "$model"
}

judge() {
  local model=$1
  unload
  step $PY scripts/track1_ps1.py judge --model "$model" --judge qwen-voice-9b
  step $PY scripts/track1_ps1.py score --model "$model" --judge qwen-voice-9b
}

measure qwen-voice-4b
judge qwen-voice-4b
measure qwen-voice-9b
judge qwen-voice-9b
log "QUEUE DONE"
