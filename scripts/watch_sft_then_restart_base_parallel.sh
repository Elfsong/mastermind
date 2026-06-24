#!/usr/bin/env bash
set -euo pipefail

ROOT="/data00/home/mz.du/Projects/mastermind"
cd "$ROOT"

SFT_RUN_ID="codex-gpt55-eval100-tinker-qwen36-sft-parallel-20260601T2115Z"
BASE_RUN_ID="codex-gpt55-eval100-tinker-qwen36-base-clean-20260601T2225Z"
WATCH_LOG="runs/codex_cybergym/sequential_self_logs/watch_sft_then_restart_base_parallel.log"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

mkdir -p "$(dirname "$WATCH_LOG")"
log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$WATCH_LOG"
}

run_pids() {
  local run_id="$1"
  pgrep -u "$(whoami)" -f "$run_id" || true
}

kill_run_by_id() {
  local run_id="$1"
  local pids pgids pgid
  pids="$(run_pids "$run_id")"
  if [[ -z "$pids" ]]; then
    log "no processes found for $run_id"
    return 0
  fi

  pgids="$(for pid in $pids; do ps -o pgid= -p "$pid" 2>/dev/null || true; done | awk '{print $1}' | sort -n | uniq)"
  log "stopping run_id=$run_id pids=$(echo "$pids" | tr '\n' ' ') pgids=$(echo "$pgids" | tr '\n' ' ')"
  for pgid in $pgids; do
    kill -TERM "-$pgid" 2>/dev/null || true
  done
  sleep 10

  pids="$(run_pids "$run_id")"
  if [[ -n "$pids" ]]; then
    pgids="$(for pid in $pids; do ps -o pgid= -p "$pid" 2>/dev/null || true; done | awk '{print $1}' | sort -n | uniq)"
    log "force-stopping remaining run_id=$run_id pids=$(echo "$pids" | tr '\n' ' ') pgids=$(echo "$pgids" | tr '\n' ' ')"
    for pgid in $pgids; do
      kill -KILL "-$pgid" 2>/dev/null || true
    done
  fi
}

log "watcher_start sft_run_id=$SFT_RUN_ID base_run_id=$BASE_RUN_ID interval_seconds=$INTERVAL_SECONDS"
while true; do
  if [[ -z "$(run_pids "$SFT_RUN_ID")" ]]; then
    log "sft_parallel_finished run_id=$SFT_RUN_ID"
    break
  fi
  sleep "$INTERVAL_SECONDS"
done

kill_run_by_id "$BASE_RUN_ID"

STAMP="$(date -u +%Y%m%dT%H%MZ)"
NEW_RUN_ID="codex-gpt55-eval100-tinker-qwen36-base-parallel-${STAMP}"
NEW_OUTPUT="runs/codex_gpt55_eval100_tinker_qwen36_base_parallel_${STAMP}_rollouts.jsonl"
log "starting_base_parallel run_id=$NEW_RUN_ID output=$NEW_OUTPUT workers=8"

nohup env \
  VARIANT="base-parallel" \
  RUN_ID="$NEW_RUN_ID" \
  OUTPUT="$NEW_OUTPUT" \
  WORKERS="8" \
  MAX_TASKS="100" \
  MAX_ATTEMPTS="8" \
  TINKER_SUMMARY_MODEL_PATH="" \
  bash scripts/run_codex_cybergym_seq_self_eval100_tinker_qwen36_matched.sh \
  >> "runs/codex_cybergym/sequential_self_logs/${NEW_RUN_ID}.launcher.log" 2>&1 &
NEW_PID="$!"
log "base_parallel_started pid=$NEW_PID run_id=$NEW_RUN_ID output=$NEW_OUTPUT"
