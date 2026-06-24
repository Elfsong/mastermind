#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"

SFT_RUN_ID="${SFT_RUN_ID:-codex-gpt55-eval100-tinker-qwen36-sft-failed8-rerun-20260602T1200Z}"
SFT_LOG="${SFT_LOG:-runs/codex_cybergym/sequential_self_logs/${SFT_RUN_ID}.log}"
LOG_DIR="runs/codex_cybergym/sequential_self_logs"
CHAIN_STAMP="${CHAIN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
CHAIN_LOG="${CHAIN_LOG:-${LOG_DIR}/chain-sft-failed8-then-vanilla-qwen36-${CHAIN_STAMP}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/chain-sft-failed8-then-vanilla-qwen36-${CHAIN_STAMP}.pid}"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"
exec >> "$CHAIN_LOG" 2>&1

echo "{\"event\":\"chain_wait_start\",\"sft_run_id\":\"${SFT_RUN_ID}\",\"sft_log\":\"${SFT_LOG}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"

while true; do
  if grep -q '"event": "sequential_complete"' "$SFT_LOG" && grep -q "\"run_id\": \"${SFT_RUN_ID}\"" "$SFT_LOG"; then
    break
  fi
  if grep -q '"event":"sequential_complete"' "$SFT_LOG" && grep -q "\"run_id\":\"${SFT_RUN_ID}\"" "$SFT_LOG"; then
    break
  fi
  if ! pgrep -f "scripts/run_codex_cybergym_sequential_self_involving.py --run-id ${SFT_RUN_ID}" >/dev/null; then
    echo "{\"event\":\"chain_wait_blocked\",\"reason\":\"sft_runner_not_found_before_complete\",\"sft_run_id\":\"${SFT_RUN_ID}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
    exit 1
  fi
  sleep 60
done

RUN_STAMP="$(date -u +%Y%m%dT%H%MZ)"
export RUN_STAMP
export VARIANT="vanilla"
export RUN_ID="codex-gpt55-eval100-tinker-qwen36-vanilla-${RUN_STAMP}"
export OUTPUT="runs/codex_gpt55_eval100_tinker_qwen36_vanilla_${RUN_STAMP}_rollouts.jsonl"
export MAX_TASKS="${MAX_TASKS:-100}"
export MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
export WORKERS="${WORKERS:-6}"
export TINKER_SUMMARY_MODEL="${TINKER_SUMMARY_MODEL:-Qwen/Qwen3.6-35B-A3B}"
unset TINKER_SUMMARY_MODEL_PATH

echo "{\"event\":\"chain_launch_vanilla_qwen36\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"workers\":${WORKERS},\"max_tasks\":${MAX_TASKS},\"max_attempts\":${MAX_ATTEMPTS},\"tinker_summary_model\":\"${TINKER_SUMMARY_MODEL}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"

scripts/run_codex_cybergym_seq_self_eval100_tinker_qwen36_matched.sh
exit_code=$?
echo "{\"event\":\"chain_complete\",\"vanilla_run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
exit "$exit_code"
