#!/usr/bin/env bash
# Iterative Improvement Experiment on the first 100 training tasks.
#
# Mirrors the eval50/remaining150 setup but targets --split train with
# --max-tasks 100. Resumes automatically on re-run (tasks already PASSED
# or already at max_attempts are skipped).
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

PY="${PYTHON_BIN:-.venv/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
RUN_ID="${RUN_ID:-codex-gateway-train-seq-self-100-${RUN_STAMP}}"
OUTPUT="${OUTPUT:-runs/codex_gateway_train_seq_self_involving_train100_rollouts.jsonl}"
LOG_DIR="runs/codex_cybergym/sequential_self_logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"
WORKERS="${WORKERS:-6}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
PROVIDER="${CODEX_PROVIDER:-llmgw}"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"

echo "{\"event\":\"seq_self_train100_start\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"workers\":${WORKERS},\"max_attempts\":${MAX_ATTEMPTS},\"split\":\"train\",\"max_tasks\":100,\"provider\":\"${PROVIDER}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"

"$PY" scripts/run_codex_cybergym_sequential_self_involving.py \
    --run-id "$RUN_ID" \
    --split train \
    --difficulty level1 \
    --output "$OUTPUT" \
    --task-offset 0 \
    --max-tasks 100 \
    --workers "$WORKERS" \
    --max-attempts "$MAX_ATTEMPTS" \
    --experience-token-budget 2048 \
    --model gpt-5.5 \
    --summary-model gpt-5.5 \
    --codex-provider "$PROVIDER" \
    >> "$LOG_FILE" 2>&1

exit_code=$?
echo "{\"event\":\"seq_self_train100_complete\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"
exit "$exit_code"
