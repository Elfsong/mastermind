#!/usr/bin/env bash
# Iterative Improvement Experiment — training tasks 101–301 (offset 100, 201 tasks).
# Mirrors run_codex_cybergym_seq_self_train100.sh but covers the rest of TASKS_TRAIN.
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

PY="${PYTHON_BIN:-.venv/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
RUN_ID="${RUN_ID:-codex-gateway-train-seq-self-remaining201-${RUN_STAMP}}"
OUTPUT="${OUTPUT:-runs/codex_gateway_train_seq_self_involving_remaining201_rollouts.jsonl}"
LOG_DIR="runs/codex_cybergym/sequential_self_logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"
WORKERS="${WORKERS:-6}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
PROVIDER="${CODEX_PROVIDER:-llmgw}"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"

echo "{\"event\":\"seq_self_train_remaining201_start\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"workers\":${WORKERS},\"max_attempts\":${MAX_ATTEMPTS},\"split\":\"train\",\"task_offset\":100,\"max_tasks\":201,\"provider\":\"${PROVIDER}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"

"$PY" scripts/run_codex_cybergym_sequential_self_involving.py \
    --run-id "$RUN_ID" \
    --split train \
    --difficulty level1 \
    --output "$OUTPUT" \
    --task-offset 100 \
    --max-tasks 201 \
    --workers "$WORKERS" \
    --max-attempts "$MAX_ATTEMPTS" \
    --experience-token-budget 2048 \
    --model gpt-5.5 \
    --summary-model gpt-5.5 \
    --codex-provider "$PROVIDER" \
    >> "$LOG_FILE" 2>&1

exit_code=$?
echo "{\"event\":\"seq_self_train_remaining201_complete\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"
exit "$exit_code"
