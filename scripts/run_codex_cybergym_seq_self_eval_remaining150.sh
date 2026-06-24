#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
RUN_ID="${RUN_ID:-codex-gateway-eval-seq-self-remaining150-${RUN_STAMP}}"
LOG_DIR="runs/codex_cybergym/sequential_self_logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"
OUTPUT="${OUTPUT:-runs/codex_gateway_eval_seq_self_involving_eval_remaining150_rollouts.jsonl}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"

echo "{\"event\":\"seq_self_eval_remaining150_start\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"task_offset\":50,\"max_tasks\":150,\"workers\":6,\"max_attempts\":${MAX_ATTEMPTS},\"experience_token_budget\":2048,\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"

.venv/bin/python scripts/run_codex_cybergym_sequential_self_involving.py \
  --run-id "$RUN_ID" \
  --split eval \
  --difficulty level1 \
  --output "$OUTPUT" \
  --task-offset 50 \
  --max-tasks 150 \
  --workers 6 \
  --max-attempts "$MAX_ATTEMPTS" \
  --experience-token-budget 2048 \
  --model gpt-5.5 \
  --summary-model gpt-5.5 >> "$LOG_FILE" 2>&1

exit_code=$?
echo "{\"event\":\"seq_self_eval_remaining150_complete\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"
exit "$exit_code"
