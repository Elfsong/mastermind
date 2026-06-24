#!/usr/bin/env bash
# Rerun five training tasks whose main remaining201 run ended in API/gateway CRASH
# (429 Too Many Requests or stream-disconnect). Lower worker count to avoid re-triggering 429s.
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

PY="${PYTHON_BIN:-.venv/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
RUN_ID="${RUN_ID:-codex-gateway-train-api429-rerun5-${RUN_STAMP}}"
OUTPUT="${OUTPUT:-runs/codex_gateway_train_api429_rerun5_rollouts.jsonl}"
LOG_DIR="runs/codex_cybergym/sequential_self_logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"
WORKERS="${WORKERS:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
PROVIDER="${CODEX_PROVIDER:-llmgw}"

mkdir -p "$LOG_DIR"
exec >> "$LOG_FILE" 2>&1
echo "$$" > "$PID_FILE"
echo "$RUN_ID" > "${LOG_DIR}/api429_rerun5.latest"

echo "{\"event\":\"api429_rerun5_start\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"workers\":${WORKERS},\"max_attempts\":${MAX_ATTEMPTS},\"provider\":\"${PROVIDER}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"

"$PY" scripts/run_codex_cybergym_sequential_self_involving.py \
  --run-id "$RUN_ID" \
  --split train \
  --difficulty level1 \
  --output "$OUTPUT" \
  --task-id arvo:11945 \
  --task-id arvo:14560 \
  --task-id arvo:14574 \
  --task-id arvo:59070 \
  --task-id arvo:61617 \
  --workers "$WORKERS" \
  --max-attempts "$MAX_ATTEMPTS" \
  --experience-token-budget 2048 \
  --model gpt-5.5 \
  --summary-model gpt-5.5 \
  --codex-provider "$PROVIDER"

exit_code=$?
echo "{\"event\":\"api429_rerun5_complete\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
exit "$exit_code"
