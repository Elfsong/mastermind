#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MASTERMIND_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT"

source scripts/mastermind_env.sh

PREV_GRPO_RUN_DIR="${PREV_GRPO_RUN_DIR:?Set PREV_GRPO_RUN_DIR to the completed/running GRPO run directory.}"
EVAL_REPORT="${EVAL_REPORT:?Set EVAL_REPORT to the current eval report JSON path, usually reports/sft_vs_grpo.json.}"
TARGETED_TASK_IDS_FILE="${TARGETED_TASK_IDS_FILE:?Set TARGETED_TASK_IDS_FILE to the targeted task list.}"

POLL_SECONDS="${POLL_SECONDS:-120}"
PASS_THRESHOLD="${PASS_THRESHOLD:-172}"
MIN_EVAL_TASKS="${MIN_EVAL_TASKS:-200}"
TASK_OFFSET="${TASK_OFFSET:-16}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
NEXT_RUN_ID="${NEXT_RUN_ID:-qwen36-strategy-grpo-targeted-offset${TASK_OFFSET}-${RUN_STAMP}}"
NEXT_RUN_DIR="${NEXT_RUN_DIR:-runs/strategy_grpo/${NEXT_RUN_ID}}"
NEXT_EVAL_RUN_ID="${NEXT_EVAL_RUN_ID:-codex-gpt55-eval200-qwen36-grpo-targeted-offset${TASK_OFFSET}-${RUN_STAMP}}"
LOG_FILE="${LOG_FILE:-runs/strategy_grpo/${NEXT_RUN_ID}.orchestrator.log}"

ADVANTAGE_GROUP_SIZE="${ADVANTAGE_GROUP_SIZE:-8}"
ROLLOUT_POOL_PER_TASK="${ROLLOUT_POOL_PER_TASK:-16}"
TASKS_PER_STEP="${TASKS_PER_STEP:-2}"
MAX_STEPS="${MAX_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
LOSS_FN="${LOSS_FN:-ppo}"
GROUPS_PER_UPDATE="${GROUPS_PER_UPDATE:-2}"
EXECUTOR_WORKERS="${EXECUTOR_WORKERS:-8}"
EXECUTOR_MODEL="${EXECUTOR_MODEL:-gpt-5.5}"
CODEX_PROVIDER="${CODEX_PROVIDER:-llmgw}"
EVAL_WORKERS="${EVAL_WORKERS:-${WORKERS:-8}}"

mkdir -p "$(dirname "$LOG_FILE")"

log_event() {
  local event="$1"
  shift || true
  printf '{"event":"%s","time":"%s"' "$event" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
  while [[ $# -gt 0 ]]; do
    local key="$1"
    local value="$2"
    shift 2
    printf ',"%s":%s' "$key" "$value" | tee -a "$LOG_FILE"
  done
  printf '}\n' | tee -a "$LOG_FILE"
}

json_string() {
  .venv/bin/python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

json_number() {
  local path="$1"
  local expr="$2"
  .venv/bin/python - "$path" "$expr" <<'PY'
import json
import sys

path, expr = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)

cur = data
for part in expr.split("."):
    if not part:
        continue
    cur = cur[part]
print(cur)
PY
}

extract_final_state() {
  .venv/bin/python - "$PREV_GRPO_RUN_DIR/result.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
value = data.get("final_state_path")
if not value:
    raise SystemExit("result.json is missing final_state_path")
print(value)
PY
}

log_event "watch_start" \
  "prev_grpo_run_dir" "$(json_string "$PREV_GRPO_RUN_DIR")" \
  "eval_report" "$(json_string "$EVAL_REPORT")" \
  "pass_threshold" "$PASS_THRESHOLD" \
  "task_offset" "$TASK_OFFSET" \
  "next_run_id" "$(json_string "$NEXT_RUN_ID")" \
  "next_eval_run_id" "$(json_string "$NEXT_EVAL_RUN_ID")"

while [[ ! -s "$EVAL_REPORT" ]]; do
  log_event "waiting_for_eval_report" "eval_report" "$(json_string "$EVAL_REPORT")"
  sleep "$POLL_SECONDS"
done

grpo_passed="$(json_number "$EVAL_REPORT" "individual.grpo.passed")"
grpo_tasks="$(json_number "$EVAL_REPORT" "individual.grpo.tasks")"
log_event "eval_report_ready" "grpo_passed" "$grpo_passed" "grpo_tasks" "$grpo_tasks"

if (( grpo_tasks >= MIN_EVAL_TASKS && grpo_passed >= PASS_THRESHOLD )); then
  log_event "threshold_met" "grpo_passed" "$grpo_passed" "pass_threshold" "$PASS_THRESHOLD"
  exit 0
fi

while [[ ! -s "$PREV_GRPO_RUN_DIR/result.json" ]]; do
  log_event "waiting_for_prev_grpo_result" "prev_grpo_run_dir" "$(json_string "$PREV_GRPO_RUN_DIR")"
  sleep "$POLL_SECONDS"
done

init_state_path="$(extract_final_state)"
log_event "starting_continuation_grpo" \
  "init_state_path" "$(json_string "$init_state_path")" \
  "next_run_dir" "$(json_string "$NEXT_RUN_DIR")"

if [[ ! -s "$NEXT_RUN_DIR/result.json" ]]; then
  RUN_ID="$NEXT_RUN_ID" \
  RUN_DIR="$NEXT_RUN_DIR" \
  INIT_STATE_PATH="$init_state_path" \
  TASK_IDS_FILE="$TARGETED_TASK_IDS_FILE" \
  TASK_OFFSET="$TASK_OFFSET" \
  SPLIT=eval \
  DIFFICULTY=level1 \
  TASK_SAMPLING=sequential \
  ADVANTAGE_GROUP_SIZE="$ADVANTAGE_GROUP_SIZE" \
  ROLLOUT_POOL_PER_TASK="$ROLLOUT_POOL_PER_TASK" \
  TASKS_PER_STEP="$TASKS_PER_STEP" \
  MAX_STEPS="$MAX_STEPS" \
  LEARNING_RATE="$LEARNING_RATE" \
  LOSS_FN="$LOSS_FN" \
  GROUPS_PER_UPDATE="$GROUPS_PER_UPDATE" \
  EXECUTOR_WORKERS="$EXECUTOR_WORKERS" \
  EXECUTOR_MODEL="$EXECUTOR_MODEL" \
  CODEX_PROVIDER="$CODEX_PROVIDER" \
  FOREGROUND=1 \
  scripts/run_strategy_grpo_background.sh
else
  log_event "continuation_grpo_already_complete" "next_run_dir" "$(json_string "$NEXT_RUN_DIR")"
fi

log_event "starting_continuation_eval" \
  "next_run_dir" "$(json_string "$NEXT_RUN_DIR")" \
  "next_eval_run_id" "$(json_string "$NEXT_EVAL_RUN_ID")"

GRPO_RUN_DIR="$NEXT_RUN_DIR" \
EVAL_RUN_ID="$NEXT_EVAL_RUN_ID" \
GRPO_WAIT_TARGET=final \
WORKERS="$EVAL_WORKERS" \
EXECUTOR_MODEL="$EXECUTOR_MODEL" \
CODEX_PROVIDER="$CODEX_PROVIDER" \
scripts/watch_strategy_grpo_then_eval200.sh

log_event "done" "next_eval_run_id" "$(json_string "$NEXT_EVAL_RUN_ID")"
