#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"

source scripts/mastermind_env.sh

RUN_ID="${RUN_ID:-claude-glm51-eval200-iterative-r1-20260620T0801Z}"
RUN_ROOT="${RUN_ROOT:-runs/claude_cybergym}"
OUTPUT="${OUTPUT:-runs/claude_glm51_eval200_iterative_r1_20260620T0801Z_rollouts.jsonl}"
WORKERS="${WORKERS:-4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
MAX_TASKS="${MAX_TASKS:-200}"
MODEL="${MODEL:-glm-5.1}"
MODEL_REASONING_EFFORT="${MODEL_REASONING_EFFORT:-}"
SUMMARY_MODEL="${SUMMARY_MODEL:-glm-5.1}"
SUMMARY_REASONING_EFFORT="${SUMMARY_REASONING_EFFORT:-}"
EXPERIENCE_TOKEN_BUDGET="${EXPERIENCE_TOKEN_BUDGET:-2048}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-900}"
SUMMARY_TIMEOUT_SECONDS="${SUMMARY_TIMEOUT_SECONDS:-600}"
SUBMIT_TIMEOUT_SECONDS="${SUBMIT_TIMEOUT_SECONDS:-1200}"
CODEX_LAUNCH_STAGGER_SECONDS="${CODEX_LAUNCH_STAGGER_SECONDS:-5}"
MAX_CONSECUTIVE_INFRA_FAILURES="${MAX_CONSECUTIVE_INFRA_FAILURES:-5}"
INFRA_FAILURE_PAUSE_SECONDS="${INFRA_FAILURE_PAUSE_SECONDS:-1800}"

if [[ -z "${LLM_GATEWAY_API_KEY:-}" ]]; then
  echo "LLM_GATEWAY_API_KEY is not set" >&2
  exit 2
fi
if [[ -z "${LLM_GATEWAY_URL:-}" ]]; then
  echo "LLM_GATEWAY_URL is not set" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export ANTHROPIC_API_KEY="$LLM_GATEWAY_API_KEY"
export ANTHROPIC_BASE_URL="${LLM_GATEWAY_URL%/v1}"

.venv/bin/python scripts/run_codex_cybergym_sequential_self_involving.py \
  --run-id "$RUN_ID" \
  --run-root "$RUN_ROOT" \
  --split eval \
  --difficulty level1 \
  --output "$OUTPUT" \
  --workers "$WORKERS" \
  --max-attempts "$MAX_ATTEMPTS" \
  --max-tasks "$MAX_TASKS" \
  --model "$MODEL" \
  --model-reasoning-effort "$MODEL_REASONING_EFFORT" \
  --summary-model "$SUMMARY_MODEL" \
  --summary-reasoning-effort "$SUMMARY_REASONING_EFFORT" \
  --experience-updater claude \
  --experience-token-budget "$EXPERIENCE_TOKEN_BUDGET" \
  --runner scripts/run_claude_cybergym_tasks.py \
  --codex-bin claude \
  --codex-timeout-seconds "$CODEX_TIMEOUT_SECONDS" \
  --summary-timeout-seconds "$SUMMARY_TIMEOUT_SECONDS" \
  --submit-timeout-seconds "$SUBMIT_TIMEOUT_SECONDS" \
  --server "${CYBERGYM_SERVER:-http://127.0.0.1:8666}" \
  --pocdb-path "${POCDB_PATH:-runs/cybergym_server/poc.db}" \
  --env-file .env \
  --max-consecutive-infra-failures "$MAX_CONSECUTIVE_INFRA_FAILURES" \
  --infra-failure-pause-seconds "$INFRA_FAILURE_PAUSE_SECONDS" \
  --codex-launch-stagger-seconds "$CODEX_LAUNCH_STAGGER_SECONDS" \
  --codex-rate-limit-retries 3 \
  --codex-rate-limit-stagger-seconds 5 \
  --codex-rate-limit-min-sleep-seconds 0 \
  --anthropic-api-key-env-key LLM_GATEWAY_API_KEY \
  --anthropic-base-url-env-key LLM_GATEWAY_URL
