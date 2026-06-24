#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

if [ -z "${LITELLM_API_KEY:-}" ] && [ -n "${QWEN36_API_KEY:-}" ]; then
    export LITELLM_API_KEY="$QWEN36_API_KEY"
fi

export LITELLM_BASE_URL="${LITELLM_BASE_URL:-http://litellm.tiktok-row.net/v1}"

START_REP="${1:-1}"
END_REP="${2:-8}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WORKERS="${WORKERS:-1}"
MODEL="${MODEL:-qwen3_6_mingzhe}"
MODEL_REASONING_EFFORT="${MODEL_REASONING_EFFORT:-minimal}"
OPENCODE_PROVIDER="${OPENCODE_PROVIDER:-qwen36_mingzhe}"
OPENCODE_BIN="${OPENCODE_BIN:-runs/opencode_tool/node_modules/.bin/opencode}"
OPENCODE_CONTEXT_LIMIT="${OPENCODE_CONTEXT_LIMIT:-70000}"
OPENCODE_OUTPUT_TOKEN_MAX="${OPENCODE_OUTPUT_TOKEN_MAX:-4096}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-900}"
SUBMIT_TIMEOUT_SECONDS="${SUBMIT_TIMEOUT_SECONDS:-1200}"
MAX_CONSECUTIVE_INFRA_FAILURES="${MAX_CONSECUTIVE_INFRA_FAILURES:-5}"
INFRA_FAILURE_PAUSE_SECONDS="${INFRA_FAILURE_PAUSE_SECONDS:-1800}"
RUN_ROOT="${RUN_ROOT:-/tmp/opencode_qwen36_mingzhe_cybergym}"
SERVER="${CYBERGYM_SERVER:-http://127.0.0.1:8666}"
POCDB_PATH="${POCDB_PATH:-runs/cybergym_server/poc.db}"
ENV_FILE="${ENV_FILE:-.env}"
SANDBOX="${SANDBOX:-workspace-write}"
NAME="${NAME:-opencode_qwen36_mingzhe_eval200_bo8}"
LOG_DIR="${LOG_DIR:-runs/opencode_qwen36_mingzhe_cybergym/eval_rep_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${NAME}_${RUN_STAMP}.log}"

if [ -z "${LITELLM_API_KEY:-}" ]; then
    echo "LITELLM_API_KEY is required for OpenCode Qwen3.6 Mingzhe." >&2
    exit 2
fi

mkdir -p "$LOG_DIR"

echo "{\"event\":\"opencode_qwen36_eval200_bo8_launch\",\"name\":\"${NAME}\",\"run_stamp\":\"${RUN_STAMP}\",\"start_rep\":${START_REP},\"end_rep\":${END_REP},\"workers\":${WORKERS},\"model\":\"${MODEL}\",\"model_reasoning_effort\":\"${MODEL_REASONING_EFFORT}\",\"provider\":\"${OPENCODE_PROVIDER}\",\"opencode_context_limit\":${OPENCODE_CONTEXT_LIMIT},\"opencode_output_token_max\":${OPENCODE_OUTPUT_TOKEN_MAX},\"log\":\"${LOG_FILE}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"

set +e
"$PYTHON_BIN" scripts/run_opencode_qwen36_mingzhe_cybergym_bo_skip_passed.py \
    --split eval \
    --difficulty level1 \
    --max-tasks 200 \
    --reps "$END_REP" \
    --start-rep "$START_REP" \
    --workers "$WORKERS" \
    --model "$MODEL" \
    --model-reasoning-effort "$MODEL_REASONING_EFFORT" \
    --opencode-provider "$OPENCODE_PROVIDER" \
    --opencode-provider-base-url "$LITELLM_BASE_URL" \
    --opencode-provider-env-key LITELLM_API_KEY \
    --opencode-context-limit "$OPENCODE_CONTEXT_LIMIT" \
    --opencode-output-token-max "$OPENCODE_OUTPUT_TOKEN_MAX" \
    --opencode-bin "$OPENCODE_BIN" \
    --codex-timeout-seconds "$CODEX_TIMEOUT_SECONDS" \
    --submit-timeout-seconds "$SUBMIT_TIMEOUT_SECONDS" \
    --max-consecutive-infra-failures "$MAX_CONSECUTIVE_INFRA_FAILURES" \
    --infra-failure-pause-seconds "$INFRA_FAILURE_PAUSE_SECONDS" \
    --server "$SERVER" \
    --pocdb-path "$POCDB_PATH" \
    --env-file "$ENV_FILE" \
    --run-root "$RUN_ROOT" \
    --output-dir runs \
    --run-stamp "$RUN_STAMP" \
    --name "$NAME" \
    --python-bin "$PYTHON_BIN" \
    --parallel-runner scripts/run_codex_cybergym_tasks_parallel.py \
    --task-runner scripts/run_opencode_cybergym_tasks.py \
    --sandbox "$SANDBOX" \
    2>&1 | tee -a "$LOG_FILE"
exit_code="${PIPESTATUS[0]}"
set -e

echo "{\"event\":\"opencode_qwen36_eval200_bo8_complete\",\"name\":\"${NAME}\",\"run_stamp\":\"${RUN_STAMP}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"
exit "$exit_code"
