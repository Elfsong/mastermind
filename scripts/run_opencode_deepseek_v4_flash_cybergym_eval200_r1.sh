#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

export HF_ROUTER_BASE_URL="${HF_ROUTER_BASE_URL:-https://router.huggingface.co/v1}"

START_REP="${1:-1}"
END_REP="${2:-1}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WORKERS="${WORKERS:-6}"
MAX_TASKS="${MAX_TASKS:-200}"
MODEL="${MODEL:-hf_router/deepseek-ai/DeepSeek-V4-Flash:novita}"
OPENCODE_PROVIDER="${OPENCODE_PROVIDER:-hf_router}"
OPENCODE_BIN="${OPENCODE_BIN:-runs/opencode_tool/node_modules/.bin/opencode}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-900}"
SUBMIT_TIMEOUT_SECONDS="${SUBMIT_TIMEOUT_SECONDS:-1200}"
MAX_CONSECUTIVE_INFRA_FAILURES="${MAX_CONSECUTIVE_INFRA_FAILURES:-5}"
INFRA_FAILURE_PAUSE_SECONDS="${INFRA_FAILURE_PAUSE_SECONDS:-1800}"
RUN_ROOT="${RUN_ROOT:-/tmp/opencode_deepseek_v4_flash_cybergym}"
SERVER="${CYBERGYM_SERVER:-http://127.0.0.1:8666}"
POCDB_PATH="${POCDB_PATH:-runs/cybergym_server/poc.db}"
ENV_FILE="${ENV_FILE:-.env}"
SANDBOX="${SANDBOX:-workspace-write}"
NAME="${NAME:-opencode_deepseek_v4_flash_eval200_r1}"
LOG_DIR="${LOG_DIR:-runs/opencode_deepseek_v4_flash_cybergym/eval_rep_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${NAME}_${RUN_STAMP}.log}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is required for OpenCode DeepSeek V4 Flash through Hugging Face Router." >&2
    exit 2
fi

mkdir -p "$LOG_DIR"

echo "{\"event\":\"opencode_deepseek_v4_flash_eval_launch\",\"name\":\"${NAME}\",\"run_stamp\":\"${RUN_STAMP}\",\"start_rep\":${START_REP},\"end_rep\":${END_REP},\"workers\":${WORKERS},\"max_tasks\":${MAX_TASKS},\"model\":\"${MODEL}\",\"provider\":\"${OPENCODE_PROVIDER}\",\"log\":\"${LOG_FILE}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"

set +e
"$PYTHON_BIN" scripts/run_opencode_qwen36_mingzhe_cybergym_bo_skip_passed.py \
    --split eval \
    --difficulty level1 \
    --max-tasks "$MAX_TASKS" \
    --reps "$END_REP" \
    --start-rep "$START_REP" \
    --workers "$WORKERS" \
    --model "$MODEL" \
    --opencode-provider "$OPENCODE_PROVIDER" \
    --opencode-provider-base-url "$HF_ROUTER_BASE_URL" \
    --opencode-provider-env-key HF_TOKEN \
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

echo "{\"event\":\"opencode_deepseek_v4_flash_eval_complete\",\"name\":\"${NAME}\",\"run_stamp\":\"${RUN_STAMP}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"
exit "$exit_code"
