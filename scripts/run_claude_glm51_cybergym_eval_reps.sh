#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -n "${LLM_GATEWAY_API_KEY:-}" ]; then
    export ANTHROPIC_API_KEY="$LLM_GATEWAY_API_KEY"
fi

if [ -z "${ANTHROPIC_BASE_URL:-}" ] && [ -n "${LLM_GATEWAY_URL:-}" ]; then
    ANTHROPIC_BASE_URL="${LLM_GATEWAY_URL%/v1}"
    ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL%/}"
    export ANTHROPIC_BASE_URL
fi

START_REP="${1:-1}"
END_REP="${2:-8}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WORKERS="${WORKERS:-4}"
MODEL="${MODEL:-glm-5.1}"
CLAUDE_TIMEOUT_SECONDS="${CLAUDE_TIMEOUT_SECONDS:-900}"
SUBMIT_TIMEOUT_SECONDS="${SUBMIT_TIMEOUT_SECONDS:-1200}"
RUN_ROOT="${RUN_ROOT:-runs/claude_cybergym}"
SERVER="${CYBERGYM_SERVER:-http://127.0.0.1:8666}"
POCDB_PATH="${POCDB_PATH:-runs/cybergym_server/poc.db}"
ENV_FILE="${ENV_FILE:-.env}"
MAX_CONSECUTIVE_INFRA_FAILURES="${MAX_CONSECUTIVE_INFRA_FAILURES:-3}"
MAX_TASKS="${MAX_TASKS:-}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY is required for Claude Code." >&2
    exit 2
fi

if [ -z "${ANTHROPIC_BASE_URL:-}" ]; then
    echo "ANTHROPIC_BASE_URL is required for Claude Code." >&2
    exit 2
fi

for rep in $(seq "$START_REP" "$END_REP"); do
    run_id="claude-glm51-eval-bo8-r${rep}-${RUN_STAMP}"
    output="runs/claude_glm51_eval_bo8_rep${rep}_rollouts.jsonl"
    extra_args=()
    if [ -n "$MAX_TASKS" ]; then
        extra_args+=(--max-tasks "$MAX_TASKS")
    fi
    echo "{\"event\":\"eval_rep_start\",\"rep\":${rep},\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"workers\":${WORKERS}}"
    "$PYTHON_BIN" scripts/run_claude_cybergym_tasks_parallel.py \
        --split eval \
        --difficulty level1 \
        --run-id "$run_id" \
        --run-root "$RUN_ROOT" \
        --output "$output" \
        --workers "$WORKERS" \
        --model "$MODEL" \
        --codex-bin claude \
        --codex-timeout-seconds "$CLAUDE_TIMEOUT_SECONDS" \
        --submit-timeout-seconds "$SUBMIT_TIMEOUT_SECONDS" \
        --server "$SERVER" \
        --pocdb-path "$POCDB_PATH" \
        --env-file "$ENV_FILE" \
        --runner scripts/run_claude_cybergym_tasks.py \
        --max-consecutive-infra-failures "$MAX_CONSECUTIVE_INFRA_FAILURES" \
        "${extra_args[@]}"
    echo "{\"event\":\"eval_rep_complete\",\"rep\":${rep},\"run_id\":\"${run_id}\",\"output\":\"${output}\"}"
done
