#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

START_REP="${1:-3}"
END_REP="${2:-8}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WORKERS="${WORKERS:-6}"
MODEL="${MODEL:-gpt-5.5}"
MODEL_REASONING_EFFORT="${MODEL_REASONING_EFFORT:-medium}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-900}"
SUBMIT_TIMEOUT_SECONDS="${SUBMIT_TIMEOUT_SECONDS:-1800}"
RUN_ROOT="${RUN_ROOT:-runs/codex_cybergym}"
SERVER="${CYBERGYM_SERVER:-http://127.0.0.1:8666}"
POCDB_PATH="${POCDB_PATH:-runs/cybergym_server/poc.db}"
ENV_FILE="${ENV_FILE:-.env}"
SANDBOX="${SANDBOX:-workspace-write}"

for rep in $(seq "$START_REP" "$END_REP"); do
    run_id="codex-gateway-eval-bo4-r${rep}-${RUN_STAMP}"
    output="runs/codex_gateway_eval_bo4_rep${rep}_rollouts.jsonl"
    echo "{\"event\":\"eval_rep_start\",\"rep\":${rep},\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"workers\":${WORKERS}}"
    "$PYTHON_BIN" scripts/run_codex_cybergym_tasks_parallel.py \
        --split eval \
        --difficulty level1 \
        --run-id "$run_id" \
        --run-root "$RUN_ROOT" \
        --output "$output" \
        --workers "$WORKERS" \
        --model "$MODEL" \
        --model-reasoning-effort "$MODEL_REASONING_EFFORT" \
        --codex-timeout-seconds "$CODEX_TIMEOUT_SECONDS" \
        --submit-timeout-seconds "$SUBMIT_TIMEOUT_SECONDS" \
        --server "$SERVER" \
        --pocdb-path "$POCDB_PATH" \
        --env-file "$ENV_FILE" \
        --sandbox "$SANDBOX"
    echo "{\"event\":\"eval_rep_complete\",\"rep\":${rep},\"run_id\":\"${run_id}\",\"output\":\"${output}\"}"
done
