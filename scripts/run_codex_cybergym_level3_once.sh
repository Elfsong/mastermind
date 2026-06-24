#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WORKERS="${WORKERS:-6}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
MODEL="${MODEL:-gpt-5.5}"
MODEL_REASONING_EFFORT="${MODEL_REASONING_EFFORT:-medium}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-900}"
SUBMIT_TIMEOUT_SECONDS="${SUBMIT_TIMEOUT_SECONDS:-1800}"
RUN_ROOT="${RUN_ROOT:-runs/codex_cybergym}"
SERVER="${CYBERGYM_SERVER:-http://127.0.0.1:8666}"
POCDB_PATH="${POCDB_PATH:-runs/cybergym_server/poc.db}"
ENV_FILE="${ENV_FILE:-.env}"
SANDBOX="${SANDBOX:-workspace-write}"

run_id="codex-gateway-eval-level3-r1-${RUN_STAMP}"
output="runs/codex_gateway_eval_level3_rep1_rollouts.jsonl"
download_manifest="runs/cybergym_assets/download_manifests/eval200_level3_${RUN_STAMP}.jsonl"

echo "{\"event\":\"level3_download_start\",\"run_id\":\"${run_id}\",\"manifest\":\"${download_manifest}\",\"workers\":${DOWNLOAD_WORKERS}}"
"$PYTHON_BIN" scripts/download_cybergym_split_data.py \
    --split eval \
    --difficulty level3 \
    --workers "$DOWNLOAD_WORKERS" \
    --output "$download_manifest"
echo "{\"event\":\"level3_download_complete\",\"run_id\":\"${run_id}\",\"manifest\":\"${download_manifest}\"}"

echo "{\"event\":\"level3_eval_start\",\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"workers\":${WORKERS}}"
"$PYTHON_BIN" scripts/run_codex_cybergym_tasks_parallel.py \
    --split eval \
    --difficulty level3 \
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
echo "{\"event\":\"level3_eval_complete\",\"run_id\":\"${run_id}\",\"output\":\"${output}\"}"
