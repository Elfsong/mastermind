#!/usr/bin/env bash
set -euo pipefail

# Matched 100-task CyberGym eval:
#   GPT-5.5 strategy generator + GPT-5.5 executor
#
# Defaults intentionally mirror the Qwen3.6 vanilla/SFT eval100 run from
# 2026-06-03, including the exact task-id file and worker count.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MASTERMIND_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT"

source scripts/mastermind_env.sh

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_TASKS="${MAX_TASKS:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
WORKERS="${WORKERS:-4}"
SPLIT="${SPLIT:-eval}"
DIFFICULTY="${DIFFICULTY:-level1}"
TASK_OFFSET="${TASK_OFFSET:-0}"

CODEX_EXECUTOR_MODEL="${CODEX_EXECUTOR_MODEL:-gpt-5.5}"
SUMMARY_MODEL="${SUMMARY_MODEL:-gpt-5.5}"
SUMMARY_REASONING_EFFORT="${SUMMARY_REASONING_EFFORT:-medium}"
MODEL_REASONING_EFFORT="${MODEL_REASONING_EFFORT:-medium}"
EXPERIENCE_TOKEN_BUDGET="${EXPERIENCE_TOKEN_BUDGET:-2048}"

PROVIDER="${CODEX_PROVIDER:-llmgw}"
CODEX_RATE_LIMIT_RETRIES="${CODEX_RATE_LIMIT_RETRIES:-3}"
CODEX_RATE_LIMIT_STAGGER_SECONDS="${CODEX_RATE_LIMIT_STAGGER_SECONDS:-5}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-900}"
SUBMIT_TIMEOUT_SECONDS="${SUBMIT_TIMEOUT_SECONDS:-1800}"
SUMMARY_TIMEOUT_SECONDS="${SUMMARY_TIMEOUT_SECONDS:-600}"
MAX_AUTO_SUBMIT_CANDIDATES="${MAX_AUTO_SUBMIT_CANDIDATES:-5}"

RUN_ROOT="${RUN_ROOT:-runs/codex_cybergym}"
LOG_DIR="${LOG_DIR:-runs/codex_cybergym/sequential_self_logs}"
EVAL_DIR="${EVAL_DIR:-runs/strategy_sft/gpt55_strategy_eval100_${RUN_STAMP}}"
mkdir -p "$EVAL_DIR" "$LOG_DIR"

ORCH_LOG="${ORCH_LOG:-${EVAL_DIR}/run.log}"
CONFIG_JSON="${CONFIG_JSON:-${EVAL_DIR}/run_config.json}"
TASK_IDS_SOURCE="${TASK_IDS_SOURCE:-runs/strategy_sft/vanilla_vs_sft_eval100_20260603T040840Z/task_ids.txt}"
TASK_IDS_PATH="${TASK_IDS_PATH:-${EVAL_DIR}/task_ids.txt}"

RUN_ID="${RUN_ID:-codex-gpt55-eval100-strategy-gpt55-${RUN_STAMP}}"
OUTPUT="${OUTPUT:-${EVAL_DIR}/gpt55_strategy_rollouts.jsonl}"
RUN_LOG="${LOG_DIR}/${RUN_ID}.log"

export RUN_STAMP EVAL_DIR
if [[ "${DETACH:-0}" == "1" && "${MM_GPT55_EVAL100_DETACHED:-0}" != "1" ]]; then
  export MM_GPT55_EVAL100_DETACHED=1
  nohup "$0" "$@" > "${EVAL_DIR}/orchestrator.nohup.log" 2>&1 &
  echo "$!" > "${EVAL_DIR}/orchestrator.pid"
  echo "Detached GPT-5.5 eval100 runner."
  echo "  pid:      $(cat "${EVAL_DIR}/orchestrator.pid")"
  echo "  eval dir: ${EVAL_DIR}"
  echo "  log:      ${EVAL_DIR}/run.log"
  echo "  nohup:    ${EVAL_DIR}/orchestrator.nohup.log"
  exit 0
fi

if [[ -f "$TASK_IDS_PATH" && "${REGENERATE_TASK_IDS:-0}" != "1" ]]; then
  echo "Using existing task-id list: ${TASK_IDS_PATH}" | tee -a "$ORCH_LOG"
elif [[ -n "${TASK_IDS_FILE:-}" ]]; then
  cp "$TASK_IDS_FILE" "$TASK_IDS_PATH"
  echo "Copied task-id list from ${TASK_IDS_FILE} to ${TASK_IDS_PATH}" | tee -a "$ORCH_LOG"
elif [[ -f "$TASK_IDS_SOURCE" ]]; then
  cp "$TASK_IDS_SOURCE" "$TASK_IDS_PATH"
  echo "Copied task-id list from ${TASK_IDS_SOURCE} to ${TASK_IDS_PATH}" | tee -a "$ORCH_LOG"
else
  "$PYTHON" - "$SPLIT" "$TASK_OFFSET" "$MAX_TASKS" "$TASK_IDS_PATH" <<'PY'
import json
import sys
from pathlib import Path

split = sys.argv[1]
offset = int(sys.argv[2])
max_tasks = int(sys.argv[3])
out = Path(sys.argv[4])

manifest = json.loads(Path("mastermind.manifest.json").read_text())
split_path = Path(manifest["task_splits"][split])
task_ids = []
for line in split_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        task_ids.append(line)
selected = task_ids[offset : offset + max_tasks]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(selected) + ("\n" if selected else ""))
print(f"Wrote {len(selected)} task ids to {out}")
PY
fi

mapfile -t TASK_IDS < "$TASK_IDS_PATH"
TASK_COUNT="${#TASK_IDS[@]}"
if [[ "$TASK_COUNT" -eq 0 ]]; then
  echo "ERROR: no task ids in ${TASK_IDS_PATH}" >&2
  exit 2
fi
if [[ "$TASK_COUNT" -ne "$MAX_TASKS" ]]; then
  echo "WARNING: task-id count (${TASK_COUNT}) != MAX_TASKS (${MAX_TASKS})." | tee -a "$ORCH_LOG"
fi

TASK_ID_ARGS=()
for task_id in "${TASK_IDS[@]}"; do
  TASK_ID_ARGS+=(--task-id "$task_id")
done

export \
  RUN_STAMP EVAL_DIR ORCH_LOG CONFIG_JSON TASK_IDS_PATH TASK_IDS_SOURCE TASK_COUNT \
  RUN_ID OUTPUT RUN_LOG MAX_TASKS MAX_ATTEMPTS WORKERS SPLIT DIFFICULTY TASK_OFFSET \
  CODEX_EXECUTOR_MODEL SUMMARY_MODEL MODEL_REASONING_EFFORT SUMMARY_REASONING_EFFORT \
  EXPERIENCE_TOKEN_BUDGET PROVIDER CODEX_RATE_LIMIT_RETRIES CODEX_RATE_LIMIT_STAGGER_SECONDS \
  CODEX_TIMEOUT_SECONDS SUBMIT_TIMEOUT_SECONDS SUMMARY_TIMEOUT_SECONDS MAX_AUTO_SUBMIT_CANDIDATES \
  RUN_ROOT LOG_DIR

"$PYTHON" - <<'PY'
import json
import os
from datetime import UTC, datetime
from pathlib import Path

keys = [
    "RUN_STAMP",
    "EVAL_DIR",
    "TASK_IDS_PATH",
    "TASK_IDS_SOURCE",
    "TASK_COUNT",
    "MAX_TASKS",
    "MAX_ATTEMPTS",
    "WORKERS",
    "SPLIT",
    "DIFFICULTY",
    "TASK_OFFSET",
    "CODEX_EXECUTOR_MODEL",
    "SUMMARY_MODEL",
    "MODEL_REASONING_EFFORT",
    "SUMMARY_REASONING_EFFORT",
    "EXPERIENCE_TOKEN_BUDGET",
    "PROVIDER",
    "RUN_ID",
    "OUTPUT",
    "RUN_ROOT",
    "LOG_DIR",
]
config = {key: os.environ.get(key) for key in keys}
config["created_at"] = datetime.now(UTC).isoformat()
Path(os.environ["CONFIG_JSON"]).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

provider_args=()
if [[ -n "$PROVIDER" && "$PROVIDER" != "none" ]]; then
  provider_args+=(--codex-provider "$PROVIDER")
  if [[ -n "${CODEX_PROVIDER_BASE_URL:-}" ]]; then
    provider_args+=(--codex-provider-base-url "$CODEX_PROVIDER_BASE_URL")
  fi
  provider_args+=(--codex-provider-wire-api "${CODEX_PROVIDER_WIRE_API:-responses}")
  provider_args+=(--codex-provider-env-key "${CODEX_PROVIDER_ENV_KEY:-LLM_GATEWAY_API_KEY}")
fi

{
  echo "===== GPT-5.5 strategy generator eval100 ====="
  echo "eval_dir=${EVAL_DIR}"
  echo "run_id=${RUN_ID}"
  echo "output=${OUTPUT}"
  echo "task_ids=${TASK_IDS_PATH} (${TASK_COUNT})"
  echo "executor_model=${CODEX_EXECUTOR_MODEL}"
  echo "summary_model=${SUMMARY_MODEL}"
  echo "workers=${WORKERS}"
  echo "{\"event\":\"variant_start\",\"variant\":\"gpt55_strategy\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
} | tee -a "$ORCH_LOG" "$RUN_LOG"

set +e
"$PYTHON" scripts/run_codex_cybergym_sequential_self_involving.py \
  --run-id "$RUN_ID" \
  --run-root "$RUN_ROOT" \
  --split "$SPLIT" \
  --difficulty "$DIFFICULTY" \
  --output "$OUTPUT" \
  "${TASK_ID_ARGS[@]}" \
  --max-tasks "$MAX_TASKS" \
  --workers "$WORKERS" \
  --max-attempts "$MAX_ATTEMPTS" \
  --model "$CODEX_EXECUTOR_MODEL" \
  --model-reasoning-effort "$MODEL_REASONING_EFFORT" \
  --experience-updater codex \
  --summary-model "$SUMMARY_MODEL" \
  --summary-reasoning-effort "$SUMMARY_REASONING_EFFORT" \
  --experience-token-budget "$EXPERIENCE_TOKEN_BUDGET" \
  --codex-timeout-seconds "$CODEX_TIMEOUT_SECONDS" \
  --summary-timeout-seconds "$SUMMARY_TIMEOUT_SECONDS" \
  --submit-timeout-seconds "$SUBMIT_TIMEOUT_SECONDS" \
  --server "${CYBERGYM_SERVER:-http://127.0.0.1:8666}" \
  --pocdb-path "${POCDB_PATH:-runs/cybergym_server/poc.db}" \
  --max-auto-submit-candidates "$MAX_AUTO_SUBMIT_CANDIDATES" \
  --codex-rate-limit-retries "$CODEX_RATE_LIMIT_RETRIES" \
  --codex-rate-limit-stagger-seconds "$CODEX_RATE_LIMIT_STAGGER_SECONDS" \
  "${provider_args[@]}" \
  2>&1 | tee -a "$ORCH_LOG" "$RUN_LOG"
exit_code=${PIPESTATUS[0]}
set -e

echo "{\"event\":\"variant_complete\",\"variant\":\"gpt55_strategy\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
  | tee -a "$ORCH_LOG" "$RUN_LOG"

exit "$exit_code"
