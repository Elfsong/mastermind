#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MASTERMIND_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT"

source scripts/mastermind_env.sh

GRPO_RUN_DIR="${GRPO_RUN_DIR:?Set GRPO_RUN_DIR to a completed or running GRPO run directory.}"
POLL_SECONDS="${POLL_SECONDS:-120}"
PYTHON="${PYTHON:-.venv/bin/python}"
GRPO_WAIT_TARGET="${GRPO_WAIT_TARGET:-final}"
MIN_CHECKPOINT_STEP="${MIN_CHECKPOINT_STEP:-1}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_RUN_ID="${EVAL_RUN_ID:-codex-gpt55-eval200-qwen36-grpo-${RUN_STAMP}}"
EVAL_OUTPUT="${EVAL_OUTPUT:-runs/strategy_grpo_eval/${EVAL_RUN_ID}/grpo_qwen36_rollouts.jsonl}"
EVAL_LOG="${EVAL_LOG:-runs/strategy_grpo_eval/${EVAL_RUN_ID}/eval.log}"
REPORT_DIR="${REPORT_DIR:-runs/strategy_grpo_eval/${EVAL_RUN_ID}/reports}"

WORKERS="${WORKERS:-6}"
MAX_TASKS="${MAX_TASKS:-200}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
TASK_OFFSET="${TASK_OFFSET:-0}"
ATTEMPT_COOLDOWN_SECONDS="${ATTEMPT_COOLDOWN_SECONDS:-0}"
CODEX_LAUNCH_STAGGER_SECONDS="${CODEX_LAUNCH_STAGGER_SECONDS:-0}"
EXECUTOR_MODEL="${EXECUTOR_MODEL:-gpt-5.5}"
EXECUTOR_REASONING_EFFORT="${EXECUTOR_REASONING_EFFORT:-medium}"
CODEX_RATE_LIMIT_RETRIES="${CODEX_RATE_LIMIT_RETRIES:-3}"
CODEX_RATE_LIMIT_STAGGER_SECONDS="${CODEX_RATE_LIMIT_STAGGER_SECONDS:-5}"
CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS="${CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS:-0}"
TINKER_SUMMARY_MODEL="${TINKER_SUMMARY_MODEL:-Qwen/Qwen3.6-35B-A3B}"
TINKER_SUMMARY_TEMPERATURE="${TINKER_SUMMARY_TEMPERATURE:-0}"
TINKER_SUMMARY_TOP_P="${TINKER_SUMMARY_TOP_P:-1}"
EXPERIENCE_TOKEN_BUDGET="${EXPERIENCE_TOKEN_BUDGET:-2048}"
CODEX_PROVIDER="${CODEX_PROVIDER:-llmgw}"

VANILLA_EVAL200="${VANILLA_EVAL200:-runs/hf_upload_staging/cybergym-codex-gpt-5-5-qwen36-vanilla-vs-sft-eval200-trajectories/data/vanilla_qwen36_rollouts.jsonl}"
SFT_EVAL200="${SFT_EVAL200:-runs/hf_upload_staging/cybergym-codex-gpt-5-5-qwen36-vanilla-vs-sft-eval200-trajectories/data/sft_qwen36_rollouts.jsonl}"

mkdir -p "$(dirname "$EVAL_OUTPUT")" "$(dirname "$EVAL_LOG")" "$REPORT_DIR"

log_event() {
  local event="$1"
  shift || true
  printf '{"event":"%s","time":"%s"' "$event" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$EVAL_LOG"
  while [[ $# -gt 0 ]]; do
    local key="$1"
    local value="$2"
    shift 2
    printf ',"%s":%s' "$key" "$value" >> "$EVAL_LOG"
  done
  printf '}\n' >> "$EVAL_LOG"
}

json_string() {
  "$PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

extract_final_sampler() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_sampler_path"])' "$GRPO_RUN_DIR/result.json"
}

extract_latest_checkpoint_sampler() {
  "$PYTHON" - "$GRPO_RUN_DIR/checkpoints.jsonl" "$MIN_CHECKPOINT_STEP" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_step = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)

rows = []
with path.open(errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(row, dict)
            and row.get("event") == "sampler_weights"
            and int(row.get("step") or 0) >= min_step
            and row.get("path")
        ):
            rows.append(row)

if not rows:
    raise SystemExit(1)
print(rows[-1]["path"])
PY
}

log_event "watch_start" \
  "grpo_run_dir" "$(json_string "$GRPO_RUN_DIR")" \
  "eval_run_id" "$(json_string "$EVAL_RUN_ID")" \
  "wait_target" "$(json_string "$GRPO_WAIT_TARGET")" \
  "min_checkpoint_step" "$MIN_CHECKPOINT_STEP" \
  "eval_output" "$(json_string "$EVAL_OUTPUT")" \
  "task_offset" "$TASK_OFFSET" \
  "codex_launch_stagger_seconds" "$CODEX_LAUNCH_STAGGER_SECONDS" \
  "attempt_cooldown_seconds" "$ATTEMPT_COOLDOWN_SECONDS" \
  "codex_rate_limit_retries" "$CODEX_RATE_LIMIT_RETRIES" \
  "codex_rate_limit_stagger_seconds" "$CODEX_RATE_LIMIT_STAGGER_SECONDS" \
  "codex_rate_limit_min_sleep_seconds" "$CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS"

FINAL_SAMPLER_PATH=""
if [[ "$GRPO_WAIT_TARGET" == "final" ]]; then
  while [[ ! -s "$GRPO_RUN_DIR/result.json" ]]; do
    log_event "waiting_for_grpo_result" "grpo_run_dir" "$(json_string "$GRPO_RUN_DIR")"
    sleep "$POLL_SECONDS"
  done
  FINAL_SAMPLER_PATH="$(extract_final_sampler)"
  log_event "grpo_result_ready" "final_sampler_path" "$(json_string "$FINAL_SAMPLER_PATH")"
elif [[ "$GRPO_WAIT_TARGET" == "latest_sampler" ]]; then
  while true; do
    if FINAL_SAMPLER_PATH="$(extract_latest_checkpoint_sampler 2>/dev/null)"; then
      break
    fi
    log_event "waiting_for_checkpoint_sampler" \
      "grpo_run_dir" "$(json_string "$GRPO_RUN_DIR")" \
      "min_checkpoint_step" "$MIN_CHECKPOINT_STEP"
    sleep "$POLL_SECONDS"
  done
  log_event "checkpoint_sampler_ready" "sampler_path" "$(json_string "$FINAL_SAMPLER_PATH")"
else
  echo "Unsupported GRPO_WAIT_TARGET=${GRPO_WAIT_TARGET}; expected final or latest_sampler" >&2
  exit 2
fi

model_path_args=(--tinker-summary-model-path "$FINAL_SAMPLER_PATH")
provider_args=()
if [[ -n "$CODEX_PROVIDER" ]]; then
  provider_args=(--codex-provider "$CODEX_PROVIDER")
fi

set +e
"$PYTHON" scripts/run_codex_cybergym_sequential_self_involving.py \
  --split eval \
  --difficulty level1 \
  --run-id "$EVAL_RUN_ID" \
  --run-root runs/codex_cybergym \
  --output "$EVAL_OUTPUT" \
  --max-tasks "$MAX_TASKS" \
  --task-offset "$TASK_OFFSET" \
  --workers "$WORKERS" \
  --max-attempts "$MAX_ATTEMPTS" \
  --attempt-cooldown-seconds "$ATTEMPT_COOLDOWN_SECONDS" \
  --codex-launch-stagger-seconds "$CODEX_LAUNCH_STAGGER_SECONDS" \
  --codex-rate-limit-retries "$CODEX_RATE_LIMIT_RETRIES" \
  --codex-rate-limit-stagger-seconds "$CODEX_RATE_LIMIT_STAGGER_SECONDS" \
  --codex-rate-limit-min-sleep-seconds "$CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS" \
  --model "$EXECUTOR_MODEL" \
  --model-reasoning-effort "$EXECUTOR_REASONING_EFFORT" \
  --experience-updater tinker \
  --tinker-summary-model "$TINKER_SUMMARY_MODEL" \
  "${model_path_args[@]}" \
  --tinker-summary-temperature "$TINKER_SUMMARY_TEMPERATURE" \
  --tinker-summary-top-p "$TINKER_SUMMARY_TOP_P" \
  --experience-token-budget "$EXPERIENCE_TOKEN_BUDGET" \
  "${provider_args[@]}" >> "$EVAL_LOG" 2>&1
eval_exit=$?
set -e
log_event "eval_complete" "exit_code" "$eval_exit"

"$PYTHON" scripts/compare_cybergym_strategy_eval.py \
  --base "$VANILLA_EVAL200" \
  --sft "$EVAL_OUTPUT" \
  --base-label vanilla \
  --sft-label grpo \
  --max-attempts "$MAX_ATTEMPTS" \
  --max-tasks "$MAX_TASKS" \
  --format json > "$REPORT_DIR/vanilla_vs_grpo.json"

"$PYTHON" scripts/compare_cybergym_strategy_eval.py \
  --base "$SFT_EVAL200" \
  --sft "$EVAL_OUTPUT" \
  --base-label sft \
  --sft-label grpo \
  --max-attempts "$MAX_ATTEMPTS" \
  --max-tasks "$MAX_TASKS" \
  --format json > "$REPORT_DIR/sft_vs_grpo.json"

log_event "compare_complete" \
  "vanilla_vs_grpo" "$(json_string "$REPORT_DIR/vanilla_vs_grpo.json")" \
  "sft_vs_grpo" "$(json_string "$REPORT_DIR/sft_vs_grpo.json")"

exit "$eval_exit"
