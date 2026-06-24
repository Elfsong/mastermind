#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MASTERMIND_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT"

source scripts/mastermind_env.sh

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-qwen36-strategy-grpo-k8-smoke-${RUN_STAMP}}"
RUN_DIR="${RUN_DIR:-runs/strategy_grpo/${RUN_ID}}"
mkdir -p "$RUN_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
ADVANTAGE_GROUP_SIZE="${ADVANTAGE_GROUP_SIZE:-${GROUP_SIZE:-8}}"
ROLLOUT_POOL_PER_TASK="${ROLLOUT_POOL_PER_TASK:-$((ADVANTAGE_GROUP_SIZE * 3))}"
TASKS_PER_STEP="${TASKS_PER_STEP:-2}"
MAX_STEPS="${MAX_STEPS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LOSS_FN="${LOSS_FN:-ppo}"
EXECUTOR_WORKERS="${EXECUTOR_WORKERS:-4}"
EXECUTOR_MODEL="${EXECUTOR_MODEL:-gpt-5.5}"
DIFFICULTY="${DIFFICULTY:-level1}"
SPLIT="${SPLIT:-train_dev}"
TASK_SAMPLING="${TASK_SAMPLING:-sequential}"
GROUPS_PER_UPDATE="${GROUPS_PER_UPDATE:-1}"
CODEX_RATE_LIMIT_RETRIES="${CODEX_RATE_LIMIT_RETRIES:-3}"
CODEX_RATE_LIMIT_STAGGER_SECONDS="${CODEX_RATE_LIMIT_STAGGER_SECONDS:-5}"

cmd=(
  "$PYTHON" scripts/train_tinker_strategy_grpo.py
  --run-id "$RUN_ID"
  --split "$SPLIT"
  --difficulty "$DIFFICULTY"
  --task-sampling "$TASK_SAMPLING"
  --advantage-group-size "$ADVANTAGE_GROUP_SIZE"
  --rollout-pool-per-task "$ROLLOUT_POOL_PER_TASK"
  --tasks-per-step "$TASKS_PER_STEP"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --loss-fn "$LOSS_FN"
  --groups-per-update "$GROUPS_PER_UPDATE"
  --executor-workers "$EXECUTOR_WORKERS"
  --executor-model "$EXECUTOR_MODEL"
  --codex-rate-limit-retries "$CODEX_RATE_LIMIT_RETRIES"
  --codex-rate-limit-stagger-seconds "$CODEX_RATE_LIMIT_STAGGER_SECONDS"
)

if [[ -n "${INIT_STATE_PATH:-}" ]]; then
  cmd+=(--init-state-path "$INIT_STATE_PATH")
fi
if [[ -n "${LOSS_FN_CONFIG_JSON:-}" ]]; then
  cmd+=(--loss-fn-config-json "$LOSS_FN_CONFIG_JSON")
fi
if [[ -n "${CODEX_PROVIDER:-}" ]]; then
  cmd+=(--codex-provider "$CODEX_PROVIDER")
fi
if [[ -n "${CODEX_PROVIDER_BASE_URL:-}" ]]; then
  cmd+=(--codex-provider-base-url "$CODEX_PROVIDER_BASE_URL")
fi
if [[ -n "${MAX_STRATEGY_TOKENS:-}" ]]; then
  cmd+=(--max-strategy-tokens "$MAX_STRATEGY_TOKENS")
fi
if [[ "${SKIP_GRPO_UPDATE:-0}" == "1" ]]; then
  cmd+=(--skip-grpo-update)
fi
if [[ "${UPDATE_AS_GROUPS_COMPLETE:-1}" == "0" ]]; then
  cmd+=(--no-update-as-groups-complete)
fi
if [[ "${SAVE_AFTER_UPDATE:-0}" == "1" ]]; then
  cmd+=(--save-after-update)
fi
if [[ -n "${TASK_IDS_FILE:-}" ]]; then
  cmd+=(--task-ids-file "$TASK_IDS_FILE")
fi
if [[ -n "${MAX_TRAIN_TASKS:-}" ]]; then
  cmd+=(--max-train-tasks "$MAX_TRAIN_TASKS")
fi
if [[ -n "${TASK_OFFSET:-}" ]]; then
  cmd+=(--task-offset "$TASK_OFFSET")
fi

printf '%q ' "${cmd[@]}" > "${RUN_DIR}/command.sh"
printf '\n' >> "${RUN_DIR}/command.sh"

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  "${cmd[@]}" > "${RUN_DIR}/nohup.log" 2>&1 < /dev/null &
else
  setsid "${cmd[@]}" > "${RUN_DIR}/nohup.log" 2>&1 < /dev/null &
fi
pid=$!
echo "$pid" > "${RUN_DIR}/pid"

cat > "${RUN_DIR}/run_env.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "run_dir": "${RUN_DIR}",
  "pid": ${pid},
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "advantage_group_size": ${ADVANTAGE_GROUP_SIZE},
  "rollout_pool_per_task": ${ROLLOUT_POOL_PER_TASK},
  "update_as_groups_complete": $([[ "${UPDATE_AS_GROUPS_COMPLETE:-1}" == "0" ]] && echo false || echo true),
  "save_after_update": $([[ "${SAVE_AFTER_UPDATE:-0}" == "1" ]] && echo true || echo false),
  "groups_per_update": ${GROUPS_PER_UPDATE},
  "tasks_per_step": ${TASKS_PER_STEP},
  "max_steps": ${MAX_STEPS},
  "task_sampling": "${TASK_SAMPLING}",
  "learning_rate": "${LEARNING_RATE}",
  "loss_fn": "${LOSS_FN}",
  "executor_workers": ${EXECUTOR_WORKERS},
  "executor_model": "${EXECUTOR_MODEL}",
  "codex_rate_limit_retries": ${CODEX_RATE_LIMIT_RETRIES},
  "codex_rate_limit_stagger_seconds": ${CODEX_RATE_LIMIT_STAGGER_SECONDS},
  "split": "${SPLIT}",
  "difficulty": "${DIFFICULTY}"
}
EOF

echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"
echo "PID=${pid}"

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  set +e
  wait "$pid"
  exit_code=$?
  set -e
  echo "$exit_code" > "${RUN_DIR}/exit_code"
  exit "$exit_code"
fi
