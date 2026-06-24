#!/usr/bin/env bash
set -euo pipefail

# Matched end-to-end evaluation for the strategy generator:
# - executor is fixed to Codex GPT-5.5;
# - strategy updater is Tinker Qwen3.6 base or a supplied SFT checkpoint;
# - task set is the deterministic first N eval/level1 tasks used by the prior baseline.

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
VARIANT="${VARIANT:-base}"
RUN_ID="${RUN_ID:-codex-gpt55-eval100-tinker-qwen36-${VARIANT}-${RUN_STAMP}}"
LOG_DIR="runs/codex_cybergym/sequential_self_logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"
OUTPUT="${OUTPUT:-runs/codex_gpt55_eval100_tinker_qwen36_${VARIANT}_matched_rollouts.jsonl}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
MAX_TASKS="${MAX_TASKS:-100}"
WORKERS="${WORKERS:-6}"
PROVIDER="${CODEX_PROVIDER:-llmgw}"
TINKER_SUMMARY_MODEL="${TINKER_SUMMARY_MODEL:-Qwen/Qwen3.6-35B-A3B}"
TINKER_SUMMARY_MODEL_PATH="${TINKER_SUMMARY_MODEL_PATH:-}"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"

echo "{\"event\":\"seq_self_eval100_tinker_qwen36_matched_start\",\"variant\":\"${VARIANT}\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"workers\":${WORKERS},\"max_tasks\":${MAX_TASKS},\"max_attempts\":${MAX_ATTEMPTS},\"experience_updater\":\"tinker\",\"tinker_summary_model\":\"${TINKER_SUMMARY_MODEL}\",\"tinker_summary_model_path\":\"${TINKER_SUMMARY_MODEL_PATH}\",\"provider\":\"${PROVIDER}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"

MODEL_PATH_ARGS=()
if [[ -n "$TINKER_SUMMARY_MODEL_PATH" ]]; then
  MODEL_PATH_ARGS+=(--tinker-summary-model-path "$TINKER_SUMMARY_MODEL_PATH")
fi

.venv/bin/python scripts/run_codex_cybergym_sequential_self_involving.py \
  --run-id "$RUN_ID" \
  --split eval \
  --difficulty level1 \
  --output "$OUTPUT" \
  --workers "$WORKERS" \
  --max-attempts "$MAX_ATTEMPTS" \
  --experience-token-budget 2048 \
  --max-tasks "$MAX_TASKS" \
  --model gpt-5.5 \
  --codex-provider "$PROVIDER" \
  --experience-updater tinker \
  --summary-model "$TINKER_SUMMARY_MODEL" \
  --tinker-summary-model "$TINKER_SUMMARY_MODEL" \
  "${MODEL_PATH_ARGS[@]}" \
  --tinker-summary-temperature 0 \
  --tinker-summary-top-p 1 >> "$LOG_FILE" 2>&1

exit_code=$?
echo "{\"event\":\"seq_self_eval100_tinker_qwen36_matched_complete\",\"variant\":\"${VARIANT}\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"
exit "$exit_code"
