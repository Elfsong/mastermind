#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"

SFT_MODEL_PATH="${SFT_MODEL_PATH:?Set SFT_MODEL_PATH=tinker://.../final-sampler}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
MAX_TASKS="${MAX_TASKS:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
WORKERS="${WORKERS:-6}"

# Resume the prior base run if present, so we do not throw away completed baseline work.
BASE_RUN_ID="${BASE_RUN_ID:-codex-gpt55-eval100-tinker-qwen36-phase0-20260531T0715Z}"
BASE_OUTPUT="${BASE_OUTPUT:-runs/codex_gpt55_eval100_tinker_qwen36_phase0_rollouts.jsonl}"

SFT_RUN_ID="${SFT_RUN_ID:-codex-gpt55-eval100-tinker-qwen36-sft-${RUN_STAMP}}"
SFT_OUTPUT="${SFT_OUTPUT:-runs/codex_gpt55_eval100_tinker_qwen36_sft_matched_rollouts.jsonl}"

PAIR_LOG="runs/codex_cybergym/sequential_self_logs/eval100_tinker_qwen36_base_vs_sft_${RUN_STAMP}.log"
mkdir -p "$(dirname "$PAIR_LOG")"

echo "{\"event\":\"pair_eval_start\",\"base_run_id\":\"${BASE_RUN_ID}\",\"sft_run_id\":\"${SFT_RUN_ID}\",\"sft_model_path\":\"${SFT_MODEL_PATH}\",\"max_tasks\":${MAX_TASKS},\"max_attempts\":${MAX_ATTEMPTS},\"workers\":${WORKERS},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$PAIR_LOG"

VARIANT=base \
RUN_ID="$BASE_RUN_ID" \
OUTPUT="$BASE_OUTPUT" \
MAX_TASKS="$MAX_TASKS" \
MAX_ATTEMPTS="$MAX_ATTEMPTS" \
WORKERS="$WORKERS" \
scripts/run_codex_cybergym_seq_self_eval100_tinker_qwen36_matched.sh | tee -a "$PAIR_LOG"

VARIANT=sft \
RUN_ID="$SFT_RUN_ID" \
OUTPUT="$SFT_OUTPUT" \
MAX_TASKS="$MAX_TASKS" \
MAX_ATTEMPTS="$MAX_ATTEMPTS" \
WORKERS="$WORKERS" \
TINKER_SUMMARY_MODEL_PATH="$SFT_MODEL_PATH" \
scripts/run_codex_cybergym_seq_self_eval100_tinker_qwen36_matched.sh | tee -a "$PAIR_LOG"

.venv/bin/python scripts/summarize_cybergym_pass_rates.py "$BASE_OUTPUT" "$SFT_OUTPUT" | tee -a "$PAIR_LOG"
echo "{\"event\":\"pair_eval_complete\",\"base_output\":\"${BASE_OUTPUT}\",\"sft_output\":\"${SFT_OUTPUT}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$PAIR_LOG"
