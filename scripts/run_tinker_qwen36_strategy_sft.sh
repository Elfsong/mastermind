#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-qwen36-strategy-sft-${RUN_STAMP}}"
DATA_DIR="${DATA_DIR:-runs/strategy_sft/qwen36_strategy_sft_data}"
OUT_DIR="${OUT_DIR:-runs/strategy_sft/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}.log}"

mkdir -p "$(dirname "$LOG_FILE")" "$OUT_DIR"

echo "{\"event\":\"prepare_start\",\"run_id\":\"${RUN_ID}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"
.venv/bin/python scripts/prepare_tinker_strategy_sft_data.py \
  --out-dir "$DATA_DIR" | tee -a "$LOG_FILE"

echo "{\"event\":\"train_start\",\"run_id\":\"${RUN_ID}\",\"out_dir\":\"${OUT_DIR}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"
.venv/bin/python scripts/train_tinker_strategy_sft.py \
  --run-id "$RUN_ID" \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --model "${MODEL:-Qwen/Qwen3.6-35B-A3B}" \
  --lora-rank "${LORA_RANK:-32}" \
  --learning-rate "${LEARNING_RATE:-5e-5}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --epochs "${EPOCHS:-1}" \
  --max-length "${MAX_LENGTH:-12288}" \
  --eval-every "${EVAL_EVERY:-25}" \
  --save-every "${SAVE_EVERY:-50}" \
  --eval-limit "${EVAL_LIMIT:-64}" \
  ${EXTRA_ARGS:-} | tee -a "$LOG_FILE"

echo "{\"event\":\"train_script_complete\",\"run_id\":\"${RUN_ID}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$LOG_FILE"
