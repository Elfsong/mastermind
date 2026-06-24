#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-codex-gpt54mini-eval200-seqself-from-bo8r1-${RUN_STAMP}}"
SEED_OUTPUT="${SEED_OUTPUT:-runs/codex_gpt54mini_eval200_20260623T112010Z.jsonl}"
OUTPUT="${OUTPUT:-runs/codex_gpt54mini_eval200_seqself_from_bo8r1_${RUN_STAMP}_rollouts.jsonl}"
LOG_DIR="${LOG_DIR:-runs/codex_cybergym/sequential_self_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_ID}.pid}"
WORKERS="${WORKERS:-16}"
SEED_WORKERS="${SEED_WORKERS:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
MODEL="${MODEL:-gpt-5.4-mini}"
MODEL_REASONING_EFFORT="${MODEL_REASONING_EFFORT:-medium}"
SUMMARY_MODEL="${SUMMARY_MODEL:-$MODEL}"
SUMMARY_REASONING_EFFORT="${SUMMARY_REASONING_EFFORT:-medium}"
EXPERIENCE_UPDATER="${EXPERIENCE_UPDATER:-codex}"
EXPERIENCE_TOKEN_BUDGET="${EXPERIENCE_TOKEN_BUDGET:-2048}"
CODEX_PROVIDER="${CODEX_PROVIDER:-none}"
CODEX_PROVIDER_WIRE_API="${CODEX_PROVIDER_WIRE_API:-responses}"
CODEX_PROVIDER_ENV_KEY="${CODEX_PROVIDER_ENV_KEY:-LLM_GATEWAY_API_KEY}"
CODEX_LAUNCH_STAGGER_SECONDS="${CODEX_LAUNCH_STAGGER_SECONDS:-5}"
CODEX_RATE_LIMIT_RETRIES="${CODEX_RATE_LIMIT_RETRIES:-3}"
CODEX_RATE_LIMIT_STAGGER_SECONDS="${CODEX_RATE_LIMIT_STAGGER_SECONDS:-5}"
CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS="${CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS:-0}"
INFRA_FAILURE_PAUSE_SECONDS="${INFRA_FAILURE_PAUSE_SECONDS:-1800}"
MAX_CONSECUTIVE_INFRA_FAILURES="${MAX_CONSECUTIVE_INFRA_FAILURES:-5}"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"

provider_args=()
if [[ -n "$CODEX_PROVIDER" && "$CODEX_PROVIDER" != "none" ]]; then
  provider_args+=(--codex-provider "$CODEX_PROVIDER")
  if [[ -n "${CODEX_PROVIDER_BASE_URL:-}" ]]; then
    provider_args+=(--codex-provider-base-url "$CODEX_PROVIDER_BASE_URL")
  fi
  provider_args+=(--codex-provider-wire-api "$CODEX_PROVIDER_WIRE_API")
  provider_args+=(--codex-provider-env-key "$CODEX_PROVIDER_ENV_KEY")
fi

echo "{\"event\":\"gpt54mini_seqself_from_bo8r1_launcher_start\",\"run_id\":\"${RUN_ID}\",\"seed_output\":\"${SEED_OUTPUT}\",\"output\":\"${OUTPUT}\",\"workers\":${WORKERS},\"seed_workers\":${SEED_WORKERS},\"max_attempts\":${MAX_ATTEMPTS},\"model\":\"${MODEL}\",\"provider\":\"${CODEX_PROVIDER}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"

.venv/bin/python scripts/seed_seq_self_from_bo8_round1.py \
  --seed-output "$SEED_OUTPUT" \
  --run-id "$RUN_ID" \
  --output "$OUTPUT" \
  --split eval \
  --difficulty level1 \
  --max-tasks 200 \
  --max-attempts "$MAX_ATTEMPTS" \
  --workers "$SEED_WORKERS" \
  --runner-workers "$WORKERS" \
  --model "$MODEL" \
  --model-reasoning-effort "$MODEL_REASONING_EFFORT" \
  --summary-model "$SUMMARY_MODEL" \
  --summary-reasoning-effort "$SUMMARY_REASONING_EFFORT" \
  --experience-updater "$EXPERIENCE_UPDATER" \
  --experience-token-budget "$EXPERIENCE_TOKEN_BUDGET" \
  "${provider_args[@]}" >> "$LOG_FILE" 2>&1

echo "{\"event\":\"gpt54mini_seqself_from_bo8r1_seed_done\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"

.venv/bin/python scripts/run_codex_cybergym_sequential_self_involving.py \
  --run-id "$RUN_ID" \
  --split eval \
  --difficulty level1 \
  --output "$OUTPUT" \
  --workers "$WORKERS" \
  --max-attempts "$MAX_ATTEMPTS" \
  --experience-token-budget "$EXPERIENCE_TOKEN_BUDGET" \
  --model "$MODEL" \
  --model-reasoning-effort "$MODEL_REASONING_EFFORT" \
  --summary-model "$SUMMARY_MODEL" \
  --summary-reasoning-effort "$SUMMARY_REASONING_EFFORT" \
  --experience-updater "$EXPERIENCE_UPDATER" \
  --codex-timeout-seconds 900 \
  --summary-timeout-seconds 600 \
  --submit-timeout-seconds 1800 \
  --server "$CYBERGYM_SERVER" \
  --pocdb-path runs/cybergym_server/poc.db \
  --env-file .env \
  --max-consecutive-infra-failures "$MAX_CONSECUTIVE_INFRA_FAILURES" \
  --infra-failure-pause-seconds "$INFRA_FAILURE_PAUSE_SECONDS" \
  --codex-launch-stagger-seconds "$CODEX_LAUNCH_STAGGER_SECONDS" \
  --codex-rate-limit-retries "$CODEX_RATE_LIMIT_RETRIES" \
  --codex-rate-limit-stagger-seconds "$CODEX_RATE_LIMIT_STAGGER_SECONDS" \
  --codex-rate-limit-min-sleep-seconds "$CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS" \
  "${provider_args[@]}" >> "$LOG_FILE" 2>&1

exit_code=$?
echo "{\"event\":\"gpt54mini_seqself_from_bo8r1_launcher_complete\",\"run_id\":\"${RUN_ID}\",\"output\":\"${OUTPUT}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"
exit "$exit_code"
