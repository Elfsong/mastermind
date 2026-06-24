#!/usr/bin/env bash
set -u

cd /data00/home/mz.du/Projects/mastermind || exit 1

LOG="runs/codex_cybergym/eval_rep_logs/eval_reps_3_8_20260527T1439Z.log"
PID_FILE="runs/codex_cybergym/eval_rep_logs/eval_reps_7_8_usage_limit_resume.pid"
echo "$$" > "$PID_FILE"

for rep in 7 8; do
  run_id="codex-gateway-eval-bo4-r${rep}-20260527T1439Z"
  output="runs/codex_gateway_eval_bo4_rep${rep}_rollouts.jsonl"
  echo "{\"event\":\"usage_limit_resume_start\",\"rep\":${rep},\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"workers\":6}" >> "$LOG"

  .venv/bin/python scripts/run_codex_cybergym_tasks_parallel.py \
    --split eval \
    --difficulty level1 \
    --run-id "$run_id" \
    --run-root runs/codex_cybergym \
    --output "$output" \
    --workers 6 \
    --model gpt-5.5 \
    --model-reasoning-effort medium \
    --codex-timeout-seconds 900 \
    --submit-timeout-seconds 1800 \
    --server http://127.0.0.1:8666 \
    --pocdb-path runs/cybergym_server/poc.db \
    --env-file .env \
    --sandbox workspace-write >> "$LOG" 2>&1
  exit_code=$?

  echo "{\"event\":\"usage_limit_resume_complete\",\"rep\":${rep},\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"exit_code\":${exit_code}}" >> "$LOG"
  if [ "$exit_code" -ne 0 ]; then
    echo "{\"event\":\"usage_limit_resume_stop\",\"rep\":${rep},\"run_id\":\"${run_id}\",\"reason\":\"runner_exit_nonzero\",\"exit_code\":${exit_code}}" >> "$LOG"
    exit "$exit_code"
  fi
done
