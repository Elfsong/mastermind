#!/usr/bin/env bash
set -u

cd /data00/home/mz.du/Projects/mastermind || exit 1

LOG="runs/codex_cybergym/level3_logs/level3_after_rep8_watch_20260528T1139Z.log"
PID_FILE="runs/codex_cybergym/level3_logs/level3_usage_limit_resume.pid"
run_id="codex-gateway-eval-level3-r1-20260528T1140Z"
output="runs/codex_gateway_eval_level3_rep1_rollouts.jsonl"

echo "$$" > "$PID_FILE"
echo "{\"event\":\"level3_usage_limit_resume_start\",\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"workers\":6}" >> "$LOG"

.venv/bin/python scripts/run_codex_cybergym_tasks_parallel.py \
  --split eval \
  --difficulty level3 \
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

echo "{\"event\":\"level3_usage_limit_resume_complete\",\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"exit_code\":${exit_code}}" >> "$LOG"
exit "$exit_code"
