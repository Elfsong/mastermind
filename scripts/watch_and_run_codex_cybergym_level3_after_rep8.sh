#!/usr/bin/env bash
set -u

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT" || exit 1

LOG_DIR="runs/codex_cybergym/level3_logs"
mkdir -p "$LOG_DIR"

WATCH_STAMP="$(date -u +%Y%m%dT%H%MZ)"
PID_FILE="$LOG_DIR/level3_after_rep8_watch_${WATCH_STAMP}.pid"
LOG_FILE="$LOG_DIR/level3_after_rep8_watch_${WATCH_STAMP}.log"
echo "$$" > "$PID_FILE"

round8_output="runs/codex_gateway_eval_bo4_rep8_rollouts.jsonl"

echo "{\"event\":\"level3_after_rep8_watch_start\",\"pid\":$$,\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"

while true; do
  lines=0
  if [ -f "$round8_output" ]; then
    lines="$(wc -l < "$round8_output")"
  fi

  active_runner="$(pgrep -af 'run_codex_cybergym_tasks_parallel.py .*codex-gateway-eval-bo4-r8-20260527T1439Z' | wc -l)"
  echo "{\"event\":\"level3_after_rep8_watch_tick\",\"round8_lines\":${lines},\"active_round8_runner\":${active_runner},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"

  if [ "$lines" -ge 200 ] && [ "$active_runner" -eq 0 ]; then
    break
  fi
  sleep 60
done

RUN_STAMP="$(date -u +%Y%m%dT%H%MZ)"
echo "{\"event\":\"level3_start_after_rep8\",\"run_stamp\":\"${RUN_STAMP}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"
RUN_STAMP="$RUN_STAMP" scripts/run_codex_cybergym_level3_once.sh >> "$LOG_FILE" 2>&1
exit_code=$?
echo "{\"event\":\"level3_after_rep8_complete\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"
exit "$exit_code"
