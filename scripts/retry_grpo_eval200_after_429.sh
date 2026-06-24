#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MASTERMIND_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT"

GRPO_RUN_DIR="${GRPO_RUN_DIR:?Set GRPO_RUN_DIR to a completed GRPO run directory.}"

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_FILE="${LOG_FILE:-runs/strategy_grpo_eval/retry_grpo_eval200_after_429_${RUN_STAMP}.log}"

INITIAL_SLEEP_SECONDS="${INITIAL_SLEEP_SECONDS:-0}"
MAX_SMOKE_TRIES="${MAX_SMOKE_TRIES:-12}"
SMOKE_SLEEP_SECONDS="${SMOKE_SLEEP_SECONDS:-1800}"
SMOKE_WORKERS="${SMOKE_WORKERS:-1}"
SMOKE_MAX_TASKS="${SMOKE_MAX_TASKS:-1}"
SMOKE_MAX_ATTEMPTS="${SMOKE_MAX_ATTEMPTS:-1}"
SMOKE_TASK_OFFSET="${SMOKE_TASK_OFFSET:-0}"
SMOKE_CODEX_RATE_LIMIT_RETRIES="${SMOKE_CODEX_RATE_LIMIT_RETRIES:-3}"
SMOKE_CODEX_RATE_LIMIT_STAGGER_SECONDS="${SMOKE_CODEX_RATE_LIMIT_STAGGER_SECONDS:-5}"
SMOKE_CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS="${SMOKE_CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS:-0}"
POST_SMOKE_SLEEP_SECONDS="${POST_SMOKE_SLEEP_SECONDS:-0}"

EVAL_RUN_ID="${EVAL_RUN_ID:-codex-gpt55-eval200-qwen36-grpo-retry-${RUN_STAMP}}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
EVAL_MAX_TASKS="${EVAL_MAX_TASKS:-200}"
EVAL_MAX_ATTEMPTS="${EVAL_MAX_ATTEMPTS:-8}"
EVAL_TASK_OFFSET="${EVAL_TASK_OFFSET:-0}"
EVAL_CODEX_RATE_LIMIT_RETRIES="${EVAL_CODEX_RATE_LIMIT_RETRIES:-3}"
EVAL_CODEX_RATE_LIMIT_STAGGER_SECONDS="${EVAL_CODEX_RATE_LIMIT_STAGGER_SECONDS:-5}"
EVAL_CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS="${EVAL_CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS:-0}"
EVAL_ABORT_ON_INFRA="${EVAL_ABORT_ON_INFRA:-1}"
EVAL_MONITOR_SECONDS="${EVAL_MONITOR_SECONDS:-30}"
EVAL_ATTEMPT_COOLDOWN_SECONDS="${EVAL_ATTEMPT_COOLDOWN_SECONDS:-0}"
EVAL_CODEX_LAUNCH_STAGGER_SECONDS="${EVAL_CODEX_LAUNCH_STAGGER_SECONDS:-0}"

EXECUTOR_MODEL="${EXECUTOR_MODEL:-gpt-5.5}"
CODEX_PROVIDER="${CODEX_PROVIDER:-llmgw}"
GRPO_WAIT_TARGET="${GRPO_WAIT_TARGET:-final}"
POLL_SECONDS="${POLL_SECONDS:-10}"
STRICT_NO_429="${STRICT_NO_429:-0}"

mkdir -p "$(dirname "$LOG_FILE")"

json_string() {
  "$PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

log_event() {
  local event="$1"
  shift || true
  printf '{"event":"%s","time":"%s"' "$event" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
  while [[ $# -gt 0 ]]; do
    local key="$1"
    local value="$2"
    shift 2
    printf ',"%s":%s' "$key" "$value" | tee -a "$LOG_FILE"
  done
  printf '}\n' | tee -a "$LOG_FILE"
}

rollout_output_is_healthy() {
  local output="$1"
  local expected_rows="$2"
  "$PYTHON" - "$output" "$expected_rows" "$STRICT_NO_429" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
strict_no_429 = str(sys.argv[3]).lower() in {"1", "true", "yes"}
if not path.exists() or path.stat().st_size == 0:
    print("missing_output")
    raise SystemExit(2)

has_429 = False
infra_failure = False
rows = 0
statuses = {}
for line in path.read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    rows += 1
    row = json.loads(line)
    status = row.get("status") or "UNKNOWN"
    statuses[status] = statuses.get(status, 0) + 1
    if status in {"AGENT_ERROR", "CRASH"}:
        infra_failure = True
    metadata = row.get("metadata") or {}
    for event in metadata.get("codex_retry_events") or []:
        if str(event.get("reason")) == "429":
            has_429 = True
    update_status = ((metadata.get("sequential") or {}).get("experience_update") or {}).get("update_status")
    if update_status == "skipped_infra_failure":
        infra_failure = True
    stdout = metadata.get("codex_stdout")
    if stdout:
        stdout_path = Path(stdout)
        if stdout_path.exists() and "429 Too Many Requests" in stdout_path.read_text(errors="replace"):
            has_429 = True

healthy = rows >= expected_rows and not infra_failure and not (strict_no_429 and has_429)
print(json.dumps({
    "rows": rows,
    "expected_rows": expected_rows,
    "statuses": statuses,
    "has_429": has_429,
    "infra_failure": infra_failure,
    "healthy": healthy,
    "strict_no_429": strict_no_429,
}, sort_keys=True))
raise SystemExit(0 if healthy else 1)
PY
}

rollout_output_has_infra_failure() {
  local output="$1"
  "$PYTHON" - "$output" "$STRICT_NO_429" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
strict_no_429 = str(sys.argv[2]).lower() in {"1", "true", "yes"}
if not path.exists() or path.stat().st_size == 0:
    print(json.dumps({
        "rows": 0,
        "has_429": False,
        "infra_failure": False,
        "strict_no_429": strict_no_429,
    }, sort_keys=True))
    raise SystemExit(0)

has_429 = False
infra_failure = False
rows = 0
statuses = {}
for line in path.read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    rows += 1
    row = json.loads(line)
    status = row.get("status") or "UNKNOWN"
    statuses[status] = statuses.get(status, 0) + 1
    if status in {"AGENT_ERROR", "CRASH"}:
        infra_failure = True
    metadata = row.get("metadata") or {}
    for event in metadata.get("codex_retry_events") or []:
        if str(event.get("reason")) == "429":
            has_429 = True
    update_status = ((metadata.get("sequential") or {}).get("experience_update") or {}).get("update_status")
    if update_status == "skipped_infra_failure":
        infra_failure = True
    stdout = metadata.get("codex_stdout")
    if stdout:
        stdout_path = Path(stdout)
        if stdout_path.exists() and "429 Too Many Requests" in stdout_path.read_text(errors="replace"):
            has_429 = True

unhealthy = infra_failure or (strict_no_429 and has_429)
print(json.dumps({
    "rows": rows,
    "statuses": statuses,
    "has_429": has_429,
    "infra_failure": infra_failure,
    "unhealthy": unhealthy,
    "strict_no_429": strict_no_429,
}, sort_keys=True))
raise SystemExit(1 if unhealthy else 0)
PY
}

log_event "start" \
  "grpo_run_dir" "$(json_string "$GRPO_RUN_DIR")" \
  "eval_run_id" "$(json_string "$EVAL_RUN_ID")" \
  "max_smoke_tries" "$MAX_SMOKE_TRIES" \
  "smoke_sleep_seconds" "$SMOKE_SLEEP_SECONDS" \
  "smoke_task_offset" "$SMOKE_TASK_OFFSET" \
  "smoke_codex_rate_limit_retries" "$SMOKE_CODEX_RATE_LIMIT_RETRIES" \
  "smoke_codex_rate_limit_stagger_seconds" "$SMOKE_CODEX_RATE_LIMIT_STAGGER_SECONDS" \
  "smoke_codex_rate_limit_min_sleep_seconds" "$SMOKE_CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS" \
  "initial_sleep_seconds" "$INITIAL_SLEEP_SECONDS" \
  "post_smoke_sleep_seconds" "$POST_SMOKE_SLEEP_SECONDS" \
  "strict_no_429" "$STRICT_NO_429" \
  "eval_abort_on_infra" "$EVAL_ABORT_ON_INFRA" \
  "eval_monitor_seconds" "$EVAL_MONITOR_SECONDS" \
  "eval_attempt_cooldown_seconds" "$EVAL_ATTEMPT_COOLDOWN_SECONDS" \
  "eval_codex_launch_stagger_seconds" "$EVAL_CODEX_LAUNCH_STAGGER_SECONDS" \
  "eval_codex_rate_limit_retries" "$EVAL_CODEX_RATE_LIMIT_RETRIES" \
  "eval_codex_rate_limit_stagger_seconds" "$EVAL_CODEX_RATE_LIMIT_STAGGER_SECONDS" \
  "eval_codex_rate_limit_min_sleep_seconds" "$EVAL_CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS"

if (( INITIAL_SLEEP_SECONDS > 0 )); then
  log_event "initial_sleep" "sleep_seconds" "$INITIAL_SLEEP_SECONDS"
  sleep "$INITIAL_SLEEP_SECONDS"
fi

for attempt in $(seq 1 "$MAX_SMOKE_TRIES"); do
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  smoke_run_id="codex-gpt55-eval-smoke-qwen36-grpo-${RUN_STAMP}-try${attempt}-${stamp}"
  smoke_output="runs/strategy_grpo_eval/${smoke_run_id}/grpo_qwen36_rollouts.jsonl"

  log_event "smoke_start" \
    "attempt" "$attempt" \
    "smoke_run_id" "$(json_string "$smoke_run_id")"

  GRPO_RUN_DIR="$GRPO_RUN_DIR" \
  EVAL_RUN_ID="$smoke_run_id" \
  GRPO_WAIT_TARGET="$GRPO_WAIT_TARGET" \
  WORKERS="$SMOKE_WORKERS" \
  MAX_TASKS="$SMOKE_MAX_TASKS" \
  MAX_ATTEMPTS="$SMOKE_MAX_ATTEMPTS" \
  TASK_OFFSET="$SMOKE_TASK_OFFSET" \
  CODEX_RATE_LIMIT_RETRIES="$SMOKE_CODEX_RATE_LIMIT_RETRIES" \
  CODEX_RATE_LIMIT_STAGGER_SECONDS="$SMOKE_CODEX_RATE_LIMIT_STAGGER_SECONDS" \
  CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS="$SMOKE_CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS" \
  EXECUTOR_MODEL="$EXECUTOR_MODEL" \
  CODEX_PROVIDER="$CODEX_PROVIDER" \
  POLL_SECONDS="$POLL_SECONDS" \
  PYTHON="$PYTHON" \
  scripts/watch_strategy_grpo_then_eval200.sh >> "$LOG_FILE" 2>&1 || true

  set +e
  smoke_summary="$(rollout_output_is_healthy "$smoke_output" "$SMOKE_MAX_TASKS" 2>&1)"
  smoke_status=$?
  set -e

  log_event "smoke_complete" \
    "attempt" "$attempt" \
    "exit_code" "$smoke_status" \
    "summary" "$(json_string "$smoke_summary")"

  if [[ "$smoke_status" -eq 0 ]]; then
    if (( POST_SMOKE_SLEEP_SECONDS > 0 )); then
      log_event "post_smoke_sleep" \
        "attempt" "$attempt" \
        "sleep_seconds" "$POST_SMOKE_SLEEP_SECONDS"
      sleep "$POST_SMOKE_SLEEP_SECONDS"
    fi

    log_event "smoke_healthy_starting_eval200" \
      "attempt" "$attempt" \
      "eval_run_id" "$(json_string "$EVAL_RUN_ID")"

    set +e
    eval_output="runs/strategy_grpo_eval/${EVAL_RUN_ID}/grpo_qwen36_rollouts.jsonl"
    if [[ "$EVAL_ABORT_ON_INFRA" =~ ^(1|true|yes)$ ]]; then
      GRPO_RUN_DIR="$GRPO_RUN_DIR" \
      EVAL_RUN_ID="$EVAL_RUN_ID" \
      GRPO_WAIT_TARGET="$GRPO_WAIT_TARGET" \
      WORKERS="$EVAL_WORKERS" \
      MAX_TASKS="$EVAL_MAX_TASKS" \
      MAX_ATTEMPTS="$EVAL_MAX_ATTEMPTS" \
      TASK_OFFSET="$EVAL_TASK_OFFSET" \
      ATTEMPT_COOLDOWN_SECONDS="$EVAL_ATTEMPT_COOLDOWN_SECONDS" \
      CODEX_LAUNCH_STAGGER_SECONDS="$EVAL_CODEX_LAUNCH_STAGGER_SECONDS" \
      CODEX_RATE_LIMIT_RETRIES="$EVAL_CODEX_RATE_LIMIT_RETRIES" \
      CODEX_RATE_LIMIT_STAGGER_SECONDS="$EVAL_CODEX_RATE_LIMIT_STAGGER_SECONDS" \
      CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS="$EVAL_CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS" \
      EXECUTOR_MODEL="$EXECUTOR_MODEL" \
      CODEX_PROVIDER="$CODEX_PROVIDER" \
      POLL_SECONDS="$POLL_SECONDS" \
      PYTHON="$PYTHON" \
      setsid scripts/watch_strategy_grpo_then_eval200.sh >> "$LOG_FILE" 2>&1 &
      eval_pid=$!
      eval_aborted=0
      while kill -0 "$eval_pid" 2>/dev/null; do
        sleep "$EVAL_MONITOR_SECONDS"
        monitor_summary="$(rollout_output_has_infra_failure "$eval_output" 2>&1)"
        monitor_status=$?
        if [[ "$monitor_status" -ne 0 ]]; then
          eval_aborted=1
          log_event "eval200_abort_on_infra" \
            "eval_run_id" "$(json_string "$EVAL_RUN_ID")" \
            "summary" "$(json_string "$monitor_summary")"
          kill -TERM -- "-$eval_pid" 2>/dev/null || kill -TERM "$eval_pid" 2>/dev/null || true
          sleep 5
          kill -KILL -- "-$eval_pid" 2>/dev/null || true
          wait "$eval_pid"
          eval_exit=$?
          break
        fi
      done
      if [[ "$eval_aborted" -eq 0 ]]; then
        wait "$eval_pid"
        eval_exit=$?
      fi
    else
      GRPO_RUN_DIR="$GRPO_RUN_DIR" \
      EVAL_RUN_ID="$EVAL_RUN_ID" \
      GRPO_WAIT_TARGET="$GRPO_WAIT_TARGET" \
      WORKERS="$EVAL_WORKERS" \
      MAX_TASKS="$EVAL_MAX_TASKS" \
      MAX_ATTEMPTS="$EVAL_MAX_ATTEMPTS" \
      TASK_OFFSET="$EVAL_TASK_OFFSET" \
      ATTEMPT_COOLDOWN_SECONDS="$EVAL_ATTEMPT_COOLDOWN_SECONDS" \
      CODEX_LAUNCH_STAGGER_SECONDS="$EVAL_CODEX_LAUNCH_STAGGER_SECONDS" \
      CODEX_RATE_LIMIT_RETRIES="$EVAL_CODEX_RATE_LIMIT_RETRIES" \
      CODEX_RATE_LIMIT_STAGGER_SECONDS="$EVAL_CODEX_RATE_LIMIT_STAGGER_SECONDS" \
      CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS="$EVAL_CODEX_RATE_LIMIT_MIN_SLEEP_SECONDS" \
      EXECUTOR_MODEL="$EXECUTOR_MODEL" \
      CODEX_PROVIDER="$CODEX_PROVIDER" \
      POLL_SECONDS="$POLL_SECONDS" \
      PYTHON="$PYTHON" \
      scripts/watch_strategy_grpo_then_eval200.sh >> "$LOG_FILE" 2>&1
      eval_exit=$?
    fi
    set -e

    set +e
    eval_summary="$(rollout_output_is_healthy "$eval_output" "$EVAL_MAX_TASKS" 2>&1)"
    eval_health_status=$?
    set -e

    if [[ "$eval_exit" -ne 0 || "$eval_health_status" -ne 0 ]]; then
      log_event "eval200_invalid" \
        "eval_run_id" "$(json_string "$EVAL_RUN_ID")" \
        "eval_exit" "$eval_exit" \
        "health_exit" "$eval_health_status" \
        "summary" "$(json_string "$eval_summary")"
      exit 76
    fi

    log_event "eval200_complete" \
      "eval_run_id" "$(json_string "$EVAL_RUN_ID")" \
      "summary" "$(json_string "$eval_summary")"
    exit 0
  fi

  if [[ "$attempt" -lt "$MAX_SMOKE_TRIES" ]]; then
    log_event "sleep_before_retry" \
      "attempt" "$attempt" \
      "sleep_seconds" "$SMOKE_SLEEP_SECONDS"
    sleep "$SMOKE_SLEEP_SECONDS"
  fi
done

log_event "exhausted_smoke_retries" "max_smoke_tries" "$MAX_SMOKE_TRIES"
exit 75
