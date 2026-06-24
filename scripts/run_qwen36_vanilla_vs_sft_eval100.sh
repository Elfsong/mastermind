#!/usr/bin/env bash
set -euo pipefail

# Run a matched 100-task CyberGym eval comparing two strategy generators:
#   A. Vanilla Qwen3.6 strategy generator + Codex 5.5 executor
#   B. SFT Qwen3.6 strategy generator     + Codex 5.5 executor
#
# The two variants use the same deterministic task-id list.  Results are
# summarized with latest-task pass rates, matched-task pass rates, and pass@N.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MASTERMIND_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT"

source scripts/mastermind_env.sh

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_TASKS="${MAX_TASKS:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
WORKERS="${WORKERS:-6}"
SPLIT="${SPLIT:-eval}"
DIFFICULTY="${DIFFICULTY:-level1}"
TASK_OFFSET="${TASK_OFFSET:-0}"

# Do not reuse scripts/mastermind_env.sh's EXECUTOR_MODEL default: that variable
# points at older local executor defaults in this workspace.  This eval must use
# Codex 5.5 unless the caller explicitly overrides CODEX_EXECUTOR_MODEL.
CODEX_EXECUTOR_MODEL="${CODEX_EXECUTOR_MODEL:-gpt-5.5}"
TINKER_SUMMARY_MODEL="${TINKER_SUMMARY_MODEL:-Qwen/Qwen3.6-35B-A3B}"
TINKER_SUMMARY_TEMPERATURE="${TINKER_SUMMARY_TEMPERATURE:-0}"
TINKER_SUMMARY_TOP_P="${TINKER_SUMMARY_TOP_P:-1}"
EXPERIENCE_TOKEN_BUDGET="${EXPERIENCE_TOKEN_BUDGET:-2048}"

PROVIDER="${CODEX_PROVIDER:-llmgw}"
CODEX_RATE_LIMIT_RETRIES="${CODEX_RATE_LIMIT_RETRIES:-3}"
CODEX_RATE_LIMIT_STAGGER_SECONDS="${CODEX_RATE_LIMIT_STAGGER_SECONDS:-5}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-900}"
SUBMIT_TIMEOUT_SECONDS="${SUBMIT_TIMEOUT_SECONDS:-1800}"
SUMMARY_TIMEOUT_SECONDS="${SUMMARY_TIMEOUT_SECONDS:-600}"
MAX_AUTO_SUBMIT_CANDIDATES="${MAX_AUTO_SUBMIT_CANDIDATES:-5}"

RUN_ROOT="${RUN_ROOT:-runs/codex_cybergym}"
LOG_DIR="${LOG_DIR:-runs/codex_cybergym/sequential_self_logs}"
EVAL_DIR="${EVAL_DIR:-runs/strategy_sft/vanilla_vs_sft_eval100_${RUN_STAMP}}"
mkdir -p "$EVAL_DIR" "$LOG_DIR"

ORCH_LOG="${ORCH_LOG:-${EVAL_DIR}/run.log}"
CONFIG_JSON="${CONFIG_JSON:-${EVAL_DIR}/run_config.json}"
TASK_IDS_PATH="${TASK_IDS_PATH:-${EVAL_DIR}/task_ids.txt}"

VANILLA_RUN_ID="${VANILLA_RUN_ID:-codex-gpt55-eval100-tinker-qwen36-vanilla-${RUN_STAMP}}"
SFT_RUN_ID="${SFT_RUN_ID:-codex-gpt55-eval100-tinker-qwen36-sft-${RUN_STAMP}}"
VANILLA_OUTPUT="${VANILLA_OUTPUT:-${EVAL_DIR}/vanilla_qwen36_rollouts.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-${EVAL_DIR}/sft_qwen36_rollouts.jsonl}"

SFT_RESULT_JSON="${SFT_RESULT_JSON:-runs/strategy_sft/qwen36-strategy-sft-20260601T1935Z/result.json}"
if [[ -z "${SFT_MODEL_PATH:-}" ]]; then
  if [[ -f "$SFT_RESULT_JSON" ]]; then
    SFT_MODEL_PATH="$("$PYTHON" - "$SFT_RESULT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
except Exception:
    data = {}
print(data.get("final_sampler_path") or "")
PY
)"
  else
    SFT_MODEL_PATH=""
  fi
fi
if [[ -z "$SFT_MODEL_PATH" ]]; then
  echo "ERROR: SFT_MODEL_PATH is empty. Set SFT_MODEL_PATH=tinker://.../final-sampler or SFT_RESULT_JSON=<result.json>." >&2
  exit 2
fi

export RUN_STAMP EVAL_DIR
if [[ "${DETACH:-0}" == "1" && "${MM_EVAL100_DETACHED:-0}" != "1" ]]; then
  export MM_EVAL100_DETACHED=1
  nohup "$0" "$@" > "${EVAL_DIR}/orchestrator.nohup.log" 2>&1 &
  echo "$!" > "${EVAL_DIR}/orchestrator.pid"
  echo "Detached eval runner."
  echo "  pid:      $(cat "${EVAL_DIR}/orchestrator.pid")"
  echo "  eval dir: ${EVAL_DIR}"
  echo "  log:      ${EVAL_DIR}/orchestrator.nohup.log"
  exit 0
fi

if [[ -f "$TASK_IDS_PATH" && "${REGENERATE_TASK_IDS:-0}" != "1" ]]; then
  echo "Using existing task-id list: ${TASK_IDS_PATH}" | tee -a "$ORCH_LOG"
elif [[ -n "${TASK_IDS_FILE:-}" ]]; then
  cp "$TASK_IDS_FILE" "$TASK_IDS_PATH"
  echo "Copied task-id list from ${TASK_IDS_FILE} to ${TASK_IDS_PATH}" | tee -a "$ORCH_LOG"
else
  "$PYTHON" - "$SPLIT" "$TASK_OFFSET" "$MAX_TASKS" "$TASK_IDS_PATH" <<'PY'
import json
import sys
from pathlib import Path

split = sys.argv[1]
offset = int(sys.argv[2])
max_tasks = int(sys.argv[3])
out = Path(sys.argv[4])

manifest_path = Path("mastermind.manifest.json")
manifest = json.loads(manifest_path.read_text())
split_path = Path(manifest["task_splits"][split])
task_ids = []
for line in split_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        task_ids.append(line)
selected = task_ids[offset : offset + max_tasks]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(selected) + ("\n" if selected else ""))
print(f"Wrote {len(selected)} task ids to {out}")
PY
fi

mapfile -t TASK_IDS < "$TASK_IDS_PATH"
TASK_COUNT="${#TASK_IDS[@]}"
if [[ "$TASK_COUNT" -eq 0 ]]; then
  echo "ERROR: no task ids in ${TASK_IDS_PATH}" >&2
  exit 2
fi
if [[ "$TASK_COUNT" -ne "$MAX_TASKS" ]]; then
  echo "WARNING: task-id count (${TASK_COUNT}) != MAX_TASKS (${MAX_TASKS})." | tee -a "$ORCH_LOG"
fi

TASK_ID_ARGS=()
for task_id in "${TASK_IDS[@]}"; do
  TASK_ID_ARGS+=(--task-id "$task_id")
done

export \
  RUN_STAMP EVAL_DIR ORCH_LOG CONFIG_JSON TASK_IDS_PATH TASK_COUNT \
  VANILLA_RUN_ID SFT_RUN_ID VANILLA_OUTPUT SFT_OUTPUT SFT_MODEL_PATH \
  MAX_TASKS MAX_ATTEMPTS WORKERS SPLIT DIFFICULTY TASK_OFFSET \
  CODEX_EXECUTOR_MODEL TINKER_SUMMARY_MODEL TINKER_SUMMARY_TEMPERATURE TINKER_SUMMARY_TOP_P \
  EXPERIENCE_TOKEN_BUDGET PROVIDER CODEX_RATE_LIMIT_RETRIES CODEX_RATE_LIMIT_STAGGER_SECONDS \
  CODEX_TIMEOUT_SECONDS SUBMIT_TIMEOUT_SECONDS SUMMARY_TIMEOUT_SECONDS MAX_AUTO_SUBMIT_CANDIDATES \
  RUN_ROOT LOG_DIR

"$PYTHON" - <<'PY'
import json
import os
from datetime import UTC, datetime
from pathlib import Path

keys = [
    "RUN_STAMP",
    "EVAL_DIR",
    "TASK_IDS_PATH",
    "TASK_COUNT",
    "MAX_TASKS",
    "MAX_ATTEMPTS",
    "WORKERS",
    "SPLIT",
    "DIFFICULTY",
    "TASK_OFFSET",
    "CODEX_EXECUTOR_MODEL",
    "TINKER_SUMMARY_MODEL",
    "TINKER_SUMMARY_TEMPERATURE",
    "TINKER_SUMMARY_TOP_P",
    "EXPERIENCE_TOKEN_BUDGET",
    "PROVIDER",
    "VANILLA_RUN_ID",
    "SFT_RUN_ID",
    "VANILLA_OUTPUT",
    "SFT_OUTPUT",
    "SFT_MODEL_PATH",
    "RUN_ROOT",
    "LOG_DIR",
]
config = {key: os.environ.get(key) for key in keys}
config["created_at"] = datetime.now(UTC).isoformat()
Path(os.environ["CONFIG_JSON"]).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

echo "===== Vanilla vs SFT Qwen3.6 eval100 =====" | tee -a "$ORCH_LOG"
echo "eval_dir=${EVAL_DIR}" | tee -a "$ORCH_LOG"
echo "vanilla_run_id=${VANILLA_RUN_ID}" | tee -a "$ORCH_LOG"
echo "sft_run_id=${SFT_RUN_ID}" | tee -a "$ORCH_LOG"
echo "task_ids=${TASK_IDS_PATH} (${TASK_COUNT})" | tee -a "$ORCH_LOG"
echo "sft_model_path=${SFT_MODEL_PATH}" | tee -a "$ORCH_LOG"

run_variant() {
  local variant="$1"
  local run_id="$2"
  local output="$3"
  local model_path="$4"
  local variant_log="${LOG_DIR}/${run_id}.log"

  local -a provider_args=()
  if [[ -n "$PROVIDER" && "$PROVIDER" != "none" ]]; then
    provider_args+=(--codex-provider "$PROVIDER")
    if [[ -n "${CODEX_PROVIDER_BASE_URL:-}" ]]; then
      provider_args+=(--codex-provider-base-url "$CODEX_PROVIDER_BASE_URL")
    fi
    provider_args+=(--codex-provider-wire-api "${CODEX_PROVIDER_WIRE_API:-responses}")
    provider_args+=(--codex-provider-env-key "${CODEX_PROVIDER_ENV_KEY:-LLM_GATEWAY_API_KEY}")
  fi

  local -a model_path_args=()
  if [[ -n "$model_path" ]]; then
    model_path_args+=(--tinker-summary-model-path "$model_path")
  fi

  {
    echo "{\"event\":\"variant_start\",\"variant\":\"${variant}\",\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
    echo "{\"event\":\"strategy_generator\",\"variant\":\"${variant}\",\"summary_model\":\"${TINKER_SUMMARY_MODEL}\",\"model_path\":\"${model_path}\"}"
  } | tee -a "$ORCH_LOG" "$variant_log"

  set +e
  "$PYTHON" scripts/run_codex_cybergym_sequential_self_involving.py \
    --run-id "$run_id" \
    --run-root "$RUN_ROOT" \
    --split "$SPLIT" \
    --difficulty "$DIFFICULTY" \
    --output "$output" \
    "${TASK_ID_ARGS[@]}" \
    --max-tasks "$MAX_TASKS" \
    --workers "$WORKERS" \
    --max-attempts "$MAX_ATTEMPTS" \
    --model "$CODEX_EXECUTOR_MODEL" \
    --experience-updater tinker \
    --summary-model "$TINKER_SUMMARY_MODEL" \
    --tinker-summary-model "$TINKER_SUMMARY_MODEL" \
    "${model_path_args[@]}" \
    --tinker-summary-temperature "$TINKER_SUMMARY_TEMPERATURE" \
    --tinker-summary-top-p "$TINKER_SUMMARY_TOP_P" \
    --experience-token-budget "$EXPERIENCE_TOKEN_BUDGET" \
    --codex-timeout-seconds "$CODEX_TIMEOUT_SECONDS" \
    --summary-timeout-seconds "$SUMMARY_TIMEOUT_SECONDS" \
    --submit-timeout-seconds "$SUBMIT_TIMEOUT_SECONDS" \
    --server "${CYBERGYM_SERVER:-http://127.0.0.1:8666}" \
    --pocdb-path "${POCDB_PATH:-runs/cybergym_server/poc.db}" \
    --max-auto-submit-candidates "$MAX_AUTO_SUBMIT_CANDIDATES" \
    --codex-rate-limit-retries "$CODEX_RATE_LIMIT_RETRIES" \
    --codex-rate-limit-stagger-seconds "$CODEX_RATE_LIMIT_STAGGER_SECONDS" \
    "${provider_args[@]}" \
    2>&1 | tee -a "$ORCH_LOG" "$variant_log"
  local exit_code=${PIPESTATUS[0]}
  set -e

  echo "{\"event\":\"variant_complete\",\"variant\":\"${variant}\",\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"exit_code\":${exit_code},\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    | tee -a "$ORCH_LOG" "$variant_log"
  return "$exit_code"
}

VANILLA_EXIT=0
SFT_EXIT=0
if [[ "${COMPARE_ONLY:-0}" != "1" ]]; then
  case "${ORDER:-vanilla_then_sft}" in
    vanilla_then_sft)
      run_variant "vanilla_qwen36" "$VANILLA_RUN_ID" "$VANILLA_OUTPUT" "" || VANILLA_EXIT=$?
      run_variant "sft_qwen36" "$SFT_RUN_ID" "$SFT_OUTPUT" "$SFT_MODEL_PATH" || SFT_EXIT=$?
      ;;
    sft_then_vanilla)
      run_variant "sft_qwen36" "$SFT_RUN_ID" "$SFT_OUTPUT" "$SFT_MODEL_PATH" || SFT_EXIT=$?
      run_variant "vanilla_qwen36" "$VANILLA_RUN_ID" "$VANILLA_OUTPUT" "" || VANILLA_EXIT=$?
      ;;
    *)
      echo "ERROR: ORDER must be vanilla_then_sft or sft_then_vanilla." >&2
      exit 2
      ;;
  esac
else
  echo "COMPARE_ONLY=1: skipping runner execution and summarizing existing outputs." | tee -a "$ORCH_LOG"
fi

SUMMARY_JSON="${EVAL_DIR}/summary.json"
COMPARE_JSON="${EVAL_DIR}/comparison.json"
COMPARE_TXT="${EVAL_DIR}/comparison.txt"
REPORT_MD="${EVAL_DIR}/report.md"

set +e
"$PYTHON" scripts/summarize_cybergym_pass_rates.py "$VANILLA_OUTPUT" "$SFT_OUTPUT" > "$SUMMARY_JSON"
SUMMARY_EXIT=$?
"$PYTHON" scripts/compare_cybergym_strategy_eval.py \
  --base "$VANILLA_OUTPUT" \
  --sft "$SFT_OUTPUT" \
  --base-label vanilla_qwen36 \
  --sft-label sft_qwen36 \
  --max-tasks "$MAX_TASKS" \
  --max-attempts "$MAX_ATTEMPTS" \
  --format json > "$COMPARE_JSON"
COMPARE_JSON_EXIT=$?
"$PYTHON" scripts/compare_cybergym_strategy_eval.py \
  --base "$VANILLA_OUTPUT" \
  --sft "$SFT_OUTPUT" \
  --base-label vanilla_qwen36 \
  --sft-label sft_qwen36 \
  --max-tasks "$MAX_TASKS" \
  --max-attempts "$MAX_ATTEMPTS" \
  --format text > "$COMPARE_TXT"
COMPARE_TXT_EXIT=$?
set -e

if [[ "$SUMMARY_EXIT" -ne 0 || "$COMPARE_JSON_EXIT" -ne 0 || "$COMPARE_TXT_EXIT" -ne 0 ]]; then
  echo "WARNING: one or more summary commands failed: summary=${SUMMARY_EXIT}, compare_json=${COMPARE_JSON_EXIT}, compare_text=${COMPARE_TXT_EXIT}" | tee -a "$ORCH_LOG"
fi

"$PYTHON" - "$COMPARE_JSON" "$SUMMARY_JSON" "$REPORT_MD" "$VANILLA_OUTPUT" "$SFT_OUTPUT" "$TASK_IDS_PATH" "$CONFIG_JSON" <<'PY'
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "scripts")
from summarize_cybergym_pass_rates import attempt_index, is_infra_failure, load_rows, milestone  # noqa: E402

compare_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
vanilla_output = Path(sys.argv[4])
sft_output = Path(sys.argv[5])
task_ids_path = Path(sys.argv[6])
config_path = Path(sys.argv[7])

def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def pct(value):
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"

def best_of_curve(path: Path, task_ids: list[str], max_attempts: int):
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in load_rows(path):
        if is_infra_failure(row):
            continue
        task_id = row.get("task_id")
        if isinstance(task_id, str) and task_id in task_ids:
            by_task[task_id].append(row)
    rows = []
    for n in range(1, max_attempts + 1):
        passed = 0
        m67 = 0
        for task_id in task_ids:
            attempts = [row for row in by_task.get(task_id, []) if attempt_index(row) <= n]
            if any(row.get("status") == "PASSED" for row in attempts):
                passed += 1
            if any(milestone(row) in {6, 7} for row in attempts):
                m67 += 1
        denom = len(task_ids)
        rows.append(
            {
                "n": n,
                "passed": passed,
                "pass_rate": passed / denom if denom else None,
                "milestone_6_or_7": m67,
                "milestone_6_or_7_rate": m67 / denom if denom else None,
            }
        )
    return rows

compare = read_json(compare_path, {})
summary = read_json(summary_path, [])
config = read_json(config_path, {})
task_ids = [line.strip() for line in task_ids_path.read_text().splitlines() if line.strip()]

matched_all = ((compare.get("matched") or {}).get("all_seen_common_tasks") or {})
matched_both = ((compare.get("matched") or {}).get("both_terminal_common_tasks") or {})
individual = compare.get("individual") or {}
progress = compare.get("progress") or {}
common_task_ids = matched_all.get("task_ids") or []
max_attempts = int(config.get("MAX_ATTEMPTS") or 8)

vanilla_curve = best_of_curve(vanilla_output, common_task_ids, max_attempts)
sft_curve = best_of_curve(sft_output, common_task_ids, max_attempts)

lines = [
    "# Vanilla Qwen3.6 vs SFT Qwen3.6 Strategy Generator Eval100 Report",
    "",
    f"- Generated at: `{datetime.now(UTC).isoformat()}`",
    f"- Executor: `Codex {config.get('CODEX_EXECUTOR_MODEL', 'gpt-5.5')}`",
    f"- Vanilla strategy generator: `{config.get('TINKER_SUMMARY_MODEL', 'Qwen/Qwen3.6-35B-A3B')}`",
    f"- SFT strategy generator: `{config.get('SFT_MODEL_PATH', 'n/a')}`",
    f"- Task list: `{task_ids_path}` (`{len(task_ids)}` tasks)",
    f"- Vanilla rollouts: `{vanilla_output}`",
    f"- SFT rollouts: `{sft_output}`",
    "",
    "## Progress",
    "",
    "| Variant | Latest tasks | Terminal tasks | Estimated remaining latest tasks |",
    "|---|---:|---:|---:|",
]
for label in ("vanilla_qwen36", "sft_qwen36"):
    row = progress.get(label, {})
    lines.append(
        f"| {label} | {row.get('latest_tasks', 0)} | {row.get('terminal_tasks', 0)} | "
        f"{row.get('estimated_remaining_latest_tasks', 0)} |"
    )

lines.extend(
    [
        "",
        "## Individual latest-task summary",
        "",
        "| Variant | Tasks | Passed | Pass rate | M6/M7 | M6/M7 rate | Infra rows ignored |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
)
for label in ("vanilla_qwen36", "sft_qwen36"):
    row = individual.get(label, {})
    lines.append(
        f"| {label} | {row.get('tasks', 0)} | {row.get('passed', 0)} | {pct(row.get('pass_rate'))} | "
        f"{row.get('milestone_6_or_7', 0)} | {pct(row.get('milestone_6_or_7_rate'))} | "
        f"{row.get('infra_rows_ignored', 0)} |"
    )

lines.extend(
    [
        "",
        "## Matched latest-task comparison (primary)",
        "",
        "| Variant | Tasks | Passed | Pass rate | M6/M7 | M6/M7 rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
)
for label in ("vanilla_qwen36", "sft_qwen36"):
    row = matched_all.get(label, {})
    lines.append(
        f"| {label} | {row.get('tasks', 0)} | {row.get('passed', 0)} | {pct(row.get('pass_rate'))} | "
        f"{row.get('milestone_6_or_7', 0)} | {pct(row.get('milestone_6_or_7_rate'))} |"
    )
lines.append("")
lines.append(f"- Pass-rate delta, SFT - Vanilla: **{pct(matched_all.get('pass_rate_delta_sft_minus_base'))}**")

lines.extend(
    [
        "",
        "## Matched both-terminal diagnostic subset",
        "",
        "| Variant | Tasks | Passed | Pass rate | M6/M7 | M6/M7 rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
)
for label in ("vanilla_qwen36", "sft_qwen36"):
    row = matched_both.get(label, {})
    lines.append(
        f"| {label} | {row.get('tasks', 0)} | {row.get('passed', 0)} | {pct(row.get('pass_rate'))} | "
        f"{row.get('milestone_6_or_7', 0)} | {pct(row.get('milestone_6_or_7_rate'))} |"
    )
lines.append("")
lines.append(f"- Both-terminal pass-rate delta, SFT - Vanilla: **{pct(matched_both.get('pass_rate_delta_sft_minus_base'))}**")

lines.extend(
    [
        "",
        "## Best-of-N curve on matched common tasks",
        "",
        "| N | Vanilla pass@N | SFT pass@N | Δ SFT-Vanilla | Vanilla M6/M7@N | SFT M6/M7@N |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
)
denom = len(common_task_ids)
for v_row, s_row in zip(vanilla_curve, sft_curve):
    v_pass = v_row["passed"]
    s_pass = s_row["passed"]
    delta = (s_pass - v_pass) / denom if denom else None
    lines.append(
        f"| {v_row['n']} | {v_pass}/{denom} ({pct(v_row['pass_rate'])}) | "
        f"{s_pass}/{denom} ({pct(s_row['pass_rate'])}) | {pct(delta)} | "
        f"{v_row['milestone_6_or_7']}/{denom} ({pct(v_row['milestone_6_or_7_rate'])}) | "
        f"{s_row['milestone_6_or_7']}/{denom} ({pct(s_row['milestone_6_or_7_rate'])}) |"
    )

lines.extend(
    [
        "",
        "## Raw summary artifacts",
        "",
        f"- Summary JSON: `{summary_path}`",
        f"- Comparison JSON: `{compare_path}`",
        f"- Config JSON: `{config_path}`",
        "",
        "Notes:",
        "",
        "- Infra/API failures are excluded by the summary scripts.",
        "- The primary comparison is the matched latest-task table; best-of-N shows how performance evolves across attempts.",
        "- If either runner is incomplete, treat this report as point-in-time and rerun the script with the same `RUN_STAMP` to resume.",
    ]
)

report_path.write_text("\n".join(lines) + "\n")
print(f"Wrote report: {report_path}")
PY

cat "$COMPARE_TXT" | tee -a "$ORCH_LOG"
echo "summary_json=${SUMMARY_JSON}" | tee -a "$ORCH_LOG"
echo "comparison_json=${COMPARE_JSON}" | tee -a "$ORCH_LOG"
echo "report_md=${REPORT_MD}" | tee -a "$ORCH_LOG"

if [[ "${COMPARE_ONLY:-0}" != "1" && ( "$VANILLA_EXIT" -ne 0 || "$SFT_EXIT" -ne 0 ) ]]; then
  echo "One or more variant runners exited non-zero: vanilla=${VANILLA_EXIT}, sft=${SFT_EXIT}" | tee -a "$ORCH_LOG"
  exit 1
fi
