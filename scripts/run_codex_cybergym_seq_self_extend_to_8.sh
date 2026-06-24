#!/usr/bin/env bash
# Extend the two sequential self-involving eval runs from max_attempts=4 to 8.
#
# Resume semantics (built into run_codex_cybergym_sequential_self_involving.py):
#   - tasks already PASSED            -> skipped
#   - tasks already at max_attempts=8 -> skipped
#   - otherwise                       -> resume from (last recorded attempt + 1) up to 8,
#                                        reusing the on-disk experience cache.
# This recovers the remaining150 tasks that crashed early on the ChatGPT usage limit
# (now routed through the LLM gateway via --codex-provider llmgw) AND extends the
# genuinely-failed tasks to attempt 8.
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"
source scripts/mastermind_env.sh

PY="${PYTHON_BIN:-.venv/bin/python}"
WORKERS="${WORKERS:-6}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
PROVIDER="${CODEX_PROVIDER:-llmgw}"
LOG_DIR="runs/codex_cybergym/sequential_self_logs"
mkdir -p "$LOG_DIR"

run_one() {
    local run_id="$1" output="$2" offset="$3" max_tasks="$4"
    local log="${LOG_DIR}/${run_id}.extend8.log"
    echo "{\"event\":\"extend8_start\",\"run_id\":\"${run_id}\",\"output\":\"${output}\",\"max_attempts\":${MAX_ATTEMPTS},\"provider\":\"${PROVIDER}\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$log"
    "$PY" scripts/run_codex_cybergym_sequential_self_involving.py \
        --run-id "$run_id" \
        --split eval --difficulty level1 \
        --output "$output" \
        --task-offset "$offset" --max-tasks "$max_tasks" \
        --workers "$WORKERS" --max-attempts "$MAX_ATTEMPTS" \
        --experience-token-budget 2048 \
        --model gpt-5.5 --summary-model gpt-5.5 \
        --codex-provider "$PROVIDER" >> "$log" 2>&1
    echo "{\"event\":\"extend8_complete\",\"run_id\":\"${run_id}\",\"exit_code\":$?,\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$log"
}

# eval50 pilot: offset 0, 50 tasks (21 genuine failures -> attempts 5..8)
run_one "codex-gateway-eval-seq-self-50-20260528T1604Z" \
        "runs/codex_gateway_eval_seq_self_involving_eval50_rollouts.jsonl" 0 50

# remaining150: offset 50, 150 tasks (22 genuine failures + 70 usage-limit crashes -> resume..8)
run_one "codex-gateway-eval-seq-self-remaining150-20260529T0458Z" \
        "runs/codex_gateway_eval_seq_self_involving_eval_remaining150_rollouts.jsonl" 50 150
