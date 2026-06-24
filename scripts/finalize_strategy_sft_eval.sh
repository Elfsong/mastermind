#!/usr/bin/env bash
set -euo pipefail

# One-shot finalization gate for the Qwen3.6 strategy-generator SFT eval.
# This script does not start or poll any evaluation.  It only validates local
# reporting utilities and then generates the final report with --require-final.

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
cd "$ROOT"

RUN_SELFTEST=1
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --no-selftest)
      RUN_SELFTEST=0
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/finalize_strategy_sft_eval.sh [--no-selftest] [report args...]

Runs the final strategy SFT evaluation gate.

Options:
  --no-selftest  Skip the offline reporting-tool self-test.
  -h, --help     Show this help.

All other arguments are forwarded to:
  .venv/bin/python scripts/report_strategy_sft_eval.py --require-final
EOF
      exit 0
      ;;
    *)
      ARGS+=("$arg")
      ;;
  esac
done

if [[ "$RUN_SELFTEST" == "1" ]]; then
  .venv/bin/python scripts/selftest_strategy_eval_tools.py
fi
.venv/bin/python scripts/report_strategy_sft_eval.py --require-final "${ARGS[@]}"
