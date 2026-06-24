# Qwen3.6 Vanilla vs SFT Strategy Generator Eval100

This runbook describes how to run a matched 100-task CyberGym evaluation for:

- **Group A / Vanilla**: Vanilla `Qwen/Qwen3.6-35B-A3B` as Strategy Generator + Codex `gpt-5.5` as Executor.
- **Group B / SFT**: SFT checkpoint of `Qwen/Qwen3.6-35B-A3B` as Strategy Generator + Codex `gpt-5.5` as Executor.

The script uses the same fixed list of 100 eval tasks for both groups and writes a report comparing latest-task pass rate, matched-task pass rate, both-terminal diagnostic pass rate, and pass@N.

## Script

```bash
scripts/run_qwen36_vanilla_vs_sft_eval100.sh
```

Default output directory:

```text
runs/strategy_sft/vanilla_vs_sft_eval100_<RUN_STAMP>/
```

Important artifacts:

- `run_config.json` — run configuration.
- `task_ids.txt` — exact task list used by both groups.
- `vanilla_qwen36_rollouts.jsonl` — Vanilla group rollouts.
- `sft_qwen36_rollouts.jsonl` — SFT group rollouts.
- `summary.json` — per-file latest-task summary.
- `comparison.json` / `comparison.txt` — matched Vanilla-vs-SFT comparison.
- `report.md` — human-readable final/point-in-time report.
- `run.log` — orchestration log.

## Prerequisites

From the repository root:

```bash
source scripts/mastermind_env.sh
```

Make sure the local CyberGym infrastructure is running before launching the eval:

```bash
scripts/start_local_docker.sh
scripts/start_cybergym_server_local.sh
```

The script defaults to Codex provider `llmgw`; override with `CODEX_PROVIDER=none` if you want to use the default Codex auth path instead.

## Basic run

The script auto-loads the SFT sampler from:

```text
runs/strategy_sft/qwen36-strategy-sft-20260601T1935Z/result.json
```

Run:

```bash
scripts/run_qwen36_vanilla_vs_sft_eval100.sh
```

Or explicitly provide the SFT sampler:

```bash
SFT_MODEL_PATH='tinker://.../sampler_weights/qwen36-strategy-sft-final-sampler' \
scripts/run_qwen36_vanilla_vs_sft_eval100.sh
```

## Detached run

For long runs:

```bash
DETACH=1 WORKERS=6 scripts/run_qwen36_vanilla_vs_sft_eval100.sh
```

The command prints the output directory, PID file, and nohup log.

Follow progress:

```bash
tail -f runs/strategy_sft/vanilla_vs_sft_eval100_<RUN_STAMP>/run.log
```

Inspect current report:

```bash
cat runs/strategy_sft/vanilla_vs_sft_eval100_<RUN_STAMP>/report.md
```

## Resume / continue a run

The underlying runner resumes from existing rollout JSONL files using the same `RUN_ID`s. To resume, reuse the same `RUN_STAMP`:

```bash
RUN_STAMP=20260602T150000Z \
scripts/run_qwen36_vanilla_vs_sft_eval100.sh
```

If you only want to regenerate summaries/reports from existing outputs:

```bash
RUN_STAMP=20260602T150000Z \
COMPARE_ONLY=1 \
scripts/run_qwen36_vanilla_vs_sft_eval100.sh
```

## Useful overrides

```bash
# Task selection
MAX_TASKS=100
TASK_OFFSET=0
SPLIT=eval
DIFFICULTY=level1

# Attempts / parallelism
MAX_ATTEMPTS=8
WORKERS=6

# Models
CODEX_EXECUTOR_MODEL=gpt-5.5
TINKER_SUMMARY_MODEL=Qwen/Qwen3.6-35B-A3B
SFT_MODEL_PATH='tinker://.../final-sampler'

# Ordering
ORDER=vanilla_then_sft      # default
ORDER=sft_then_vanilla      # alternate

# Provider
CODEX_PROVIDER=llmgw        # default
CODEX_PROVIDER=none         # do not pass --codex-provider
```

## Interpreting the report

Use `report.md` as the primary artifact.

- **Individual latest-task summary**: each variant over its own completed/latest task set.
- **Matched latest-task comparison**: primary fair comparison on common task IDs.
- **Matched both-terminal diagnostic subset**: only tasks terminal for both variants.
- **Best-of-N curve**: pass@N and M6/M7@N over matched common tasks.

Infra/API failures are excluded by the existing summary scripts. If either runner is incomplete, treat the report as point-in-time and resume with the same `RUN_STAMP`.
