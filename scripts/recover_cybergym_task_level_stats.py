#!/usr/bin/env python3
"""Recover task-level CyberGym outcomes and paired reliability statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from summarize_cybergym_pass_rates import attempt_index, is_infra_failure, load_rows


DEFAULT_VANILLA = Path(
    "runs/hf_upload_staging/"
    "cybergym-codex-gpt-5-5-qwen36-vanilla-vs-sft-eval200-trajectories/"
    "data/vanilla_qwen36_rollouts.jsonl"
)
DEFAULT_SFT = Path(
    "runs/hf_upload_staging/"
    "cybergym-codex-gpt-5-5-qwen36-vanilla-vs-sft-eval200-trajectories/"
    "data/sft_qwen36_rollouts.jsonl"
)
DEFAULT_ITERATIVE = Path("runs/codex_gateway_iterative_improvement_experiment_rollouts.jsonl")
DEFAULT_BEST8_TASKS = Path(
    "runs/hf_upload_staging/"
    "cybergym-codex-gpt-5-5-round1-8-independent-eval/"
    "data/tasks.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover per-task solved/not-solved columns and compute paired "
            "bootstrap CIs plus exact McNemar tests."
        )
    )
    parser.add_argument("--vanilla-rollouts", type=Path, default=DEFAULT_VANILLA)
    parser.add_argument("--sft-rollouts", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--iterative-rollouts", type=Path, default=DEFAULT_ITERATIVE)
    parser.add_argument("--best8-tasks", type=Path, default=DEFAULT_BEST8_TASKS)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/statistical_reliability"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rollout-condition",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Add another condition from a rollout JSONL, using latest scored row per task. "
            "Example: --rollout-condition rl=runs/rl_rollouts.jsonl"
        ),
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="LABEL:A_COL:B_COL",
        help=(
            "Add a paired comparison. Defaults cover SFT vs Vanilla and "
            "Iterative vs Best-of-8."
        ),
    )
    return parser.parse_args()


def latest_pass_by_task(path: Path) -> dict[str, bool]:
    rows = [row for row in load_rows(path) if not is_infra_failure(row)]
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        key = (attempt_index(row), row.get("finished_at") or "")
        old = latest.get(task_id)
        old_key = (attempt_index(old), old.get("finished_at") or "") if old else (-1, "")
        if key >= old_key:
            latest[task_id] = row
    return {task_id: row.get("status") == "PASSED" for task_id, row in latest.items()}


def load_best8_tasks(path: Path) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if not isinstance(task_id, str):
                continue
            if "best_of_8_passed" in row:
                outcomes[task_id] = bool(row["best_of_8_passed"])
            else:
                outcomes[task_id] = any(row.get(f"round{i}_status") == "PASSED" for i in range(1, 9))
    return outcomes


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing condition name in {value!r}")
    return name, Path(path)


def parse_comparison(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected LABEL:A_COL:B_COL, got {value!r}")
    label, a_col, b_col = [part.strip() for part in parts]
    if not label or not a_col or not b_col:
        raise ValueError(f"Malformed comparison {value!r}")
    return label, a_col, b_col


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take quantile of empty list")
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def paired_bootstrap(
    a: list[int],
    b: list[int],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(a)
    diffs: list[float] = []
    for _ in range(n_boot):
        total = 0
        for _ in range(n):
            idx = rng.randrange(n)
            total += a[idx] - b[idx]
        diffs.append(total / n)
    diffs.sort()
    point = sum(x - y for x, y in zip(a, b)) / n
    return point, quantile(diffs, 0.025), quantile(diffs, 0.975)


def mcnemar_exact(a: list[int], b: list[int]) -> tuple[int, int, float]:
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    lower_tail = sum(choose(discordant, i) / (2**discordant) for i in range(k + 1))
    return a_only, b_only, min(1.0, 2.0 * lower_tail)


def choose(n: int, k: int) -> int:
    try:
        return math.comb(n, k)
    except AttributeError:
        if k < 0 or k > n:
            return 0
        k = min(k, n - k)
        result = 1
        for i in range(1, k + 1):
            result = result * (n - k + i) // i
        return result


def compare_conditions(
    label: str,
    a_col: str,
    b_col: str,
    conditions: dict[str, dict[str, bool]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    missing = [name for name in (a_col, b_col) if name not in conditions]
    if missing:
        raise ValueError(f"Unknown condition(s) for {label}: {', '.join(missing)}")
    task_ids = sorted(set(conditions[a_col]) & set(conditions[b_col]))
    a = [1 if conditions[a_col][task_id] else 0 for task_id in task_ids]
    b = [1 if conditions[b_col][task_id] else 0 for task_id in task_ids]
    point, lo, hi = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
    a_only, b_only, p_value = mcnemar_exact(a, b)
    return {
        "label": label,
        "a_col": a_col,
        "b_col": b_col,
        "tasks": len(task_ids),
        "a_passed": sum(a),
        "b_passed": sum(b),
        "delta_pass_rate": point,
        "bootstrap_ci_95": [lo, hi],
        "a_only": a_only,
        "b_only": b_only,
        "mcnemar_exact_p": p_value,
        "missing_from_a": sorted(set(conditions[b_col]) - set(conditions[a_col])),
        "missing_from_b": sorted(set(conditions[a_col]) - set(conditions[b_col])),
    }


def write_task_csv(path: Path, conditions: dict[str, dict[str, bool]], columns: list[str]) -> None:
    task_ids = sorted({task_id for condition in conditions.values() for task_id in condition})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", *columns])
        writer.writeheader()
        for task_id in task_ids:
            row: dict[str, Any] = {"task_id": task_id}
            for col in columns:
                value = conditions[col].get(task_id)
                row[col] = "" if value is None else int(value)
            writer.writerow(row)


def fmt_pp(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}] pp"


def write_markdown(path: Path, comparisons: list[dict[str, Any]]) -> None:
    lines = [
        "# Paired Reliability Statistics",
        "",
        "| Comparison | Pass Counts | Delta | 95% Paired Bootstrap CI | a_only / b_only | McNemar exact p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            "| {label} | {a_passed}/{tasks} vs {b_passed}/{tasks} | {delta} | {ci} | "
            "{a_only} / {b_only} | {p:.4g} |".format(
                label=item["label"],
                a_passed=item["a_passed"],
                b_passed=item["b_passed"],
                tasks=item["tasks"],
                delta=fmt_pp(item["delta_pass_rate"]),
                ci=fmt_ci(item["bootstrap_ci_95"]),
                a_only=item["a_only"],
                b_only=item["b_only"],
                p=item["mcnemar_exact_p"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Best-of-8 is recovered as task-level OR: a task passes if any of the 8 independent rounds passed.",
            "- Rollout JSONL conditions use the latest non-infra row per task.",
            "- McNemar's exact test uses only discordant task pairs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    conditions: dict[str, dict[str, bool]] = {
        "best8": load_best8_tasks(args.best8_tasks),
        "iterative": latest_pass_by_task(args.iterative_rollouts),
        "vanilla": latest_pass_by_task(args.vanilla_rollouts),
        "sft": latest_pass_by_task(args.sft_rollouts),
    }
    input_paths: dict[str, str] = {
        "best8": str(args.best8_tasks),
        "iterative": str(args.iterative_rollouts),
        "vanilla": str(args.vanilla_rollouts),
        "sft": str(args.sft_rollouts),
    }

    for item in args.rollout_condition:
        name, path = parse_named_path(item)
        if name in conditions:
            raise ValueError(f"Condition {name!r} already exists")
        conditions[name] = latest_pass_by_task(path)
        input_paths[name] = str(path)

    columns = list(conditions)
    comparisons = [
        ("SFT vs Vanilla", "sft", "vanilla"),
        ("Iterative vs Best-of-8", "iterative", "best8"),
    ]
    comparisons.extend(parse_comparison(item) for item in args.comparison)
    comparison_rows = [
        compare_conditions(label, a_col, b_col, conditions, n_boot=args.n_boot, seed=args.seed)
        for label, a_col, b_col in comparisons
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "task_level_results.csv"
    json_path = args.output_dir / "paired_stats.json"
    md_path = args.output_dir / "paired_stats.md"

    write_task_csv(csv_path, conditions, columns)
    summary = {
        "inputs": input_paths,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "conditions": {
            name: {
                "tasks": len(values),
                "passed": sum(1 for passed in values.values() if passed),
                "pass_rate": sum(1 for passed in values.values() if passed) / len(values)
                if values
                else None,
            }
            for name, values in conditions.items()
        },
        "comparisons": comparison_rows,
        "outputs": {
            "task_level_csv": str(csv_path),
            "paired_stats_json": str(json_path),
            "paired_stats_md": str(md_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, comparison_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
