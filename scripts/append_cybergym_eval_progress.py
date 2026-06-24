#!/usr/bin/env python3
"""Append periodic CyberGym eval progress snapshots to a Markdown document."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--start-rep", type=int, default=3)
    parser.add_argument("--end-rep", type=int, default=8)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def count_rows(rows: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str]]:
    statuses: Counter[str] = Counter()
    milestones: Counter[str] = Counter()
    for row in rows:
        statuses[str(row.get("status") or "unknown")] += 1
        milestone = (row.get("milestone") or {}).get("milestone")
        milestones[str(milestone)] += 1
    return statuses, milestones


def fmt_counter(counter: Counter[str]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items()))


def active_process_lines(run_stamp: str) -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return []
    lines: list[str] = []
    for line in result.stdout.splitlines():
        if "codex-gateway-eval-bo4-r" in line and run_stamp in line:
            if "append_cybergym_eval_progress.py" in line:
                continue
            lines.append(line.strip())
    return lines


def build_snapshot(args: argparse.Namespace) -> str:
    now = now_iso()
    reps: list[tuple[int, list[dict[str, Any]], Counter[str], Counter[str]]] = []
    all_rows: list[dict[str, Any]] = []
    for rep in range(args.start_rep, args.end_rep + 1):
        rows = load_rows(Path(f"runs/codex_gateway_eval_bo4_rep{rep}_rollouts.jsonl"))
        statuses, milestones = count_rows(rows)
        reps.append((rep, rows, statuses, milestones))
        all_rows.extend(rows)

    total_statuses, total_milestones = count_rows(all_rows)
    process_lines = active_process_lines(args.run_stamp)
    active_codex = sum(1 for line in process_lines if "codex exec" in line)
    active_task_runners = sum(1 for line in process_lines if "run_codex_cybergym_tasks.py" in line)
    active_parallel = sum(1 for line in process_lines if "run_codex_cybergym_tasks_parallel.py" in line)

    lines = [
        f"## Progress Snapshot - {now}",
        "",
        f"- Run stamp: `{args.run_stamp}`",
        f"- Target reps: `{args.start_rep}-{args.end_rep}`",
        f"- Total completed rollouts: `{len(all_rows)}` / `{(args.end_rep - args.start_rep + 1) * 200}`",
        f"- Overall status distribution: {fmt_counter(total_statuses)}",
        f"- Overall milestone distribution: {fmt_counter(total_milestones)}",
        f"- Active processes: parallel={active_parallel}, per-task-runner={active_task_runners}, codex-exec={active_codex}",
        "",
        "| Rep | Completed | Status Distribution | Milestone Distribution |",
        "|---:|---:|---|---|",
    ]
    for rep, rows, statuses, milestones in reps:
        lines.append(
            f"| {rep} | {len(rows)}/200 | {fmt_counter(statuses)} | {fmt_counter(milestones)} |"
        )
    lines.append("")
    return "\n".join(lines)


def append_snapshot(args: argparse.Namespace) -> None:
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    if not args.doc.exists():
        args.doc.write_text(
            "# GPT-5.5 CyberGym Eval Progress\n\n"
            "Lark-friendly Markdown progress log for the running CyberGym eval reps.\n\n"
        )
    with args.doc.open("a") as f:
        f.write(build_snapshot(args))
        f.write("\n")


def main() -> int:
    args = parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    while True:
        append_snapshot(args)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
