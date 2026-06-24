#!/usr/bin/env python3
"""Run Codex CyberGym best-of-N while skipping tasks already solved."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mastermind.config import load_manifest
from mastermind.tasks import load_split_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="eval")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--start-rep", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--model-reasoning-effort", default="medium")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-provider")
    parser.add_argument("--codex-provider-base-url")
    parser.add_argument("--codex-provider-wire-api", default="responses")
    parser.add_argument("--codex-provider-env-key", default="LLM_GATEWAY_API_KEY")
    parser.add_argument("--codex-rate-limit-retries", type=int)
    parser.add_argument("--codex-rate-limit-stagger-seconds", type=float)
    parser.add_argument("--codex-rate-limit-min-sleep-seconds", type=float)
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)
    parser.add_argument("--submit-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-consecutive-infra-failures", type=int, default=5)
    parser.add_argument("--infra-failure-pause-seconds", type=int, default=1800)
    parser.add_argument("--server", default=os.environ.get("CYBERGYM_SERVER", "http://127.0.0.1:8666"))
    parser.add_argument("--pocdb-path", type=Path, default=Path("runs/cybergym_server/poc.db"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/codex_cybergym"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-stamp", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--name", default="codex-gpt54mini-eval200-bo8")
    parser.add_argument("--seed-output", type=Path, action="append", default=[])
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--parallel-runner", type=Path, default=Path("scripts/run_codex_cybergym_tasks_parallel.py"))
    parser.add_argument("--task-runner", type=Path, default=Path("scripts/run_codex_cybergym_tasks.py"))
    parser.add_argument("--sandbox", default="workspace-write")
    parser.add_argument("--runner-no-bare", action="store_true")
    return parser.parse_args()


def output_path(args: argparse.Namespace, rep: int) -> Path:
    return args.output_dir / f"{args.name}_rep{rep}_{args.run_stamp}_rollouts.jsonl"


def run_id(args: argparse.Namespace, rep: int) -> str:
    return f"{args.name}-r{rep}-{args.run_stamp}"


def row_passed(row: dict[str, Any]) -> bool:
    if row.get("status") == "PASSED":
        return True
    verification = row.get("verification") or {}
    return verification.get("passed") is True


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_passed_tasks(paths: list[Path]) -> set[str]:
    passed: set[str] = set()
    for path in paths:
        for row in read_rows(path):
            task_id = row.get("task_id")
            if isinstance(task_id, str) and row_passed(row):
                passed.add(task_id)
    return passed


def summarize_output(path: Path) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_rows(path):
        task_id = row.get("task_id")
        if isinstance(task_id, str):
            latest[task_id] = row
    status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in latest.values())
    passed = sum(1 for row in latest.values() if row_passed(row))
    return {
        "output": str(path),
        "completed": len(latest),
        "passed": passed,
        "status_counts": dict(sorted(status_counts.items())),
    }


def summary_path(args: argparse.Namespace) -> Path:
    if args.summary_output is not None:
        return args.summary_output
    return args.output_dir / f"{args.name}_{args.run_stamp}_summary.json"


def main() -> int:
    args = parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be >= 1")
    if not 1 <= args.start_rep <= args.reps:
        raise ValueError("--start-rep must be in [1, --reps]")

    load_dotenv(args.env_file)
    env = os.environ.copy()

    manifest = load_manifest()
    task_ids = load_split_ids(manifest, args.split)
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_event = {
        "event": "codex_bo_skip_passed_start",
        "name": args.name,
        "run_stamp": args.run_stamp,
        "split": args.split,
        "difficulty": args.difficulty,
        "max_tasks": args.max_tasks,
        "task_count": len(task_ids),
        "reps": args.reps,
        "start_rep": args.start_rep,
        "workers": args.workers,
        "model": args.model,
        "model_reasoning_effort": args.model_reasoning_effort,
        "seed_outputs": [str(path) for path in args.seed_output],
        "max_consecutive_infra_failures": args.max_consecutive_infra_failures,
        "infra_failure_pause_seconds": args.infra_failure_pause_seconds,
    }
    print(json.dumps(start_event, sort_keys=True), flush=True)

    rep_summaries: list[dict[str, Any]] = []
    exit_code = 0
    for rep in range(args.start_rep, args.reps + 1):
        previous_outputs = [*args.seed_output, *[output_path(args, previous) for previous in range(1, rep)]]
        passed_before = read_passed_tasks(previous_outputs)
        remaining = [task_id for task_id in task_ids if task_id not in passed_before]
        rep_event = {
            "event": "eval_rep_start",
            "rep": rep,
            "run_id": run_id(args, rep),
            "output": str(output_path(args, rep)),
            "previous_passed": len(passed_before),
            "remaining": len(remaining),
            "workers": args.workers,
        }
        print(json.dumps(rep_event, sort_keys=True), flush=True)
        if not remaining:
            print(json.dumps({"event": "all_tasks_passed", "rep": rep}, sort_keys=True), flush=True)
            break

        command = [
            args.python_bin,
            str(args.parallel_runner),
            "--split",
            args.split,
            "--difficulty",
            args.difficulty,
            "--run-id",
            run_id(args, rep),
            "--run-root",
            str(args.run_root),
            "--output",
            str(output_path(args, rep)),
            "--workers",
            str(args.workers),
            "--model",
            args.model,
            "--codex-bin",
            args.codex_bin,
            "--codex-timeout-seconds",
            str(args.codex_timeout_seconds),
            "--submit-timeout-seconds",
            str(args.submit_timeout_seconds),
            "--server",
            args.server,
            "--pocdb-path",
            str(args.pocdb_path),
            "--env-file",
            str(args.env_file),
            "--sandbox",
            args.sandbox,
            "--runner",
            str(args.task_runner),
            "--max-consecutive-infra-failures",
            str(args.max_consecutive_infra_failures),
            "--infra-failure-pause-seconds",
            str(args.infra_failure_pause_seconds),
        ]
        if args.model_reasoning_effort:
            command.extend(["--model-reasoning-effort", args.model_reasoning_effort])
        if args.codex_provider:
            command.extend(["--codex-provider", args.codex_provider])
        if args.codex_provider_base_url:
            command.extend(["--codex-provider-base-url", args.codex_provider_base_url])
        if args.codex_provider_wire_api:
            command.extend(["--codex-provider-wire-api", args.codex_provider_wire_api])
        if args.codex_provider_env_key:
            command.extend(["--codex-provider-env-key", args.codex_provider_env_key])
        if args.codex_rate_limit_retries is not None:
            command.extend(["--codex-rate-limit-retries", str(args.codex_rate_limit_retries)])
        if args.codex_rate_limit_stagger_seconds is not None:
            command.extend(["--codex-rate-limit-stagger-seconds", str(args.codex_rate_limit_stagger_seconds)])
        if args.codex_rate_limit_min_sleep_seconds is not None:
            command.extend(["--codex-rate-limit-min-sleep-seconds", str(args.codex_rate_limit_min_sleep_seconds)])
        if args.runner_no_bare:
            command.append("--runner-no-bare")
        for task_id in remaining:
            command.extend(["--task-id", task_id])

        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode != 0:
            exit_code = result.returncode
            print(
                json.dumps(
                    {
                        "event": "eval_rep_failed",
                        "rep": rep,
                        "run_id": run_id(args, rep),
                        "returncode": result.returncode,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break

        passed_after = read_passed_tasks([*previous_outputs, output_path(args, rep)])
        rep_summary = {
            "rep": rep,
            "run_id": run_id(args, rep),
            "previous_passed": len(passed_before),
            "new_passed": len(passed_after - passed_before),
            "best_of_passed": len(passed_after),
            "remaining_after": len([task_id for task_id in task_ids if task_id not in passed_after]),
            "output_summary": summarize_output(output_path(args, rep)),
            "output": str(output_path(args, rep)),
            "trajectory_dir": str((args.run_root / run_id(args, rep) / "trajectories").resolve()),
        }
        rep_summaries.append(rep_summary)
        print(json.dumps({"event": "eval_rep_complete", **rep_summary}, sort_keys=True), flush=True)

    # Include previous rep outputs when resuming with --start-rep > 1; those
    # were used for skip decisions and must also count in the final best-of.
    all_outputs = [*args.seed_output, *[output_path(args, rep) for rep in range(1, args.reps + 1)]]
    best_of_passed = read_passed_tasks(all_outputs)
    final_summary = {
        "event": "codex_bo_skip_passed_complete",
        "exit_code": exit_code,
        "name": args.name,
        "run_stamp": args.run_stamp,
        "model": args.model,
        "model_reasoning_effort": args.model_reasoning_effort,
        "workers": args.workers,
        "task_count": len(task_ids),
        "best_of_passed": len(best_of_passed),
        "best_of_pass_rate": len(best_of_passed) / len(task_ids) if task_ids else None,
        "remaining": len([task_id for task_id in task_ids if task_id not in best_of_passed]),
        "seed_outputs": [str(path) for path in args.seed_output],
        "rep_summaries": rep_summaries,
    }
    path = summary_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(final_summary, indent=2, sort_keys=True) + "\n")
    final_summary["summary_output"] = str(path)
    print(json.dumps(final_summary, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
