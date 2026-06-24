#!/usr/bin/env python3
"""Run Claude CyberGym best-of-N while skipping tasks already solved."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mastermind.config import load_manifest
from mastermind.tasks import load_split_ids


def anthropic_base_url_from_gateway(url: str) -> str:
    return url.removesuffix("/v1").rstrip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="eval")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--start-rep", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="glm-5.1")
    parser.add_argument("--codex-bin", default="claude")
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)
    parser.add_argument("--submit-timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-consecutive-infra-failures", type=int, default=5)
    parser.add_argument("--infra-failure-pause-seconds", type=int, default=1800)
    parser.add_argument("--server", default=os.environ.get("CYBERGYM_SERVER", "http://127.0.0.1:8666"))
    parser.add_argument("--pocdb-path", type=Path, default=Path("runs/cybergym_server/poc.db"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/claude_cybergym"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-stamp", default=datetime.now(UTC).strftime("%Y%m%dT%H%MZ"))
    parser.add_argument("--name", default="claude-glm51-eval200-bo8")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--parallel-runner", type=Path, default=Path("scripts/run_claude_cybergym_tasks_parallel.py"))
    parser.add_argument("--task-runner", type=Path, default=Path("scripts/run_claude_cybergym_tasks.py"))
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


def read_passed_tasks(paths: list[Path]) -> set[str]:
    passed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = row.get("task_id")
                if isinstance(task_id, str) and row_passed(row):
                    passed.add(task_id)
    return passed


def main() -> int:
    args = parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be >= 1")
    if not 1 <= args.start_rep <= args.reps:
        raise ValueError("--start-rep must be in [1, --reps]")

    load_dotenv(args.env_file)
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("LLM_GATEWAY_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["LLM_GATEWAY_API_KEY"]
    if not os.environ.get("ANTHROPIC_BASE_URL") and os.environ.get("LLM_GATEWAY_URL"):
        os.environ["ANTHROPIC_BASE_URL"] = anthropic_base_url_from_gateway(os.environ["LLM_GATEWAY_URL"])

    missing_env = [name for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL") if not os.environ.get(name)]
    if missing_env:
        raise SystemExit(f"Missing required environment variable(s): {', '.join(missing_env)}")

    manifest = load_manifest()
    task_ids = load_split_ids(manifest, args.split)
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_event = {
        "event": "bo_skip_passed_start",
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
        "max_consecutive_infra_failures": args.max_consecutive_infra_failures,
        "infra_failure_pause_seconds": args.infra_failure_pause_seconds,
    }
    print(json.dumps(log_event, sort_keys=True), flush=True)

    exit_code = 0
    for rep in range(args.start_rep, args.reps + 1):
        previous_outputs = [output_path(args, previous) for previous in range(1, rep)]
        passed = read_passed_tasks(previous_outputs)
        remaining = [task_id for task_id in task_ids if task_id not in passed]
        print(
            json.dumps(
                {
                    "event": "eval_rep_start",
                    "rep": rep,
                    "run_id": run_id(args, rep),
                    "output": str(output_path(args, rep)),
                    "previous_passed": len(passed),
                    "remaining": len(remaining),
                    "workers": args.workers,
                },
                sort_keys=True,
            ),
            flush=True,
        )
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
            "--runner",
            str(args.task_runner),
            "--max-consecutive-infra-failures",
            str(args.max_consecutive_infra_failures),
            "--infra-failure-pause-seconds",
            str(args.infra_failure_pause_seconds),
        ]
        for task_id in remaining:
            command.extend(["--task-id", task_id])

        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
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

        print(
            json.dumps(
                {
                    "event": "eval_rep_complete",
                    "rep": rep,
                    "run_id": run_id(args, rep),
                    "output": str(output_path(args, rep)),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    print(json.dumps({"event": "bo_skip_passed_complete", "exit_code": exit_code}, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
