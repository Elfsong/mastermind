#!/usr/bin/env python3
"""Run OpenCode Qwen3.6 CyberGym best-of-N while skipping solved tasks."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="eval")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--start-rep", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="qwen3_6_mingzhe")
    parser.add_argument("--model-reasoning-effort", default=os.environ.get("MODEL_REASONING_EFFORT"))
    parser.add_argument("--opencode-provider", default="qwen36_mingzhe")
    parser.add_argument(
        "--opencode-provider-base-url",
        default=os.environ.get("LITELLM_BASE_URL", "http://litellm.tiktok-row.net/v1"),
    )
    parser.add_argument("--opencode-provider-env-key", default="LITELLM_API_KEY")
    parser.add_argument("--opencode-context-limit", type=int, default=int(os.environ.get("OPENCODE_CONTEXT_LIMIT", "70000")))
    parser.add_argument("--opencode-output-token-max", type=int, default=int(os.environ.get("OPENCODE_OUTPUT_TOKEN_MAX", "4096")))
    parser.add_argument("--opencode-bin", default="runs/opencode_tool/node_modules/.bin/opencode")
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)
    parser.add_argument("--submit-timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-consecutive-infra-failures", type=int, default=5)
    parser.add_argument("--infra-failure-pause-seconds", type=int, default=1800)
    parser.add_argument("--server", default=os.environ.get("CYBERGYM_SERVER", "http://127.0.0.1:8666"))
    parser.add_argument("--pocdb-path", type=Path, default=Path("runs/cybergym_server/poc.db"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--run-root", type=Path, default=Path("/tmp/opencode_qwen36_mingzhe_cybergym"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-stamp", default=datetime.now(UTC).strftime("%Y%m%dT%H%MZ"))
    parser.add_argument("--name", default="opencode_qwen36_mingzhe_eval200_bo8")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--parallel-runner", type=Path, default=Path("scripts/run_codex_cybergym_tasks_parallel.py"))
    parser.add_argument("--task-runner", type=Path, default=Path("scripts/run_opencode_cybergym_tasks.py"))
    parser.add_argument("--sandbox", default="workspace-write")
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


def prepare_env(args: argparse.Namespace) -> dict[str, str]:
    load_dotenv(args.env_file)
    env = os.environ.copy()
    if not env.get(args.opencode_provider_env_key) and env.get("QWEN36_API_KEY"):
        env[args.opencode_provider_env_key] = env["QWEN36_API_KEY"]
    if not env.get(args.opencode_provider_env_key):
        raise SystemExit(f"Missing required environment variable: {args.opencode_provider_env_key}")
    return env


def main() -> int:
    args = parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be >= 1")
    if not 1 <= args.start_rep <= args.reps:
        raise ValueError("--start-rep must be in [1, --reps]")

    env = prepare_env(args)
    manifest = load_manifest()
    task_ids = load_split_ids(manifest, args.split)
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "opencode_qwen36_bo_skip_passed_start",
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
                "opencode_provider": args.opencode_provider,
                "opencode_provider_base_url": args.opencode_provider_base_url,
                "opencode_context_limit": args.opencode_context_limit,
                "opencode_output_token_max": args.opencode_output_token_max,
                "max_consecutive_infra_failures": args.max_consecutive_infra_failures,
                "infra_failure_pause_seconds": args.infra_failure_pause_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )

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
            "--opencode-provider",
            args.opencode_provider,
            "--opencode-provider-base-url",
            args.opencode_provider_base_url,
            "--opencode-provider-env-key",
            args.opencode_provider_env_key,
            "--opencode-context-limit",
            str(args.opencode_context_limit),
            "--opencode-output-token-max",
            str(args.opencode_output_token_max),
            "--codex-bin",
            args.opencode_bin,
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
        for task_id in remaining:
            command.extend(["--task-id", task_id])

        child_env = env.copy()
        child_env.setdefault("LITELLM_BASE_URL", args.opencode_provider_base_url)
        child_env.setdefault("OPENCODE_PROVIDER", args.opencode_provider)
        result = subprocess.run(command, cwd=ROOT, env=child_env, check=False)
        if result.returncode != 0:
            exit_code = result.returncode
            print(
                json.dumps(
                    {"event": "eval_rep_failed", "rep": rep, "run_id": run_id(args, rep), "returncode": result.returncode},
                    sort_keys=True,
                ),
                flush=True,
            )
            break

        print(
            json.dumps(
                {"event": "eval_rep_complete", "rep": rep, "run_id": run_id(args, rep), "output": str(output_path(args, rep))},
                sort_keys=True,
            ),
            flush=True,
        )

    print(json.dumps({"event": "opencode_qwen36_bo_skip_passed_complete", "exit_code": exit_code}, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
