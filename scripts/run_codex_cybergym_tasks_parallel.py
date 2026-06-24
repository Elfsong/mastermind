#!/usr/bin/env python3
"""Parallel orchestrator for per-task Codex CyberGym rollouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mastermind.config import load_manifest
from mastermind.rollout import (
    MilestoneSummary,
    RolloutRecord,
    VerificationSummary,
    append_rollout,
)
from mastermind.tasks import load_split_ids


INFRA_FAILURE_MARKERS = (
    "quota exceeded",
    "daily limit",
    "daily limitation",
    "daily limits exceeded",
    "usage limit",
    "payment required",
    "monthly included credits",
    "included credits",
    "pre-paid credits",
    "depleted",
    "inference providers",
    "too many requests",
    "rate limit",
    "rate limited",
    "rate_limit",
    "temporarily unavailable",
    "overloaded",
    "backend request failed",
    "prefill stall",
    "no data from backend",
    "database is locked",
)
INFRA_FAILURE_STATUS_CODES = {429, 500, 502, 503, 504, 529}
INFRA_FAILURE_STATUS_CODES_WITH_MARKER = {401, 402, 403}
AGENT_ERROR_STATUS = "AGENT_ERROR"
AGENT_ERROR_STATUSES = {AGENT_ERROR_STATUS, "CRASH"}  # CRASH is legacy rollout data.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train_dev")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/codex_cybergym"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model")
    parser.add_argument("--model-reasoning-effort")
    parser.add_argument("--opencode-provider")
    parser.add_argument("--opencode-provider-base-url")
    parser.add_argument("--opencode-provider-env-key")
    parser.add_argument("--opencode-context-limit", type=int)
    parser.add_argument("--opencode-output-token-max", type=int)
    parser.add_argument("--codex-provider")
    parser.add_argument("--codex-provider-base-url")
    parser.add_argument("--codex-provider-wire-api", default="responses")
    parser.add_argument("--codex-provider-env-key", default="LLM_GATEWAY_API_KEY")
    parser.add_argument("--codex-rate-limit-retries", type=int)
    parser.add_argument("--codex-rate-limit-stagger-seconds", type=float)
    parser.add_argument("--codex-rate-limit-min-sleep-seconds", type=float)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)
    parser.add_argument("--submit-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-consecutive-infra-failures", type=int, default=5)
    parser.add_argument("--infra-failure-pause-seconds", type=int, default=1800)
    parser.add_argument("--server", default="http://127.0.0.1:8666")
    parser.add_argument("--pocdb-path", type=Path, default=Path("runs/cybergym_server/poc.db"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--sandbox", default="workspace-write")
    parser.add_argument("--usage-token-budget", type=int)
    parser.add_argument("--usage-stop-fraction", type=float, default=0.9)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_codex_cybergym_tasks.py"),
    )
    parser.add_argument(
        "--runner-no-bare",
        action="store_true",
        help="Pass --no-bare to runners that support Claude Code non-bare auth.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def safe_name(task_id: str) -> str:
    return task_id.replace(":", "_").replace("/", "_")


def coerce_status_code(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def has_infra_marker(text: str) -> bool:
    lowered = text.lower()
    return "you've hit your usage limit" in text or any(marker in lowered for marker in INFRA_FAILURE_MARKERS)


def api_error_is_infra(status_code: Any, text: str) -> bool:
    status = coerce_status_code(status_code)
    if status in INFRA_FAILURE_STATUS_CODES:
        return True
    if status in INFRA_FAILURE_STATUS_CODES_WITH_MARKER and has_infra_marker(text):
        return True
    return has_infra_marker(text)


def read_head_tail_text(path: Path, *, max_bytes_each: int = 8000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) <= max_bytes_each * 2:
        return data.decode(errors="ignore")
    return (data[:max_bytes_each] + b"\n" + data[-max_bytes_each:]).decode(errors="ignore")


def read_opencode_api_errors(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    errors: list[dict[str, Any]] = []
    with path.open(errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "error":
                continue
            error = event.get("error") or {}
            data = error.get("data") if isinstance(error, dict) else {}
            data = data if isinstance(data, dict) else {}
            response_headers = data.get("responseHeaders") if isinstance(data.get("responseHeaders"), dict) else {}
            message_parts = [
                str(data.get("message") or ""),
                str(data.get("responseBody") or ""),
                str(response_headers.get("x-error-message") or ""),
            ]
            errors.append(
                {
                    "status_code": coerce_status_code(data.get("statusCode")),
                    "message": "\n".join(part for part in message_parts if part),
                }
            )
    return errors


def row_is_retryable_infra_failure(row: dict[str, Any]) -> bool:
    if str(row.get("status") or "") not in AGENT_ERROR_STATUSES:
        return False
    verification = row.get("verification") or {}
    if verification.get("status") == "runner_failed":
        return True
    metadata = row.get("metadata") or {}
    claude_result = metadata.get("claude_result") or {}
    api_error_status = claude_result.get("api_error_status")
    result_text = str(claude_result.get("result") or "").lower()
    if api_error_is_infra(api_error_status, result_text):
        return True
    metadata_api_status = metadata.get("api_error_status")
    if metadata_api_status is None:
        metadata_api_status = metadata.get("opencode_api_error_status")
    metadata_api_text = "\n".join(
        str(metadata.get(key) or "")
        for key in ("api_error_message", "opencode_api_error_message", "error", "generation_error")
    )
    if api_error_is_infra(metadata_api_status, metadata_api_text):
        return True
    stdout_path = metadata.get("codex_stdout")
    if isinstance(stdout_path, str):
        path = Path(stdout_path)
        if path.exists():
            for api_error in read_opencode_api_errors(path):
                if api_error_is_infra(api_error.get("status_code"), str(api_error.get("message") or "")):
                    return True
            if has_infra_marker(read_head_tail_text(path)):
                return True
    return False


def read_completed_tasks(path: Path, run_id: str) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id") != run_id:
                continue
            verification = row.get("verification") or {}
            if verification.get("status") == "missing_benchmark_data":
                continue
            if row_is_retryable_infra_failure(row):
                continue
            task_id = row.get("task_id")
            if isinstance(task_id, str):
                completed.add(task_id)
    return completed


def read_task_rows(path: Path, run_id: str, task_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id") == run_id and row.get("task_id") == task_id:
                rows.append(row)
    return rows


def synthesize_failure_record(
    *,
    run_id: str,
    task_id: str,
    returncode: int | None,
    timed_out: bool,
    stdout_log: Path,
    stderr_log: Path,
    wall_seconds: float,
) -> RolloutRecord:
    status = "TIMEOUT" if timed_out else AGENT_ERROR_STATUS
    details = {
        "source": "run_codex_cybergym_tasks_parallel.py",
        "runner_returncode": returncode,
        "runner_stdout": str(stdout_log),
        "runner_stderr": str(stderr_log),
        "runner_timed_out": timed_out,
    }
    return RolloutRecord(
        run_id=run_id,
        task_id=task_id,
        agent=f"{run_id}-{safe_name(task_id)}",
        model="codex-cli-default",
        executor="codex.exec",
        status=status,
        milestone=MilestoneSummary(milestone=0, reasoning="per-task runner did not produce a rollout row"),
        verification=VerificationSummary(status="runner_failed", passed=False, submit_count=0, details=details),
        wall_seconds=wall_seconds,
        started_at=now_iso(),
        finished_at=now_iso(),
        metadata=details,
    )


def run_one_task(args: argparse.Namespace, task_id: str, attempts_dir: Path, logs_dir: Path) -> dict[str, Any]:
    safe_task = safe_name(task_id)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    attempt_output = attempts_dir / f"{safe_task}.{stamp}.jsonl"
    stdout_log = logs_dir / f"{safe_task}.{stamp}.stdout.log"
    stderr_log = logs_dir / f"{safe_task}.{stamp}.stderr.log"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(args.runner),
        "--split",
        args.split,
        "--difficulty",
        args.difficulty,
        "--run-id",
        args.run_id,
        "--run-root",
        str(args.run_root),
        "--output",
        str(attempt_output),
        "--task-id",
        task_id,
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
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.model_reasoning_effort:
        command.extend(["--model-reasoning-effort", args.model_reasoning_effort])
    if args.opencode_provider:
        command.extend(["--opencode-provider", args.opencode_provider])
    if args.opencode_provider_base_url:
        command.extend(["--opencode-provider-base-url", args.opencode_provider_base_url])
    if args.opencode_provider_env_key:
        command.extend(["--opencode-provider-env-key", args.opencode_provider_env_key])
    if args.opencode_context_limit is not None:
        command.extend(["--opencode-context-limit", str(args.opencode_context_limit)])
    if args.opencode_output_token_max is not None:
        command.extend(["--opencode-output-token-max", str(args.opencode_output_token_max)])
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
        command.append("--no-bare")

    started = time.monotonic()
    timeout_seconds = args.codex_timeout_seconds + args.submit_timeout_seconds + 600
    timed_out = False
    with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
        try:
            proc = subprocess.run(
                command,
                cwd=Path.cwd(),
                env=os.environ.copy(),
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
            )
            returncode: int | None = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = None
            timed_out = True
            stderr.write(b"\nPARALLEL_RUNNER_TIMEOUT\n")
    wall_seconds = time.monotonic() - started
    rows = read_task_rows(attempt_output, args.run_id, task_id)
    if rows:
        return {
            "task_id": task_id,
            "rows": rows,
            "returncode": returncode,
            "timed_out": timed_out,
            "wall_seconds": wall_seconds,
            "attempt_output": str(attempt_output),
        }
    return {
        "task_id": task_id,
        "rows": [
            synthesize_failure_record(
                run_id=args.run_id,
                task_id=task_id,
                returncode=returncode,
                timed_out=timed_out,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                wall_seconds=wall_seconds,
            ).to_dict()
        ],
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "attempt_output": str(attempt_output),
    }


def append_rows(output: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        append_rollout(output, RolloutRecord.from_dict(row))


def trajectory_usage(row: dict[str, Any]) -> dict[str, int] | None:
    path_value = row.get("trajectory_path")
    if not isinstance(path_value, str):
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    usage: dict[str, int] | None = None
    with path.open(errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = {
                    "input_tokens": int(event["usage"].get("input_tokens") or 0),
                    "cached_input_tokens": int(event["usage"].get("cached_input_tokens") or 0),
                    "output_tokens": int(event["usage"].get("output_tokens") or 0),
                    "reasoning_output_tokens": int(event["usage"].get("reasoning_output_tokens") or 0),
                }
    return usage


def usage_net_total_tokens(usage: dict[str, int] | None) -> int:
    if not usage:
        return 0
    net_input = max(usage["input_tokens"] - usage["cached_input_tokens"], 0)
    return net_input + usage["output_tokens"] + usage["reasoning_output_tokens"]


def is_infra_failure(row: dict[str, Any]) -> bool:
    return row_is_retryable_infra_failure(row)


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.max_consecutive_infra_failures < 1:
        raise ValueError("--max-consecutive-infra-failures must be >= 1")
    if args.infra_failure_pause_seconds < 0:
        raise ValueError("--infra-failure-pause-seconds must be >= 0")
    if not 0 < args.usage_stop_fraction <= 1:
        raise ValueError("--usage-stop-fraction must be in (0, 1]")
    manifest = load_manifest()
    task_ids = args.task_id or load_split_ids(manifest, args.split)
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    output = args.output.resolve()
    run_root = (args.run_root / args.run_id).resolve()
    attempts_dir = run_root / "per_task_rollouts"
    logs_dir = run_root / "runner_logs"
    completed = read_completed_tasks(output, args.run_id)
    pending = [task_id for task_id in task_ids if task_id not in completed]
    print(
        json.dumps(
            {
                "event": "parallel_start",
                "run_id": args.run_id,
                "workers": args.workers,
                "completed": len(completed),
                "pending": len(pending),
                "usage_token_budget": args.usage_token_budget,
                "usage_stop_fraction": args.usage_stop_fraction,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    exit_code = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        pending_queue = deque(pending)

        stop_scheduling = False
        consecutive_infra_failures = 0
        pause_until: float | None = None
        probe_mode = False
        pending_exhausted = False
        usage_tokens = 0
        usage_stop_tokens = (
            int(args.usage_token_budget * args.usage_stop_fraction)
            if args.usage_token_budget is not None
            else None
        )

        def schedule_available_slots(limit: int | None = None) -> None:
            nonlocal pending_exhausted
            if stop_scheduling or pause_until is not None:
                return
            slots = max(args.workers - len(futures), 0)
            if limit is not None:
                slots = min(slots, limit)
            for _ in range(slots):
                try:
                    task_id = pending_queue.popleft()
                except IndexError:
                    pending_exhausted = True
                    return
                futures[executor.submit(run_one_task, args, task_id, attempts_dir, logs_dir)] = task_id
                print(json.dumps({"event": "task_scheduled", "task_id": task_id}, sort_keys=True), flush=True)

        def start_infra_pause(reason: str) -> None:
            nonlocal pause_until
            if pause_until is not None:
                return
            pause_until = time.monotonic() + args.infra_failure_pause_seconds
            resume_at = datetime.now(UTC) + timedelta(seconds=args.infra_failure_pause_seconds)
            print(
                json.dumps(
                    {
                        "event": "infra_pause_start",
                        "reason": reason,
                        "consecutive_infra_failures": consecutive_infra_failures,
                        "pause_seconds": args.infra_failure_pause_seconds,
                        "resume_at": resume_at.isoformat(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        schedule_available_slots()

        finished_count = 0
        while futures or (not pending_exhausted and not stop_scheduling):
            if not futures:
                if pause_until is not None:
                    sleep_seconds = max(pause_until - time.monotonic(), 0)
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
                    pause_until = None
                    probe_mode = True
                    consecutive_infra_failures = 0
                    print(
                        json.dumps(
                            {
                                "event": "infra_pause_complete",
                                "probe_tasks": 1,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    schedule_available_slots(limit=1)
                    continue
                schedule_available_slots()
                if not futures:
                    break

            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = futures.pop(future)
                try:
                    result = future.result()
                    append_rows(output, result["rows"])
                    row = result["rows"][-1]
                    finished_count += 1
                    row_usage = trajectory_usage(row)
                    usage_delta = usage_net_total_tokens(row_usage)
                    usage_tokens += usage_delta
                    if usage_stop_tokens is not None and usage_tokens >= usage_stop_tokens:
                        stop_scheduling = True
                        print(
                            json.dumps(
                                {
                                    "event": "stop_scheduling",
                                    "reason": "usage_soft_limit",
                                    "usage_tokens": usage_tokens,
                                    "usage_stop_tokens": usage_stop_tokens,
                                    "usage_token_budget": args.usage_token_budget,
                                    "usage_stop_fraction": args.usage_stop_fraction,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    infra_failure = is_infra_failure(row)
                    if result["returncode"] not in {0, None} and not infra_failure:
                        exit_code = 1
                    if infra_failure:
                        pending_queue.appendleft(task_id)
                        pending_exhausted = False
                        consecutive_infra_failures += 1
                        if probe_mode:
                            start_infra_pause("infra_probe_failed")
                        elif consecutive_infra_failures >= args.max_consecutive_infra_failures:
                            start_infra_pause("consecutive_infra_failures")
                    else:
                        if probe_mode:
                            print(
                                json.dumps(
                                    {
                                        "event": "infra_probe_succeeded",
                                        "task_id": task_id,
                                        "status": row.get("status"),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            probe_mode = False
                        consecutive_infra_failures = 0
                    print(
                        json.dumps(
                            {
                                "event": "task_done",
                                "task_id": task_id,
                                "finished": finished_count,
                                "total_pending": len(pending),
                                "returncode": result["returncode"],
                                "status": row.get("status"),
                                "milestone": (row.get("milestone") or {}).get("milestone"),
                                "infra_failure": infra_failure,
                                "pending_queue": len(pending_queue),
                                "wall_seconds": result["wall_seconds"],
                                "usage_delta_tokens": usage_delta,
                                "usage_tokens": usage_tokens,
                                "usage_stop_tokens": usage_stop_tokens,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the queue moving.
                    exit_code = 1
                    print(
                        json.dumps(
                            {"event": "task_orchestrator_error", "task_id": task_id, "error": repr(exc)},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if not probe_mode and pause_until is None:
                schedule_available_slots()

    print(json.dumps({"event": "parallel_complete", "exit_code": exit_code}, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
