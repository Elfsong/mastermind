#!/usr/bin/env python3
"""Finalize the Codex GPT-5.5 Iterative Improvement Experiment.

This merges the historical eval50 and remaining150 rollout files into one
canonical 200-task experiment artifact. Source run provenance is retained in
metadata, but downstream consumers should treat the merged outputs as the
authoritative Iterative Improvement Experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "iterative_improvement_experiment"
EXPERIMENT_NAME = "Iterative Improvement Experiment"


@dataclass(frozen=True)
class Source:
    run_id: str
    path: Path
    task_offset: int
    task_count: int
    log_path: Path


DEFAULT_SOURCES = (
    Source(
        run_id="codex-gateway-eval-seq-self-50-20260528T1604Z",
        path=Path("runs/codex_gateway_eval_seq_self_involving_eval50_rollouts.jsonl"),
        task_offset=0,
        task_count=50,
        log_path=Path(
            "runs/codex_cybergym/sequential_self_logs/"
            "codex-gateway-eval-seq-self-50-20260528T1604Z.extend8.log"
        ),
    ),
    Source(
        run_id="codex-gateway-eval-seq-self-remaining150-20260529T0458Z",
        path=Path("runs/codex_gateway_eval_seq_self_involving_eval_remaining150_rollouts.jsonl"),
        task_offset=50,
        task_count=150,
        log_path=Path(
            "runs/codex_cybergym/sequential_self_logs/"
            "codex-gateway-eval-seq-self-remaining150-20260529T0458Z.extend8.log"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--eval-split", type=Path, default=Path("cybergym/TASKS_EVAL"))
    parser.add_argument(
        "--output-rollouts",
        type=Path,
        default=Path("runs/codex_gateway_iterative_improvement_experiment_rollouts.jsonl"),
    )
    parser.add_argument(
        "--output-tasks",
        type=Path,
        default=Path("runs/codex_gateway_iterative_improvement_experiment_tasks.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("runs/codex_gateway_iterative_improvement_experiment_summary.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("runs/codex_gateway_iterative_improvement_experiment_summary.md"),
    )
    parser.add_argument("--expected-tasks", type=int, default=200)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--require-extend8-complete", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def wait_for_pid(pid: int, poll_seconds: float, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    proc_path = Path("/proc") / str(pid)
    while proc_path.exists():
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for pid {pid}")
        time.sleep(poll_seconds)


def read_task_ids(path: Path) -> list[str]:
    task_ids: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                task_ids.append(line)
    return task_ids


def read_jsonl(path: Path, run_id: str) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_json = 0
    with path.open(errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if row.get("run_id") == run_id:
                rows.append(row)
    return rows, bad_json


def has_complete_event(path: Path, run_id: str) -> bool:
    if not path.exists():
        return False
    with path.open(errors="replace") as f:
        for line in f:
            if '"event":"extend8_complete"' in line and f'"run_id":"{run_id}"' in line:
                return True
    return False


def attempt_index(row: dict[str, Any]) -> int:
    metadata = row.get("metadata") or {}
    sequential = metadata.get("sequential") or {}
    for value in (sequential.get("attempt_index"), metadata.get("attempt_index")):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def max_attempts(row: dict[str, Any]) -> int | None:
    metadata = row.get("metadata") or {}
    sequential = metadata.get("sequential") or {}
    value = sequential.get("max_attempts")
    return value if isinstance(value, int) else None


def milestone_value(row: dict[str, Any]) -> Any:
    milestone = row.get("milestone")
    if isinstance(milestone, dict):
        return milestone.get("milestone")
    return milestone


def milestone_reason(row: dict[str, Any]) -> str:
    milestone = row.get("milestone")
    if isinstance(milestone, dict):
        return str(milestone.get("reasoning") or "")
    return ""


def sort_key(row: dict[str, Any], task_order: dict[str, int]) -> tuple[Any, ...]:
    task_id = str(row.get("task_id") or "")
    return (
        task_order.get(task_id, 10**9),
        attempt_index(row),
        row.get("started_at") or "",
        row.get("finished_at") or "",
        row.get("run_id") or "",
    )


def annotate_row(
    row: dict[str, Any],
    *,
    source: Source,
    task_order: dict[str, int],
    expected_tasks: int,
) -> dict[str, Any]:
    annotated = dict(row)
    task_id = str(annotated.get("task_id") or "")
    task_index = task_order.get(task_id)
    metadata = dict(annotated.get("metadata") or {})
    experiment_metadata = dict(metadata.get(EXPERIMENT_ID) or {})
    experiment_metadata.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "task_index": task_index,
            "task_ordinal": None if task_index is None else task_index + 1,
            "task_total": expected_tasks,
            "source_run_id": source.run_id,
            "source_output": str(source.path),
            "source_task_offset": source.task_offset,
            "source_task_count": source.task_count,
        }
    )
    metadata[EXPERIMENT_ID] = experiment_metadata
    annotated["metadata"] = metadata
    annotated["experiment_id"] = EXPERIMENT_ID
    annotated["experiment_name"] = EXPERIMENT_NAME
    return annotated


def latest_by_task(rows: list[dict[str, Any]], task_order: dict[str, int]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str):
            grouped[task_id].append(row)

    latest: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in grouped.items():
        task_rows.sort(key=lambda row: sort_key(row, task_order))
        latest[task_id] = task_rows[-1]
    return latest


def build_task_summary(
    task_ids: list[str],
    rows: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str):
            rows_by_task[task_id].append(row)

    summaries: list[dict[str, Any]] = []
    for index, task_id in enumerate(task_ids):
        task_rows = rows_by_task.get(task_id, [])
        latest_row = latest.get(task_id)
        if latest_row is None:
            summaries.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "experiment_name": EXPERIMENT_NAME,
                    "task_index": index,
                    "task_ordinal": index + 1,
                    "task_total": len(task_ids),
                    "task_id": task_id,
                    "status": "MISSING",
                    "milestone": None,
                    "milestone_reason": "No rollout rows found for task",
                    "latest_attempt": 0,
                    "max_attempts": None,
                    "rollout_rows": 0,
                    "unique_attempts": [],
                    "started_at_first": None,
                    "finished_at_latest": None,
                    "latest_run_id": None,
                    "latest_agent": None,
                    "model": None,
                }
            )
            continue

        attempts = sorted({attempt_index(row) for row in task_rows if attempt_index(row)})
        summaries.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "experiment_name": EXPERIMENT_NAME,
                "task_index": index,
                "task_ordinal": index + 1,
                "task_total": len(task_ids),
                "task_id": task_id,
                "status": latest_row.get("status"),
                "milestone": milestone_value(latest_row),
                "milestone_reason": milestone_reason(latest_row),
                "latest_attempt": attempt_index(latest_row),
                "max_attempts": max_attempts(latest_row),
                "rollout_rows": len(task_rows),
                "unique_attempts": attempts,
                "started_at_first": min((row.get("started_at") or "" for row in task_rows), default=None),
                "finished_at_latest": latest_row.get("finished_at"),
                "latest_run_id": latest_row.get("run_id"),
                "latest_agent": latest_row.get("agent"),
                "model": latest_row.get("model"),
                "strategy": latest_row.get("strategy"),
                "trajectory_path": latest_row.get("trajectory_path"),
            }
        )
    return summaries


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Canonical rollout JSONL: `{summary['outputs']['rollouts']}`",
        f"- Canonical task JSONL: `{summary['outputs']['tasks']}`",
        f"- Unified task count: `{summary['tasks_seen']}` / `{summary['expected_tasks']}`",
        f"- Rollout rows: `{summary['rollout_rows']}`",
        f"- First started: `{summary['first_started']}`",
        f"- Last finished: `{summary['last_finished']}`",
        "",
        "## Latest Task Status",
        "",
    ]
    for key, value in sorted(summary["latest_status_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Latest Milestones", ""])
    for key, value in sorted(summary["latest_milestone_counts"].items(), key=lambda item: str(item[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Notes", ""])
    lines.append(
        "- Treat these files as the canonical Iterative Improvement Experiment outputs. "
        "The source run IDs are retained in JSON metadata only for provenance."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()

    if args.wait_for_pid is not None:
        print(
            json.dumps(
                {
                    "event": "waiting_for_runner",
                    "pid": args.wait_for_pid,
                    "poll_seconds": args.poll_seconds,
                    "time": now_iso(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        wait_for_pid(args.wait_for_pid, args.poll_seconds, args.timeout_seconds)

    task_ids = read_task_ids(repo_path(root, args.eval_split))
    if len(task_ids) < args.expected_tasks:
        raise ValueError(f"Expected at least {args.expected_tasks} eval tasks, found {len(task_ids)}")
    task_ids = task_ids[: args.expected_tasks]
    task_order = {task_id: index for index, task_id in enumerate(task_ids)}

    if args.require_extend8_complete:
        incomplete = [
            str(source.log_path)
            for source in DEFAULT_SOURCES
            if not has_complete_event(repo_path(root, source.log_path), source.run_id)
        ]
        if incomplete:
            raise RuntimeError(f"Missing extend8_complete event in: {', '.join(incomplete)}")

    annotated_rows: list[dict[str, Any]] = []
    bad_json_by_source: dict[str, int] = {}
    rows_by_source: dict[str, int] = {}
    for source in DEFAULT_SOURCES:
        rows, bad_json = read_jsonl(repo_path(root, source.path), source.run_id)
        bad_json_by_source[source.run_id] = bad_json
        rows_by_source[source.run_id] = len(rows)
        annotated_rows.extend(
            annotate_row(row, source=source, task_order=task_order, expected_tasks=args.expected_tasks)
            for row in rows
        )

    annotated_rows.sort(key=lambda row: sort_key(row, task_order))
    latest = latest_by_task(annotated_rows, task_order)
    task_summaries = build_task_summary(task_ids, annotated_rows, latest)

    missing = [row["task_id"] for row in task_summaries if row["status"] == "MISSING"]
    if missing and not args.allow_partial:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Missing {len(missing)} expected task summaries: {preview}")

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "generated_at": now_iso(),
        "expected_tasks": args.expected_tasks,
        "tasks_seen": len(task_summaries) - len(missing),
        "missing_tasks": missing,
        "rollout_rows": len(annotated_rows),
        "first_started": min((row.get("started_at") or "" for row in annotated_rows), default=None),
        "last_finished": max((row.get("finished_at") or "" for row in annotated_rows), default=None),
        "latest_status_counts": dict(Counter(row["status"] for row in task_summaries)),
        "latest_milestone_counts": dict(Counter(str(row["milestone"]) for row in task_summaries)),
        "latest_attempt_counts": dict(Counter(str(row["latest_attempt"]) for row in task_summaries)),
        "source_rows": rows_by_source,
        "source_bad_json": bad_json_by_source,
        "outputs": {
            "rollouts": str(args.output_rollouts),
            "tasks": str(args.output_tasks),
            "summary_json": str(args.summary_json),
            "summary_md": str(args.summary_md),
        },
    }

    write_jsonl(repo_path(root, args.output_rollouts), annotated_rows)
    write_jsonl(repo_path(root, args.output_tasks), task_summaries)
    write_json(repo_path(root, args.summary_json), summary)
    write_summary_md(repo_path(root, args.summary_md), summary)
    print(json.dumps({"event": "finalized", **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "finalize_failed",
                    "error": str(exc),
                    "time": now_iso(),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
