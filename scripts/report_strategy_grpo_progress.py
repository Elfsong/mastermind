#!/usr/bin/env python3
"""Report progress for a Strategy Generator GRPO run directory."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def milestone_value(row: dict[str, Any]) -> int | None:
    milestone = row.get("milestone")
    value = milestone.get("milestone") if isinstance(milestone, dict) else milestone
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def latest_by_event(rows: list[dict[str, Any]], event: str) -> dict[str, Any] | None:
    selected = [row for row in rows if row.get("event") == event]
    return selected[-1] if selected else None


def parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def estimate_progress(
    *,
    metrics: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    strategies: list[dict[str, Any]],
    hyperparameters: dict[str, Any],
    current_step: int,
    complete: bool,
) -> dict[str, Any]:
    rollout_events = [row for row in metrics if row.get("event") == "rollout_done"]
    sample_done_times = [
        parsed
        for parsed in (parse_time(row.get("time")) for row in metrics if row.get("event") == "sample_done")
        if parsed is not None
    ]
    now = dt.datetime.now(dt.timezone.utc)
    start_time = min(sample_done_times) if sample_done_times else None
    elapsed_seconds = (now - start_time).total_seconds() if start_time else None
    completed_rollouts = len(rollouts)

    configured_steps = as_int(hyperparameters.get("max_steps"))
    tasks_per_step = as_int(hyperparameters.get("tasks_per_step"))
    pool_per_task = as_int(hyperparameters.get("rollout_pool_per_task") or hyperparameters.get("group_size"))
    expected_total = (
        configured_steps * tasks_per_step * pool_per_task
        if configured_steps and tasks_per_step and pool_per_task
        else None
    )

    current_step_expected = sum(1 for row in strategies if as_int(row.get("step")) == current_step)
    current_step_completed = sum(
        1
        for row in rollout_events
        if as_int(row.get("step")) == current_step
    )

    throughput_per_hour = None
    eta_remaining_hours = None
    eta_current_step_hours = None
    if elapsed_seconds and elapsed_seconds > 0 and completed_rollouts > 0:
        throughput_per_hour = completed_rollouts / elapsed_seconds * 3600.0
        if expected_total is not None and not complete:
            remaining = max(expected_total - completed_rollouts, 0)
            eta_remaining_hours = remaining / throughput_per_hour if throughput_per_hour else None
        if current_step_expected and current_step_completed < current_step_expected:
            eta_current_step_hours = (current_step_expected - current_step_completed) / throughput_per_hour

    wall_seconds = [
        float(row["wall_seconds"])
        for row in rollout_events
        if isinstance(row.get("wall_seconds"), (int, float))
    ]
    wall_seconds_sorted = sorted(wall_seconds)
    p90_index = int(0.9 * (len(wall_seconds_sorted) - 1)) if wall_seconds_sorted else 0

    return {
        "now": now.isoformat(),
        "start_time": start_time.isoformat() if start_time else None,
        "elapsed_hours": elapsed_seconds / 3600.0 if elapsed_seconds else None,
        "completed_rollouts": completed_rollouts,
        "expected_total_rollouts": expected_total,
        "remaining_rollouts": max(expected_total - completed_rollouts, 0) if expected_total is not None else None,
        "current_step_completed_rollouts": current_step_completed,
        "current_step_expected_rollouts": current_step_expected or None,
        "current_step_remaining_rollouts": (
            max(current_step_expected - current_step_completed, 0) if current_step_expected else None
        ),
        "throughput_rollouts_per_hour": throughput_per_hour,
        "eta_remaining_hours": eta_remaining_hours,
        "eta_current_step_hours": eta_current_step_hours,
        "executor_wall_seconds_mean": sum(wall_seconds) / len(wall_seconds) if wall_seconds else None,
        "executor_wall_seconds_median": (
            wall_seconds_sorted[len(wall_seconds_sorted) // 2] if wall_seconds_sorted else None
        ),
        "executor_wall_seconds_p90": wall_seconds_sorted[p90_index] if wall_seconds_sorted else None,
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    result = load_json(run_dir / "result.json")
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    groups = load_jsonl(run_dir / "groups.jsonl")
    rollouts = load_jsonl(run_dir / "rollouts.jsonl")
    checkpoints = load_jsonl(run_dir / "checkpoints.jsonl")
    strategies = load_jsonl(run_dir / "strategies.jsonl")

    step_completes = [row for row in metrics if row.get("event") == "step_complete"]
    updates = [row for row in metrics if row.get("event") in {"update", "update_skipped"}]
    current_step = max([int(row.get("step") or 0) for row in metrics if row.get("step") is not None] or [0])
    completed_step = max([int(row.get("step") or 0) for row in step_completes] or [0])

    rewards = []
    for group in groups:
        for reward in group.get("rewards") or []:
            if isinstance(reward, (int, float)):
                rewards.append(float(reward))

    latest_checkpoint = next(
        (row for row in reversed(checkpoints) if row.get("event") == "checkpoint"),
        None,
    )
    latest_sampler = next(
        (row for row in reversed(checkpoints) if row.get("event") == "sampler_weights"),
        None,
    )

    hyperparameters = manifest.get("hyperparameters") or {}
    rollout_sample_ids = {
        str(row.get("strategy_id"))
        for row in rollouts
        if row.get("strategy_id") is not None
    }
    rollout_by_sample = {
        str(row.get("strategy_id")): row
        for row in rollouts
        if row.get("strategy_id") is not None
    }
    group_progress_map: dict[tuple[int, str, int], dict[str, Any]] = {}
    for strategy in strategies:
        try:
            step = int(strategy.get("step"))
            task_id = str(strategy.get("task_id"))
            group_index = int(strategy.get("group_index"))
            group_sample_index = int(strategy.get("group_sample_index"))
        except (TypeError, ValueError):
            continue
        key = (step, task_id, group_index)
        record = group_progress_map.setdefault(
            key,
            {
                "step": step,
                "task_id": task_id,
                "group_index": group_index,
                "expected": 0,
                "completed": 0,
                "completed_group_sample_indices": [],
                "statuses": {},
                "milestones": {},
            },
        )
        record["expected"] += 1
        sample_id = str(strategy.get("sample_id"))
        if sample_id in rollout_sample_ids:
            rollout = rollout_by_sample[sample_id]
            record["completed"] += 1
            record["completed_group_sample_indices"].append(group_sample_index)
            status = str(rollout.get("status") or "UNKNOWN")
            milestone = str(milestone_value(rollout))
            record["statuses"][status] = int(record["statuses"].get(status, 0)) + 1
            record["milestones"][milestone] = int(record["milestones"].get(milestone, 0)) + 1
    group_progress = []
    for record in group_progress_map.values():
        record["completed_group_sample_indices"] = sorted(record["completed_group_sample_indices"])
        group_progress.append(record)
    group_progress.sort(key=lambda row: (row["step"], row["task_id"], row["group_index"]))
    complete = bool(result)
    estimate = estimate_progress(
        metrics=metrics,
        rollouts=rollouts,
        strategies=strategies,
        hyperparameters=hyperparameters,
        current_step=current_step,
        complete=complete,
    )

    return {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id") or result.get("run_id"),
        "complete": complete,
        "current_step": current_step,
        "completed_step": completed_step,
        "configured_steps": hyperparameters.get("max_steps"),
        "configured_group_size": hyperparameters.get("group_size"),
        "configured_advantage_group_size": hyperparameters.get("advantage_group_size")
        or hyperparameters.get("group_size"),
        "configured_rollout_pool_per_task": hyperparameters.get("rollout_pool_per_task")
        or hyperparameters.get("group_size"),
        "configured_rollout_groups_per_task": hyperparameters.get("rollout_groups_per_task"),
        "configured_update_as_groups_complete": hyperparameters.get("update_as_groups_complete"),
        "configured_groups_per_update": hyperparameters.get("groups_per_update"),
        "configured_tasks_per_step": hyperparameters.get("tasks_per_step"),
        "strategies": len(strategies),
        "rollouts": len(rollouts),
        "groups": len(groups),
        "uniform_groups": sum(bool(row.get("uniform")) for row in groups),
        "skipped_groups": sum(bool(row.get("skipped_update")) for row in groups),
        "updates": len([row for row in updates if row.get("event") == "update"]),
        "updates_skipped": len([row for row in updates if row.get("event") == "update_skipped"]),
        "latest_update": updates[-1] if updates else None,
        "latest_step": step_completes[-1] if step_completes else None,
        "status_counts": dict(Counter(str(row.get("status") or "UNKNOWN") for row in rollouts)),
        "milestone_counts": dict(Counter(str(milestone_value(row)) for row in rollouts)),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "latest_checkpoint": latest_checkpoint,
        "latest_sampler": latest_sampler,
        "group_progress": group_progress,
        "estimate": estimate,
        "result": result or None,
    }


def print_text(summary: dict[str, Any]) -> None:
    print(f"run_dir: {summary['run_dir']}")
    print(f"run_id: {summary.get('run_id')}")
    print(
        "steps: "
        f"current={summary['current_step']} completed={summary['completed_step']} "
        f"configured={summary.get('configured_steps')}"
    )
    print(
        "config: "
        f"tasks_per_step={summary.get('configured_tasks_per_step')} "
        f"advantage_group_size={summary.get('configured_advantage_group_size')} "
        f"rollout_pool_per_task={summary.get('configured_rollout_pool_per_task')} "
        f"rollout_groups_per_task={summary.get('configured_rollout_groups_per_task')} "
        f"update_as_groups_complete={summary.get('configured_update_as_groups_complete')} "
        f"groups_per_update={summary.get('configured_groups_per_update')}"
    )
    print(
        "counts: "
        f"strategies={summary['strategies']} rollouts={summary['rollouts']} "
        f"groups={summary['groups']} updates={summary['updates']} skipped_updates={summary['updates_skipped']}"
    )
    estimate = summary.get("estimate") or {}
    if estimate:
        print(
            "eta: "
            f"rollouts={estimate.get('completed_rollouts')}/{estimate.get('expected_total_rollouts')} "
            f"current_step={estimate.get('current_step_completed_rollouts')}/"
            f"{estimate.get('current_step_expected_rollouts')} "
            f"throughput_per_hour={fmt_float(estimate.get('throughput_rollouts_per_hour'), 2)} "
            f"step_remaining_hours={fmt_float(estimate.get('eta_current_step_hours'), 2)} "
            f"train_remaining_hours={fmt_float(estimate.get('eta_remaining_hours'), 2)}"
        )
        print(
            "executor_wall_minutes: "
            f"mean={fmt_float(minutes(estimate.get('executor_wall_seconds_mean')), 2)} "
            f"median={fmt_float(minutes(estimate.get('executor_wall_seconds_median')), 2)} "
            f"p90={fmt_float(minutes(estimate.get('executor_wall_seconds_p90')), 2)}"
        )
    print(
        "groups: "
        f"uniform={summary['uniform_groups']} skipped={summary['skipped_groups']} "
        f"mean_reward={summary['mean_reward']}"
    )
    print(f"status: {summary['status_counts']}")
    print(f"milestone: {summary['milestone_counts']}")
    group_progress = summary.get("group_progress") or []
    if group_progress:
        current_step = summary.get("current_step")
        visible = [
            row
            for row in group_progress
            if row.get("step") == current_step and row.get("completed") != row.get("expected")
        ][:12]
        if not visible:
            visible = [row for row in group_progress if row.get("step") == current_step][:12]
        print("group_progress:")
        for row in visible:
            print(
                "  "
                f"step={row.get('step')} task={row.get('task_id')} group={row.get('group_index')} "
                f"completed={row.get('completed')}/{row.get('expected')} "
                f"k={row.get('completed_group_sample_indices')} "
                f"status={row.get('statuses')} milestone={row.get('milestones')}"
            )
    latest_step = summary.get("latest_step") or {}
    if latest_step:
        print(
            "latest_step: "
            f"step={latest_step.get('step')} mean_reward={latest_step.get('mean_reward')} "
            f"datums={latest_step.get('datums')} elapsed={latest_step.get('elapsed')}"
        )
    latest_update = summary.get("latest_update") or {}
    if latest_update:
        print(
            "latest_update: "
            f"event={latest_update.get('event')} step={latest_update.get('step')} "
            f"datums={latest_update.get('datums')} reason={latest_update.get('reason')}"
        )
    latest_sampler = summary.get("latest_sampler") or {}
    if latest_sampler:
        print(f"latest_sampler: {latest_sampler.get('path')}")
    latest_checkpoint = summary.get("latest_checkpoint") or {}
    if latest_checkpoint:
        print(f"latest_checkpoint: {latest_checkpoint.get('path')}")
    if summary.get("complete"):
        result = summary.get("result") or {}
        print(f"complete: final_sampler={result.get('final_sampler_path')}")


def fmt_float(value: Any, digits: int) -> str:
    if not isinstance(value, (int, float)):
        return "None"
    return f"{value:.{digits}f}"


def minutes(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return value / 60.0


def main() -> int:
    args = parse_args()
    summary = summarize(args.run_dir)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
