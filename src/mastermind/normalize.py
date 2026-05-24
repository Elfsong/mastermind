"""Normalize runner-specific outputs into Mastermind rollout records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .rollout import MilestoneSummary, RolloutRecord, VerificationSummary


def status_from_execution(entry: dict[str, Any]) -> str:
    if entry.get("cancelled"):
        return "CANCELLED"
    if entry.get("trajectory_path"):
        return "OK"
    wall_seconds = entry.get("wall_seconds")
    timeout = entry.get("executor_timeout")
    if isinstance(wall_seconds, (int, float)) and isinstance(timeout, (int, float)):
        if wall_seconds >= timeout - 5:
            return "TIMEOUT"
    return "CRASH"


def rollout_from_dual_loop_entry(
    *,
    run_id: str,
    entry: dict[str, Any],
    model: str,
    executor: str = "dual_loops.openhands",
    milestone: int | None = None,
    milestone_reasoning: str | None = None,
    strategy: str | None = None,
) -> RolloutRecord:
    """Convert one `dual_loops` execution JSONL row to the shared schema."""
    status = status_from_execution(entry)
    submit_count = entry.get("submit_count")
    return RolloutRecord(
        run_id=run_id,
        task_id=str(entry.get("task_id", "")),
        agent=str(entry.get("agent_id", "")),
        model=model,
        executor=executor,
        status=status,
        milestone=MilestoneSummary(
            milestone=milestone,
            reasoning=milestone_reasoning,
        ),
        verification=VerificationSummary(
            status="unverified" if milestone is None else "milestone_scored",
            passed=(milestone == 7 if milestone is not None else None),
            submit_count=submit_count if isinstance(submit_count, int) else None,
        ),
        trajectory_path=(
            str(Path(entry["trajectory_path"]))
            if entry.get("trajectory_path")
            else None
        ),
        wall_seconds=entry.get("wall_seconds"),
        strategy_id=(
            f"{entry.get('task_id')}:{entry.get('group_id')}"
            if "group_id" in entry
            else None
        ),
        strategy=strategy,
        metadata={
            "subprocess_returncode": entry.get("subprocess_returncode"),
            "log_dir": entry.get("log_dir"),
            "group_id": entry.get("group_id"),
            "source": "dual_loops.executions.jsonl",
        },
    )
