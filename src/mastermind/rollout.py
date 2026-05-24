"""Stable rollout record schema and JSONL store helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MilestoneSummary:
    milestone: int | None
    reasoning: str | None = None
    verified_fix: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.milestone is not None and not 0 <= self.milestone <= 7:
            raise ValueError(f"milestone must be in [0, 7], got {self.milestone}")


@dataclass(frozen=True)
class VerificationSummary:
    status: str
    passed: bool | None = None
    submit_count: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.status:
            raise ValueError("verification.status is required")
        if self.submit_count is not None and self.submit_count < 0:
            raise ValueError("verification.submit_count must be non-negative")


@dataclass(frozen=True)
class RolloutRecord:
    run_id: str
    task_id: str
    agent: str
    model: str
    executor: str
    status: str
    milestone: MilestoneSummary
    verification: VerificationSummary
    trajectory_path: str | None = None
    cost: float | None = None
    wall_seconds: float | None = None
    strategy_id: str | None = None
    strategy: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent": self.agent,
            "model": self.model,
            "executor": self.executor,
            "status": self.status,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"missing required rollout fields: {', '.join(missing)}")
        if self.cost is not None and self.cost < 0:
            raise ValueError("cost must be non-negative")
        if self.wall_seconds is not None and self.wall_seconds < 0:
            raise ValueError("wall_seconds must be non-negative")
        self.milestone.validate()
        self.verification.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RolloutRecord":
        milestone_value = value.get("milestone") or {}
        verification_value = value.get("verification") or {}
        record = cls(
            run_id=str(value.get("run_id", "")),
            task_id=str(value.get("task_id", "")),
            agent=str(value.get("agent", "")),
            model=str(value.get("model", "")),
            executor=str(value.get("executor", "")),
            status=str(value.get("status", "")),
            milestone=MilestoneSummary(**milestone_value),
            verification=VerificationSummary(**verification_value),
            trajectory_path=value.get("trajectory_path"),
            cost=value.get("cost"),
            wall_seconds=value.get("wall_seconds"),
            strategy_id=value.get("strategy_id"),
            strategy=value.get("strategy"),
            started_at=value.get("started_at"),
            finished_at=value.get("finished_at"),
            metadata=value.get("metadata") or {},
        )
        record.validate()
        return record


def append_rollout(path: Path, record: RolloutRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.to_json())
        f.write("\n")


def read_rollouts(path: Path) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(RolloutRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid rollout JSONL at {path}:{line_no}") from exc
    return records
