"""CyberGym task manifest helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Manifest


@dataclass(frozen=True)
class TaskMetadata:
    task_id: str
    project_name: str
    project_homepage: str | None = None
    project_main_repo: str | None = None
    project_language: str | None = None
    vulnerability_description: str | None = None
    task_difficulty: dict[str, list[str]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> str:
        return self.task_id.split(":", 1)[0]


def read_task_ids(path: Path) -> list[str]:
    """Read task IDs from a TASKS file, skipping comments and blanks."""
    task_ids: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                task_ids.append(line)
    return task_ids


def load_task_metadata(path: Path) -> dict[str, TaskMetadata]:
    """Load CyberGym `tasks.json` keyed by task_id."""
    with path.open() as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in tasks metadata: {path}")

    tasks: dict[str, TaskMetadata] = {}
    for row in rows:
        if not isinstance(row, dict) or "task_id" not in row:
            continue
        metadata = TaskMetadata(
            task_id=str(row["task_id"]),
            project_name=str(row.get("project_name", "")),
            project_homepage=row.get("project_homepage"),
            project_main_repo=row.get("project_main_repo"),
            project_language=row.get("project_language"),
            vulnerability_description=row.get("vulnerability_description"),
            task_difficulty={
                str(level): [str(item) for item in files]
                for level, files in row.get("task_difficulty", {}).items()
            },
            raw=row,
        )
        tasks[metadata.task_id] = metadata
    return tasks


def load_split_ids(manifest: Manifest, split: str) -> list[str]:
    try:
        split_path = manifest.task_splits[split]
    except KeyError as exc:
        known = ", ".join(sorted(manifest.task_splits))
        raise KeyError(f"Unknown split {split!r}; known splits: {known}") from exc
    return read_task_ids(split_path)


def load_split_metadata(manifest: Manifest, split: str) -> list[TaskMetadata]:
    metadata = load_task_metadata(manifest.benchmark.tasks_json)
    missing: list[str] = []
    selected: list[TaskMetadata] = []
    for task_id in load_split_ids(manifest, split):
        task = metadata.get(task_id)
        if task is None:
            missing.append(task_id)
        else:
            selected.append(task)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} task IDs from split {split!r} are missing in tasks.json: {preview}"
        )
    return selected
