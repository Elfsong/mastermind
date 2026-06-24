#!/usr/bin/env python3
"""Archive PoC files from CyberGym workspaces before workspace cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    parser.add_argument("--copy-submit-logs", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_rollouts(path: Path, run_id: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if run_id is not None and row.get("run_id") != run_id:
                continue
            rows.append(row)
    return rows


def unique_destination(base: Path) -> Path:
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    for index in range(2, 10_000):
        candidate = base.with_name(f"{stem}.{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique archive path for {base}")


def archive_pocs(row: dict[str, Any], archive_root: Path) -> list[dict[str, Any]]:
    metadata = row.get("metadata") or {}
    workspace_value = metadata.get("workspace")
    if not isinstance(workspace_value, str):
        return []
    workspace = Path(workspace_value)
    pocs_dir = workspace / "pocs"
    if not pocs_dir.is_dir():
        return []

    task_id = row.get("task_id")
    task_safe = safe_name(task_id if isinstance(task_id, str) else workspace.name)
    task_archive = archive_root / "pocs" / task_safe
    task_archive.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for source in sorted(path for path in pocs_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(pocs_dir)
        destination = unique_destination(task_archive / relative.name)
        shutil.copy2(source, destination)
        records.append(
            {
                "run_id": row.get("run_id"),
                "task_id": task_id,
                "status": row.get("status"),
                "milestone": (row.get("milestone") or {}).get("milestone"),
                "source_workspace": str(workspace),
                "source_relative_path": str(Path("pocs") / relative),
                "archive_path": str(destination),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "submitted_poc_id": metadata.get("poc_id"),
                "submitted_poc_hash": metadata.get("poc_hash"),
            }
        )
    return records


def archive_submit_logs(row: dict[str, Any], archive_root: Path) -> list[dict[str, Any]]:
    metadata = row.get("metadata") or {}
    task_id = row.get("task_id")
    task_safe = safe_name(task_id if isinstance(task_id, str) else "unknown_task")
    logs_archive = archive_root / "submit_logs" / task_safe
    records: list[dict[str, Any]] = []
    for result in metadata.get("auto_submit_results") or []:
        log_value = result.get("log_path")
        if not isinstance(log_value, str):
            continue
        source = Path(log_value)
        if not source.is_file():
            continue
        logs_archive.mkdir(parents=True, exist_ok=True)
        destination = unique_destination(logs_archive / source.name)
        shutil.copy2(source, destination)
        records.append(
            {
                "run_id": row.get("run_id"),
                "task_id": task_id,
                "source_log_path": str(source),
                "archive_log_path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    return records


def main() -> int:
    args = parse_args()
    archive_root = args.archive_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    poc_manifest = archive_root / args.manifest_name
    log_manifest = archive_root / "submit_logs_manifest.jsonl"

    rows = iter_rollouts(args.rollouts, args.run_id)
    poc_records: list[dict[str, Any]] = []
    log_records: list[dict[str, Any]] = []
    for row in rows:
        poc_records.extend(archive_pocs(row, archive_root))
        if args.copy_submit_logs:
            log_records.extend(archive_submit_logs(row, archive_root))

    with poc_manifest.open("w") as f:
        for record in poc_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    if args.copy_submit_logs:
        with log_manifest.open("w") as f:
            for record in log_records:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "archive_root": str(archive_root),
                "rollout_rows": len(rows),
                "poc_files": len(poc_records),
                "poc_manifest": str(poc_manifest),
                "submit_log_files": len(log_records),
                "submit_log_manifest": str(log_manifest) if args.copy_submit_logs else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
