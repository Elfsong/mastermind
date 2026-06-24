#!/usr/bin/env python3
"""Monitor disk space and clean generated CyberGym artifacts when low."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


GB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("."))
    parser.add_argument("--threshold-gb", type=float, default=50.0)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def safe_task_name(task_id: str) -> str:
    return task_id.replace(":", "_").replace("/", "_")


def ps_lines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "args"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.splitlines()


def token_after(tokens: list[str], option: str) -> str | None:
    try:
        index = tokens.index(option)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def active_workspaces(root: Path) -> set[Path]:
    active: set[Path] = set()
    for line in ps_lines():
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if not tokens:
            continue

        if any(Path(token).name == "codex" for token in tokens) and "-C" in tokens:
            cwd = token_after(tokens, "-C")
            if cwd:
                active.add(Path(cwd).resolve())

        if any(token.endswith("scripts/run_codex_cybergym_tasks.py") for token in tokens):
            run_id = token_after(tokens, "--run-id")
            task_id = token_after(tokens, "--task-id")
            run_root = token_after(tokens, "--run-root") or "runs/codex_cybergym"
            if run_id and task_id:
                active.add((root / run_root / run_id / "workspaces" / safe_task_name(task_id)).resolve())
    return active


def dir_size(path: Path) -> int:
    total = 0
    for base, dirs, files in os.walk(path, topdown=True, onerror=lambda _err: None):
        for name in files:
            file_path = Path(base) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
        for name in dirs:
            dir_path = Path(base) / name
            try:
                total += dir_path.stat().st_size
            except OSError:
                pass
    return total


def remove_path(path: Path, *, dry_run: bool) -> int:
    try:
        size = dir_size(path) if path.is_dir() else path.stat().st_size
    except OSError:
        size = 0
    if not dry_run:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return size


def cleanup(root: Path, *, dry_run: bool) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    active = active_workspaces(root)

    workspace_roots = sorted((root / "runs" / "codex_cybergym").glob("*/workspaces"))
    for workspace_root in workspace_roots:
        if not workspace_root.is_dir():
            continue
        for task_workspace in sorted(path for path in workspace_root.iterdir() if path.is_dir()):
            resolved = task_workspace.resolve()
            if resolved in active:
                actions.append({"action": "skip_active_workspace", "path": str(task_workspace)})
                continue
            size = remove_path(task_workspace, dry_run=dry_run)
            actions.append(
                {
                    "action": "remove_workspace",
                    "path": str(task_workspace),
                    "bytes": size,
                    "dry_run": dry_run,
                }
            )
        if not dry_run:
            try:
                workspace_root.rmdir()
            except OSError:
                pass

    cache_paths = [
        root / "runs" / "cybergym_assets" / ".cache",
        root / "runs" / "cache" / "uv",
        root / "runs" / "uv-cache",
    ]
    for cache_path in cache_paths:
        if cache_path.exists():
            size = remove_path(cache_path, dry_run=dry_run)
            actions.append(
                {
                    "action": "remove_cache",
                    "path": str(cache_path),
                    "bytes": size,
                    "dry_run": dry_run,
                }
            )
    return actions


def append_log(log_path: Path, record: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    monitor_path = args.path.resolve()
    threshold_bytes = int(args.threshold_gb * GB)
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")

    while True:
        free_before = disk_free_bytes(monitor_path)
        record: dict[str, object] = {
            "event": "disk_check",
            "time": now_iso(),
            "path": str(monitor_path),
            "threshold_gb": args.threshold_gb,
            "free_gb": round(free_before / GB, 2),
            "dry_run": args.dry_run,
        }
        if free_before < threshold_bytes:
            actions = cleanup(root, dry_run=args.dry_run)
            free_after = disk_free_bytes(monitor_path)
            record.update(
                {
                    "event": "disk_cleanup",
                    "free_after_gb": round(free_after / GB, 2),
                    "actions": actions,
                    "removed_gb": round(
                        sum(int(action.get("bytes") or 0) for action in actions) / GB,
                        2,
                    ),
                }
            )
        append_log(args.log, record)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
