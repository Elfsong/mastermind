#!/usr/bin/env python3
"""Download CyberGym files required by a task split."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from mastermind.config import load_manifest
from mastermind.tasks import load_split_ids, load_task_metadata


@dataclass(frozen=True)
class DownloadResult:
    task_id: str
    filename: str
    status: str
    path: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train_dev")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--repo-id", default="sunblaze-ucb/cybergym")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/cybergym_assets/download_manifests/train_dev_level1.jsonl"),
    )
    return parser.parse_args()


def required_files(split: str, difficulty: str, manifest_path: Path | None) -> list[tuple[str, str]]:
    manifest = load_manifest(manifest_path)
    metadata = load_task_metadata(manifest.benchmark.tasks_json)
    files: list[tuple[str, str]] = []
    for task_id in load_split_ids(manifest, split):
        task = metadata[task_id]
        try:
            rel_paths = task.task_difficulty[difficulty]
        except KeyError as exc:
            raise KeyError(f"{task_id} has no difficulty {difficulty!r}") from exc
        for rel_path in rel_paths:
            files.append((task_id, rel_path))
    return files


def download_one(
    *,
    task_id: str,
    filename: str,
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    token: str | None,
    force: bool,
    local_files_only: bool,
) -> DownloadResult:
    target = local_dir / filename
    if target.exists() and not force:
        return DownloadResult(task_id, filename, "present", str(target))
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=local_dir,
            token=token,
            force_download=force,
            local_files_only=local_files_only,
            etag_timeout=60,
        )
    except Exception as exc:  # noqa: BLE001 - record all per-file failures.
        return DownloadResult(task_id, filename, "failed", error=repr(exc))
    return DownloadResult(task_id, filename, "downloaded", path)


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    manifest = load_manifest(args.manifest)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    files = required_files(args.split, args.difficulty, args.manifest)
    if args.limit is not None:
        files = files[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    print(
        json.dumps(
            {
                "event": "start",
                "split": args.split,
                "difficulty": args.difficulty,
                "repo_id": args.repo_id,
                "local_dir": str(manifest.benchmark.root),
                "files": len(files),
                "started_at": started_at,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    workers = max(1, args.workers)
    failures = 0
    with args.output.open("a") as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    download_one,
                    task_id=task_id,
                    filename=filename,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    local_dir=manifest.benchmark.root,
                    token=token,
                    force=args.force,
                    local_files_only=args.local_files_only,
                )
                for task_id, filename in files
            ]
            for future in as_completed(futures):
                result = future.result()
                if result.status == "failed":
                    failures += 1
                out.write(json.dumps(asdict(result), sort_keys=True))
                out.write("\n")
                out.flush()
                print(json.dumps(asdict(result), sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "event": "finish",
                "files": len(files),
                "failures": failures,
                "finished_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
