#!/usr/bin/env python3
"""Upload one local folder as a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--namespace", help="Defaults to the HF token owner.")
    parser.add_argument("--public", action="store_true", help="Create/update as public. Default is private.")
    parser.add_argument(
        "--method",
        choices=("large-folder", "folder"),
        default="large-folder",
        help="large-folder is resumable; folder creates one normal commit.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--commit-message", default="Upload CyberGym eval trajectories")
    parser.add_argument("--commit-description", default="")
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def load_env_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = ROOT / ".env"
    for line in env_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            return value.strip().strip('"').strip("'")
    raise RuntimeError("HF_TOKEN not found in environment or .env")


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def main() -> int:
    args = parse_args()
    local_dir = args.local_dir.resolve()
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    token = load_env_token()
    api = HfApi(token=token)
    namespace = args.namespace or api.whoami()["name"]
    repo_id = f"{namespace}/{args.repo_name}"
    private = not args.public

    plan = {
        "event": "upload_plan",
        "repo_id": repo_id,
        "repo_type": "dataset",
        "private": private,
        "local_dir": str(local_dir),
        "files": file_count(local_dir),
        "size_bytes": folder_size(local_dir),
        "method": args.method,
        "time": datetime.now(UTC).isoformat(),
    }
    print(json.dumps(plan, sort_keys=True), flush=True)

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(json.dumps({"event": "repo_ready", "repo_id": repo_id}, sort_keys=True), flush=True)

    commit_url = None
    commit_oid = None
    if args.method == "large-folder":
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=local_dir,
            private=private,
            num_workers=args.num_workers,
            print_report=True,
            print_report_every=60,
        )
    else:
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=local_dir,
            commit_message=args.commit_message,
            commit_description=args.commit_description,
        )
        commit_url = getattr(commit, "commit_url", None)
        commit_oid = getattr(commit, "oid", None)

    result = {
        "event": "upload_complete",
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "commit_url": commit_url,
        "commit_oid": commit_oid,
        "method": args.method,
        "time": datetime.now(UTC).isoformat(),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
