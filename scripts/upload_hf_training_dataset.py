#!/usr/bin/env python3
"""Upload the staged CyberGym GPT-5.5 training dataset to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / "runs/hf_upload_staging"
DATASET_NAME = "cybergym-codex-gpt-5-5-iterative-improvement-train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually upload. Default is dry-run.")
    parser.add_argument("--namespace", help="HF namespace. Defaults to token owner.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--staging-root", type=Path, default=STAGING_ROOT)
    parser.add_argument("--public", action="store_true", help="Create public repo. Default is private.")
    parser.add_argument("--method", choices=("large-folder", "folder"), default="large-folder")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--commit-message", default="Upload CyberGym Codex GPT-5.5 training dataset")
    parser.add_argument(
        "--commit-description",
        default=(
            "Merged train split sequential self-improvement rollouts, reruns, raw trajectories, "
            "feedback, strategy updates, and one-step strategy-generator SFT examples."
        ),
    )
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


def main() -> int:
    args = parse_args()
    local_dir = args.staging_root / args.dataset_name
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    namespace = args.namespace
    api = None
    if args.execute:
        token = load_env_token()
        api = HfApi(token=token)
        namespace = namespace or api.whoami()["name"]
    repo_id = f"{namespace or '<namespace>'}/{args.dataset_name}"
    summary_path = local_dir / "metadata/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    plan = {
        "event": "plan",
        "mode": "execute" if args.execute else "dry_run",
        "repo_id": repo_id,
        "private": not args.public,
        "method": args.method,
        "local_dir": str(local_dir),
        "files": sum(1 for p in local_dir.rglob("*") if p.is_file()),
        "size_bytes": folder_size(local_dir),
        "summary_overall": summary.get("overall"),
        "time": datetime.now(UTC).isoformat(),
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        print("Dry run only. Re-run with --execute to upload.", flush=True)
        return 0

    assert api is not None and namespace is not None
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    print(json.dumps({"event": "upload_start", "repo_id": repo_id, "time": datetime.now(UTC).isoformat()}), flush=True)
    if args.method == "large-folder":
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=local_dir,
            private=not args.public,
            num_workers=args.num_workers,
            print_report=True,
            print_report_every=60,
        )
        commit_url = None
        commit_oid = None
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
        "commit_url": commit_url,
        "commit_oid": commit_oid,
        "method": args.method,
        "time": datetime.now(UTC).isoformat(),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    (args.staging_root / f"{args.dataset_name}_upload_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
