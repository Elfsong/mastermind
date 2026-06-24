#!/usr/bin/env python3
"""Upload staged CyberGym evaluation datasets to Hugging Face Hub.

Default behavior is a local dry run. Pass --execute to create/update repos and
upload folders. The default upload method is upload_large_folder because the
trajectory bundles contain many files and can be larger than 1GB.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / "runs/hf_upload_staging"
DATASETS = (
    "cybergym-codex-gpt-5-5-round1-8-independent-eval",
    "cybergym-codex-gpt-5-5-level3-eval",
    "cybergym-codex-gpt-5-5-iterative-improvement-eval",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually create/update Hugging Face dataset repos and upload files. Default is dry-run only.",
    )
    parser.add_argument(
        "--namespace",
        help="Hugging Face namespace. Defaults to token owner from whoami when --execute is used.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create repos as public. Default is private.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="Upload only one dataset. Repeat to select multiple. Default: all datasets.",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=STAGING_ROOT,
        help="Directory containing staged dataset folders.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload CyberGym Codex GPT-5.5 evaluation artifacts",
    )
    parser.add_argument(
        "--commit-description",
        default=(
            "Includes rollout summaries, enriched rollout rows with full visible trajectory events, "
            "extracted prompt/output items, raw Codex trajectories, and aggregate metadata."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("large-folder", "folder"),
        default="large-folder",
        help="Upload method. large-folder is resumable and recommended for these datasets.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Worker count for --method large-folder.",
    )
    return parser.parse_args()


def load_env_token() -> str:
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    env_path = ROOT / ".env"
    for line in env_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            return value.strip().strip('"').strip("'")
    raise RuntimeError("HF_TOKEN not found")


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def dataset_plan(staging_root: Path, dataset: str, namespace: str | None) -> dict[str, object]:
    local_dir = staging_root / dataset
    summary_path = local_dir / "metadata/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    repo_id = f"{namespace or '<namespace>'}/{dataset}"
    return {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "exists": local_dir.exists(),
        "files": sum(1 for p in local_dir.rglob("*") if p.is_file()) if local_dir.exists() else 0,
        "size_bytes": folder_size(local_dir) if local_dir.exists() else 0,
        "overall": summary.get("overall"),
        "trajectory_files_found": summary.get("trajectory_files_found"),
        "trajectory_manifest_rows": summary.get("trajectory_manifest_rows"),
        "trajectory_files_missing": summary.get("trajectory_files_missing"),
    }


def main() -> int:
    args = parse_args()
    selected = tuple(args.dataset or DATASETS)

    namespace = args.namespace
    if args.execute:
        token = load_env_token()
        api = HfApi(token=token)
        namespace = namespace or api.whoami()["name"]
    else:
        api = None

    print(
        json.dumps(
            {
                "event": "plan",
                "mode": "execute" if args.execute else "dry_run",
                "method": args.method,
                "visibility": "public" if args.public else "private",
                "namespace": namespace or "<resolved on --execute>",
                "datasets": [dataset_plan(args.staging_root, dataset, namespace) for dataset in selected],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.execute:
        print("Dry run only. Re-run with --execute to upload.", flush=True)
        return 0

    results = []
    for dataset in selected:
        local_dir = args.staging_root / dataset
        if not local_dir.exists():
            raise FileNotFoundError(local_dir)
        repo_id = f"{namespace}/{dataset}"
        print(
            json.dumps(
                {
                    "event": "create_repo",
                    "repo_id": repo_id,
                    "private": not args.public,
                    "local_dir": str(local_dir),
                    "size_bytes": folder_size(local_dir),
                    "time": datetime.now(UTC).isoformat(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        assert api is not None
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
        print(
            json.dumps({"event": "upload_start", "repo_id": repo_id, "time": datetime.now(UTC).isoformat()}, sort_keys=True),
            flush=True,
        )
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
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
    (args.staging_root / "upload_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
