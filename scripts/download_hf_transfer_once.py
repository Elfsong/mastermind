#!/usr/bin/env python3
"""Download one Hugging Face file with the low-level hf_transfer API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import hf_transfer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_url
from huggingface_hub.file_download import get_hf_file_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("filename")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--parallel-failures", type=int, default=16)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)

    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists; pass --force to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    url = hf_hub_url(args.repo_id, args.filename, repo_type=args.repo_type)
    metadata = get_hf_file_metadata(url, token=token, timeout=60)
    if metadata.size is None:
        raise SystemExit("Could not determine remote file size")

    print(f"url: {url}", flush=True)
    print(f"etag: {metadata.etag}", flush=True)
    print(f"size: {metadata.size}", flush=True)
    print(f"output: {output}", flush=True)

    headers = {}
    if token:
        headers["authorization"] = f"Bearer {token}"

    hf_transfer.download(
        metadata.location,
        str(output),
        max_files=args.max_files,
        chunk_size=args.chunk_size,
        parallel_failures=args.parallel_failures,
        max_retries=args.max_retries,
        headers=headers,
    )

    actual_size = output.stat().st_size
    print(f"downloaded: {actual_size}", flush=True)
    if actual_size != metadata.size:
        raise SystemExit(f"size mismatch: expected {metadata.size}, got {actual_size}")


if __name__ == "__main__":
    main()
