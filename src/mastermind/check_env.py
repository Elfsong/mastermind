"""Check the local Mastermind/CyberGym environment."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any

from .config import CheckResult, check_manifest, load_manifest
from .tasks import load_split_ids, load_task_metadata


def _probe_url(name: str, url: str, timeout: float = 3.0) -> CheckResult:
    if not url:
        return CheckResult(name=name, ok=False, detail="missing URL")
    try:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return CheckResult(
                name=name,
                ok=200 <= response.status < 500,
                detail=f"HTTP {response.status} {url}",
            )
    except urllib.error.HTTPError as exc:
        return CheckResult(name=name, ok=exc.code < 500, detail=f"HTTP {exc.code} {url}")
    except Exception as exc:
        return CheckResult(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}")


def _as_json(checks: list[CheckResult], split_counts: dict[str, int]) -> str:
    return json.dumps(
        {
            "ok": all(check.ok for check in checks),
            "checks": [asdict(check) for check in checks],
            "split_counts": split_counts,
        },
        indent=2,
        sort_keys=True,
    )


def _print_table(checks: list[CheckResult], split_counts: dict[str, int]) -> None:
    for check in checks:
        marker = "OK" if check.ok else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")
    if split_counts:
        print("split counts:")
        for name in sorted(split_counts):
            print(f"  {name}: {split_counts[name]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--probe-services",
        action="store_true",
        help="Probe CyberGym and executor HTTP endpoints.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    checks = check_manifest(manifest)

    metadata = load_task_metadata(manifest.benchmark.tasks_json)
    checks.append(
        CheckResult(
            name="benchmark.tasks_json.count",
            ok=bool(metadata),
            detail=str(len(metadata)),
        )
    )

    split_counts: dict[str, int] = {}
    for split in sorted(manifest.task_splits):
        task_ids = load_split_ids(manifest, split)
        split_counts[split] = len(task_ids)
        missing = [task_id for task_id in task_ids if task_id not in metadata]
        checks.append(
            CheckResult(
                name=f"task_splits.{split}.metadata_match",
                ok=not missing,
                detail="all present" if not missing else ", ".join(missing[:5]),
            )
        )

    if args.probe_services:
        checks.append(
            _probe_url("service.cybergym_server", manifest.cybergym_server_url)
        )
        executor_url = manifest.executor_base_url.rstrip("/")
        checks.append(
            _probe_url("service.executor_models", f"{executor_url}/models")
        )

    if args.json:
        print(_as_json(checks, split_counts))
    else:
        _print_table(checks, split_counts)
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
