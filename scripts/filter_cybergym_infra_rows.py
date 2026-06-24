#!/usr/bin/env python3
"""Remove known infrastructure/API failure rows from a CyberGym rollout JSONL.

The scorer/report scripts ignore infra rows, but the sequential runner's resume
logic reads the raw output file.  If a high-concurrency run appends many Codex
429 rows, filtering them out before resuming keeps those tasks eligible for a
clean retry.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from summarize_cybergym_pass_rates import is_infra_failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="Backup path. Defaults to <jsonl>.pre_infra_filter_<UTC timestamp>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report counts; do not write files.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    rows = load_rows(args.jsonl)
    kept = [row for row in rows if not is_infra_failure(row)]
    removed = len(rows) - len(kept)
    backup = args.backup
    if backup is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = Path(str(args.jsonl) + f".pre_infra_filter_{stamp}")

    result = {
        "path": str(args.jsonl),
        "rows_before": len(rows),
        "rows_after": len(kept),
        "infra_rows_removed": removed,
        "backup": str(backup),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
        shutil.copy2(args.jsonl, backup)
        args.jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
