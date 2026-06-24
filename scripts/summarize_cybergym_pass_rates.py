#!/usr/bin/env python3
"""Summarize CyberGym rollout JSONL pass rates by latest attempt per task."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


AGENT_ERROR_STATUSES = {"AGENT_ERROR", "CRASH"}  # CRASH is legacy rollout data.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, nargs="+")
    parser.add_argument(
        "--include-infra",
        action="store_true",
        help="Include known infrastructure/API failures, e.g. Codex 429 rows, in latest-attempt task counts.",
    )
    return parser.parse_args()


def attempt_index(row: dict[str, Any]) -> int:
    metadata = row.get("metadata") or {}
    sequential = metadata.get("sequential") or {}
    for value in (sequential.get("attempt_index"), metadata.get("attempt_index"), row.get("attempt")):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return 0


def milestone(row: dict[str, Any]) -> int | None:
    value = row.get("milestone")
    if isinstance(value, dict):
        value = value.get("milestone")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_excerpt(path: Any, *, max_chars: int = 4000) -> str:
    if not path:
        return ""
    try:
        text = Path(str(path)).read_text(errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]


def is_infra_failure(row: dict[str, Any]) -> bool:
    """Return true for API/runner failures that should be rerun, not scored."""
    if str(row.get("status") or "") not in AGENT_ERROR_STATUSES:
        return False
    metadata = row.get("metadata") or {}
    haystacks = [
        read_excerpt(metadata.get("codex_stdout")),
        read_excerpt(metadata.get("codex_stderr")),
    ]
    verification = row.get("verification") or {}
    details = verification.get("details") or {}
    haystacks.extend(
        [
            read_excerpt(details.get("codex_stdout")),
            read_excerpt(details.get("codex_stderr")),
        ]
    )
    for text in haystacks:
        lower = text.lower()
        if (
            "you've hit your usage limit" in lower
            or "429 too many requests" in lower
            or "exceeded retry limit" in lower
            or "rate limit" in lower
            or "quota exceeded" in lower
            or "daily limits exceeded" in lower
            or "backend request failed" in lower
            or "prefill stall" in lower
            or "no data from backend" in lower
        ):
            return True
    return verification.get("status") == "runner_failed"


def summarize(path: Path, *, include_infra: bool) -> dict[str, Any]:
    rows = load_rows(path)
    infra_rows = [row for row in rows if is_infra_failure(row)]
    scored_rows = rows if include_infra else [row for row in rows if not is_infra_failure(row)]
    latest: dict[str, dict[str, Any]] = {}
    for row in scored_rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        current_key = (attempt_index(row), row.get("finished_at") or "")
        previous = latest.get(task_id)
        previous_key = (attempt_index(previous), previous.get("finished_at") or "") if previous else (-1, "")
        if current_key >= previous_key:
            latest[task_id] = row

    status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in latest.values())
    milestone_counts = Counter(str(milestone(row)) for row in latest.values())
    attempts = [attempt_index(row) for row in latest.values()]
    n = len(latest)
    passed = status_counts.get("PASSED", 0)
    m67 = sum(1 for row in latest.values() if milestone(row) in {6, 7})
    return {
        "path": str(path),
        "rows": len(rows),
        "scored_rows": len(scored_rows),
        "infra_rows_ignored": 0 if include_infra else len(infra_rows),
        "tasks": n,
        "passed": passed,
        "pass_rate": passed / n if n else None,
        "milestone_6_or_7": m67,
        "milestone_6_or_7_rate": m67 / n if n else None,
        "status_counts_latest": dict(status_counts),
        "milestone_counts_latest": dict(sorted(milestone_counts.items())),
        "attempts_latest": {
            "mean": sum(attempts) / len(attempts) if attempts else None,
            "max": max(attempts) if attempts else None,
        },
    }


def main() -> int:
    args = parse_args()
    summaries = [summarize(path, include_infra=args.include_infra) for path in args.jsonl]
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
