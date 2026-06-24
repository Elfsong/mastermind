#!/usr/bin/env python3
"""Compare base vs SFT CyberGym strategy-generator eval rollouts.

This is intentionally a one-shot reporter, not a poller.  It summarizes each
rollout file independently and then computes matched-task comparisons so the
base generator and SFT generator are compared on the same task set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from summarize_cybergym_pass_rates import (
    attempt_index,
    is_infra_failure,
    load_rows,
    milestone,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--sft-label", default="sft")
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument(
        "--include-infra",
        action="store_true",
        help="Include known infrastructure/API failures instead of filtering them out.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-task matched comparison rows.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format.",
    )
    return parser.parse_args()


def latest_by_task(path: Path, *, include_infra: bool) -> tuple[dict[str, dict[str, Any]], int, int]:
    rows = load_rows(path)
    infra = [row for row in rows if is_infra_failure(row)]
    scored = rows if include_infra else [row for row in rows if not is_infra_failure(row)]
    latest: dict[str, dict[str, Any]] = {}
    for row in scored:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        key = (attempt_index(row), row.get("finished_at") or "")
        old = latest.get(task_id)
        old_key = (attempt_index(old), old.get("finished_at") or "") if old else (-1, "")
        if key >= old_key:
            latest[task_id] = row
    return latest, len(rows), len(infra)


def is_terminal(row: dict[str, Any], *, max_attempts: int) -> bool:
    return row.get("status") == "PASSED" or attempt_index(row) >= max_attempts


def summarize_subset(
    label: str,
    rows: list[dict[str, Any]],
    *,
    max_attempts: int,
) -> dict[str, Any]:
    n = len(rows)
    passed = sum(row.get("status") == "PASSED" for row in rows)
    m67 = sum(milestone(row) in {6, 7} for row in rows)
    terminal = sum(is_terminal(row, max_attempts=max_attempts) for row in rows)
    attempts = [attempt_index(row) for row in rows]
    return {
        "label": label,
        "tasks": n,
        "terminal_tasks": terminal,
        "nonterminal_tasks": n - terminal,
        "passed": passed,
        "pass_rate": passed / n if n else None,
        "milestone_6_or_7": m67,
        "milestone_6_or_7_rate": m67 / n if n else None,
        "status_counts": dict(Counter(str(row.get("status") or "UNKNOWN") for row in rows)),
        "milestone_counts": dict(sorted(Counter(str(milestone(row)) for row in rows).items())),
        "attempts": {
            "mean": sum(attempts) / len(attempts) if attempts else None,
            "max": max(attempts) if attempts else None,
        },
    }


def matched_report(
    name: str,
    task_ids: list[str],
    base_latest: dict[str, dict[str, Any]],
    sft_latest: dict[str, dict[str, Any]],
    *,
    base_label: str,
    sft_label: str,
    max_attempts: int,
    include_details: bool,
) -> dict[str, Any]:
    base_rows = [base_latest[task_id] for task_id in task_ids]
    sft_rows = [sft_latest[task_id] for task_id in task_ids]
    report: dict[str, Any] = {
        "name": name,
        "tasks": len(task_ids),
        "task_ids": task_ids,
        base_label: summarize_subset(base_label, base_rows, max_attempts=max_attempts),
        sft_label: summarize_subset(sft_label, sft_rows, max_attempts=max_attempts),
    }
    base_pass = report[base_label]["pass_rate"]
    sft_pass = report[sft_label]["pass_rate"]
    if base_pass is not None and sft_pass is not None:
        report["pass_rate_delta_sft_minus_base"] = sft_pass - base_pass
    if include_details:
        report["details"] = [
            {
                "task_id": task_id,
                base_label: {
                    "attempt": attempt_index(base_latest[task_id]),
                    "status": base_latest[task_id].get("status"),
                    "milestone": milestone(base_latest[task_id]),
                    "terminal": is_terminal(base_latest[task_id], max_attempts=max_attempts),
                },
                sft_label: {
                    "attempt": attempt_index(sft_latest[task_id]),
                    "status": sft_latest[task_id].get("status"),
                    "milestone": milestone(sft_latest[task_id]),
                    "terminal": is_terminal(sft_latest[task_id], max_attempts=max_attempts),
                },
            }
            for task_id in task_ids
        ]
    return report


def fmt_rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def main() -> int:
    args = parse_args()
    base_latest, base_rows, base_infra = latest_by_task(args.base, include_infra=args.include_infra)
    sft_latest, sft_rows, sft_infra = latest_by_task(args.sft, include_infra=args.include_infra)

    common = sorted(set(base_latest) & set(sft_latest))
    common_terminal_both = [
        task_id
        for task_id in common
        if is_terminal(base_latest[task_id], max_attempts=args.max_attempts)
        and is_terminal(sft_latest[task_id], max_attempts=args.max_attempts)
    ]
    sft_terminal_common = [
        task_id
        for task_id in common
        if is_terminal(sft_latest[task_id], max_attempts=args.max_attempts)
    ]

    report = {
        "inputs": {
            args.base_label: {
                "path": str(args.base),
                "rows": base_rows,
                "infra_rows_detected": base_infra,
            },
            args.sft_label: {
                "path": str(args.sft),
                "rows": sft_rows,
                "infra_rows_detected": sft_infra,
            },
            "include_infra": args.include_infra,
            "max_attempts": args.max_attempts,
            "max_tasks": args.max_tasks,
        },
        "individual": {
            args.base_label: summarize(args.base, include_infra=args.include_infra),
            args.sft_label: summarize(args.sft, include_infra=args.include_infra),
        },
        "progress": {
            args.base_label: {
                "latest_tasks": len(base_latest),
                "terminal_tasks": sum(is_terminal(row, max_attempts=args.max_attempts) for row in base_latest.values()),
                "estimated_remaining_latest_tasks": max(args.max_tasks - len(base_latest), 0),
            },
            args.sft_label: {
                "latest_tasks": len(sft_latest),
                "terminal_tasks": sum(is_terminal(row, max_attempts=args.max_attempts) for row in sft_latest.values()),
                "estimated_remaining_latest_tasks": max(args.max_tasks - len(sft_latest), 0),
            },
        },
        "matched": {
            "all_seen_common_tasks": matched_report(
                "all_seen_common_tasks",
                common,
                base_latest,
                sft_latest,
                base_label=args.base_label,
                sft_label=args.sft_label,
                max_attempts=args.max_attempts,
                include_details=args.details,
            ),
            "sft_terminal_common_tasks": matched_report(
                "sft_terminal_common_tasks",
                sft_terminal_common,
                base_latest,
                sft_latest,
                base_label=args.base_label,
                sft_label=args.sft_label,
                max_attempts=args.max_attempts,
                include_details=args.details,
            ),
            "both_terminal_common_tasks": matched_report(
                "both_terminal_common_tasks",
                common_terminal_both,
                base_latest,
                sft_latest,
                base_label=args.base_label,
                sft_label=args.sft_label,
                max_attempts=args.max_attempts,
                include_details=args.details,
            ),
        },
    }
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        base_ind = report["individual"][args.base_label]
        sft_ind = report["individual"][args.sft_label]
        matched = report["matched"]["all_seen_common_tasks"]
        base_matched = matched[args.base_label]
        sft_matched = matched[args.sft_label]
        both = report["matched"]["both_terminal_common_tasks"]
        base_both = both[args.base_label]
        sft_both = both[args.sft_label]
        print("CyberGym strategy-generator eval comparison")
        print()
        print("Individual latest-task summaries, infra/API failures excluded:")
        print(f"- {args.base_label}: {base_ind['passed']}/{base_ind['tasks']} pass ({fmt_rate(base_ind['pass_rate'])})")
        print(f"- {args.sft_label}: {sft_ind['passed']}/{sft_ind['tasks']} pass ({fmt_rate(sft_ind['pass_rate'])})")
        print()
        print("Matched latest-task comparison:")
        print(
            f"- {args.base_label}: {base_matched['passed']}/{base_matched['tasks']} "
            f"pass ({fmt_rate(base_matched['pass_rate'])})"
        )
        print(
            f"- {args.sft_label}: {sft_matched['passed']}/{sft_matched['tasks']} "
            f"pass ({fmt_rate(sft_matched['pass_rate'])})"
        )
        print(f"- delta SFT-base: {fmt_rate(matched.get('pass_rate_delta_sft_minus_base'))}")
        print()
        print("Matched both-terminal diagnostic subset:")
        print(
            f"- {args.base_label}: {base_both['passed']}/{base_both['tasks']} "
            f"pass ({fmt_rate(base_both['pass_rate'])})"
        )
        print(
            f"- {args.sft_label}: {sft_both['passed']}/{sft_both['tasks']} "
            f"pass ({fmt_rate(sft_both['pass_rate'])})"
        )
        print(f"- delta SFT-base: {fmt_rate(both.get('pass_rate_delta_sft_minus_base'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
