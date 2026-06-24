#!/usr/bin/env python3
"""Summarize Mastermind rollout JSONL records."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from mastermind.rollout import read_rollouts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("runs/teacher_corpus/trajectories.jsonl"))
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--model")
    parser.add_argument("--milestone", type=int)
    parser.add_argument("--min-cost", type=float)
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    records = read_rollouts(args.input_path or args.path)
    if args.task_id:
        records = [record for record in records if record.task_id == args.task_id]
    if args.model:
        records = [record for record in records if record.model == args.model]
    if args.milestone is not None:
        records = [record for record in records if record.milestone.milestone == args.milestone]
    if args.min_cost is not None:
        records = [record for record in records if record.cost is not None and record.cost >= args.min_cost]
    if args.max_cost is not None:
        records = [record for record in records if record.cost is not None and record.cost <= args.max_cost]

    by_status = Counter(record.status for record in records)
    by_milestone = Counter(record.milestone.milestone for record in records)
    by_model = Counter(record.model for record in records)
    print(f"records: {len(records)}")
    print("status:", dict(sorted(by_status.items())))
    print("milestone:", dict(sorted(by_milestone.items(), key=lambda item: (-1 if item[0] is None else item[0]))))
    print("model:", dict(sorted(by_model.items())))
    for record in records[: args.limit]:
        print(
            f"{record.run_id}\t{record.task_id}\t{record.model}\t"
            f"m={record.milestone.milestone}\tcost={record.cost}\t{record.status}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
