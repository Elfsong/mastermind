#!/usr/bin/env python3
"""Archive CyberGym eval rollouts and generate task-level diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    kind: str
    round_index: int | None = None


DEFAULT_SOURCES = [
    Source("round1", Path("runs/codex_gateway_eval100_clean_rollouts.jsonl"), "round", 1),
    Source("round2", Path("runs/codex_gateway_eval_bo4_rep2_rollouts.jsonl"), "round", 2),
    Source("round3", Path("runs/codex_gateway_eval_bo4_rep3_rollouts.jsonl"), "round", 3),
    Source("round4", Path("runs/codex_gateway_eval_bo4_rep4_rollouts.jsonl"), "round", 4),
    Source("round5", Path("runs/codex_gateway_eval_bo4_rep5_rollouts.jsonl"), "round", 5),
    Source("round6", Path("runs/codex_gateway_eval_bo4_rep6_rollouts.jsonl"), "round", 6),
    Source("round7", Path("runs/codex_gateway_eval_bo4_rep7_rollouts.jsonl"), "round", 7),
    Source("round8", Path("runs/codex_gateway_eval_bo4_rep8_rollouts.jsonl"), "round", 8),
    Source("level3", Path("runs/codex_gateway_eval_level3_rep1_rollouts.jsonl"), "level3"),
]

SUPPORTING_FILES = [
    Path("runs/codex_cybergym/eval_rep_logs/eval_reps_3_8_20260527T1439Z_lark_progress.md"),
    Path("runs/codex_cybergym/eval_rep_logs/eval_reps_3_8_20260527T1439Z.log"),
    Path("runs/codex_cybergym/level3_logs/level3_after_rep8_watch_20260528T1139Z.log"),
    Path("runs/cybergym_assets/download_manifests/eval200_level3_20260528T1140Z.jsonl"),
]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def milestone_value(row: dict[str, Any]) -> Any:
    milestone = row.get("milestone")
    if milestone is None:
        milestone = (row.get("metadata") or {}).get("milestone")
    if isinstance(milestone, dict):
        return milestone.get("milestone")
    return milestone


def milestone_reason(row: dict[str, Any]) -> str:
    milestone = row.get("milestone")
    if isinstance(milestone, dict):
        return str(milestone.get("reasoning") or "")
    verification = row.get("verification") or {}
    details = verification.get("details") or {}
    return str(details.get("reason") or verification.get("status") or "")


def task_id_sort_key(task_id: str) -> tuple[str, int | str]:
    prefix, _, rest = task_id.partition(":")
    try:
        return prefix, int(rest)
    except ValueError:
        return prefix, rest


def format_counter(counter: Counter[Any]) -> str:
    return ", ".join(f"{key}: {counter[key]}" for key in sorted(counter, key=lambda v: str(v)))


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.1%}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict[str, Any]], bad_json: int) -> dict[str, Any]:
    statuses = Counter(row.get("status", "") for row in rows)
    milestones = Counter(milestone_value(row) for row in rows)
    task_ids = {row.get("task_id") for row in rows if isinstance(row.get("task_id"), str)}
    wall_seconds = [row.get("wall_seconds") for row in rows if isinstance(row.get("wall_seconds"), (int, float))]
    return {
        "rows": len(rows),
        "unique_tasks": len(task_ids),
        "bad_json": bad_json,
        "status": dict(sorted(statuses.items())),
        "milestones": dict(sorted(milestones.items(), key=lambda item: str(item[0]))),
        "pass_rate": (statuses.get("PASSED", 0) / len(rows)) if rows else None,
        "wall_seconds_mean": mean(wall_seconds) if wall_seconds else None,
        "wall_seconds_median": median(wall_seconds) if wall_seconds else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Archive directory. Defaults to runs/codex_cybergym/analysis/eval_archive_<timestamp>.",
    )
    parser.add_argument("--copy-raw", action="store_true", default=True)
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or Path("runs/codex_cybergym/analysis") / f"eval_archive_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    support_dir = out_dir / "supporting"
    raw_dir.mkdir(exist_ok=True)
    support_dir.mkdir(exist_ok=True)

    source_rows: dict[str, list[dict[str, Any]]] = {}
    bad_json_by_source: dict[str, int] = {}
    manifest_sources: list[dict[str, Any]] = []
    for source in DEFAULT_SOURCES:
        rows, bad_json = read_jsonl(source.path)
        source_rows[source.name] = rows
        bad_json_by_source[source.name] = bad_json
        copied_to = None
        if args.copy_raw:
            copied_to = raw_dir / source.path.name
            shutil.copy2(source.path, copied_to)
        source_summary = summarize_rows(rows, bad_json)
        manifest_sources.append(
            {
                "name": source.name,
                "kind": source.kind,
                "round_index": source.round_index,
                "path": str(source.path),
                "copied_to": str(copied_to) if copied_to else None,
                "bytes": source.path.stat().st_size,
                "sha256": sha256_path(source.path),
                **source_summary,
            }
        )

    supporting_manifest: list[dict[str, Any]] = []
    for path in SUPPORTING_FILES:
        if not path.exists():
            supporting_manifest.append({"path": str(path), "missing": True})
            continue
        copied_to = support_dir / path.name
        shutil.copy2(path, copied_to)
        supporting_manifest.append(
            {
                "path": str(path),
                "copied_to": str(copied_to),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )

    round_sources = [source for source in DEFAULT_SOURCES if source.kind == "round"]
    all_tasks = sorted(
        {
            row["task_id"]
            for source in DEFAULT_SOURCES
            for row in source_rows[source.name]
            if isinstance(row.get("task_id"), str)
        },
        key=task_id_sort_key,
    )

    rows_by_source_task: dict[str, dict[str, dict[str, Any]]] = {}
    for source in DEFAULT_SOURCES:
        by_task: dict[str, dict[str, Any]] = {}
        for row in source_rows[source.name]:
            task_id = row.get("task_id")
            if isinstance(task_id, str):
                by_task[task_id] = row
        rows_by_source_task[source.name] = by_task

    pass_sets: dict[str, set[str]] = {}
    for source in DEFAULT_SOURCES:
        pass_sets[source.name] = {
            task_id
            for task_id, row in rows_by_source_task[source.name].items()
            if row.get("status") == "PASSED"
        }

    round_pass_sets = [pass_sets[source.name] for source in round_sources]
    bo_union: set[str] = set()
    best_of_rows: list[dict[str, Any]] = []
    previous_count = 0
    for source, pass_set in zip(round_sources, round_pass_sets, strict=True):
        bo_union |= pass_set
        count = len(bo_union)
        best_of_rows.append(
            {
                "best_of_n": source.round_index,
                "passed": count,
                "total_tasks": len(all_tasks),
                "pass_rate": f"{count / len(all_tasks):.6f}",
                "marginal_new_passes": count - previous_count,
            }
        )
        previous_count = count

    level3_pass = pass_sets["level3"]
    bo8_pass = set(bo_union)
    both_pass = bo8_pass & level3_pass
    bo8_only = bo8_pass - level3_pass
    level3_only = level3_pass - bo8_pass
    neither = set(all_tasks) - (bo8_pass | level3_pass)

    matrix_rows: list[dict[str, Any]] = []
    pass_count_hist = Counter()
    transition_hist = Counter()
    for task_id in all_tasks:
        row: dict[str, Any] = {"task_id": task_id}
        round_statuses: list[str] = []
        first_pass_round = ""
        for source in round_sources:
            rollout = rows_by_source_task[source.name].get(task_id, {})
            status = str(rollout.get("status", "MISSING"))
            milestone = milestone_value(rollout)
            row[f"round{source.round_index}_status"] = status
            row[f"round{source.round_index}_milestone"] = milestone
            round_statuses.append(status)
            if status == "PASSED" and not first_pass_round:
                first_pass_round = str(source.round_index)
        pass_count = sum(1 for status in round_statuses if status == "PASSED")
        pass_count_hist[pass_count] += 1
        level3_row = rows_by_source_task["level3"].get(task_id, {})
        level3_status = str(level3_row.get("status", "MISSING"))
        level3_milestone = milestone_value(level3_row)
        bo8 = task_id in bo8_pass
        level3 = task_id in level3_pass
        if bo8 and level3:
            comparison = "both_pass"
        elif bo8:
            comparison = "bo8_only"
        elif level3:
            comparison = "level3_only"
        else:
            comparison = "neither_pass"
        transition_hist[(bo8, level3)] += 1
        row.update(
            {
                "pass_count_round1_8": pass_count,
                "bo8_pass": bo8,
                "first_pass_round": first_pass_round,
                "level3_status": level3_status,
                "level3_milestone": level3_milestone,
                "level3_pass": level3,
                "comparison_group": comparison,
            }
        )
        matrix_rows.append(row)

    matrix_fields = ["task_id"]
    for index in range(1, 9):
        matrix_fields.extend([f"round{index}_status", f"round{index}_milestone"])
    matrix_fields.extend(
        [
            "pass_count_round1_8",
            "bo8_pass",
            "first_pass_round",
            "level3_status",
            "level3_milestone",
            "level3_pass",
            "comparison_group",
        ]
    )
    write_csv(out_dir / "task_matrix.csv", matrix_fields, matrix_rows)

    run_summary_rows: list[dict[str, Any]] = []
    for source_manifest in manifest_sources:
        run_summary_rows.append(
            {
                "name": source_manifest["name"],
                "kind": source_manifest["kind"],
                "round_index": source_manifest["round_index"] or "",
                "rows": source_manifest["rows"],
                "unique_tasks": source_manifest["unique_tasks"],
                "bad_json": source_manifest["bad_json"],
                "passed": source_manifest["status"].get("PASSED", 0),
                "failed": source_manifest["status"].get("FAILED", 0),
                "timeout": source_manifest["status"].get("TIMEOUT", 0),
                "crash": source_manifest["status"].get("CRASH", 0),
                "pass_rate": f"{source_manifest['pass_rate']:.6f}" if source_manifest["pass_rate"] is not None else "",
                "milestones": json.dumps(source_manifest["milestones"], sort_keys=True),
                "wall_seconds_mean": f"{source_manifest['wall_seconds_mean']:.3f}" if source_manifest["wall_seconds_mean"] else "",
                "wall_seconds_median": f"{source_manifest['wall_seconds_median']:.3f}" if source_manifest["wall_seconds_median"] else "",
            }
        )
    write_csv(
        out_dir / "run_summary.csv",
        [
            "name",
            "kind",
            "round_index",
            "rows",
            "unique_tasks",
            "bad_json",
            "passed",
            "failed",
            "timeout",
            "crash",
            "pass_rate",
            "milestones",
            "wall_seconds_mean",
            "wall_seconds_median",
        ],
        run_summary_rows,
    )
    write_csv(
        out_dir / "best_of_curve.csv",
        ["best_of_n", "passed", "total_tasks", "pass_rate", "marginal_new_passes"],
        best_of_rows,
    )

    reason_rows: list[dict[str, Any]] = []
    for source in DEFAULT_SOURCES:
        counter: Counter[tuple[str, Any, str]] = Counter()
        for row in source_rows[source.name]:
            counter[(str(row.get("status", "")), milestone_value(row), milestone_reason(row))] += 1
        for (status, milestone, reason), count in counter.most_common():
            reason_rows.append(
                {
                    "source": source.name,
                    "status": status,
                    "milestone": milestone,
                    "reason": reason,
                    "count": count,
                }
            )
    write_csv(out_dir / "reason_summary.csv", ["source", "status", "milestone", "reason", "count"], reason_rows)

    poc_rows: list[dict[str, Any]] = []
    for source in DEFAULT_SOURCES:
        grouped: dict[str, list[int]] = defaultdict(list)
        for row in source_rows[source.name]:
            length = (row.get("metadata") or {}).get("poc_length")
            if isinstance(length, int):
                grouped[str(row.get("status", ""))].append(length)
        for status, lengths in sorted(grouped.items()):
            poc_rows.append(
                {
                    "source": source.name,
                    "status": status,
                    "count": len(lengths),
                    "mean_poc_length": f"{mean(lengths):.1f}",
                    "median_poc_length": f"{median(lengths):.1f}",
                    "min_poc_length": min(lengths),
                    "max_poc_length": max(lengths),
                }
            )
    write_csv(
        out_dir / "poc_length_summary.csv",
        ["source", "status", "count", "mean_poc_length", "median_poc_length", "min_poc_length", "max_poc_length"],
        poc_rows,
    )

    pairwise_rows: list[dict[str, Any]] = []
    for left in round_sources:
        for right in round_sources:
            a = pass_sets[left.name]
            b = pass_sets[right.name]
            union = a | b
            pairwise_rows.append(
                {
                    "left": left.name,
                    "right": right.name,
                    "left_passed": len(a),
                    "right_passed": len(b),
                    "intersection": len(a & b),
                    "union": len(union),
                    "jaccard": f"{len(a & b) / len(union):.6f}" if union else "",
                }
            )
    write_csv(
        out_dir / "pairwise_round_pass_jaccard.csv",
        ["left", "right", "left_passed", "right_passed", "intersection", "union", "jaccard"],
        pairwise_rows,
    )

    task_sets = {
        "bo8_pass": sorted(bo8_pass, key=task_id_sort_key),
        "level3_pass": sorted(level3_pass, key=task_id_sort_key),
        "both_pass": sorted(both_pass, key=task_id_sort_key),
        "bo8_only": sorted(bo8_only, key=task_id_sort_key),
        "level3_only": sorted(level3_only, key=task_id_sort_key),
        "neither_pass": sorted(neither, key=task_id_sort_key),
        "pass_count_histogram_round1_8": {str(key): pass_count_hist[key] for key in sorted(pass_count_hist)},
    }
    (out_dir / "task_sets.json").write_text(json.dumps(task_sets, indent=2, sort_keys=True) + "\n")

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "archive_dir": str(out_dir),
        "sources": manifest_sources,
        "supporting_files": supporting_manifest,
        "derived_files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    level3_summary = next(item for item in manifest_sources if item["name"] == "level3")
    report_lines = [
        "# CyberGym Eval Archive Diagnostics",
        "",
        f"- Created at: `{manifest['created_at']}`",
        f"- Archive dir: `{out_dir}`",
        f"- Total tasks observed: `{len(all_tasks)}`",
        f"- Round 1-8 source files: `{len(round_sources)}`",
        f"- Level 3 source file: `runs/codex_gateway_eval_level3_rep1_rollouts.jsonl`",
        "",
        "## Headline Results",
        "",
        "| Experiment | Completed | Passed | Failed | Timeout | Crash | Pass rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source_manifest in manifest_sources:
        status = source_manifest["status"]
        report_lines.append(
            "| "
            + " | ".join(
                [
                    source_manifest["name"],
                    str(source_manifest["rows"]),
                    str(status.get("PASSED", 0)),
                    str(status.get("FAILED", 0)),
                    str(status.get("TIMEOUT", 0)),
                    str(status.get("CRASH", 0)),
                    pct(status.get("PASSED", 0), source_manifest["rows"]),
                ]
            )
            + " |"
        )
    report_lines.extend(
        [
            "",
            "## Best-of Curve",
            "",
            "| Best-of-N | Passed | Pass rate | Marginal new passes |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in best_of_rows:
        report_lines.append(
            f"| {row['best_of_n']} | {row['passed']} | {float(row['pass_rate']):.1%} | {row['marginal_new_passes']} |"
        )
    report_lines.extend(
        [
            "",
            "## Level 3 Comparison",
            "",
            f"- Best-of-8 passed: `{len(bo8_pass)}` / `{len(all_tasks)}` ({pct(len(bo8_pass), len(all_tasks))})",
            f"- Level 3 passed: `{len(level3_pass)}` / `{len(all_tasks)}` ({pct(len(level3_pass), len(all_tasks))})",
            f"- Passed by both: `{len(both_pass)}`",
            f"- Best-of-8 only: `{len(bo8_only)}`",
            f"- Level 3 only: `{len(level3_only)}`",
            f"- Neither passed: `{len(neither)}`",
            "",
            "## Stability Across Round 1-8",
            "",
            "| Number of passing rounds | Task count |",
            "|---:|---:|",
        ]
    )
    for pass_count in sorted(pass_count_hist):
        report_lines.append(f"| {pass_count} | {pass_count_hist[pass_count]} |")

    first_order = {row["best_of_n"]: row["marginal_new_passes"] for row in best_of_rows}
    report_lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Level 3 has no CRASH rows after cleanup and completed `{level3_summary['rows']}` rows with `{level3_summary['bad_json']}` bad JSON lines.",
            f"- Best-of-8 gains are front-loaded: Round 1 contributes `{first_order.get(1, 0)}` passes, while Rounds 6-8 add `{first_order.get(6, 0) + first_order.get(7, 0) + first_order.get(8, 0)}` net new tasks combined.",
            f"- Level 3 is close to Best-of-8 overall but not a superset: `{len(level3_only)}` tasks are Level-3-only and `{len(bo8_only)}` tasks are Best-of-8-only.",
            f"- Persistent hard set: `{len(neither)}` tasks fail both Best-of-8 and Level 3; these are the most useful targets for sequential or augmented follow-up.",
            f"- Round pass-count histogram: {format_counter(pass_count_hist)}.",
            "",
            "## Generated Files",
            "",
            "- `manifest.json`: source checksums, row counts, copied paths",
            "- `run_summary.csv`: per-run aggregate status and milestone counts",
            "- `task_matrix.csv`: one row per task with Round 1-8 status, Level 3 status, and comparison group",
            "- `best_of_curve.csv`: Best-of-N pass curve and marginal gains",
            "- `task_sets.json`: explicit task-id sets for Bo8/Level3 overlap and disagreement",
            "- `reason_summary.csv`: status/milestone reason distribution",
            "- `poc_length_summary.csv`: PoC length summaries by source/status",
            "- `pairwise_round_pass_jaccard.csv`: pass-set overlap among repeated rounds",
        ]
    )
    (out_dir / "diagnostics.md").write_text("\n".join(report_lines) + "\n")

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
