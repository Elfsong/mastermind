#!/usr/bin/env python3
"""Stage Codex GPT-5.5 CyberGym training artifacts for Hugging Face.

This builds a single training dataset from:
- the first-100 train sequential self-improvement run;
- the remaining-201 train sequential self-improvement run;
- the escalated infra rerun used to replace missing-data failures;
- the 5-task API/429 rerun, when present.

The staged dataset includes canonical rollout JSONL files, a best-by-task view,
milestone-6/7 subsets, one-step strategy-generator examples, and raw trajectory
/ feedback / strategy-update files referenced by those records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
STAGING_ROOT = RUNS / "hf_upload_staging"
DATASET_NAME = "cybergym-codex-gpt-5-5-iterative-improvement-train"
DEFAULT_REPO_ID = f"Elfsong/{DATASET_NAME}"


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    path: Path
    run_id: str
    priority: int
    task_ids: tuple[str, ...] | None = None


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_task_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def load_jsonl(path: Path, run_id: str | None = None) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_json = 0
    if not path.exists():
        return rows, 0
    with path.open(errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if run_id is not None and row.get("run_id") != run_id:
                continue
            rows.append(row)
    return rows, bad_json


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_text(path_value: Any, max_chars: int | None = None) -> str:
    if not isinstance(path_value, str) or not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() or not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    if max_chars is not None and len(text) > max_chars:
        half = max_chars // 2
        return text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]
    return text


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


def milestone_value(row: dict[str, Any]) -> int | None:
    milestone = row.get("milestone")
    if isinstance(milestone, dict):
        value = milestone.get("milestone")
    else:
        value = milestone
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def latest_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str):
            grouped[task_id].append(row)
    latest: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in grouped.items():
        task_rows.sort(key=lambda row: (attempt_index(row), row.get("finished_at") or "", row.get("run_id") or ""))
        latest[task_id] = task_rows[-1]
    return latest


def best_row_score(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    """Choose useful replacement/best rows: pass > high milestone > priority > later attempt."""
    status = str(row.get("status") or "")
    status_rank = {"PASSED": 5, "FAILED": 3, "TIMEOUT": 2, "AGENT_ERROR": 1, "CRASH": 1}.get(status, 0)
    milestone = milestone_value(row) or 0
    meta = row.get("metadata") or {}
    td = meta.get("training_dataset") or {}
    priority = int(td.get("source_priority") or 0)
    return (status_rank, milestone, priority, attempt_index(row), str(row.get("finished_at") or ""))


def best_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str):
            grouped[task_id].append(row)
    return {task_id: max(task_rows, key=best_row_score) for task_id, task_rows in grouped.items()}


def copy_artifact(src_value: Any, dataset_dir: Path, rel_dir: str, prefix: str = "") -> dict[str, Any]:
    if not isinstance(src_value, str) or not src_value:
        return {"found": False, "error": "missing path"}
    src = Path(src_value)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists() or not src.is_file():
        return {"found": False, "source_path": str(src), "error": "file does not exist"}
    rel = Path(rel_dir) / f"{prefix}{src.name}"
    dst = dataset_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return {
        "found": True,
        "path": str(rel),
        "source_path": relative_to_root(src),
        "bytes": src.stat().st_size,
        "sha256": sha256_file(src),
    }


def companion_paths(trajectory_path: str) -> list[Path]:
    src = Path(trajectory_path)
    if not src.is_absolute():
        src = ROOT / src
    out: list[Path] = []
    if src.name.endswith(".codex.jsonl"):
        stem = src.name[: -len(".codex.jsonl")]
        for suffix in (".codex.last.txt", ".codex.stderr.log"):
            p = src.with_name(stem + suffix)
            if p.exists():
                out.append(p)
    return out


def annotate_row(row: dict[str, Any], source: Source, task_order: dict[str, int]) -> dict[str, Any]:
    annotated = json.loads(json.dumps(row))
    task_id = str(annotated.get("task_id") or "")
    metadata = dict(annotated.get("metadata") or {})
    td = dict(metadata.get("training_dataset") or {})
    task_index = task_order.get(task_id)
    td.update(
        {
            "dataset_name": DATASET_NAME,
            "source_name": source.name,
            "source_kind": source.kind,
            "source_run_id": source.run_id,
            "source_output": relative_to_root(source.path),
            "source_priority": source.priority,
            "task_index": task_index,
            "task_ordinal": None if task_index is None else task_index + 1,
            "task_total": len(task_order),
            "attempt_index": attempt_index(annotated),
            "milestone": milestone_value(annotated),
        }
    )
    metadata["training_dataset"] = td
    annotated["metadata"] = metadata
    annotated["training_dataset"] = {
        "dataset_name": DATASET_NAME,
        "source_name": source.name,
        "source_kind": source.kind,
        "source_priority": source.priority,
    }
    return annotated


def stage_rows(
    dataset_dir: Path,
    sources: list[Source],
    task_order: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    copied_manifests: list[dict[str, Any]] = []
    trajectory_cache: dict[str, dict[str, Any]] = {}
    artifact_cache: dict[str, dict[str, Any]] = {}

    for source in sources:
        rows, bad_json = load_jsonl(source.path, source.run_id)
        if source.task_ids is not None:
            allowed = set(source.task_ids)
            rows = [row for row in rows if row.get("task_id") in allowed]
        annotated = [annotate_row(row, source, task_order) for row in rows]
        all_rows.extend(annotated)

        write_jsonl(dataset_dir / "data/source_rollouts" / f"{source.name}.jsonl", annotated)

        for row in annotated:
            run_id = safe_name(str(row.get("run_id") or source.run_id))
            trajectory_value = row.get("trajectory_path")
            if isinstance(trajectory_value, str) and trajectory_value:
                src = str((ROOT / trajectory_value).resolve() if not Path(trajectory_value).is_absolute() else Path(trajectory_value).resolve())
                if src not in trajectory_cache:
                    manifest = copy_artifact(src, dataset_dir, f"trajectories/raw/{run_id}")
                    trajectory_cache[src] = manifest
                    if manifest.get("found"):
                        copied_manifests.append({"kind": "trajectory", **manifest})
                        for companion in companion_paths(src):
                            comp = copy_artifact(str(companion), dataset_dir, f"trajectories/raw/{run_id}")
                            copied_manifests.append({"kind": "trajectory_companion", **comp})
                row.setdefault("metadata", {}).setdefault("training_dataset", {})["staged_trajectory"] = trajectory_cache[src]

            sequential = (row.get("metadata") or {}).get("sequential") or {}
            for kind, key, rel_dir in (
                ("feedback", "feedback_path", "feedback"),
                ("strategy_update", "experience_update.last_message", "strategy_updates"),
            ):
                if key == "experience_update.last_message":
                    value = (sequential.get("experience_update") or {}).get("last_message")
                else:
                    value = sequential.get(key)
                if not isinstance(value, str) or not value:
                    continue
                src = str((ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve())
                if src not in artifact_cache:
                    artifact_cache[src] = copy_artifact(src, dataset_dir, f"{rel_dir}/{run_id}")
                    copied_manifests.append({"kind": kind, **artifact_cache[src]})
                row.setdefault("metadata", {}).setdefault("training_dataset", {})[f"staged_{kind}"] = artifact_cache[src]

        latest = latest_by_task(annotated)
        source_summaries.append(
            {
                "name": source.name,
                "kind": source.kind,
                "run_id": source.run_id,
                "path": relative_to_root(source.path),
                "bad_json": bad_json,
                "rows": len(annotated),
                "unique_tasks": len(latest),
                "status_counts_latest": dict(Counter(str(row.get("status") or "UNKNOWN") for row in latest.values())),
                "milestone_counts_rows": dict(Counter(str(milestone_value(row)) for row in annotated)),
                "sha256": sha256_file(source.path) if source.path.exists() else None,
                "bytes": source.path.stat().st_size if source.path.exists() else 0,
            }
        )

    all_rows.sort(
        key=lambda row: (
            (row.get("metadata") or {}).get("training_dataset", {}).get("task_index") or 10**9,
            (row.get("metadata") or {}).get("training_dataset", {}).get("source_priority") or 0,
            attempt_index(row),
            row.get("finished_at") or "",
        )
    )
    return all_rows, source_summaries, copied_manifests


def strategy_update_text(row: dict[str, Any]) -> tuple[str, str, str]:
    sequential = (row.get("metadata") or {}).get("sequential") or {}
    update = sequential.get("experience_update") or {}
    status = str(update.get("update_status") or "")
    path = str(update.get("last_message") or "")
    return read_text(path), status, path


def build_strategy_examples(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = row.get("task_id")
        run_id = row.get("run_id")
        if isinstance(task_id, str) and isinstance(run_id, str):
            grouped[(run_id, task_id)].append(row)

    examples: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    for (run_id, task_id), task_rows in sorted(grouped.items()):
        by_attempt: dict[int, dict[str, Any]] = {}
        for row in task_rows:
            by_attempt[attempt_index(row)] = row
        previous_strategy_by_attempt: dict[int, str] = {1: ""}
        for attempt in sorted(by_attempt):
            row = by_attempt[attempt]
            text, update_status, _ = strategy_update_text(row)
            if text:
                previous_strategy_by_attempt[attempt + 1] = text

        for attempt in sorted(by_attempt):
            if attempt + 1 not in by_attempt:
                continue
            current = by_attempt[attempt]
            nxt = by_attempt[attempt + 1]
            target_strategy, update_status, update_path = strategy_update_text(current)
            if not target_strategy.strip():
                continue
            sequential = (current.get("metadata") or {}).get("sequential") or {}
            feedback_path = sequential.get("feedback_path")
            previous_strategy = previous_strategy_by_attempt.get(attempt, "")
            usable_for_sft = update_status in {"codex_updated", "tinker_updated"} and bool(target_strategy.strip())
            example = {
                "example_id": f"{safe_name(run_id)}__{safe_name(task_id)}__after_attempt_{attempt}",
                "task_id": task_id,
                "run_id": run_id,
                "source_name": ((current.get("metadata") or {}).get("training_dataset") or {}).get("source_name"),
                "input_attempt_index": attempt,
                "target_attempt_index": attempt + 1,
                "input": {
                    "previous_strategy": previous_strategy,
                    "attempt_rollout": {
                        "task_id": current.get("task_id"),
                        "run_id": current.get("run_id"),
                        "agent": current.get("agent"),
                        "model": current.get("model"),
                        "status": current.get("status"),
                        "milestone": milestone_value(current),
                        "milestone_reason": (current.get("milestone") or {}).get("reasoning")
                        if isinstance(current.get("milestone"), dict)
                        else "",
                        "verification": current.get("verification"),
                        "metadata": {
                            "codex_returncode": (current.get("metadata") or {}).get("codex_returncode"),
                            "codex_timed_out": (current.get("metadata") or {}).get("codex_timed_out"),
                            "wall_seconds": current.get("wall_seconds"),
                            "poc_length": (current.get("metadata") or {}).get("poc_length"),
                            "submit_count": (current.get("verification") or {}).get("submit_count"),
                        },
                    },
                    "feedback": read_text(feedback_path, max_chars=20000),
                    "trajectory_path": (((current.get("metadata") or {}).get("training_dataset") or {}).get("staged_trajectory") or {}).get("path"),
                    "feedback_path": (((current.get("metadata") or {}).get("training_dataset") or {}).get("staged_feedback") or {}).get("path"),
                },
                "target_strategy": target_strategy,
                "target_strategy_update_status": update_status,
                "target_strategy_source_path": relative_to_root(Path(update_path)) if update_path else "",
                "next_attempt_outcome": {
                    "status": nxt.get("status"),
                    "milestone": milestone_value(nxt),
                    "milestone_reason": (nxt.get("milestone") or {}).get("reasoning") if isinstance(nxt.get("milestone"), dict) else "",
                    "passed": nxt.get("status") == "PASSED",
                },
                "quality": {
                    "usable_for_sft": usable_for_sft,
                    "next_milestone_6_or_7": milestone_value(nxt) in {6, 7},
                    "next_passed": nxt.get("status") == "PASSED",
                    "milestone_delta": (milestone_value(nxt) or 0) - (milestone_value(current) or 0),
                },
            }
            examples.append(example)
            if usable_for_sft:
                sft_rows.append(
                    {
                        "id": example["example_id"],
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a one-step CyberGym strategy generator. Given the previous "
                                    "strategy, complete attempt trajectory/result summary, and verifier feedback, "
                                    "write the updated strategy for the next attempt only."
                                ),
                            },
                            {
                                "role": "user",
                                "content": json.dumps(example["input"], ensure_ascii=False, sort_keys=True),
                            },
                            {"role": "assistant", "content": target_strategy},
                        ],
                        "metadata": {
                            "task_id": task_id,
                            "run_id": run_id,
                            "input_attempt_index": attempt,
                            "target_attempt_index": attempt + 1,
                            "next_status": nxt.get("status"),
                            "next_milestone": milestone_value(nxt),
                            "next_passed": nxt.get("status") == "PASSED",
                        },
                    }
                )
    return examples, sft_rows


def build_readme(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    return f"""---
license: other
task_categories:
- text-generation
language:
- en
pretty_name: CyberGym Codex GPT-5.5 Iterative Improvement Train
---

# CyberGym Codex GPT-5.5 Iterative Improvement Train

This private dataset contains Codex GPT-5.5 CyberGym training-split sequential
self-improvement artifacts. It is intended for training and evaluating a
one-step strategy generator: given the previous strategy plus the previous
attempt trajectory/result/feedback, generate the next strategy.

Generated at: `{summary['generated_at']}`

## Key counts

- Training tasks covered by main runs: **{overall['main_tasks']}**
- Main completed tasks at staging time: **{overall['main_completed_tasks']}**
- All rollout rows staged: **{overall['all_rollout_rows']}**
- Best-by-task rows: **{overall['best_by_task_rows']}**
- Milestone 6/7 rollout rows: **{overall['milestone_6_or_7_rows']}**
- Strategy-generator examples: **{overall['strategy_examples']}**
- SFT-ready strategy examples: **{overall['sft_examples']}**

## Files

- `data/all_rollouts.jsonl`: all annotated rollout rows from main and rerun sources.
- `data/best_by_task.jsonl`: one best row per task, preferring pass/high-milestone reruns.
- `data/latest_by_task.jsonl`: latest row per task among staged rows.
- `data/milestone_6_or_7_rollouts.jsonl`: rollout rows with milestone 6 or 7.
- `data/strategy_generator_examples.jsonl`: structured one-step generator examples.
- `data/strategy_generator_sft.jsonl`: chat-style SFT rows.
- `trajectories/raw/`: raw Codex JSONL trajectories plus companion stderr/last-message files.
- `feedback/`: per-attempt verifier/runner feedback files.
- `strategy_updates/`: per-attempt strategy update texts.
- `metadata/summary.json`: full staging summary.

## Notes

The main train run is supplemented with reruns for known infra failures:

- `infra_rerun_escalated`: replaces missing-data failures where available.
- `api429_rerun5`: reruns API/gateway 429 or stream-disconnect failures.

Keep task-level splits for evaluation; do not randomly split examples from the
same CyberGym task across train/eval.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, default=STAGING_ROOT)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--clean", action="store_true", help="Remove the staged dataset directory first.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.staging_root / args.dataset_name
    if args.clean and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_ids = read_task_ids(ROOT / "cybergym/TASKS_TRAIN")
    task_order = {task_id: index for index, task_id in enumerate(train_ids)}
    api429_latest = RUNS / "codex_cybergym/sequential_self_logs/api429_rerun5.latest"
    sources = [
        Source(
            "train100",
            "main",
            RUNS / "codex_gateway_train_seq_self_involving_train100_rollouts.jsonl",
            "codex-gateway-train-seq-self-100-20260531T1640Z",
            10,
            tuple(train_ids[:100]),
        ),
        Source(
            "remaining201",
            "main",
            RUNS / "codex_gateway_train_seq_self_involving_remaining201_rollouts.jsonl",
            "codex-gateway-train-seq-self-remaining201-20260601T0445Z",
            10,
            tuple(train_ids[100:301]),
        ),
        Source(
            "infra_rerun_escalated",
            "infra_rerun",
            RUNS / "codex_gateway_train_infra_rerun_escalated_rollouts.jsonl",
            "codex-gateway-train-infra-rerun-escalated-20260601T0856Z",
            20,
            None,
        ),
    ]
    if api429_latest.exists():
        api429_run_id = api429_latest.read_text().strip()
        sources.append(
            Source(
                "api429_rerun5",
                "api429_rerun",
                RUNS / "codex_gateway_train_api429_rerun5_rollouts.jsonl",
                api429_run_id,
                30,
                ("arvo:11945", "arvo:14560", "arvo:14574", "arvo:59070", "arvo:61617"),
            )
        )

    all_rows, source_summaries, artifact_manifest = stage_rows(dataset_dir, sources, task_order)
    latest_rows = list(latest_by_task(all_rows).values())
    best_rows = list(best_by_task(all_rows).values())
    m67_rows = [row for row in all_rows if milestone_value(row) in {6, 7}]
    strategy_examples, sft_rows = build_strategy_examples(all_rows)

    write_jsonl(dataset_dir / "data/all_rollouts.jsonl", all_rows)
    write_jsonl(dataset_dir / "data/latest_by_task.jsonl", latest_rows)
    write_jsonl(dataset_dir / "data/best_by_task.jsonl", best_rows)
    write_jsonl(dataset_dir / "data/milestone_6_or_7_rollouts.jsonl", m67_rows)
    write_jsonl(dataset_dir / "data/strategy_generator_examples.jsonl", strategy_examples)
    write_jsonl(dataset_dir / "data/strategy_generator_sft.jsonl", sft_rows)
    write_jsonl(dataset_dir / "metadata/artifact_manifest.jsonl", artifact_manifest)

    main_rows = [
        row
        for row in all_rows
        if ((row.get("metadata") or {}).get("training_dataset") or {}).get("source_kind") == "main"
    ]
    main_latest = latest_by_task(main_rows)
    summary = {
        "dataset_name": args.dataset_name,
        "repo_id": args.repo_id,
        "generated_at": now_iso(),
        "sources": source_summaries,
        "overall": {
            "main_tasks": 301,
            "main_completed_tasks": len(main_latest),
            "all_rollout_rows": len(all_rows),
            "latest_by_task_rows": len(latest_rows),
            "best_by_task_rows": len(best_rows),
            "milestone_6_or_7_rows": len(m67_rows),
            "milestone_7_rows": sum(1 for row in all_rows if milestone_value(row) == 7),
            "strategy_examples": len(strategy_examples),
            "sft_examples": len(sft_rows),
            "status_counts_best_by_task": dict(Counter(str(row.get("status") or "UNKNOWN") for row in best_rows)),
            "status_counts_main_latest": dict(Counter(str(row.get("status") or "UNKNOWN") for row in main_latest.values())),
            "milestone_counts_all_rows": dict(Counter(str(milestone_value(row)) for row in all_rows)),
            "artifacts_copied": len([row for row in artifact_manifest if row.get("found")]),
            "artifacts_missing": len([row for row in artifact_manifest if not row.get("found")]),
        },
    }
    write_json(dataset_dir / "metadata/summary.json", summary)
    (dataset_dir / "README.md").write_text(build_readme(summary))

    staging_summary = {
        "generated_at": now_iso(),
        "dataset": {
            "repo_id": args.repo_id,
            "local_dir": str(dataset_dir),
            "files": sum(1 for path in dataset_dir.rglob("*") if path.is_file()),
            "bytes": sum(path.stat().st_size for path in dataset_dir.rglob("*") if path.is_file()),
            "summary": summary,
        },
    }
    write_json(args.staging_root / f"{args.dataset_name}_staging_summary.json", staging_summary)
    print(json.dumps(staging_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
