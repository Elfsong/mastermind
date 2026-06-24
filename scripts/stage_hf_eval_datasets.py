#!/usr/bin/env python3
"""Stage CyberGym Codex GPT-5.5 evaluation artifacts for Hugging Face datasets."""

from __future__ import annotations

import csv
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
NAMESPACE = "Elfsong"


@dataclass(frozen=True)
class SourceFile:
    name: str
    path: Path
    run_id: str | None = None
    round_index: int | None = None


@dataclass(frozen=True)
class DatasetSpec:
    repo_name: str
    title: str
    short_name: str
    description: str
    sources: tuple[SourceFile, ...]
    analysis_dir: Path | None = None
    summary_path: Path | None = None


ROUND_SOURCES = (
    SourceFile("round1", RUNS / "codex_gateway_eval100_clean_rollouts.jsonl", "codex-gateway-eval100-20260526T1655Z", 1),
    SourceFile("round2", RUNS / "codex_gateway_eval_bo4_rep2_rollouts.jsonl", "codex-gateway-eval-bo4-r2-20260527T0905Z", 2),
    SourceFile("round3", RUNS / "codex_gateway_eval_bo4_rep3_rollouts.jsonl", "codex-gateway-eval-bo4-r3-20260527T1439Z", 3),
    SourceFile("round4", RUNS / "codex_gateway_eval_bo4_rep4_rollouts.jsonl", "codex-gateway-eval-bo4-r4-20260527T1439Z", 4),
    SourceFile("round5", RUNS / "codex_gateway_eval_bo4_rep5_rollouts.jsonl", "codex-gateway-eval-bo4-r5-20260527T1439Z", 5),
    SourceFile("round6", RUNS / "codex_gateway_eval_bo4_rep6_rollouts.jsonl", "codex-gateway-eval-bo4-r6-20260527T1439Z", 6),
    SourceFile("round7", RUNS / "codex_gateway_eval_bo4_rep7_rollouts.jsonl", "codex-gateway-eval-bo4-r7-20260527T1439Z", 7),
    SourceFile("round8", RUNS / "codex_gateway_eval_bo4_rep8_rollouts.jsonl", "codex-gateway-eval-bo4-r8-20260527T1439Z", 8),
)


DATASETS = (
    DatasetSpec(
        repo_name="cybergym-codex-gpt-5-5-round1-8-independent-eval",
        title="CyberGym Codex GPT-5.5 Round 1-8 Independent Evaluation",
        short_name="Round 1-8 Independent Evaluation",
        description="Eight independent 200-task CyberGym evaluation rounds and the derived Best-of-8 analysis.",
        sources=ROUND_SOURCES,
        analysis_dir=RUNS / "codex_cybergym/analysis/eval_archive_20260528T153449Z",
    ),
    DatasetSpec(
        repo_name="cybergym-codex-gpt-5-5-level3-eval",
        title="CyberGym Codex GPT-5.5 Level 3 Evaluation",
        short_name="Level 3 Experiment",
        description="Single 200-task CyberGym Level 3 evaluation run.",
        sources=(
            SourceFile(
                "level3",
                RUNS / "codex_gateway_eval_level3_rep1_rollouts.jsonl",
                "codex-gateway-eval-level3-r1-20260528T1140Z",
            ),
        ),
        analysis_dir=RUNS / "codex_cybergym/analysis/eval_archive_20260528T153449Z",
    ),
    DatasetSpec(
        repo_name="cybergym-codex-gpt-5-5-iterative-improvement-eval",
        title="CyberGym Codex GPT-5.5 Iterative Improvement Evaluation",
        short_name="Iterative Improvement Experiment",
        description="Canonical 200-task iterative improvement experiment with up to 8 sequential attempts per task.",
        sources=(
            SourceFile(
                "iterative_improvement",
                RUNS / "codex_gateway_iterative_improvement_experiment_rollouts.jsonl",
                None,
            ),
        ),
        summary_path=RUNS / "codex_gateway_iterative_improvement_experiment_summary.json",
    ),
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_run_id(run_id: str | None) -> str:
    return (run_id or "mixed").replace("/", "_").replace(":", "_")


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path, run_id: str | None = None) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_json = 0
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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


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


def latest_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str):
            grouped[task_id].append(row)
    latest: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in grouped.items():
        task_rows.sort(
            key=lambda row: (
                attempt_index(row),
                row.get("started_at") or row.get("started_at_first") or "",
                row.get("finished_at") or row.get("finished_at_latest") or "",
                row.get("run_id") or row.get("latest_run_id") or "",
            )
        )
        latest[task_id] = task_rows[-1]
    return latest


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    latest = latest_by_task(rows)
    return dict(Counter(str(row.get("status") or "UNKNOWN") for row in latest.values()))


def companion_paths(trajectory_path: Path) -> tuple[Path | None, Path | None]:
    if trajectory_path.name.endswith(".codex.jsonl"):
        stem = trajectory_path.name[: -len(".codex.jsonl")]
        last = trajectory_path.with_name(f"{stem}.codex.last.txt")
        stderr = trajectory_path.with_name(f"{stem}.codex.stderr.log")
    else:
        last = trajectory_path.with_suffix(".last.txt")
        stderr = trajectory_path.with_suffix(".stderr.log")
    return (last if last.exists() else None, stderr if stderr.exists() else None)


def text_from_item(item: dict[str, Any]) -> tuple[str | None, str | None]:
    if isinstance(item.get("text"), str):
        return "text", item["text"]
    if isinstance(item.get("aggregated_output"), str):
        return "aggregated_output", item["aggregated_output"]
    return None, None


def read_trajectory(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    text_items: list[dict[str, Any]] = []
    bad_json = 0
    with path.open(errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            events.append(event)
            item = event.get("item")
            if isinstance(item, dict):
                field, text = text_from_item(item)
                if text is not None:
                    text_items.append(
                        {
                            "line": line_no,
                            "event_type": event.get("type"),
                            "item_id": item.get("id"),
                            "item_type": item.get("type"),
                            "field": field,
                            "text": text,
                        }
                    )
    return events, text_items, bad_json


def stage_trajectory(
    *,
    dataset_dir: Path,
    row: dict[str, Any],
    source: SourceFile,
    trajectory_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trajectory_value = row.get("trajectory_path")
    if not isinstance(trajectory_value, str) or not trajectory_value:
        return {"found": False, "error": "missing trajectory_path"}
    src = Path(trajectory_value)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        return {"found": False, "source_path": str(src), "error": "trajectory file does not exist"}

    cache_key = str(src.resolve())
    if cache_key in trajectory_cache:
        return trajectory_cache[cache_key]

    run_id = safe_run_id(row.get("run_id") or source.run_id)
    raw_rel = Path("trajectories/raw") / run_id / src.name
    raw_dst = dataset_dir / raw_rel
    copy_file(src, raw_dst)

    last_src, stderr_src = companion_paths(src)
    last_rel = None
    stderr_rel = None
    if last_src is not None:
        last_rel_path = Path("trajectories/last_messages") / run_id / last_src.name
        copy_file(last_src, dataset_dir / last_rel_path)
        last_rel = str(last_rel_path)
    if stderr_src is not None:
        stderr_rel_path = Path("trajectories/stderr") / run_id / stderr_src.name
        copy_file(stderr_src, dataset_dir / stderr_rel_path)
        stderr_rel = str(stderr_rel_path)

    events, text_items, bad_json = read_trajectory(src)
    text_chars = sum(len(item["text"]) for item in text_items)
    bundle = {
        "found": True,
        "source_path": relative_to_root(src),
        "raw_path": str(raw_rel),
        "last_message_path": last_rel,
        "stderr_path": stderr_rel,
        "bytes": src.stat().st_size,
        "sha256": sha256_file(src),
        "event_count": len(events),
        "bad_json": bad_json,
        "text_item_count": len(text_items),
        "text_chars": text_chars,
        "trajectory_events": events,
        "prompt_output_items": text_items,
    }
    trajectory_cache[cache_key] = bundle
    return bundle


def public_bundle_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in bundle.items()
        if key not in {"trajectory_events", "prompt_output_items"}
    }


def enrich_rows(dataset_dir: Path, source: SourceFile, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    enriched_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    trajectory_cache: dict[str, dict[str, Any]] = {}

    for row_index, row in enumerate(rows):
        enriched = dict(row)
        enriched["source_name"] = source.name
        if source.round_index is not None:
            enriched["round_index"] = source.round_index
        bundle = stage_trajectory(
            dataset_dir=dataset_dir,
            row=row,
            source=source,
            trajectory_cache=trajectory_cache,
        )
        enriched["trajectory_bundle"] = public_bundle_metadata(bundle)
        enriched["trajectory_events"] = bundle.get("trajectory_events", [])
        enriched["prompt_output_items"] = bundle.get("prompt_output_items", [])
        enriched_rows.append(enriched)

        manifest = public_bundle_metadata(bundle)
        manifest.update(
            {
                "source_name": source.name,
                "round_index": source.round_index,
                "row_index": row_index,
                "task_id": row.get("task_id"),
                "run_id": row.get("run_id") or source.run_id,
                "status": row.get("status"),
                "attempt_index": attempt_index(row),
            }
        )
        manifest_rows.append(manifest)
    return enriched_rows, manifest_rows


def copy_analysis(spec: DatasetSpec, dataset_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    if spec.analysis_dir is None:
        return copied
    include_names = {
        "best_of_curve.csv",
        "diagnostics.md",
        "manifest.json",
        "pairwise_round_pass_jaccard.csv",
        "poc_length_summary.csv",
        "reason_summary.csv",
        "run_summary.csv",
        "task_matrix.csv",
        "task_sets.json",
    }
    for src in sorted(spec.analysis_dir.iterdir()):
        if src.name not in include_names or not src.is_file():
            continue
        rel = Path("analysis") / src.name
        copy_file(src, dataset_dir / rel)
        copied.append({"path": str(rel), "source_path": relative_to_root(src), "bytes": src.stat().st_size, "sha256": sha256_file(src)})
    return copied


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest = latest_by_task(rows)
    counts = Counter(str(row.get("status") or "UNKNOWN") for row in latest.values())
    passed = counts.get("PASSED", 0)
    total = len(latest)
    return {
        "rows": len(rows),
        "tasks": total,
        "status_counts": dict(counts),
        "passed": passed,
        "pass_rate": passed / total if total else None,
    }


def read_best_of_curve(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def read_run_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def read_task_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_round_task_matrix(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            converted = dict(row)
            bo8_pass = row.get("bo8_pass") == "True"
            converted["status"] = "PASSED" if bo8_pass else "NOT_PASSED"
            converted["best_of_8_passed"] = bo8_pass
            converted["pass_count_round1_8"] = int(row["pass_count_round1_8"])
            converted["first_pass_round"] = int(row["first_pass_round"]) if row.get("first_pass_round") else None
            converted["level3_passed"] = row.get("level3_pass") == "True"
            rows.append(converted)
    return rows


def is_round_dataset(spec: DatasetSpec) -> bool:
    return "round1-8" in spec.repo_name


def make_readme(spec: DatasetSpec, summary: dict[str, Any]) -> str:
    lines = [
        "---",
        "license: other",
        "task_categories:",
        "- text-generation",
        "language:",
        "- en",
        "tags:",
        "- cybergym",
        "- codex",
        "- gpt-5.5",
        "- cybersecurity",
        "- evaluation",
        "pretty_name: " + spec.title,
        "---",
        "",
        f"# {spec.title}",
        "",
        spec.description,
        "",
        "This dataset preserves both task-level rollout summaries and the raw Codex trajectory logs.",
        "The enriched JSONL files inline visible trajectory events and extracted prompt/output text items for each rollout.",
        "",
        "## Results",
        "",
        f"- Tasks: `{summary['overall']['tasks']}`",
        f"- Rollout rows: `{summary['rollout_summary']['rows']}`",
        f"- Passed: `{summary['overall']['passed']}`",
        f"- Pass rate: `{summary['overall']['pass_rate']:.2%}`" if summary["overall"]["pass_rate"] is not None else "- Pass rate: `n/a`",
        f"- Status counts: `{json.dumps(summary['overall']['status_counts'], sort_keys=True)}`",
        "",
        "## Files",
        "",
        "- `data/rollouts.jsonl`: normalized rollout rows.",
        "- `data/rollouts_enriched.jsonl`: rollout rows with full `trajectory_events` and extracted `prompt_output_items`.",
        "- `data/tasks.jsonl`: latest task-level summary rows when available.",
        "- `trajectories/raw/`: original `.codex.jsonl` trajectory files.",
        "- `trajectories/last_messages/`: final Codex visible messages when available.",
        "- `trajectories/stderr/`: Codex stderr logs when available.",
        "- `metadata/trajectory_manifest.jsonl`: per-rollout trajectory coverage and checksums.",
        "- `metadata/summary.json`: aggregate counts and source file metadata.",
        "",
        "## Completeness Notes",
        "",
        "The dataset includes visible prompts, outputs, command execution logs, and assistant messages present in the local Codex trajectory files.",
        "Hidden system prompts or provider-side raw HTTP payloads are not reconstructable unless they were already recorded in the trajectory logs.",
        "",
    ]
    if summary.get("best_of_curve") and is_round_dataset(spec):
        lines.extend(["## Best-of Curve", ""])
        lines.append("| Best-of-N | Passed | Pass rate | Marginal new passes |")
        lines.append("|---:|---:|---:|---:|")
        for row in summary["best_of_curve"]:
            lines.append(
                f"| {row['best_of_n']} | {row['passed']} | {float(row['pass_rate']):.1%} | {row['marginal_new_passes']} |"
            )
        lines.append("")
    return "\n".join(lines)


def stage_dataset(spec: DatasetSpec) -> dict[str, Any]:
    dataset_dir = STAGING_ROOT / spec.repo_name
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)

    all_rows: list[dict[str, Any]] = []
    all_enriched: list[dict[str, Any]] = []
    all_manifest: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    for source in spec.sources:
        rows, bad_json = load_jsonl(source.path, source.run_id)
        source_rel = Path("data/source_rollouts") / f"{source.name}.jsonl"
        write_jsonl(dataset_dir / source_rel, rows)
        enriched_rows, manifest_rows = enrich_rows(dataset_dir, source, rows)
        enriched_rel = Path("data/source_rollouts_enriched") / f"{source.name}.enriched.jsonl"
        write_jsonl(dataset_dir / enriched_rel, enriched_rows)

        source_summary = summarize_rows(rows)
        source_summary.update(
            {
                "name": source.name,
                "source_path": relative_to_root(source.path),
                "dataset_path": str(source_rel),
                "enriched_dataset_path": str(enriched_rel),
                "run_id": source.run_id,
                "round_index": source.round_index,
                "bad_json": bad_json,
                "source_bytes": source.path.stat().st_size,
                "source_sha256": sha256_file(source.path),
            }
        )
        source_summaries.append(source_summary)
        all_rows.extend(rows)
        all_enriched.extend(enriched_rows)
        all_manifest.extend(manifest_rows)

    write_jsonl(dataset_dir / "data/rollouts.jsonl", all_rows)
    write_jsonl(dataset_dir / "data/rollouts_enriched.jsonl", all_enriched)
    write_jsonl(dataset_dir / "metadata/trajectory_manifest.jsonl", all_manifest)

    task_rows: list[dict[str, Any]]
    if is_round_dataset(spec):
        task_rows = read_round_task_matrix(spec.analysis_dir / "task_matrix.csv")
    elif spec.summary_path and (RUNS / "codex_gateway_iterative_improvement_experiment_tasks.jsonl").exists():
        task_rows = read_task_summary(RUNS / "codex_gateway_iterative_improvement_experiment_tasks.jsonl")
    else:
        task_rows = list(latest_by_task(all_rows).values())
    write_jsonl(dataset_dir / "data/tasks.jsonl", task_rows)

    copied_analysis = copy_analysis(spec, dataset_dir) if is_round_dataset(spec) else []
    if spec.summary_path is not None and spec.summary_path.exists():
        copy_file(spec.summary_path, dataset_dir / "metadata/original_summary.json")

    if is_round_dataset(spec):
        bo8_passed = sum(1 for row in task_rows if row.get("best_of_8_passed"))
        overall = {
            "rows": len(all_rows),
            "tasks": len(task_rows),
            "status_counts": {"NOT_PASSED": len(task_rows) - bo8_passed, "PASSED": bo8_passed},
            "passed": bo8_passed,
            "pass_rate": bo8_passed / len(task_rows) if task_rows else None,
        }
    else:
        overall = summarize_rows(task_rows if spec.summary_path else all_rows)

    summary = {
        "title": spec.title,
        "repo_id": f"{NAMESPACE}/{spec.repo_name}",
        "generated_at": now_iso(),
        "overall": overall,
        "rollout_summary": summarize_rows(all_rows),
        "source_summaries": source_summaries,
        "trajectory_manifest_rows": len(all_manifest),
        "trajectory_files_found": sum(1 for row in all_manifest if row.get("found")),
        "trajectory_files_missing": sum(1 for row in all_manifest if not row.get("found")),
        "analysis_files": copied_analysis,
        "best_of_curve": read_best_of_curve(dataset_dir / "analysis/best_of_curve.csv"),
        "run_summary": read_run_summary(dataset_dir / "analysis/run_summary.csv"),
    }
    write_json(dataset_dir / "metadata/summary.json", summary)
    (dataset_dir / "README.md").write_text(make_readme(spec, summary), encoding="utf-8")

    return {
        "repo_id": f"{NAMESPACE}/{spec.repo_name}",
        "local_dir": str(dataset_dir),
        "summary": summary,
        "bytes": sum(path.stat().st_size for path in dataset_dir.rglob("*") if path.is_file()),
        "files": sum(1 for path in dataset_dir.rglob("*") if path.is_file()),
    }


def main() -> int:
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    results = [stage_dataset(spec) for spec in DATASETS]
    write_json(STAGING_ROOT / "staging_summary.json", {"generated_at": now_iso(), "datasets": results})
    for result in results:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
