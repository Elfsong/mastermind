#!/usr/bin/env python3
"""Prepare one-step strategy-generator SFT data for Tinker training.

The HF-staged dataset contains structured strategy-generator examples. This
script converts them into the exact prompt shape used by
run_codex_cybergym_sequential_self_involving.py when
--experience-updater=tinker is enabled: a single user message asking for an
updated task-local experience/strategy, with the assistant response starting at
"1. Current best hypothesis".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "runs/hf_upload_staging/cybergym-codex-gpt-5-5-iterative-improvement-train"
    / "data/strategy_generator_examples.jsonl"
)
DEFAULT_OUT_DIR = ROOT / "runs/strategy_sft/qwen36_strategy_sft_data"
DEFAULT_TASKS_JSON = ROOT / "runs/cybergym_assets/cybergym_data/tasks.json"
DEFAULT_DATA_DIR = ROOT / "runs/cybergym_assets/cybergym_data/data"
ASSISTANT_PREFIX = "1. Current best hypothesis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS_JSON)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--min-next-milestone",
        type=int,
        default=None,
        help="Optional quality filter. Example: 6 keeps transitions whose next attempt reached m>=6.",
    )
    parser.add_argument(
        "--only-next-passed",
        action="store_true",
        help="Optional strict quality filter: keep only transitions whose next attempt passed.",
    )
    parser.add_argument("--experience-token-budget", type=int, default=2048)
    parser.add_argument("--task-context-max-chars", type=int, default=1500)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def trim_prefix(text: str, prefix: str) -> str:
    stripped = text.lstrip()
    leading = text[: len(text) - len(stripped)]
    if stripped.startswith(prefix):
        return leading + stripped[len(prefix) :]
    return text


def build_task_context(
    task_meta: dict[str, Any] | None,
    data_dir: Path | None,
    task_id: str,
    max_chars: int = 1500,
) -> str:
    parts: list[str] = []
    if task_meta:
        project = task_meta.get("project_name", "")
        language = task_meta.get("project_language", "")
        vuln = task_meta.get("vulnerability_description", "")
        header = f"Project: {project} ({language})" if project else ""
        if header:
            parts.append(header)
        if vuln:
            parts.append(f"Vulnerability summary: {vuln}")

    if data_dir and task_id.startswith("arvo:"):
        arvo_id = task_id.split(":", 1)[1]
        task_dir = data_dir / "arvo" / arvo_id
        for fname in ("description.txt", "error.txt"):
            fpath = task_dir / fname
            if fpath.exists():
                text = fpath.read_text(errors="replace").strip()
                if text:
                    label = "Vulnerability description" if fname == "description.txt" else "Reference error output"
                    parts.append(f"{label}:\n```text\n{text[:max_chars]}\n```")

    return "\n\n".join(parts)


def build_experience_update_prompt(
    *,
    task_id: str,
    attempt: int,
    token_budget: int,
    previous_experience: str,
    feedback: str,
    task_context: str = "",
) -> str:
    previous_block = previous_experience or "(No prior experience for this task.)"
    context_block = f"\nTask context:\n{task_context}\n" if task_context else ""
    return f"""You are maintaining a task-local experience object for CyberGym task {task_id}.

Update the experience after attempt {attempt}.
{context_block}
Constraints:
- Output only the updated experience text.
- Do not output hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Start directly with "1. Current best hypothesis".
- Keep it under {token_budget} tokens.
- Preserve concrete lessons that help the next attempt solve this same task.
- Remove stale or disproven hypotheses unless they are useful as warnings.
- Include verifier facts such as whether the vulnerable build crashed, whether the fixed build crashed, and which PoC patterns failed.
- Do not invent results not present in the feedback.
- Do not include generic CyberGym instructions.

Recommended structure:
1. Current best hypothesis
2. Evidence from attempts so far
3. Failed approaches to avoid
4. Concrete next-trial plan

Previous experience:
```text
{previous_block}
```

Newest feedback:
```text
{feedback}
```"""


def convert_example(
    row: dict[str, Any],
    token_budget: int,
    task_context: str = "",
) -> dict[str, Any] | None:
    if not (row.get("quality") or {}).get("usable_for_sft", False):
        return None
    target = str(row.get("target_strategy") or "").strip()
    if not target:
        return None

    input_obj = row.get("input") or {}
    task_id = str(row.get("task_id") or "")
    attempt = int(row.get("input_attempt_index") or 0)
    previous_strategy = str(input_obj.get("previous_strategy") or "")
    feedback = str(input_obj.get("feedback") or "")
    prompt = build_experience_update_prompt(
        task_id=task_id,
        attempt=attempt,
        token_budget=token_budget,
        previous_experience=previous_strategy,
        feedback=feedback,
        task_context=task_context,
    )
    target_suffix = trim_prefix(target, ASSISTANT_PREFIX)
    return {
        "id": row.get("example_id"),
        "task_id": task_id,
        "run_id": row.get("run_id"),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "prompt_assistant_prefix": ASSISTANT_PREFIX,
        "target_completion": target_suffix,
        "metadata": {
            "source_name": row.get("source_name"),
            "input_attempt_index": row.get("input_attempt_index"),
            "target_attempt_index": row.get("target_attempt_index"),
            "next_status": (row.get("next_attempt_outcome") or {}).get("status"),
            "next_milestone": (row.get("next_attempt_outcome") or {}).get("milestone"),
            "next_passed": (row.get("next_attempt_outcome") or {}).get("passed"),
            "milestone_delta": (row.get("quality") or {}).get("milestone_delta"),
            "next_milestone_6_or_7": (row.get("quality") or {}).get(
                "next_milestone_6_or_7"
            ),
            "original_example_id": row.get("example_id"),
            "has_task_context": bool(task_context),
        },
    }


def split_by_task(
    rows: list[dict[str, Any]],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    task_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    n = len(task_ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_tasks = set(task_ids[:n_train])
    val_tasks = set(task_ids[n_train : n_train + n_val])
    test_tasks = set(task_ids[n_train + n_val :])
    splits = {"train": [], "val": [], "test": []}
    for task_id, task_rows in grouped.items():
        if task_id in train_tasks:
            split = "train"
        elif task_id in val_tasks:
            split = "val"
        elif task_id in test_tasks:
            split = "test"
        else:
            raise AssertionError(task_id)
        for row in task_rows:
            row = dict(row)
            row["split"] = split
            splits[split].append(row)
    for split_rows in splits.values():
        split_rows.sort(key=lambda r: (r["task_id"], r["metadata"]["input_attempt_index"], r["id"]))
    return splits


def main() -> int:
    args = parse_args()

    tasks_by_id: dict[str, dict[str, Any]] = {}
    if args.tasks_json.exists():
        try:
            raw_tasks = json.loads(args.tasks_json.read_text())
            if isinstance(raw_tasks, list):
                tasks_by_id = {t["task_id"]: t for t in raw_tasks if isinstance(t, dict) and "task_id" in t}
        except Exception:
            pass

    raw = read_jsonl(args.source)
    converted: list[dict[str, Any]] = []
    for row in raw:
        next_milestone = (row.get("next_attempt_outcome") or {}).get("milestone")
        next_passed = bool((row.get("next_attempt_outcome") or {}).get("passed"))
        if args.min_next_milestone is not None:
            try:
                if int(next_milestone) < args.min_next_milestone:
                    continue
            except (TypeError, ValueError):
                continue
        if args.only_next_passed and not next_passed:
            continue
        task_id = str(row.get("task_id") or "")
        task_context = build_task_context(
            tasks_by_id.get(task_id),
            args.data_dir if args.data_dir.exists() else None,
            task_id,
            max_chars=args.task_context_max_chars,
        )
        converted_row = convert_example(row, args.experience_token_budget, task_context=task_context)
        if converted_row is not None:
            converted.append(converted_row)

    splits = split_by_task(
        converted,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", splits["train"])
    write_jsonl(args.out_dir / "val.jsonl", splits["val"])
    write_jsonl(args.out_dir / "test.jsonl", splits["test"])
    write_jsonl(args.out_dir / "all.jsonl", converted)

    summary = {
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "tasks_json": str(args.tasks_json),
        "tasks_loaded": len(tasks_by_id),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "filters": {
            "min_next_milestone": args.min_next_milestone,
            "only_next_passed": args.only_next_passed,
        },
        "assistant_prefix": ASSISTANT_PREFIX,
        "rows_total": len(converted),
        "rows_with_task_context": sum(1 for r in converted if r["metadata"].get("has_task_context")),
        "splits": {
            split: {
                "rows": len(rows),
                "tasks": len({r["task_id"] for r in rows}),
                "next_status": dict(Counter(r["metadata"]["next_status"] for r in rows)),
                "next_milestone": dict(Counter(str(r["metadata"]["next_milestone"]) for r in rows)),
            }
            for split, rows in splits.items()
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
