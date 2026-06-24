#!/usr/bin/env python3
"""Seed a sequential self-improving run from an existing BO round-1 output."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from mastermind.config import load_manifest
from mastermind.tasks import load_split_ids

import run_codex_cybergym_sequential_self_involving as seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/codex_cybergym"))
    parser.add_argument("--split", default="eval")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--runner-workers", type=int, default=16)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--model-reasoning-effort", default="medium")
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--summary-reasoning-effort", default="medium")
    parser.add_argument("--experience-updater", choices=("codex", "openai"), default="codex")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-provider")
    parser.add_argument("--codex-provider-base-url")
    parser.add_argument("--codex-provider-wire-api", default="responses")
    parser.add_argument("--codex-provider-env-key", default="LLM_GATEWAY_API_KEY")
    parser.add_argument("--openai-base-url")
    parser.add_argument("--openai-api-key-env-key", default="OPENAI_API_KEY")
    parser.add_argument("--openai-summary-max-tokens", type=int)
    parser.add_argument("--summary-timeout-seconds", type=int, default=600)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--sandbox", default="workspace-write")
    parser.add_argument("--experience-token-budget", type=int, default=2048)
    parser.add_argument("--experience-max-chars", type=int, default=9000)
    parser.add_argument("--feedback-max-chars", type=int, default=9000)
    parser.add_argument("--tasks-json", type=Path, default=Path("runs/cybergym_assets/cybergym_data/tasks.json"))
    parser.add_argument("--task-context-data-dir", type=Path, default=Path("runs/cybergym_assets/cybergym_data/data"))
    parser.add_argument("--task-context-max-chars", type=int, default=1500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-experience-updates",
        action="store_true",
        help="Write feedback-derived fallback experience without calling a summary model.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def row_passed(row: dict[str, Any]) -> bool:
    if row.get("status") == "PASSED":
        return True
    verification = row.get("verification") or {}
    return verification.get("passed") is True


def latest_seed_rows(seed_output: Path, task_ids: list[str]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    wanted = set(task_ids)
    for row in read_rows(seed_output):
        task_id = row.get("task_id")
        if isinstance(task_id, str) and task_id in wanted:
            latest[task_id] = row
    return latest


def build_tasks_by_id(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if not args.tasks_json.exists():
        return {}
    try:
        raw = json.loads(args.tasks_json.read_text(errors="replace"))
    except Exception:
        return {}
    if not isinstance(raw, list):
        return {}
    return {item["task_id"]: item for item in raw if isinstance(item, dict) and isinstance(item.get("task_id"), str)}


def decorate_seed_row(
    *,
    args: argparse.Namespace,
    source_row: dict[str, Any],
    feedback_path: Path,
    experience_md: Path,
    update_metadata: dict[str, Any],
) -> dict[str, Any]:
    row = deepcopy(source_row)
    task_id = str(row["task_id"])
    safe_task = seq.safe_name(task_id)
    metadata = row.setdefault("metadata", {})
    verification = row.setdefault("verification", {})
    details = verification.setdefault("details", {})
    source_run_id = row.get("run_id")
    source_agent = row.get("agent")
    old_source = metadata.get("source")
    sequential = {
        "strategy": "sequential_improvement_self_involving",
        "source": seq.SOURCE,
        "base_runner_source": old_source,
        "attempt_index": 1,
        "max_attempts": args.max_attempts,
        "stop_on_pass": True,
        "infra_failure": False,
        "effective_attempt": True,
        "attempt_output": str(args.seed_output),
        "experience_before": str(experience_md),
        "feedback_path": str(feedback_path),
        "experience_after": str(experience_md),
        "experience_update": update_metadata,
        "experience_token_budget": args.experience_token_budget,
        "seeded_from_bo8_round1": True,
        "seed_source_output": str(args.seed_output),
        "seed_source_run_id": source_run_id,
        "seed_source_agent": source_agent,
        "seeded_at": now_iso(),
    }
    row["run_id"] = args.run_id
    row["agent"] = f"{args.run_id}-{safe_task}_seed1"
    row["model"] = args.model
    row["strategy_id"] = "sequential_self_involving_attempt_1"
    row["strategy"] = "sequential_improvement_self_involving"
    metadata["source"] = seq.SOURCE
    metadata["configured_model"] = args.model
    metadata["attempt_index"] = 1
    metadata["seed_source_run_id"] = source_run_id
    metadata["seed_source_agent"] = source_agent
    metadata["sequential"] = sequential
    details["source"] = seq.SOURCE
    details["sequential"] = sequential
    return row


def write_seed_output(path: Path, rows: list[dict[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def existing_seed_output_valid(path: Path, run_id: str) -> bool:
    if not path.exists():
        return False
    for row in read_rows(path):
        if row.get("run_id") != run_id:
            continue
        sequential = ((row.get("metadata") or {}).get("sequential") or {})
        if sequential.get("strategy") == "sequential_improvement_self_involving":
            return True
    return False


def update_one_experience(
    *,
    args: argparse.Namespace,
    task_id: str,
    source_row: dict[str, Any],
    feedback: str,
    feedback_path: Path,
    experience_json: Path,
    experience_md: Path,
    update_dir: Path,
    env: dict[str, str],
    task_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    if row_passed(source_row):
        return {"update_status": "skipped_seed_passed", "reason": "round1 already passed"}
    if experience_md.exists() and experience_json.exists() and not args.force:
        return {
            "update_status": "existing_seed_experience_reused",
            "experience_md": str(experience_md),
            "experience_json": str(experience_json),
        }

    task_context = seq.build_task_context(
        task_meta,
        args.task_context_data_dir,
        task_id,
        max_chars=args.task_context_max_chars,
    )
    previous_experience = ""
    if args.skip_experience_updates:
        update_metadata = {
            "update_status": "fallback_seed_feedback_only",
            "reason": "--skip-experience-updates",
        }
        updated_experience = seq.trim_text(
            "\n\n".join(
                [
                    "1. Current best hypothesis",
                    "(No distilled hypothesis yet; seeded from BO8 round 1 feedback.)",
                    "2. Evidence from attempts so far",
                    feedback,
                    "3. Failed approaches to avoid",
                    "Avoid repeating the exact submitted PoC patterns unless there is a concrete new reason.",
                    "4. Concrete next-trial plan",
                    "Use the verifier feedback and task context above to plan the next attempt.",
                ]
            ),
            args.experience_max_chars,
        )
    else:
        workspace = Path((source_row.get("metadata") or {}).get("workspace") or ".")
        if not workspace.exists():
            workspace = ROOT
        if args.experience_updater == "openai":
            updated_experience, update_metadata = seq.update_experience_with_openai(
                args=args,
                task_id=task_id,
                attempt=1,
                previous_experience=previous_experience,
                feedback=feedback,
                env=env,
                task_context=task_context,
            )
        else:
            updated_experience, update_metadata = seq.update_experience_with_codex(
                args=args,
                task_id=task_id,
                safe_task=seq.safe_name(task_id),
                attempt=1,
                workspace=workspace,
                previous_experience=previous_experience,
                feedback=feedback,
                update_dir=update_dir,
                env=env,
                task_context=task_context,
            )

    seq.store_experience(
        experience_json=experience_json,
        experience_md=experience_md,
        task_id=task_id,
        run_id=args.run_id,
        attempt=1,
        experience=updated_experience,
        token_budget=args.experience_token_budget,
        update_metadata=update_metadata,
    )
    return update_metadata


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    env = os.environ.copy()
    pythonpath = [str((ROOT / "cybergym/src").resolve()), str((ROOT / "src").resolve())]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    manifest = load_manifest()
    task_ids = load_split_ids(manifest, args.split)
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    latest = latest_seed_rows(args.seed_output, task_ids)
    missing = [task_id for task_id in task_ids if task_id not in latest]
    if missing:
        raise SystemExit(f"seed output is missing {len(missing)} tasks, first missing: {missing[:5]}")

    source_rows = [latest[task_id] for task_id in task_ids]
    passed = sum(1 for row in source_rows if row_passed(row))
    unpassed = len(source_rows) - passed
    run_root = (args.run_root / args.run_id).resolve()
    experiences_dir = run_root / "experiences"
    feedback_dir = run_root / "feedback"
    update_dir = run_root / "experience_updates"

    start_event = {
        "event": "seed_seq_self_from_bo8_round1_start",
        "run_id": args.run_id,
        "seed_output": str(args.seed_output),
        "output": str(args.output),
        "task_count": len(source_rows),
        "passed_seed": passed,
        "unpassed_seed": unpassed,
        "workers": args.workers,
        "model": args.model,
        "summary_model": args.summary_model or args.model,
        "experience_updater": args.experience_updater,
        "dry_run": args.dry_run,
        "time": now_iso(),
    }
    print(json.dumps(start_event, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    if args.output.exists() and args.force:
        args.output.unlink()
    output_already_seeded = existing_seed_output_valid(args.output, args.run_id)

    tasks_by_id = build_tasks_by_id(args)
    seed_update_by_task: dict[str, dict[str, Any]] = {}

    def process_task(row: dict[str, Any]) -> tuple[str, dict[str, Any], Path, Path]:
        task_id = str(row["task_id"])
        safe_task = seq.safe_name(task_id)
        feedback = seq.build_feedback(row, attempt=1, max_chars=args.feedback_max_chars)
        feedback_path = feedback_dir / f"{safe_task}.attempt1.seed.feedback.md"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text(feedback.rstrip() + "\n")
        experience_json = experiences_dir / f"{safe_task}.experience.json"
        experience_md = experiences_dir / f"{safe_task}.experience.md"
        update_metadata = update_one_experience(
            args=args,
            task_id=task_id,
            source_row=row,
            feedback=feedback,
            feedback_path=feedback_path,
            experience_json=experience_json,
            experience_md=experience_md,
            update_dir=update_dir,
            env=env,
            task_meta=tasks_by_id.get(task_id),
        )
        return task_id, update_metadata, feedback_path, experience_md

    futures = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in source_rows:
            futures[executor.submit(process_task, row)] = str(row["task_id"])
        completed = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = futures.pop(future)
                completed += 1
                try:
                    task_id, update_metadata, feedback_path, experience_md = future.result()
                    seed_update_by_task[task_id] = {
                        "update_metadata": update_metadata,
                        "feedback_path": str(feedback_path),
                        "experience_md": str(experience_md),
                    }
                    print(
                        json.dumps(
                            {
                                "event": "seed_task_ready",
                                "task_id": task_id,
                                "completed": completed,
                                "total": len(source_rows),
                                "update_status": update_metadata.get("update_status"),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - fail fast with the task id.
                    print(
                        json.dumps(
                            {
                                "event": "seed_task_error",
                                "task_id": task_id,
                                "error": repr(exc),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    raise

    if not output_already_seeded:
        decorated_rows: list[dict[str, Any]] = []
        for row in source_rows:
            task_id = str(row["task_id"])
            safe_task = seq.safe_name(task_id)
            info = seed_update_by_task[task_id]
            decorated_rows.append(
                decorate_seed_row(
                    args=args,
                    source_row=row,
                    feedback_path=Path(info["feedback_path"]),
                    experience_md=Path(info["experience_md"]),
                    update_metadata=info["update_metadata"],
                )
            )
        write_seed_output(args.output, decorated_rows, force=args.force)

    seed_config = run_root / "seed_from_bo8_round1_config.json"
    seed_config.parent.mkdir(parents=True, exist_ok=True)
    seed_config.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "seed_output": str(args.seed_output),
                "output": str(args.output),
                "task_count": len(source_rows),
                "passed_seed": passed,
                "unpassed_seed": unpassed,
                "max_attempts": args.max_attempts,
                "model": args.model,
                "model_reasoning_effort": args.model_reasoning_effort,
                "summary_model": args.summary_model or args.model,
                "summary_reasoning_effort": args.summary_reasoning_effort,
                "experience_updater": args.experience_updater,
                "experience_token_budget": args.experience_token_budget,
                "seeded_at": now_iso(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    complete_event = {
        "event": "seed_seq_self_from_bo8_round1_complete",
        "run_id": args.run_id,
        "output": str(args.output),
        "task_count": len(source_rows),
        "passed_seed": passed,
        "unpassed_seed": unpassed,
        "output_already_seeded": output_already_seeded,
        "wall_seconds": time.monotonic() - started,
        "time": now_iso(),
    }
    print(json.dumps(complete_event, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
