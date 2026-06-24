#!/usr/bin/env python3
"""Wait for training experiments to finish, then stage and upload to HF."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "runs/codex_cybergym/sequential_self_logs"
DATASET_NAME = "cybergym-codex-gpt-5-5-iterative-improvement-train"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def read_json_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open(errors="replace") as f:
        for line in f:
            if not line.strip().startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def latest_done(events: list[dict[str, Any]], target: set[str] | None) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    for event in events:
        task_id = event.get("task_id")
        if event.get("event") == "sequential_task_done" and isinstance(task_id, str):
            if target is None or task_id in target:
                done[task_id] = event
    return done


def active_attempts(events: list[dict[str, Any]], target: set[str] | None) -> list[tuple[str, int]]:
    start_indices = [index for index, event in enumerate(events) if event.get("event") == "sequential_start"]
    segment = events[start_indices[-1] :] if start_indices else events
    starts: Counter[tuple[str, int]] = Counter()
    dones: Counter[tuple[str, int]] = Counter()
    for event in segment:
        task_id = event.get("task_id")
        attempt = event.get("attempt")
        if not isinstance(task_id, str) or not isinstance(attempt, int):
            continue
        if target is not None and task_id not in target:
            continue
        key = (task_id, attempt)
        if event.get("event") == "sequential_attempt_start":
            starts[key] += 1
        if event.get("event") in {"sequential_attempt_done", "sequential_attempt_missing_row"}:
            dones[key] += 1
    active: list[tuple[str, int]] = []
    for key, count in starts.items():
        active.extend([key] * max(0, count - dones.get(key, 0)))
    return active


def load_train_ids() -> list[str]:
    path = ROOT / "cybergym/TASKS_TRAIN"
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def run_status(name: str, run_id: str, expected: int, target: set[str] | None) -> dict[str, Any]:
    log_path = LOG_DIR / f"{run_id}.log"
    events = read_json_events(log_path)
    done = latest_done(events, target)
    active = active_attempts(events, target)
    complete = len(done) >= expected
    return {
        "name": name,
        "run_id": run_id,
        "expected": expected,
        "completed": len(done),
        "not_done": max(0, expected - len(done)),
        "active": [{"task_id": task_id, "attempt": attempt} for task_id, attempt in active],
        "completed_status": dict(Counter(str(event.get("status") or "UNKNOWN") for event in done.values())),
        "complete": complete,
        "log_path": str(log_path),
        "log_mtime": datetime.fromtimestamp(log_path.stat().st_mtime, UTC).isoformat() if log_path.exists() else None,
    }


def api429_run_id() -> str | None:
    latest = LOG_DIR / "api429_rerun5.latest"
    if latest.exists():
        return latest.read_text().strip()
    return None


def all_statuses() -> list[dict[str, Any]]:
    train_ids = load_train_ids()
    statuses = [
        run_status(
            "train100",
            "codex-gateway-train-seq-self-100-20260531T1640Z",
            100,
            set(train_ids[:100]),
        ),
        run_status(
            "remaining201",
            "codex-gateway-train-seq-self-remaining201-20260601T0445Z",
            201,
            set(train_ids[100:301]),
        ),
        run_status(
            "infra_rerun_escalated",
            "codex-gateway-train-infra-rerun-escalated-20260601T0856Z",
            20,
            None,
        ),
    ]
    rid = api429_run_id()
    if rid:
        statuses.append(
            run_status(
                "api429_rerun5",
                rid,
                5,
                {"arvo:11945", "arvo:14560", "arvo:14574", "arvo:59070", "arvo:61617"},
            )
        )
    return statuses


def run_command(command: list[str], *, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "command_start", "command": command, "time": now_iso()}), flush=True)
    with log_file.open("a") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)
    print(json.dumps({"event": "command_complete", "command": command, "time": now_iso()}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="0 means no timeout.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--namespace", default="Elfsong")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--upload-method", choices=("large-folder", "folder"), default="large-folder")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    orchestration_log = LOG_DIR / f"{args.dataset_name}.wait_stage_upload.log"
    while True:
        statuses = all_statuses()
        payload = {"event": "poll", "time": now_iso(), "statuses": statuses}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        with orchestration_log.open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        if all(status["complete"] for status in statuses):
            break
        if args.timeout_seconds and time.monotonic() - started > args.timeout_seconds:
            raise TimeoutError("Timed out waiting for experiments to finish")
        time.sleep(args.poll_seconds)

    stage_log = LOG_DIR / f"{args.dataset_name}.stage.log"
    upload_log = LOG_DIR / f"{args.dataset_name}.upload.log"
    run_command(
        [
            sys.executable,
            "scripts/stage_hf_training_dataset.py",
            "--clean",
            "--dataset-name",
            args.dataset_name,
            "--repo-id",
            f"{args.namespace}/{args.dataset_name}",
        ],
        log_file=stage_log,
    )
    if not args.skip_upload:
        command = [
            sys.executable,
            "scripts/upload_hf_training_dataset.py",
            "--execute",
            "--dataset-name",
            args.dataset_name,
            "--namespace",
            args.namespace,
            "--method",
            args.upload_method,
            "--num-workers",
            str(args.num_workers),
        ]
        if args.public:
            command.append("--public")
        run_command(command, log_file=upload_log)
    print(json.dumps({"event": "all_done", "time": now_iso(), "dataset_name": args.dataset_name}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
