#!/usr/bin/env python3
"""Self-test the strategy SFT eval reporting utilities with synthetic artifacts.

This test is intentionally offline: it does not call Codex, Tinker, CyberGym, or
network APIs.  It verifies the finality gate and infra-row filtering semantics
used by the final report tooling.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def rollout_row(task_id: str, status: str, milestone: int, attempt: int, **metadata_extra) -> dict:
    metadata = {
        "attempt_index": attempt,
        "sequential": {"attempt_index": attempt},
    }
    metadata.update(metadata_extra)
    return {
        "task_id": task_id,
        "status": status,
        "milestone": {"milestone": milestone},
        "metadata": metadata,
        "finished_at": f"2026-01-01T00:00:{attempt:02d}Z",
    }


def write_eval_log(path: Path, *, run_id: str, completed: bool) -> None:
    events = [
        {
            "event": "seq_self_eval100_tinker_qwen36_matched_start",
            "run_id": run_id,
            "workers": 2,
            "max_tasks": 2,
            "max_attempts": 8,
        },
        {
            "event": "sequential_start",
            "run_id": run_id,
            "task_count": 2,
            "pending": 2,
            "workers": 2,
        },
    ]
    if completed:
        events.extend(
            [
                {
                    "event": "sequential_complete",
                    "run_id": run_id,
                    "tasks_completed": 2,
                    "attempts_run": 3,
                    "infra_failures": 0,
                    "exit_code": 0,
                },
                {
                    "event": "seq_self_eval100_tinker_qwen36_matched_complete",
                    "run_id": run_id,
                    "exit_code": 0,
                },
            ]
        )
    write_jsonl(path, events)


def write_sft_artifacts(sft_dir: Path) -> Path:
    sft_dir.mkdir(parents=True)
    (sft_dir / "result.json").write_text(
        json.dumps(
            {
                "event": "complete",
                "run_id": "synthetic-sft",
                "step": 2,
                "final_sampler_path": "tinker://synthetic/final-sampler",
            }
        )
    )
    write_jsonl(
        sft_dir / "metrics.jsonl",
        [
            {"event": "eval", "split": "val", "step": 0, "loss_mean_unweighted": 10.0, "rows": 2},
            {"event": "eval", "split": "val", "step": 2, "loss_mean_unweighted": 7.0, "rows": 2},
        ],
    )
    offline = sft_dir / "offline_eval.json"
    offline.write_text(
        json.dumps(
            {
                "rows": 2,
                "summaries": [
                    {"client": "base", "rows": 2, "target_nll_mean": 1.0, "target_nll_median": 1.0},
                    {"client": "sft", "rows": 2, "target_nll_mean": 0.7, "target_nll_median": 0.7},
                ],
            }
        )
    )
    return offline


def run(cmd: list[str], *, expect_code: int = 0) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != expect_code:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise AssertionError(f"Expected exit {expect_code}, got {proc.returncode}: {' '.join(cmd)}")
    return proc


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="strategy-eval-tools-") as td:
        tmp = Path(td)
        stdout_429 = tmp / "codex_429.jsonl"
        stdout_429.write_text('{"type":"error","message":"429 Too Many Requests"}\n')

        base = tmp / "base.jsonl"
        sft = tmp / "sft.jsonl"
        write_jsonl(
            base,
            [
                rollout_row("task-a", "PASSED", 7, 1),
                rollout_row("task-b", "FAILED", 4, 8),
                rollout_row("task-infra", "CRASH", 0, 1, codex_stdout=str(stdout_429)),
            ],
        )
        write_jsonl(
            sft,
            [
                rollout_row("task-a", "PASSED", 7, 1),
                rollout_row("task-b", "PASSED", 7, 2),
            ],
        )
        base_log = tmp / "base.log"
        sft_log = tmp / "sft.log"
        sft_incomplete_log = tmp / "sft_incomplete.log"
        write_eval_log(base_log, run_id="base", completed=True)
        write_eval_log(sft_log, run_id="sft", completed=True)
        write_eval_log(sft_incomplete_log, run_id="sft", completed=False)
        offline = write_sft_artifacts(tmp / "sft_artifacts")

        filter_proc = run(
            [
                sys.executable,
                "scripts/filter_cybergym_infra_rows.py",
                str(base),
                "--dry-run",
            ]
        )
        filter_summary = json.loads(filter_proc.stdout)
        assert filter_summary["infra_rows_removed"] == 1, filter_summary

        report = tmp / "report.md"
        report_json = tmp / "report.json"
        proc = run(
            [
                sys.executable,
                "scripts/report_strategy_sft_eval.py",
                "--sft-dir",
                str(tmp / "sft_artifacts"),
                "--offline-eval",
                str(offline),
                "--base-rollouts",
                str(base),
                "--sft-rollouts",
                str(sft),
                "--base-log",
                str(base_log),
                "--sft-log",
                str(sft_log),
                "--max-tasks",
                "2",
                "--max-attempts",
                "8",
                "--output-md",
                str(report),
                "--output-json",
                str(report_json),
                "--require-final",
            ]
        )
        result = json.loads(proc.stdout)
        assert result["ready_for_final_conclusion"] is True, result
        payload = json.loads(report_json.read_text())
        delta = payload["end_to_end"]["matched_all_common"]["pass_rate_delta_sft_minus_base"]
        assert abs(delta - 0.5) < 1e-9, delta
        assert payload["conclusion"]["end_to_end_status"] == "final", payload["conclusion"]
        assert payload["conclusion"]["end_to_end_direction"] == "sft_higher", payload["conclusion"]
        assert payload["conclusion"]["offline_nll_improved"] is True, payload["conclusion"]
        assert payload["conclusion"]["training_loss_decreased"] is True, payload["conclusion"]

        proc = run(
            [
                sys.executable,
                "scripts/report_strategy_sft_eval.py",
                "--sft-dir",
                str(tmp / "sft_artifacts"),
                "--offline-eval",
                str(offline),
                "--base-rollouts",
                str(base),
                "--sft-rollouts",
                str(sft),
                "--base-log",
                str(base_log),
                "--sft-log",
                str(sft_incomplete_log),
                "--max-tasks",
                "2",
                "--max-attempts",
                "8",
                "--output-md",
                str(tmp / "report_incomplete.md"),
                "--output-json",
                str(tmp / "report_incomplete.json"),
                "--require-final",
            ],
            expect_code=2,
        )
        result = json.loads(proc.stdout)
        assert result["ready_for_final_conclusion"] is False, result
        assert result["finality_blockers"], result
        incomplete_payload = json.loads((tmp / "report_incomplete.json").read_text())
        assert incomplete_payload["conclusion"]["end_to_end_status"] == "pending", incomplete_payload["conclusion"]
        assert incomplete_payload["conclusion"]["end_to_end_direction"] == "pending", incomplete_payload["conclusion"]

        # Also cover the opposite final direction: SFT lower than base.
        sft_lower = tmp / "sft_lower.jsonl"
        write_jsonl(
            sft_lower,
            [
                rollout_row("task-a", "FAILED", 4, 8),
                rollout_row("task-b", "FAILED", 4, 8),
            ],
        )
        proc = run(
            [
                sys.executable,
                "scripts/report_strategy_sft_eval.py",
                "--sft-dir",
                str(tmp / "sft_artifacts"),
                "--offline-eval",
                str(offline),
                "--base-rollouts",
                str(base),
                "--sft-rollouts",
                str(sft_lower),
                "--base-log",
                str(base_log),
                "--sft-log",
                str(sft_log),
                "--max-tasks",
                "2",
                "--max-attempts",
                "8",
                "--output-md",
                str(tmp / "report_sft_lower.md"),
                "--output-json",
                str(tmp / "report_sft_lower.json"),
                "--require-final",
            ]
        )
        result = json.loads(proc.stdout)
        assert result["ready_for_final_conclusion"] is True, result
        lower_payload = json.loads((tmp / "report_sft_lower.json").read_text())
        assert lower_payload["conclusion"]["end_to_end_status"] == "final", lower_payload["conclusion"]
        assert lower_payload["conclusion"]["end_to_end_direction"] == "sft_lower", lower_payload["conclusion"]

    print("strategy eval reporting self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
