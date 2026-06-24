#!/usr/bin/env python3
"""Run Claude Code on CyberGym tasks and record rollout JSONL.

This mirrors the Codex runner's CyberGym workspace, submission, and verification
flow while swapping the agent invocation to `claude -p`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mastermind.config import load_manifest
from mastermind.rollout import (
    MilestoneSummary,
    RolloutRecord,
    VerificationSummary,
    append_rollout,
)
from mastermind.tasks import load_split_ids, load_task_metadata

from scripts.run_codex_cybergym_tasks import (
    RETRY_BIN_DIRNAME,
    build_prompt,
    generate_task_workspace,
    install_tool_retry_wrappers,
    load_poc_records,
    milestone_from_record,
    read_experience_context,
    read_recorded_tasks,
    read_strategy_context,
    run_command,
    safe_name,
    safe_suffix,
    submit_candidates,
    verify_agent,
)


CLAUDE_RATE_LIMIT_PATTERNS = [
    "429",
    "rate limit",
    "ratelimit",
    "rate_limit",
    "too many requests",
    "overloaded",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train_dev")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path, default=Path("runs/claude_cybergym"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/claude_cybergym/rollouts.jsonl"),
    )
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--model")
    parser.add_argument("--model-reasoning-effort")
    parser.add_argument("--claude-bin", default=None)
    parser.add_argument(
        "--codex-bin",
        default="claude",
        help="Compatibility alias used by the parallel orchestrator; treated as --claude-bin.",
    )
    parser.add_argument("--claude-timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=2400,
        help="Compatibility alias used by the parallel orchestrator.",
    )
    parser.add_argument("--server", default="http://127.0.0.1:8666")
    parser.add_argument(
        "--pocdb-path",
        type=Path,
        default=Path("runs/cybergym_server/poc.db"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--force-workspace", action="store_true")
    parser.add_argument("--rerun-recorded", action="store_true")
    parser.add_argument("--skip-missing-data", action="store_true")
    parser.add_argument("--agent-suffix", default="")
    parser.add_argument("--attempt-index", type=int)
    parser.add_argument("--experience-file", type=Path)
    parser.add_argument("--strategy-file", type=Path)
    parser.add_argument("--no-auto-submit-candidates", action="store_true")
    parser.add_argument("--max-auto-submit-candidates", type=int, default=5)
    parser.add_argument("--submit-timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--sandbox",
        default="workspace-write",
        help="Compatibility argument accepted from the shared parallel orchestrator.",
    )
    parser.add_argument(
        "--codex-provider",
        default=None,
        help="Compatibility argument accepted from the shared parallel orchestrator; ignored by Claude runner.",
    )
    parser.add_argument(
        "--codex-provider-base-url",
        default=None,
        help="Compatibility argument accepted from the shared parallel orchestrator; ignored by Claude runner.",
    )
    parser.add_argument(
        "--codex-provider-wire-api",
        default=None,
        help="Compatibility argument accepted from the shared parallel orchestrator; ignored by Claude runner.",
    )
    parser.add_argument(
        "--codex-provider-env-key",
        default=None,
        help="Compatibility argument accepted from the shared parallel orchestrator; ignored by Claude runner.",
    )
    parser.add_argument("--permission-mode", default="bypassPermissions")
    parser.add_argument(
        "--no-bare",
        action="store_true",
        help="Do not pass --bare to Claude Code, allowing normal Claude Code auth sources.",
    )
    parser.add_argument(
        "--allowed-tools",
        default="Bash,Read,Write,Edit,MultiEdit,Glob,Grep,LS",
    )
    parser.add_argument(
        "--anthropic-api-key-env-key",
        default="ANTHROPIC_API_KEY",
        help="Environment variable that contains the API key. If not ANTHROPIC_API_KEY, it is copied there.",
    )
    parser.add_argument(
        "--anthropic-base-url-env-key",
        default="ANTHROPIC_BASE_URL",
        help="Environment variable that contains the Anthropic-compatible base URL.",
    )
    parser.add_argument(
        "--claude-rate-limit-retries",
        "--codex-rate-limit-retries",
        dest="claude_rate_limit_retries",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--claude-rate-limit-stagger-seconds",
        "--codex-rate-limit-stagger-seconds",
        dest="claude_rate_limit_stagger_seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--claude-rate-limit-min-sleep-seconds",
        "--codex-rate-limit-min-sleep-seconds",
        dest="claude_rate_limit_min_sleep_seconds",
        type=float,
        default=0.0,
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def read_tail_text(path: Path, *, max_bytes: int = 200_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes), os.SEEK_SET)
        return f.read().decode(errors="replace")


def claude_retry_reason(
    *,
    returncode: int | None,
    timed_out: bool,
    stdout_path: Path,
    stderr_path: Path,
) -> str | None:
    if timed_out or returncode in {None, 0}:
        return None
    text = f"{read_tail_text(stdout_path)}\n{read_tail_text(stderr_path)}".lower()
    for pattern in CLAUDE_RATE_LIMIT_PATTERNS:
        if pattern in text:
            return pattern
    return None


def extract_claude_result(stdout_path: Path, last_message_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not stdout_path.exists():
        return result
    for line in stdout_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
    message = result.get("result")
    if isinstance(message, str):
        last_message_path.write_text(message)
    return result


def append_missing_data_record(
    *,
    output: Path,
    run_id: str,
    task_id: str,
    missing: list[str],
) -> None:
    record = RolloutRecord(
        run_id=run_id,
        task_id=task_id,
        agent=f"{run_id}-{safe_name(task_id)}",
        model="claude-code",
        executor="claude-code.print",
        status="SKIPPED",
        milestone=MilestoneSummary(
            milestone=None,
            reasoning="missing benchmark data",
        ),
        verification=VerificationSummary(
            status="missing_benchmark_data",
            passed=None,
            submit_count=0,
            details={"missing_files": missing},
        ),
        started_at=now_iso(),
        finished_at=now_iso(),
        metadata={"source": "run_claude_cybergym_tasks.py"},
    )
    append_rollout(output, record)


def prepare_claude_env(args: argparse.Namespace, env: dict[str, str]) -> dict[str, str]:
    prepared = env.copy()
    if args.no_bare:
        prepared.pop("ANTHROPIC_API_KEY", None)
        prepared.pop("ANTHROPIC_BASE_URL", None)
        return prepared
    api_key_env = args.anthropic_api_key_env_key
    base_url_env = args.anthropic_base_url_env_key
    if api_key_env != "ANTHROPIC_API_KEY" and prepared.get(api_key_env):
        prepared["ANTHROPIC_API_KEY"] = prepared[api_key_env]
    if base_url_env != "ANTHROPIC_BASE_URL" and prepared.get(base_url_env):
        prepared["ANTHROPIC_BASE_URL"] = prepared[base_url_env].removesuffix("/v1").rstrip("/")
    if not args.no_bare and not prepared.get("ANTHROPIC_API_KEY"):
        raise ValueError(f"missing {api_key_env}/ANTHROPIC_API_KEY for Claude Code")
    if not prepared.get("ANTHROPIC_BASE_URL") and prepared.get("LLM_GATEWAY_URL"):
        prepared["ANTHROPIC_BASE_URL"] = prepared["LLM_GATEWAY_URL"].removesuffix("/v1").rstrip("/")
    if not args.no_bare and not prepared.get("ANTHROPIC_BASE_URL"):
        raise ValueError(f"missing {base_url_env}/ANTHROPIC_BASE_URL for Claude Code")
    return prepared


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    manifest = load_manifest()
    metadata = load_task_metadata(manifest.benchmark.tasks_json)
    task_ids = args.task_id or load_split_ids(manifest, args.split)
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    run_id = args.run_id or f"claude-{args.split}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_root = (args.run_root / run_id).resolve()
    trajectories_dir = run_root / "trajectories"
    workspaces_dir = run_root / "workspaces"
    output = args.output.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run_config = run_root / "run_config.json"
    claude_bin = args.claude_bin or args.codex_bin
    claude_timeout_seconds = args.claude_timeout_seconds or args.codex_timeout_seconds
    if not run_config.exists():
        run_config.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "split": args.split,
                    "difficulty": args.difficulty,
                    "task_count": len(task_ids),
                    "server": args.server,
                    "pocdb_path": str(args.pocdb_path),
                    "agent": "claude-code",
                    "model": args.model,
                    "started_at": now_iso(),
                },
                indent=2,
                sort_keys=True,
            )
        )

    recorded = set() if args.rerun_recorded else read_recorded_tasks(output, run_id)
    env = os.environ.copy()
    pythonpath = [str(Path("cybergym/src").resolve()), str(Path("src").resolve())]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env = prepare_claude_env(args, env)

    exit_code = 0
    for index, task_id in enumerate(task_ids, start=1):
        if task_id in recorded:
            print(json.dumps({"event": "skip_recorded", "task_id": task_id}), flush=True)
            continue
        task = metadata[task_id]
        required = task.task_difficulty[args.difficulty]
        missing = [
            rel_path
            for rel_path in required
            if not (manifest.benchmark.root / rel_path).exists()
        ]
        if missing:
            print(
                json.dumps(
                    {
                        "event": "missing_data",
                        "task_id": task_id,
                        "missing_files": missing,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.skip_missing_data:
                append_missing_data_record(
                    output=output,
                    run_id=run_id,
                    task_id=task_id,
                    missing=missing,
                )
                continue
            return 2

        safe_task = safe_name(task_id)
        agent_suffix = safe_suffix(args.agent_suffix)
        attempt_name = f"{safe_task}{agent_suffix}"
        agent_id = f"{run_id}-{attempt_name}"
        workspace = workspaces_dir / attempt_name
        claude_stdout = trajectories_dir / f"{attempt_name}.claude.jsonl"
        claude_stderr = trajectories_dir / f"{attempt_name}.claude.stderr.log"
        last_message = trajectories_dir / f"{attempt_name}.claude.last.txt"
        experience = read_experience_context(args.experience_file)
        strategy = read_strategy_context(args.strategy_file)
        started_at = now_iso()
        print(
            json.dumps(
                {"event": "task_start", "index": index, "total": len(task_ids), "task_id": task_id},
                sort_keys=True,
            ),
            flush=True,
        )

        status = "FAILED"
        returncode: int | None = None
        timed_out = False
        wall_seconds = 0.0
        generation_error: str | None = None
        claude_retry_events: list[dict[str, Any]] = []
        claude_result: dict[str, Any] = {}
        attempt_env = env
        try:
            generate_task_workspace(
                task_id=task_id,
                agent_id=agent_id,
                workspace=workspace,
                difficulty=args.difficulty,
                server=args.server,
                force=args.force_workspace,
                env=env,
            )
            attempt_env = install_tool_retry_wrappers(workspace, env)
            prompt = build_prompt(
                task_id,
                args.difficulty,
                experience=experience,
                strategy=strategy,
                agent_budget_seconds=claude_timeout_seconds,
                submit_budget_seconds=args.submit_timeout_seconds,
            )
            max_invocations = max(1, 1 + args.claude_rate_limit_retries)
            for invocation in range(1, max_invocations + 1):
                suffix = "" if invocation == 1 else f".retry{invocation}"
                current_stdout = trajectories_dir / f"{attempt_name}.claude{suffix}.jsonl"
                current_stderr = trajectories_dir / f"{attempt_name}.claude{suffix}.stderr.log"
                current_last_message = trajectories_dir / f"{attempt_name}.claude{suffix}.last.txt"
                command = [
                    claude_bin,
                    "--model",
                    args.model or "sonnet",
                    "-p",
                    "--verbose",
                    "--output-format",
                    "stream-json",
                    "--no-session-persistence",
                    "--permission-mode",
                    args.permission_mode,
                    f"--allowedTools={args.allowed_tools}",
                ]
                if not args.no_bare:
                    command.insert(1, "--bare")
                if args.model_reasoning_effort:
                    command.extend(["--effort", args.model_reasoning_effort])
                command.append(prompt)
                current_returncode, current_timed_out, current_wall_seconds = run_command(
                    command,
                    cwd=workspace,
                    env=attempt_env,
                    stdout_path=current_stdout,
                    stderr_path=current_stderr,
                    timeout_seconds=claude_timeout_seconds,
                )
                returncode = current_returncode
                timed_out = current_timed_out
                wall_seconds += current_wall_seconds
                claude_stdout = current_stdout
                claude_stderr = current_stderr
                last_message = current_last_message
                claude_result = extract_claude_result(claude_stdout, last_message)
                retry_reason = claude_retry_reason(
                    returncode=returncode,
                    timed_out=timed_out,
                    stdout_path=claude_stdout,
                    stderr_path=claude_stderr,
                )
                if retry_reason and invocation < max_invocations:
                    sleep_seconds = max(0.0, args.claude_rate_limit_min_sleep_seconds) + random.uniform(
                        0.0,
                        max(0.0, args.claude_rate_limit_stagger_seconds),
                    )
                    claude_retry_events.append(
                        {
                            "invocation": invocation,
                            "reason": retry_reason,
                            "returncode": returncode,
                            "sleep_seconds": sleep_seconds,
                            "stdout": str(claude_stdout),
                            "stderr": str(claude_stderr),
                        }
                    )
                    time.sleep(sleep_seconds)
                    continue
                break
            is_error = bool(claude_result.get("is_error")) if claude_result else False
            status = "TIMEOUT" if timed_out else ("OK" if returncode == 0 and not is_error else "AGENT_ERROR")
        except Exception as exc:  # noqa: BLE001 - convert per-task errors to rollout rows.
            generation_error = repr(exc)
            status = "AGENT_ERROR"
            exit_code = 1

        auto_submit_results: list[dict[str, Any]] = []
        records = load_poc_records(args.pocdb_path, agent_id, task_id)
        if not records and not args.no_auto_submit_candidates:
            auto_submit_results = submit_candidates(
                workspace=workspace,
                task_safe_name=safe_task,
                agent_id=agent_id,
                task_id=task_id,
                db_path=args.pocdb_path,
                server=args.server,
                env=attempt_env,
                timeout_seconds=args.submit_timeout_seconds,
                max_candidates=args.max_auto_submit_candidates,
                logs_dir=run_root / "submit_logs",
            )

        verify_result = verify_agent(args.server, agent_id)
        records = load_poc_records(args.pocdb_path, agent_id, task_id)
        latest = records[0] if records else None
        milestone, reasoning, passed = milestone_from_record(latest)
        if milestone == 7:
            status = "PASSED"
        elif status == "OK":
            status = "FAILED"

        details: dict[str, Any] = {
            "agent_id": agent_id,
            "workspace": str(workspace),
            "claude_stdout": str(claude_stdout),
            "claude_stderr": str(claude_stderr),
            "last_message": str(last_message),
            "claude_returncode": returncode,
            "claude_timed_out": timed_out,
            "verify_result": verify_result,
            "auto_submit_results": auto_submit_results,
            "source": "run_claude_cybergym_tasks.py",
            "configured_model": args.model,
            "model_reasoning_effort": args.model_reasoning_effort,
            "agent_suffix": agent_suffix,
            "attempt_index": args.attempt_index,
            "experience_file": str(args.experience_file) if args.experience_file else None,
            "experience_chars": len(experience),
            "strategy_file": str(args.strategy_file) if args.strategy_file else None,
            "strategy_chars": len(strategy),
            "tool_retry_bin": str(workspace / RETRY_BIN_DIRNAME),
            "claude_result": claude_result,
        }
        # Compatibility with existing summarizers that look for codex_* fields.
        details["codex_stdout"] = details["claude_stdout"]
        details["codex_stderr"] = details["claude_stderr"]
        if claude_retry_events:
            details["claude_retry_events"] = claude_retry_events
        if latest is not None:
            details.update(
                {
                    "poc_id": latest.poc_id,
                    "poc_hash": latest.poc_hash,
                    "poc_length": latest.poc_length,
                    "vul_exit_code": latest.vul_exit_code,
                    "fix_exit_code": latest.fix_exit_code,
                }
            )
        if generation_error:
            details["generation_error"] = generation_error

        rollout = RolloutRecord(
            run_id=run_id,
            task_id=task_id,
            agent=agent_id,
            model=args.model or "claude-code-default",
            executor="claude-code.print",
            status=status,
            milestone=MilestoneSummary(
                milestone=milestone,
                reasoning=reasoning,
                verified_fix=passed if milestone == 7 else None,
            ),
            verification=VerificationSummary(
                status="verified" if latest and latest.fix_exit_code is not None else "partial",
                passed=passed,
                submit_count=len(records),
                details=details,
            ),
            trajectory_path=str(claude_stdout),
            wall_seconds=wall_seconds,
            strategy_id=f"strategy_file:{args.strategy_file.stem}" if args.strategy_file else None,
            strategy=strategy if strategy else None,
            started_at=started_at,
            finished_at=now_iso(),
            metadata=details,
        )
        append_rollout(output, rollout)
        print(rollout.to_json(), flush=True)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
