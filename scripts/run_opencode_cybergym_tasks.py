#!/usr/bin/env python3
"""Run OpenCode on CyberGym tasks and record rollout JSONL."""

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


OPENCODE_RETRYABLE_ERROR_STATUSES = {429, 500, 502, 503, 504, 529}
OPENCODE_RETRYABLE_ERROR_PATTERNS = [
    "backend request failed",
    "prefill stall",
    "no data from backend",
    "rate limit",
    "ratelimit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "temporarily unavailable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train_dev")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path, default=Path("runs/opencode_cybergym"))
    parser.add_argument("--output", type=Path, default=Path("runs/opencode_cybergym/rollouts.jsonl"))
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--model", default="qwen3_6_mingzhe")
    parser.add_argument("--model-reasoning-effort")
    parser.add_argument("--opencode-provider", default="qwen36_mingzhe")
    parser.add_argument(
        "--opencode-provider-base-url",
        default=os.environ.get("LITELLM_BASE_URL", "http://litellm.tiktok-row.net/v1"),
    )
    parser.add_argument("--opencode-provider-env-key", default="LITELLM_API_KEY")
    parser.add_argument(
        "--opencode-context-limit",
        type=int,
        default=int(os.environ.get("OPENCODE_CONTEXT_LIMIT", "1000000")),
        help="Context window advertised to OpenCode for this provider/model.",
    )
    parser.add_argument(
        "--opencode-output-token-max",
        type=int,
        default=int(os.environ.get("OPENCODE_OUTPUT_TOKEN_MAX", "20000")),
        help="Maximum output tokens advertised/requested through OpenCode.",
    )
    parser.add_argument("--opencode-bin", default=None)
    parser.add_argument(
        "--codex-bin",
        default="opencode",
        help="Compatibility alias used by the parallel orchestrator; treated as --opencode-bin.",
    )
    parser.add_argument("--opencode-timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=900,
        help="Compatibility alias used by the shared parallel orchestrator.",
    )
    parser.add_argument("--server", default="http://127.0.0.1:8666")
    parser.add_argument("--pocdb-path", type=Path, default=Path("runs/cybergym_server/poc.db"))
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
    parser.add_argument("--sandbox", default="workspace-write")
    parser.add_argument(
        "--opencode-rate-limit-retries",
        "--codex-rate-limit-retries",
        dest="opencode_rate_limit_retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--opencode-rate-limit-stagger-seconds",
        "--codex-rate-limit-stagger-seconds",
        dest="opencode_rate_limit_stagger_seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--opencode-rate-limit-min-sleep-seconds",
        "--codex-rate-limit-min-sleep-seconds",
        dest="opencode_rate_limit_min_sleep_seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument("--anthropic-api-key-env-key", default=None)
    parser.add_argument("--anthropic-base-url-env-key", default=None)
    parser.add_argument("--codex-provider-wire-api", default=None)
    parser.add_argument("--codex-provider-env-key", default=None)
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


def coerce_status_code(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def read_opencode_api_errors(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    errors: list[dict[str, Any]] = []
    with path.open(errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "error":
                continue
            error = event.get("error") or {}
            data = error.get("data") if isinstance(error, dict) else {}
            data = data if isinstance(data, dict) else {}
            response_headers = data.get("responseHeaders") if isinstance(data.get("responseHeaders"), dict) else {}
            message_parts = [
                str(data.get("message") or ""),
                str(data.get("responseBody") or ""),
                str(response_headers.get("x-error-message") or ""),
            ]
            errors.append(
                {
                    "name": error.get("name") if isinstance(error, dict) else None,
                    "status_code": coerce_status_code(data.get("statusCode")),
                    "is_retryable": data.get("isRetryable"),
                    "message": "\n".join(part for part in message_parts if part),
                }
            )
    return errors


def latest_opencode_api_error(path: Path) -> dict[str, Any] | None:
    errors = read_opencode_api_errors(path)
    return errors[-1] if errors else None


def opencode_retry_reason(
    *,
    returncode: int | None,
    timed_out: bool,
    stdout_path: Path,
    stderr_path: Path,
) -> str | None:
    if timed_out or returncode in {None, 0}:
        return None
    api_error = latest_opencode_api_error(stdout_path)
    if api_error:
        status_code = api_error.get("status_code")
        message = str(api_error.get("message") or "").lower()
        if status_code in OPENCODE_RETRYABLE_ERROR_STATUSES or any(
            pattern in message for pattern in OPENCODE_RETRYABLE_ERROR_PATTERNS
        ):
            return f"api_error_{status_code}" if status_code is not None else "api_error"
        return None

    text = f"{read_tail_text(stdout_path)}\n{read_tail_text(stderr_path)}".lower()
    for pattern in OPENCODE_RETRYABLE_ERROR_PATTERNS:
        if pattern in text:
            return pattern
    return None


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
        model="opencode",
        executor="opencode.run",
        status="SKIPPED",
        milestone=MilestoneSummary(milestone=None, reasoning="missing benchmark data"),
        verification=VerificationSummary(
            status="missing_benchmark_data",
            passed=None,
            submit_count=0,
            details={"missing_files": missing},
        ),
        started_at=now_iso(),
        finished_at=now_iso(),
        metadata={"source": "run_opencode_cybergym_tasks.py"},
    )
    append_rollout(output, record)


def configured_model(provider: str, model: str | None) -> str:
    if not model:
        return f"{provider}/qwen3_6_mingzhe"
    if "/" in model:
        return model
    return f"{provider}/{model}"


def opencode_config_content(args: argparse.Namespace) -> str:
    model_name = args.model.split("/", 1)[1] if args.model and "/" in args.model else args.model
    model_name = model_name or "qwen3_6_mingzhe"
    return json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                args.opencode_provider: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": args.opencode_provider,
                    "options": {
                        "baseURL": args.opencode_provider_base_url,
                        "apiKey": f"{{env:{args.opencode_provider_env_key}}}",
                    },
                    "models": {
                        model_name: {
                            "name": model_name,
                            "limit": {
                                "context": args.opencode_context_limit,
                                "output": args.opencode_output_token_max,
                            },
                        }
                    },
                }
            },
            "model": configured_model(args.opencode_provider, args.model),
            "small_model": configured_model(args.opencode_provider, args.model),
        },
        sort_keys=True,
    )


def prepare_opencode_env(args: argparse.Namespace, env: dict[str, str]) -> dict[str, str]:
    prepared = env.copy()
    key_env = args.opencode_provider_env_key
    if not prepared.get(key_env) and prepared.get("QWEN36_API_KEY"):
        prepared[key_env] = prepared["QWEN36_API_KEY"]
    if not prepared.get(key_env):
        raise ValueError(f"missing {key_env} for OpenCode")
    prepared["OPENCODE_CONFIG_CONTENT"] = opencode_config_content(args)
    prepared.setdefault("OPENCODE_DISABLE_AUTOUPDATE", "true")
    prepared.setdefault("OPENCODE_DISABLE_PRUNE", "true")
    prepared.setdefault("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", str(args.opencode_output_token_max))
    return prepared


def isolate_opencode_env(env: dict[str, str], workspace: Path) -> dict[str, str]:
    isolated = env.copy()
    opencode_home = workspace / ".opencode_home"
    isolated["XDG_DATA_HOME"] = str(opencode_home / "data")
    isolated["XDG_STATE_HOME"] = str(opencode_home / "state")
    isolated["XDG_CACHE_HOME"] = str(opencode_home / "cache")
    return isolated


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    manifest = load_manifest()
    metadata = load_task_metadata(manifest.benchmark.tasks_json)
    task_ids = args.task_id or load_split_ids(manifest, args.split)
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    run_id = args.run_id or f"opencode-{args.split}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_root = (args.run_root / run_id).resolve()
    trajectories_dir = run_root / "trajectories"
    workspaces_dir = run_root / "workspaces"
    output = args.output.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    opencode_bin = args.opencode_bin or args.codex_bin
    opencode_bin_path = Path(opencode_bin)
    if not opencode_bin_path.is_absolute() and (ROOT / opencode_bin_path).exists():
        opencode_bin = str((ROOT / opencode_bin_path).resolve())
    opencode_timeout_seconds = args.opencode_timeout_seconds or args.codex_timeout_seconds
    model = configured_model(args.opencode_provider, args.model)

    run_config = run_root / "run_config.json"
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
                    "agent": "opencode",
                    "model": model,
                    "provider": args.opencode_provider,
                    "provider_base_url": args.opencode_provider_base_url,
                    "opencode_context_limit": args.opencode_context_limit,
                    "opencode_output_token_max": args.opencode_output_token_max,
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
    env = prepare_opencode_env(args, env)

    exit_code = 0
    for index, task_id in enumerate(task_ids, start=1):
        if task_id in recorded:
            print(json.dumps({"event": "skip_recorded", "task_id": task_id}), flush=True)
            continue
        task = metadata[task_id]
        required = task.task_difficulty[args.difficulty]
        missing = [rel_path for rel_path in required if not (manifest.benchmark.root / rel_path).exists()]
        if missing:
            print(
                json.dumps(
                    {"event": "missing_data", "task_id": task_id, "missing_files": missing},
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.skip_missing_data:
                append_missing_data_record(output=output, run_id=run_id, task_id=task_id, missing=missing)
                continue
            return 2

        safe_task = safe_name(task_id)
        agent_suffix = safe_suffix(args.agent_suffix)
        attempt_name = f"{safe_task}{agent_suffix}"
        agent_id = f"{run_id}-{attempt_name}"
        workspace = workspaces_dir / attempt_name
        opencode_stdout = trajectories_dir / f"{attempt_name}.opencode.jsonl"
        opencode_stderr = trajectories_dir / f"{attempt_name}.opencode.stderr.log"
        last_message = trajectories_dir / f"{attempt_name}.opencode.last.txt"
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
        opencode_retry_events: list[dict[str, Any]] = []
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
            attempt_env = isolate_opencode_env(attempt_env, workspace)
            prompt = build_prompt(
                task_id,
                args.difficulty,
                experience=experience,
                strategy=strategy,
                agent_budget_seconds=opencode_timeout_seconds,
                submit_budget_seconds=args.submit_timeout_seconds,
            )
            max_invocations = max(1, 1 + args.opencode_rate_limit_retries)
            for invocation in range(1, max_invocations + 1):
                suffix = "" if invocation == 1 else f".retry{invocation}"
                current_stdout = trajectories_dir / f"{attempt_name}.opencode{suffix}.jsonl"
                current_stderr = trajectories_dir / f"{attempt_name}.opencode{suffix}.stderr.log"
                command = [
                    opencode_bin,
                    "run",
                    "--format",
                    "json",
                    "--dir",
                    str(workspace),
                    "--model",
                    model,
                    "--title",
                    task_id,
                ]
                if args.model_reasoning_effort:
                    command.extend(["--variant", args.model_reasoning_effort])
                command.extend(["--dangerously-skip-permissions", prompt])
                current_returncode, current_timed_out, current_wall_seconds = run_command(
                    command,
                    cwd=workspace,
                    env=attempt_env,
                    stdout_path=current_stdout,
                    stderr_path=current_stderr,
                    timeout_seconds=opencode_timeout_seconds,
                )
                returncode = current_returncode
                timed_out = current_timed_out
                wall_seconds += current_wall_seconds
                opencode_stdout = current_stdout
                opencode_stderr = current_stderr
                retry_reason = opencode_retry_reason(
                    returncode=returncode,
                    timed_out=timed_out,
                    stdout_path=opencode_stdout,
                    stderr_path=opencode_stderr,
                )
                if retry_reason and invocation < max_invocations:
                    sleep_seconds = max(0.0, args.opencode_rate_limit_min_sleep_seconds) + random.uniform(
                        0.0,
                        max(0.0, args.opencode_rate_limit_stagger_seconds),
                    )
                    opencode_retry_events.append(
                        {
                            "invocation": invocation,
                            "reason": retry_reason,
                            "returncode": returncode,
                            "sleep_seconds": sleep_seconds,
                            "stdout": str(opencode_stdout),
                            "stderr": str(opencode_stderr),
                        }
                    )
                    time.sleep(sleep_seconds)
                    continue
                break
            if last_message.exists():
                last_message.unlink()
            if opencode_stdout.exists():
                last_message.write_text(read_tail_text(opencode_stdout, max_bytes=10_000))
            status = "TIMEOUT" if timed_out else ("OK" if returncode == 0 else "AGENT_ERROR")
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

        api_error = latest_opencode_api_error(opencode_stdout)
        details: dict[str, Any] = {
            "agent_id": agent_id,
            "workspace": str(workspace),
            "opencode_stdout": str(opencode_stdout),
            "opencode_stderr": str(opencode_stderr),
            "last_message": str(last_message),
            "opencode_returncode": returncode,
            "opencode_timed_out": timed_out,
            "verify_result": verify_result,
            "auto_submit_results": auto_submit_results,
            "source": "run_opencode_cybergym_tasks.py",
            "configured_model": model,
            "model_reasoning_effort": args.model_reasoning_effort,
            "agent_suffix": agent_suffix,
            "attempt_index": args.attempt_index,
            "experience_file": str(args.experience_file) if args.experience_file else None,
            "experience_chars": len(experience),
            "strategy_file": str(args.strategy_file) if args.strategy_file else None,
            "strategy_chars": len(strategy),
            "tool_retry_bin": str(workspace / RETRY_BIN_DIRNAME),
        }
        details["codex_stdout"] = details["opencode_stdout"]
        details["codex_stderr"] = details["opencode_stderr"]
        if api_error:
            details["opencode_api_error_status"] = api_error.get("status_code")
            details["opencode_api_error_message"] = api_error.get("message")
            details["opencode_api_error_retryable"] = api_error.get("is_retryable")
            details["api_error_status"] = api_error.get("status_code")
            details["api_error_message"] = api_error.get("message")
        if opencode_retry_events:
            details["opencode_retry_events"] = opencode_retry_events
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
            model=model,
            executor="opencode.run",
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
            trajectory_path=str(opencode_stdout),
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
