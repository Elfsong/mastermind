#!/usr/bin/env python3
"""Sequential self-involving CyberGym runner.

For each task, this runner alternates between:
1. a Codex attempt to create/submit a PoC;
2. feedback extraction from the rollout and submit results;
3. a Codex-authored task-local experience update capped by a token budget;
4. the next attempt with that updated experience injected into the prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mastermind.config import load_manifest
from mastermind.rollout import RolloutRecord, append_rollout
from mastermind.tasks import load_split_ids


SOURCE = "run_codex_cybergym_sequential_self_involving.py"
TINKER_SAMPLER_LOCK = threading.Lock()
TINKER_SAMPLER_CACHE: dict[str, tuple[Any, Any]] = {}
AGENT_ERROR_STATUS = "AGENT_ERROR"
AGENT_ERROR_STATUSES = {AGENT_ERROR_STATUS, "CRASH"}  # CRASH is legacy rollout data.
INFRA_FAILURE_MARKERS = (
    "you've hit your usage limit",
    "usage limit",
    "quota exceeded",
    "daily limitation",
    "daily limit",
    "rate limited",
    "rate limit",
    "ratelimit",
    "rate_limit",
    "429",
    "too many requests",
    "overloaded",
    "temporarily unavailable",
    "backend request failed",
    "prefill stall",
    "no data from backend",
    "service unavailable",
    "resource exhausted",
    "payment required",
    "included credits",
    "pre-paid credits",
    "insufficient quota",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="eval")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/codex_cybergym"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/codex_gateway_eval_seq_self_involving_rollouts.jsonl"),
    )
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--no-stop-on-pass", action="store_true")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--model-reasoning-effort", default="medium")
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--summary-reasoning-effort", default="medium")
    parser.add_argument(
        "--experience-updater",
        choices=("codex", "claude", "openai", "tinker"),
        default="codex",
        help="Backend used to update the task-local experience after each attempt.",
    )
    parser.add_argument(
        "--tinker-summary-model",
        default=None,
        help="Tinker base model for --experience-updater=tinker.",
    )
    parser.add_argument(
        "--tinker-summary-model-path",
        default=None,
        help=(
            "Optional Tinker checkpoint/sampler path for --experience-updater=tinker. "
            "If set, sampling uses this fine-tuned model instead of --tinker-summary-model."
        ),
    )
    parser.add_argument("--tinker-summary-temperature", type=float, default=0.2)
    parser.add_argument("--tinker-summary-top-p", type=float, default=0.9)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--summary-bin",
        default=None,
        help="Optional binary for experience updates. Defaults to --codex-bin.",
    )
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)
    parser.add_argument("--summary-timeout-seconds", type=int, default=600)
    parser.add_argument("--submit-timeout-seconds", type=int, default=1800)
    parser.add_argument("--server", default="http://127.0.0.1:8666")
    parser.add_argument("--pocdb-path", type=Path, default=Path("runs/cybergym_server/poc.db"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--sandbox", default="workspace-write")
    parser.add_argument("--experience-token-budget", type=int, default=2048)
    parser.add_argument(
        "--experience-max-chars",
        type=int,
        default=9000,
        help="Hard local size guard for the experience text; Codex is still asked for the token budget.",
    )
    parser.add_argument("--feedback-max-chars", type=int, default=9000)
    parser.add_argument("--max-auto-submit-candidates", type=int, default=5)
    parser.add_argument(
        "--tasks-json",
        type=Path,
        default=Path(os.environ.get("CYBERGYM_TASKS_JSON", "runs/cybergym_assets/cybergym_data/tasks.json")),
        help="Path to tasks.json for injecting task context into the experience update prompt.",
    )
    parser.add_argument(
        "--task-context-data-dir",
        type=Path,
        default=Path(os.environ.get("CYBERGYM_DATA_DIR", "runs/cybergym_assets/cybergym_data/data")),
        help="Root data dir for reading description.txt and error.txt per task.",
    )
    parser.add_argument(
        "--task-context-max-chars",
        type=int,
        default=1500,
        help="Max chars for each of description.txt and error.txt in the task context block.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_codex_cybergym_tasks.py"),
    )
    parser.add_argument("--force-workspace", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument(
        "--codex-provider",
        default=None,
        help=(
            "If set, route Codex (both attempts and experience updates) through a custom "
            "model_provider of this id via -c flags (e.g. 'llmgw') instead of ChatGPT auth."
        ),
    )
    parser.add_argument(
        "--codex-provider-base-url",
        default=None,
        help="Base URL for --codex-provider (defaults to $LLM_GATEWAY_URL).",
    )
    parser.add_argument("--codex-provider-wire-api", default="responses")
    parser.add_argument("--codex-provider-env-key", default="LLM_GATEWAY_API_KEY")
    parser.add_argument("--opencode-provider", default=None)
    parser.add_argument("--opencode-provider-base-url", default=None)
    parser.add_argument("--opencode-provider-env-key", default="LITELLM_API_KEY")
    parser.add_argument("--opencode-context-limit", type=int)
    parser.add_argument("--opencode-output-token-max", type=int)
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI-compatible base URL used when --experience-updater=openai.",
    )
    parser.add_argument(
        "--openai-api-key-env-key",
        default="OPENAI_API_KEY",
        help="Environment variable holding the API key for --experience-updater=openai.",
    )
    parser.add_argument(
        "--openai-summary-max-tokens",
        type=int,
        default=None,
        help="max_tokens for OpenAI-compatible experience updates.",
    )
    parser.add_argument("--codex-rate-limit-retries", type=int, default=3)
    parser.add_argument("--codex-rate-limit-stagger-seconds", type=float, default=5.0)
    parser.add_argument("--codex-rate-limit-min-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-consecutive-infra-failures", type=int, default=5)
    parser.add_argument("--infra-failure-pause-seconds", type=int, default=1800)
    parser.add_argument(
        "--anthropic-api-key-env-key",
        default="ANTHROPIC_API_KEY",
        help="Environment variable to copy into ANTHROPIC_API_KEY for Claude-based runners/updaters.",
    )
    parser.add_argument(
        "--anthropic-base-url-env-key",
        default="ANTHROPIC_BASE_URL",
        help="Environment variable to copy into ANTHROPIC_BASE_URL for Claude-based runners/updaters.",
    )
    parser.add_argument(
        "--attempt-cooldown-seconds",
        type=float,
        default=0.0,
        help="Sleep between sequential attempts for the same task to avoid executor rate limits.",
    )
    parser.add_argument(
        "--codex-launch-stagger-seconds",
        type=float,
        default=0.0,
        help="Globally stagger Codex attempt launches across workers to avoid executor request bursts.",
    )
    return parser.parse_args()


def codex_provider_flags(args: argparse.Namespace) -> list[str]:
    """Build `codex exec -c ...` flags routing through a custom model_provider."""
    provider = args.codex_provider
    if not provider:
        return []
    base_url = args.codex_provider_base_url or os.getenv("LLM_GATEWAY_URL")
    if not base_url:
        raise ValueError(
            "--codex-provider was set but no base URL is available "
            "(pass --codex-provider-base-url or export LLM_GATEWAY_URL)."
        )
    flags: list[str] = []

    def add(key: str, value: str) -> None:
        flags.extend(["-c", f'{key}="{value}"'])

    add("model_provider", provider)
    add(f"model_providers.{provider}.name", provider)
    add(f"model_providers.{provider}.base_url", base_url)
    add(f"model_providers.{provider}.wire_api", args.codex_provider_wire_api)
    add(f"model_providers.{provider}.env_key", args.codex_provider_env_key)
    return flags


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def safe_name(task_id: str) -> str:
    return task_id.replace(":", "_").replace("/", "_")


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int | None,
) -> tuple[int | None, bool, float]:
    started = time.monotonic()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                returncode = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait()
    return returncode, timed_out, time.monotonic() - started


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_existing_attempts(output: Path, run_id: str) -> dict[str, list[dict[str, Any]]]:
    attempts: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl_rows(output):
        if row.get("run_id") != run_id:
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        metadata = row.get("metadata") or {}
        sequential = metadata.get("sequential") or {}
        if sequential.get("strategy") != "sequential_improvement_self_involving":
            continue
        if sequential.get("infra_failure") or sequential.get("effective_attempt") is False:
            continue
        attempts.setdefault(task_id, []).append(row)
    for task_rows in attempts.values():
        task_rows.sort(key=attempt_index)
    return attempts


def attempt_index(row: dict[str, Any]) -> int:
    metadata = row.get("metadata") or {}
    sequential = metadata.get("sequential") or {}
    for value in (sequential.get("attempt_index"), metadata.get("attempt_index")):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def milestone_value(row: dict[str, Any]) -> Any:
    milestone = row.get("milestone")
    if isinstance(milestone, dict):
        return milestone.get("milestone")
    return milestone


def milestone_reason(row: dict[str, Any]) -> str:
    milestone = row.get("milestone")
    if isinstance(milestone, dict):
        return str(milestone.get("reasoning") or "")
    return ""


def read_text_excerpt(path_value: Any, *, max_chars: int) -> str:
    if not isinstance(path_value, str):
        return ""
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ""
    text = path.read_text(errors="replace").strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]


def trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[truncated by runner size guard]"


def has_infra_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in INFRA_FAILURE_MARKERS)


def prepare_claude_env(args: argparse.Namespace, env: dict[str, str]) -> dict[str, str]:
    prepared = env.copy()
    api_key_env = args.anthropic_api_key_env_key
    base_url_env = args.anthropic_base_url_env_key
    if api_key_env != "ANTHROPIC_API_KEY" and prepared.get(api_key_env):
        prepared["ANTHROPIC_API_KEY"] = prepared[api_key_env]
    if base_url_env != "ANTHROPIC_BASE_URL" and prepared.get(base_url_env):
        prepared["ANTHROPIC_BASE_URL"] = prepared[base_url_env].removesuffix("/v1").rstrip("/")
    if not prepared.get("ANTHROPIC_BASE_URL") and prepared.get("LLM_GATEWAY_URL"):
        prepared["ANTHROPIC_BASE_URL"] = prepared["LLM_GATEWAY_URL"].removesuffix("/v1").rstrip("/")
    return prepared


def load_experience(experience_json: Path, experience_md: Path) -> str:
    if experience_json.exists():
        try:
            data = json.loads(experience_json.read_text(errors="replace"))
            value = data.get("experience")
            if isinstance(value, str):
                return value.strip()
        except json.JSONDecodeError:
            pass
    if experience_md.exists():
        return experience_md.read_text(errors="replace").strip()
    return ""


def store_experience(
    *,
    experience_json: Path,
    experience_md: Path,
    task_id: str,
    run_id: str,
    attempt: int,
    experience: str,
    token_budget: int,
    update_metadata: dict[str, Any],
) -> None:
    experience_json.parent.mkdir(parents=True, exist_ok=True)
    experience_md.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "task_id": task_id,
        "run_id": run_id,
        "updated_at": now_iso(),
        "attempt_index": attempt,
        "experience_token_budget": token_budget,
        "experience": experience,
        "update_metadata": update_metadata,
    }
    experience_json.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    experience_md.write_text(experience.rstrip() + "\n")


def build_feedback(row: dict[str, Any], *, attempt: int, max_chars: int) -> str:
    metadata = row.get("metadata") or {}
    verification = row.get("verification") or {}
    details = verification.get("details") or {}
    auto_submit_results = metadata.get("auto_submit_results") or details.get("auto_submit_results") or []
    submit_summaries: list[dict[str, Any]] = []
    submit_log_excerpts: list[str] = []
    for index, result in enumerate(auto_submit_results[:5], start=1):
        if not isinstance(result, dict):
            continue
        submit_summaries.append(
            {
                "index": index,
                "candidate": Path(str(result.get("candidate", ""))).name,
                "returncode": result.get("returncode"),
                "timed_out": result.get("timed_out"),
                "vul_exit_code": result.get("vul_exit_code"),
                "fix_exit_code": result.get("fix_exit_code"),
                "stop_reason": result.get("stop_reason"),
                "wall_seconds": result.get("wall_seconds"),
            }
        )
        excerpt = read_text_excerpt(result.get("log_path"), max_chars=1200)
        if excerpt:
            submit_log_excerpts.append(f"submit log {index}:\n{excerpt}")

    last_message = read_text_excerpt(metadata.get("last_message"), max_chars=2500)
    payload = {
        "task_id": row.get("task_id"),
        "attempt_index": attempt,
        "status": row.get("status"),
        "milestone": milestone_value(row),
        "milestone_reason": milestone_reason(row),
        "verification_status": verification.get("status"),
        "verification_passed": verification.get("passed"),
        "submit_count": verification.get("submit_count"),
        "codex_returncode": metadata.get("codex_returncode"),
        "codex_timed_out": metadata.get("codex_timed_out"),
        "wall_seconds": row.get("wall_seconds"),
        "poc_length": metadata.get("poc_length"),
        "vul_exit_code": metadata.get("vul_exit_code"),
        "fix_exit_code": metadata.get("fix_exit_code"),
        "boundary_violations": metadata.get("boundary_violations"),
        "auto_submit_results": submit_summaries,
    }
    parts = [
        "Newest task feedback, produced by the verifier and runner:",
        json.dumps(payload, indent=2, sort_keys=True),
    ]
    if submit_log_excerpts:
        parts.append("\n\n".join(submit_log_excerpts))
    if last_message:
        parts.append("Codex final message from this attempt:\n" + last_message)
    return trim_text("\n\n".join(parts), max_chars)


def build_task_context(
    task_meta: dict[str, Any] | None,
    data_dir: Path | None,
    task_id: str,
    max_chars: int = 1500,
) -> str:
    """Build a static task context block from tasks.json metadata and data files."""
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


def update_experience_with_codex(
    *,
    args: argparse.Namespace,
    task_id: str,
    safe_task: str,
    attempt: int,
    workspace: Path,
    previous_experience: str,
    feedback: str,
    update_dir: Path,
    env: dict[str, str],
    task_context: str = "",
) -> tuple[str, dict[str, Any]]:
    update_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = update_dir / f"{safe_task}.attempt{attempt}.experience.codex.jsonl"
    stderr_path = update_dir / f"{safe_task}.attempt{attempt}.experience.stderr.log"
    last_message = update_dir / f"{safe_task}.attempt{attempt}.experience.md"
    prompt = build_experience_update_prompt(
        task_id=task_id,
        attempt=attempt,
        token_budget=args.experience_token_budget,
        previous_experience=previous_experience,
        feedback=feedback,
        task_context=task_context,
    )
    command = [
        args.codex_bin,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "-C",
        str(workspace),
        "-o",
        str(last_message),
        "--sandbox",
        args.sandbox,
        "--model",
        args.summary_model or args.model,
        *codex_provider_flags(args),
        "-c",
        f'model_reasoning_effort="{args.summary_reasoning_effort}"',
        prompt,
    ]
    returncode, timed_out, wall_seconds = run_command(
        command,
        cwd=workspace,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=args.summary_timeout_seconds,
    )
    update_metadata = {
        "codex_stdout": str(stdout_path),
        "codex_stderr": str(stderr_path),
        "last_message": str(last_message),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "summary_model": args.summary_model or args.model,
        "summary_reasoning_effort": args.summary_reasoning_effort,
    }
    if returncode == 0 and not timed_out and last_message.exists():
        updated = trim_text(last_message.read_text(errors="replace"), args.experience_max_chars)
        if updated:
            update_metadata["update_status"] = "codex_updated"
            return updated, update_metadata

    fallback_parts = [
        previous_experience.strip(),
        "\n\nLatest feedback that still needs incorporation:\n",
        feedback,
    ]
    update_metadata["update_status"] = "fallback_after_summary_failure"
    return trim_text("\n".join(part for part in fallback_parts if part), args.experience_max_chars), update_metadata


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


def update_experience_with_claude(
    *,
    args: argparse.Namespace,
    task_id: str,
    safe_task: str,
    attempt: int,
    workspace: Path,
    previous_experience: str,
    feedback: str,
    update_dir: Path,
    env: dict[str, str],
    task_context: str = "",
) -> tuple[str, dict[str, Any]]:
    update_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = update_dir / f"{safe_task}.attempt{attempt}.experience.claude.jsonl"
    stderr_path = update_dir / f"{safe_task}.attempt{attempt}.experience.claude.stderr.log"
    last_message = update_dir / f"{safe_task}.attempt{attempt}.experience.md"
    prompt = build_experience_update_prompt(
        task_id=task_id,
        attempt=attempt,
        token_budget=args.experience_token_budget,
        previous_experience=previous_experience,
        feedback=feedback,
        task_context=task_context,
    )
    claude_bin = args.summary_bin or args.codex_bin
    command = [
        claude_bin,
        "--bare",
        "--model",
        args.summary_model or args.model,
        "-p",
        "--verbose",
        "--output-format",
        "stream-json",
        "--no-session-persistence",
        "--permission-mode",
        "bypassPermissions",
        "--allowedTools=Read,Grep,LS",
    ]
    if args.summary_reasoning_effort:
        command.extend(["--effort", args.summary_reasoning_effort])
    command.append(prompt)
    returncode, timed_out, wall_seconds = run_command(
        command,
        cwd=workspace,
        env=prepare_claude_env(args, env),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=args.summary_timeout_seconds,
    )
    claude_result = extract_claude_result(stdout_path, last_message)
    update_metadata = {
        "summary_backend": "claude",
        "claude_stdout": str(stdout_path),
        "claude_stderr": str(stderr_path),
        "codex_stdout": str(stdout_path),
        "codex_stderr": str(stderr_path),
        "last_message": str(last_message),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "summary_model": args.summary_model or args.model,
        "summary_reasoning_effort": args.summary_reasoning_effort,
        "claude_result": claude_result,
    }
    is_error = bool(claude_result.get("is_error")) if claude_result else False
    if returncode == 0 and not timed_out and not is_error and last_message.exists():
        updated = trim_text(last_message.read_text(errors="replace"), args.experience_max_chars)
        if updated:
            update_metadata["update_status"] = "claude_updated"
            return updated, update_metadata

    fallback_parts = [
        previous_experience.strip(),
        "\n\nLatest feedback that still needs incorporation:\n",
        feedback,
    ]
    update_metadata["update_status"] = "fallback_after_summary_failure"
    return trim_text("\n".join(part for part in fallback_parts if part), args.experience_max_chars), update_metadata


def update_experience_with_openai(
    *,
    args: argparse.Namespace,
    task_id: str,
    attempt: int,
    previous_experience: str,
    feedback: str,
    env: dict[str, str],
    task_context: str = "",
) -> tuple[str, dict[str, Any]]:
    prompt = build_experience_update_prompt(
        task_id=task_id,
        attempt=attempt,
        token_budget=args.experience_token_budget,
        previous_experience=previous_experience,
        feedback=feedback,
        task_context=task_context,
    )
    model = args.summary_model or args.model
    base_url = args.openai_base_url or args.opencode_provider_base_url or args.codex_provider_base_url
    api_key = env.get(args.openai_api_key_env_key) or os.environ.get(args.openai_api_key_env_key)
    max_tokens = args.openai_summary_max_tokens or max(4096, args.experience_token_budget * 3)
    update_metadata: dict[str, Any] = {
        "summary_backend": "openai",
        "summary_model": model,
        "summary_base_url": base_url,
        "summary_api_key_env_key": args.openai_api_key_env_key,
        "summary_max_tokens": max_tokens,
    }

    try:
        if not api_key:
            raise ValueError(f"missing {args.openai_api_key_env_key}")
        if not base_url:
            raise ValueError("missing OpenAI-compatible base URL")
        from openai import OpenAI

        started = time.monotonic()
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You update concise task-local CyberGym experience notes. "
                        "Output only the updated notes; do not include hidden reasoning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        update_metadata["wall_seconds"] = time.monotonic() - started
        update_metadata["finish_reason"] = response.choices[0].finish_reason if response.choices else None
        usage = getattr(response, "usage", None)
        if usage is not None:
            update_metadata["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        content = response.choices[0].message.content if response.choices else ""
        updated = trim_text((content or "").strip(), args.experience_max_chars)
        if updated:
            update_metadata["update_status"] = "openai_updated"
            return updated, update_metadata
        update_metadata["update_status"] = "fallback_after_empty_summary"
    except Exception as exc:
        update_metadata["update_status"] = "fallback_after_summary_failure"
        update_metadata["error"] = f"{type(exc).__name__}: {exc}"

    fallback_parts = [
        previous_experience.strip(),
        "\n\nLatest feedback that still needs incorporation:\n",
        feedback,
    ]
    return trim_text("\n".join(part for part in fallback_parts if part), args.experience_max_chars), update_metadata


def get_tinker_sampler(base_model: str | None, model_path: str | None = None) -> tuple[Any, Any]:
    with TINKER_SAMPLER_LOCK:
        cache_key = model_path or f"base:{base_model}"
        cached = TINKER_SAMPLER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        import tinker

        service = tinker.ServiceClient()
        if model_path:
            sampler = service.create_sampling_client(model_path=model_path)
        else:
            if not base_model:
                raise ValueError("Tinker base_model is required when model_path is not set")
            sampler = service.create_sampling_client(base_model=base_model)
        tokenizer = sampler.get_tokenizer()
        cached = (sampler, tokenizer)
        TINKER_SAMPLER_CACHE[cache_key] = cached
        return cached


def render_tinker_prompt(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.encode(f"User: {prompt}\nAssistant:")


def clean_tinker_generation(text: str) -> str:
    text = text.replace("<|endoftext|>", "").replace("<|im_end|>", "")
    text = re.split(r"(?m)^\s*(User:|<\|im_start\|>user)", text, maxsplit=1)[0]
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    if "<think>" in text.lower():
        heading = re.search(r"(?im)^\s*1\.\s+Current best hypothesis", text)
        if heading:
            text = text[heading.start() :]
        else:
            return ""
    return text.strip()


def trim_to_token_budget(tokenizer: Any, text: str, token_budget: int) -> str:
    """Hard-cap saved strategy text to the requested token budget.

    Tinker sampling's max_tokens limits sampled continuation tokens.  The runner
    prepends a fixed assistant prefix before cleaning, so the saved strategy can
    otherwise exceed the user-facing budget by a few tokens.
    """
    if token_budget <= 0 or not text:
        return text.strip()
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)
    if len(tokens) <= token_budget:
        return text.strip()
    try:
        trimmed = tokenizer.decode(tokens[:token_budget], skip_special_tokens=True)
    except TypeError:
        trimmed = tokenizer.decode(tokens[:token_budget])
    return clean_tinker_generation(trimmed)


def update_experience_with_tinker(
    *,
    args: argparse.Namespace,
    task_id: str,
    attempt: int,
    previous_experience: str,
    feedback: str,
    task_context: str = "",
) -> tuple[str, dict[str, Any]]:
    from tinker import types

    base_model = args.tinker_summary_model or args.summary_model
    model_path = getattr(args, "tinker_summary_model_path", None)
    if not base_model and not model_path:
        raise ValueError(
            "--experience-updater=tinker requires --tinker-summary-model/--summary-model "
            "or --tinker-summary-model-path"
        )
    started = time.monotonic()
    update_metadata = {
        "summary_backend": "tinker",
        "summary_model": base_model,
        "summary_model_path": model_path,
        "tinker_summary_temperature": args.tinker_summary_temperature,
        "tinker_summary_top_p": args.tinker_summary_top_p,
    }
    try:
        sampler, tokenizer = get_tinker_sampler(base_model, model_path)
        prompt = build_experience_update_prompt(
            task_id=task_id,
            attempt=attempt,
            token_budget=args.experience_token_budget,
            previous_experience=previous_experience,
            feedback=feedback,
            task_context=task_context,
        )
        response_prefix = "1. Current best hypothesis"
        prompt_tokens = render_tinker_prompt(tokenizer, prompt)
        prompt_tokens.extend(tokenizer.encode(response_prefix, add_special_tokens=False))
        params = types.SamplingParams(
            max_tokens=args.experience_token_budget,
            stop=["<|im_end|>", "\nUser:", "\n<|im_start|>user"],
            temperature=args.tinker_summary_temperature,
            top_p=args.tinker_summary_top_p,
        )
        result = sampler.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=params,
        ).result(timeout=args.summary_timeout_seconds)
        sequence = result.sequences[0]
        try:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        except TypeError:
            text = tokenizer.decode(sequence.tokens)
        if not text.lstrip().startswith(response_prefix):
            text = response_prefix + text
        updated = trim_text(
            trim_to_token_budget(
                tokenizer,
                clean_tinker_generation(text),
                args.experience_token_budget,
            ),
            args.experience_max_chars,
        )
        update_metadata.update(
            {
                "update_status": "tinker_updated" if updated else "empty_tinker_generation",
                "wall_seconds": time.monotonic() - started,
                "stop_reason": str(getattr(sequence, "stop_reason", "")),
            }
        )
        if updated:
            return updated, update_metadata
    except Exception as exc:  # noqa: BLE001 - recorded in rollout metadata for recovery.
        update_metadata.update(
            {
                "update_status": "fallback_after_tinker_failure",
                "error": repr(exc),
                "wall_seconds": time.monotonic() - started,
            }
        )

    fallback_parts = [
        previous_experience.strip(),
        "\n\nLatest feedback that still needs incorporation:\n",
        feedback,
    ]
    return trim_text("\n".join(part for part in fallback_parts if part), args.experience_max_chars), update_metadata


def is_infra_failure(row: dict[str, Any]) -> bool:
    if str(row.get("status") or "") not in AGENT_ERROR_STATUSES:
        return False
    metadata = row.get("metadata") or {}
    text_parts: list[str] = []
    for key in (
        "codex_stdout",
        "codex_stderr",
        "claude_stdout",
        "claude_stderr",
        "generation_error",
        "opencode_api_error_message",
        "api_error_message",
    ):
        value = metadata.get(key)
        if key.endswith("_stdout") or key.endswith("_stderr"):
            text_parts.append(read_text_excerpt(value, max_chars=4000))
        elif value:
            text_parts.append(str(value))
    for key in ("claude_result", "codex_result", "opencode_result"):
        value = metadata.get(key)
        if value:
            text_parts.append(json.dumps(value, sort_keys=True, default=str))
    verification = row.get("verification") or {}
    details = verification.get("details") or {}
    if details is not metadata:
        for key in ("codex_stdout", "codex_stderr", "claude_stdout", "claude_stderr", "generation_error"):
            value = details.get(key)
            if key.endswith("_stdout") or key.endswith("_stderr"):
                text_parts.append(read_text_excerpt(value, max_chars=4000))
            elif value:
                text_parts.append(str(value))
    return has_infra_marker("\n".join(part for part in text_parts if part))


def register_infra_failure(
    *,
    args: argparse.Namespace,
    infra_lock: threading.Lock,
    infra_state: dict[str, Any],
    print_lock: threading.Lock,
    task_id: str,
    attempt: int,
) -> int:
    with infra_lock:
        consecutive = int(infra_state.get("consecutive") or 0) + 1
        infra_state["consecutive"] = consecutive
        should_pause = consecutive >= args.max_consecutive_infra_failures
        if should_pause:
            pause_until = time.monotonic() + args.infra_failure_pause_seconds
            resume_at = datetime.fromtimestamp(time.time() + args.infra_failure_pause_seconds, UTC).isoformat()
            infra_state["pause_until"] = max(float(infra_state.get("pause_until") or 0.0), pause_until)
            infra_state["consecutive"] = 0
        else:
            resume_at = None
    if should_pause:
        with print_lock:
            print(
                json.dumps(
                    {
                        "event": "sequential_infra_pause_start",
                        "task_id": task_id,
                        "attempt": attempt,
                        "consecutive_infra_failures": consecutive,
                        "pause_seconds": args.infra_failure_pause_seconds,
                        "resume_at": resume_at,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return consecutive


def reset_infra_failures(*, infra_lock: threading.Lock, infra_state: dict[str, Any]) -> None:
    with infra_lock:
        infra_state["consecutive"] = 0


def wait_for_infra_pause(
    *,
    infra_lock: threading.Lock,
    infra_state: dict[str, Any],
    print_lock: threading.Lock,
    task_id: str,
    attempt: int,
) -> None:
    reported = False
    while True:
        with infra_lock:
            pause_until = float(infra_state.get("pause_until") or 0.0)
        remaining = pause_until - time.monotonic()
        if remaining <= 0:
            if reported:
                with print_lock:
                    print(
                        json.dumps(
                            {
                                "event": "sequential_infra_pause_complete",
                                "task_id": task_id,
                                "attempt": attempt,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            return
        if not reported:
            reported = True
            with print_lock:
                print(
                    json.dumps(
                        {
                            "event": "sequential_infra_pause_wait",
                            "task_id": task_id,
                            "attempt": attempt,
                            "remaining_seconds": remaining,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        time.sleep(min(remaining, 60.0))


def decorate_attempt_row(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    attempt: int,
    infra_failure: bool = False,
    attempt_output: Path,
    experience_before: Path,
    feedback_path: Path,
    experience_after: Path | None,
    experience_update: dict[str, Any] | None,
) -> dict[str, Any]:
    row = json.loads(json.dumps(row))
    row["strategy_id"] = f"sequential_self_involving_attempt_{attempt}"
    row["strategy"] = "sequential_improvement_self_involving"
    metadata = row.setdefault("metadata", {})
    verification = row.setdefault("verification", {})
    details = verification.setdefault("details", {})
    old_source = metadata.get("source")
    sequential = {
        "strategy": "sequential_improvement_self_involving",
        "source": SOURCE,
        "base_runner_source": old_source,
        "attempt_index": attempt,
        "max_attempts": args.max_attempts,
        "stop_on_pass": not args.no_stop_on_pass,
        "infra_failure": infra_failure,
        "effective_attempt": not infra_failure,
        "attempt_output": str(attempt_output),
        "experience_before": str(experience_before),
        "feedback_path": str(feedback_path),
        "experience_after": str(experience_after) if experience_after else None,
        "experience_update": experience_update or {},
        "experience_token_budget": args.experience_token_budget,
    }
    metadata["source"] = SOURCE
    metadata["sequential"] = sequential
    details["source"] = SOURCE
    details["sequential"] = sequential
    return row


def append_row_locked(output: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        append_rollout(output, RolloutRecord.from_dict(row))


def wait_for_launch_slot(
    *,
    args: argparse.Namespace,
    task_id: str,
    attempt: int,
    launch_lock: threading.Lock,
    next_launch_time: list[float],
    print_lock: threading.Lock,
) -> None:
    interval = max(0.0, float(args.codex_launch_stagger_seconds or 0.0))
    if interval <= 0:
        return

    with launch_lock:
        now = time.monotonic()
        sleep_seconds = max(0.0, next_launch_time[0] - now)
        next_launch_time[0] = max(now, next_launch_time[0]) + interval

    if sleep_seconds <= 0:
        return

    with print_lock:
        print(
            json.dumps(
                {
                    "event": "sequential_attempt_launch_stagger",
                    "task_id": task_id,
                    "attempt": attempt,
                    "sleep_seconds": sleep_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    time.sleep(sleep_seconds)


def run_attempt(
    *,
    args: argparse.Namespace,
    task_id: str,
    attempt: int,
    run_root: Path,
    experience_md: Path,
    env: dict[str, str],
) -> tuple[dict[str, Any] | None, Path, int | None, bool, float]:
    safe_task = safe_name(task_id)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    attempt_output = run_root / "per_attempt_rollouts" / f"{safe_task}.attempt{attempt}.{stamp}.jsonl"
    stdout_log = run_root / "sequential_runner_logs" / f"{safe_task}.attempt{attempt}.stdout.log"
    stderr_log = run_root / "sequential_runner_logs" / f"{safe_task}.attempt{attempt}.stderr.log"
    command = [
        sys.executable,
        str(args.runner),
        "--split",
        args.split,
        "--difficulty",
        args.difficulty,
        "--run-id",
        args.run_id,
        "--run-root",
        str(args.run_root),
        "--output",
        str(attempt_output),
        "--task-id",
        task_id,
        "--rerun-recorded",
        "--agent-suffix",
        f"_si{attempt}",
        "--attempt-index",
        str(attempt),
        "--experience-file",
        str(experience_md),
        "--codex-bin",
        args.codex_bin,
        "--codex-timeout-seconds",
        str(args.codex_timeout_seconds),
        "--submit-timeout-seconds",
        str(args.submit_timeout_seconds),
        "--server",
        args.server,
        "--pocdb-path",
        str(args.pocdb_path),
        "--env-file",
        str(args.env_file),
        "--sandbox",
        args.sandbox,
        "--max-auto-submit-candidates",
        str(args.max_auto_submit_candidates),
        "--codex-rate-limit-retries",
        str(args.codex_rate_limit_retries),
        "--codex-rate-limit-stagger-seconds",
        str(args.codex_rate_limit_stagger_seconds),
        "--codex-rate-limit-min-sleep-seconds",
        str(args.codex_rate_limit_min_sleep_seconds),
    ]
    if args.runner.name != "run_codex_cybergym_tasks.py":
        command.extend(
            [
                "--anthropic-api-key-env-key",
                args.anthropic_api_key_env_key,
                "--anthropic-base-url-env-key",
                args.anthropic_base_url_env_key,
            ]
        )
    if args.codex_provider:
        command.extend(["--codex-provider", args.codex_provider])
        if args.codex_provider_base_url:
            command.extend(["--codex-provider-base-url", args.codex_provider_base_url])
        command.extend(
            [
                "--codex-provider-wire-api",
                args.codex_provider_wire_api,
                "--codex-provider-env-key",
                args.codex_provider_env_key,
            ]
        )
    if args.opencode_provider:
        command.extend(["--opencode-provider", args.opencode_provider])
        if args.opencode_provider_base_url:
            command.extend(["--opencode-provider-base-url", args.opencode_provider_base_url])
        command.extend(["--opencode-provider-env-key", args.opencode_provider_env_key])
        if args.opencode_context_limit is not None:
            command.extend(["--opencode-context-limit", str(args.opencode_context_limit)])
        if args.opencode_output_token_max is not None:
            command.extend(["--opencode-output-token-max", str(args.opencode_output_token_max)])
    if args.force_workspace:
        command.append("--force-workspace")
    if args.model:
        command.extend(["--model", args.model])
    if args.model_reasoning_effort:
        command.extend(["--model-reasoning-effort", args.model_reasoning_effort])

    returncode, timed_out, wall_seconds = run_command(
        command,
        cwd=Path.cwd(),
        env=env,
        stdout_path=stdout_log,
        stderr_path=stderr_log,
        timeout_seconds=args.codex_timeout_seconds + args.submit_timeout_seconds + 600,
    )
    rows = [row for row in read_jsonl_rows(attempt_output) if row.get("task_id") == task_id]
    return (rows[-1] if rows else None), attempt_output, returncode, timed_out, wall_seconds


def run_task_sequence(
    *,
    args: argparse.Namespace,
    task_id: str,
    run_root: Path,
    existing_rows: list[dict[str, Any]],
    output_lock: threading.Lock,
    print_lock: threading.Lock,
    launch_lock: threading.Lock,
    infra_lock: threading.Lock,
    infra_state: dict[str, Any],
    next_launch_time: list[float],
    env: dict[str, str],
    task_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_task = safe_name(task_id)
    experiences_dir = run_root / "experiences"
    feedback_dir = run_root / "feedback"
    update_dir = run_root / "experience_updates"
    experience_json = experiences_dir / f"{safe_task}.experience.json"
    experience_md = experiences_dir / f"{safe_task}.experience.md"

    existing_by_attempt = {attempt_index(row): row for row in existing_rows}
    if not args.rerun_completed:
        if any(row.get("status") == "PASSED" for row in existing_rows) and not args.no_stop_on_pass:
            return {"task_id": task_id, "status": "SKIPPED_PASSED", "attempts_run": 0, "infra_failures": 0}
        if existing_by_attempt and max(existing_by_attempt) >= args.max_attempts:
            return {"task_id": task_id, "status": "SKIPPED_MAX_ATTEMPTS", "attempts_run": 0, "infra_failures": 0}

    task_context = build_task_context(
        task_meta,
        args.task_context_data_dir if hasattr(args, "task_context_data_dir") else None,
        task_id,
        max_chars=getattr(args, "task_context_max_chars", 1500),
    )

    attempts_run = 0
    infra_failures = 0
    final_status = "NOT_RUN"
    start_attempt = 1
    if existing_by_attempt and not args.rerun_completed:
        start_attempt = max(existing_by_attempt) + 1

    attempt = start_attempt
    while attempt <= args.max_attempts:
        wait_for_infra_pause(
            infra_lock=infra_lock,
            infra_state=infra_state,
            print_lock=print_lock,
            task_id=task_id,
            attempt=attempt,
        )
        previous_experience = load_experience(experience_json, experience_md)
        with print_lock:
            print(
                json.dumps(
                    {
                        "event": "sequential_attempt_start",
                        "task_id": task_id,
                        "attempt": attempt,
                        "experience_chars": len(previous_experience),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        wait_for_launch_slot(
            args=args,
            task_id=task_id,
            attempt=attempt,
            launch_lock=launch_lock,
            next_launch_time=next_launch_time,
            print_lock=print_lock,
        )
        row, attempt_output, returncode, timed_out, wall_seconds = run_attempt(
            args=args,
            task_id=task_id,
            attempt=attempt,
            run_root=run_root,
            experience_md=experience_md,
            env=env,
        )
        if row is None:
            attempts_run += 1
            final_status = AGENT_ERROR_STATUS
            with print_lock:
                print(
                    json.dumps(
                        {
                            "event": "sequential_attempt_missing_row",
                            "task_id": task_id,
                            "attempt": attempt,
                            "returncode": returncode,
                            "timed_out": timed_out,
                            "wall_seconds": wall_seconds,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            break

        feedback = build_feedback(row, attempt=attempt, max_chars=args.feedback_max_chars)
        feedback_path = feedback_dir / f"{safe_task}.attempt{attempt}.feedback.md"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text(feedback.rstrip() + "\n")

        row_is_infra_failure = is_infra_failure(row)
        workspace = Path((row.get("metadata") or {}).get("workspace", "."))
        if not workspace.exists():
            workspace = Path.cwd()
        if row_is_infra_failure:
            updated_experience = previous_experience
            infra_failures += 1
            consecutive_infra_failures = register_infra_failure(
                args=args,
                infra_lock=infra_lock,
                infra_state=infra_state,
                print_lock=print_lock,
                task_id=task_id,
                attempt=attempt,
            )
            update_metadata = {
                "update_status": "skipped_infra_failure",
                "reason": "attempt did not produce task feedback suitable for experience distillation",
                "consecutive_infra_failures": consecutive_infra_failures,
            }
            decorated = decorate_attempt_row(
                row,
                args=args,
                attempt=attempt,
                infra_failure=True,
                attempt_output=attempt_output,
                experience_before=experience_md,
                feedback_path=feedback_path,
                experience_after=experience_md,
                experience_update=update_metadata,
            )
            append_row_locked(args.output.resolve(), decorated, output_lock)

            final_status = str(decorated.get("status"))
            with print_lock:
                print(
                    json.dumps(
                        {
                            "event": "sequential_attempt_done",
                            "task_id": task_id,
                            "attempt": attempt,
                            "status": final_status,
                            "milestone": milestone_value(decorated),
                            "infra_failure": True,
                            "returncode": returncode,
                            "timed_out": timed_out,
                            "experience_update_status": update_metadata.get("update_status"),
                            "experience_chars": len(updated_experience),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            continue

        reset_infra_failures(infra_lock=infra_lock, infra_state=infra_state)
        attempts_run += 1
        if args.experience_updater == "tinker":
            updated_experience, update_metadata = update_experience_with_tinker(
                args=args,
                task_id=task_id,
                attempt=attempt,
                previous_experience=previous_experience,
                feedback=feedback,
                task_context=task_context,
            )
        elif args.experience_updater == "claude":
            updated_experience, update_metadata = update_experience_with_claude(
                args=args,
                task_id=task_id,
                safe_task=safe_task,
                attempt=attempt,
                workspace=workspace,
                previous_experience=previous_experience,
                feedback=feedback,
                update_dir=update_dir,
                env=env,
                task_context=task_context,
            )
        elif args.experience_updater == "openai":
            updated_experience, update_metadata = update_experience_with_openai(
                args=args,
                task_id=task_id,
                attempt=attempt,
                previous_experience=previous_experience,
                feedback=feedback,
                env=env,
                task_context=task_context,
            )
        else:
            updated_experience, update_metadata = update_experience_with_codex(
                args=args,
                task_id=task_id,
                safe_task=safe_task,
                attempt=attempt,
                workspace=workspace,
                previous_experience=previous_experience,
                feedback=feedback,
                update_dir=update_dir,
                env=env,
                task_context=task_context,
            )
        store_experience(
            experience_json=experience_json,
            experience_md=experience_md,
            task_id=task_id,
            run_id=args.run_id,
            attempt=attempt,
            experience=updated_experience,
            token_budget=args.experience_token_budget,
            update_metadata=update_metadata,
        )

        decorated = decorate_attempt_row(
            row,
            args=args,
            attempt=attempt,
            infra_failure=False,
            attempt_output=attempt_output,
            experience_before=experience_md,
            feedback_path=feedback_path,
            experience_after=experience_md,
            experience_update=update_metadata,
        )
        append_row_locked(args.output.resolve(), decorated, output_lock)

        final_status = str(decorated.get("status"))
        with print_lock:
            print(
                json.dumps(
                    {
                        "event": "sequential_attempt_done",
                        "task_id": task_id,
                        "attempt": attempt,
                        "status": final_status,
                        "milestone": milestone_value(decorated),
                        "infra_failure": False,
                        "returncode": returncode,
                        "timed_out": timed_out,
                        "experience_update_status": update_metadata.get("update_status"),
                        "experience_chars": len(updated_experience),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if final_status == "PASSED" and not args.no_stop_on_pass:
            break
        if attempt < args.max_attempts and args.attempt_cooldown_seconds > 0:
            with print_lock:
                print(
                    json.dumps(
                        {
                            "event": "sequential_attempt_cooldown",
                            "task_id": task_id,
                            "attempt": attempt,
                            "sleep_seconds": args.attempt_cooldown_seconds,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            time.sleep(args.attempt_cooldown_seconds)
        attempt += 1

    return {
        "task_id": task_id,
        "status": final_status,
        "attempts_run": attempts_run,
        "infra_failures": infra_failures,
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be >= 1")
    if args.max_consecutive_infra_failures < 1:
        raise ValueError("--max-consecutive-infra-failures must be >= 1")
    if args.infra_failure_pause_seconds < 0:
        raise ValueError("--infra-failure-pause-seconds must be >= 0")

    load_dotenv(args.env_file)
    manifest = load_manifest()
    task_ids = args.task_id or load_split_ids(manifest, args.split)
    if args.task_offset:
        task_ids = task_ids[args.task_offset :]
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    output = args.output.resolve()
    run_root = (args.run_root / args.run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = run_root / "sequential_self_involving_config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "split": args.split,
                    "difficulty": args.difficulty,
                    "task_count": len(task_ids),
                    "max_attempts": args.max_attempts,
                    "workers": args.workers,
                    "codex_launch_stagger_seconds": args.codex_launch_stagger_seconds,
                    "attempt_cooldown_seconds": args.attempt_cooldown_seconds,
                    "model": args.model,
                    "model_reasoning_effort": args.model_reasoning_effort,
                    "summary_model": args.summary_model or args.model,
                    "experience_updater": args.experience_updater,
                    "tinker_summary_model": args.tinker_summary_model,
                    "summary_reasoning_effort": args.summary_reasoning_effort,
                    "summary_bin": args.summary_bin or args.codex_bin,
                    "experience_token_budget": args.experience_token_budget,
                    "max_consecutive_infra_failures": args.max_consecutive_infra_failures,
                    "infra_failure_pause_seconds": args.infra_failure_pause_seconds,
                    "started_at": now_iso(),
                    "source": SOURCE,
                },
                indent=2,
                sort_keys=True,
            )
        )

    env = os.environ.copy()
    pythonpath = [str(Path("cybergym/src").resolve()), str(Path("src").resolve())]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    tasks_by_id: dict[str, dict[str, Any]] = {}
    if args.tasks_json.exists():
        try:
            raw_tasks = json.loads(args.tasks_json.read_text())
            if isinstance(raw_tasks, list):
                tasks_by_id = {t["task_id"]: t for t in raw_tasks if isinstance(t, dict) and "task_id" in t}
        except Exception:
            pass

    existing = read_existing_attempts(output, args.run_id)
    pending = []
    for task_id in task_ids:
        rows = existing.get(task_id, [])
        if args.rerun_completed:
            pending.append(task_id)
            continue
        if any(row.get("status") == "PASSED" for row in rows) and not args.no_stop_on_pass:
            continue
        if rows and max(attempt_index(row) for row in rows) >= args.max_attempts:
            continue
        pending.append(task_id)

    print(
        json.dumps(
            {
                "event": "sequential_start",
                "run_id": args.run_id,
                "workers": args.workers,
                "max_attempts": args.max_attempts,
                "codex_launch_stagger_seconds": args.codex_launch_stagger_seconds,
                "attempt_cooldown_seconds": args.attempt_cooldown_seconds,
                "max_consecutive_infra_failures": args.max_consecutive_infra_failures,
                "infra_failure_pause_seconds": args.infra_failure_pause_seconds,
                "task_count": len(task_ids),
                "pending": len(pending),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    output_lock = threading.Lock()
    print_lock = threading.Lock()
    launch_lock = threading.Lock()
    infra_lock = threading.Lock()
    infra_state: dict[str, Any] = {"consecutive": 0, "pause_until": 0.0}
    next_launch_time = [time.monotonic()]
    exit_code = 0
    completed = 0
    total_attempts = 0
    total_infra_failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_task_sequence,
                args=args,
                task_id=task_id,
                run_root=run_root,
                existing_rows=existing.get(task_id, []),
                output_lock=output_lock,
                print_lock=print_lock,
                launch_lock=launch_lock,
                infra_lock=infra_lock,
                infra_state=infra_state,
                next_launch_time=next_launch_time,
                env=env,
                task_meta=tasks_by_id.get(task_id),
            ): task_id
            for task_id in pending
        }
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = futures.pop(future)
                try:
                    result = future.result()
                    completed += 1
                    total_attempts += int(result.get("attempts_run") or 0)
                    total_infra_failures += int(result.get("infra_failures") or 0)
                    with print_lock:
                        print(
                            json.dumps(
                                {
                                    "event": "sequential_task_done",
                                    "task_id": task_id,
                                    "completed": completed,
                                    "pending": len(pending),
                                    **result,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                except Exception as exc:  # noqa: BLE001 - report task-level failures.
                    exit_code = 1
                    with print_lock:
                        print(
                            json.dumps(
                                {"event": "sequential_task_error", "task_id": task_id, "error": repr(exc)},
                                sort_keys=True,
                            ),
                            flush=True,
                        )

    print(
        json.dumps(
            {
                "event": "sequential_complete",
                "run_id": args.run_id,
                "exit_code": exit_code,
                "tasks_completed": completed,
                "attempts_run": total_attempts,
                "infra_failures": total_infra_failures,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
