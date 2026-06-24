#!/usr/bin/env python3
"""Train a CyberGym strategy generator with GRPO from an SFT checkpoint.

The policy is the strategy generator.  The executor is fixed: each sampled
strategy is written to disk and passed to run_codex_cybergym_tasks.py via
--strategy-file.  The CyberGym verifier milestone is converted to reward and
normalized within each task group.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mastermind.config import load_manifest
from mastermind.tasks import TaskMetadata, load_split_ids, load_task_metadata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "runs/strategy_grpo"
DEFAULT_SFT_RESULT = ROOT / "runs/strategy_sft/qwen36-strategy-sft-20260601T1935Z/result.json"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
ASSISTANT_PREFIX = "1. Current best hypothesis"
MILESTONE_REWARD = {
    0: 0.0,
    1: 0.5,
    2: 1.5,
    3: 2.5,
    4: 4.0,
    5: 5.5,
    6: 8.0,
    7: 12.0,
}


@dataclass
class StrategySample:
    sample_id: str
    step: int
    task_id: str
    task_index: int
    sample_index: int
    group_index: int
    group_sample_index: int
    prompt_tokens: list[int]
    tokens: list[int]
    logprobs: list[float]
    text: str
    stop_reason: str
    strategy_path: str
    invalid_format: bool
    sampled_tokens: int
    prompt_token_count: int
    sampling_error: str | None = None


@dataclass
class RolloutResult:
    sample_id: str
    task_id: str
    sample_index: int
    row: dict[str, Any]
    returncode: int | None
    timed_out: bool
    wall_seconds: float
    output_path: str
    stdout_path: str
    stderr_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--init-state-path", default=None)
    parser.add_argument(
        "--init-result-json",
        type=Path,
        default=DEFAULT_SFT_RESULT,
        help="Used to infer --init-state-path from final_state_path when not supplied.",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260608)

    parser.add_argument("--split", default="train_dev")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--task-ids-file", type=Path)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--max-train-tasks", type=int)
    parser.add_argument(
        "--task-sampling",
        choices=("sequential", "random"),
        default="sequential",
        help=(
            "How to choose tasks for each GRPO step. 'sequential' walks the "
            "selected list and shuffles only between full passes; 'random' "
            "samples a fresh task group each step."
        ),
    )
    parser.add_argument("--tasks-per-step", type=int, default=2)
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Compatibility alias for --advantage-group-size.",
    )
    parser.add_argument(
        "--advantage-group-size",
        type=int,
        default=None,
        help="Number of rollout rewards used in one task-local GRPO advantage group.",
    )
    parser.add_argument(
        "--rollout-pool-per-task",
        type=int,
        default=None,
        help=(
            "Number of strategies to sample and execute per task each step. "
            "Must be a multiple of --advantage-group-size."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=1)

    parser.add_argument("--max-strategy-tokens", type=int, default=1024)
    parser.add_argument("--strategy-temperature", type=float, default=0.7)
    parser.add_argument("--strategy-top-p", type=float, default=0.95)
    parser.add_argument("--task-context-max-chars", type=int, default=1500)

    parser.add_argument("--loss-fn", choices=("ppo", "importance_sampling"), default="ppo")
    parser.add_argument(
        "--loss-fn-config-json",
        default=None,
        help="Optional JSON object passed to Tinker forward_backward_async loss_fn_config.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--skip-grpo-update", action="store_true")
    parser.add_argument("--skip-uniform-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--update-as-groups-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run GRPO updates as soon as fixed task-local rollout groups complete.",
    )
    parser.add_argument(
        "--groups-per-update",
        type=int,
        default=1,
        help="Number of completed non-uniform rollout groups to accumulate per optimizer update.",
    )
    parser.add_argument(
        "--save-after-update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist a sampler checkpoint after each optimizer update for early evaluation.",
    )
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--reward-compression", choices=("none", "log1p"), default="log1p")
    parser.add_argument("--timeout-penalty", type=float, default=0.2)
    parser.add_argument("--invalid-format-penalty", type=float, default=0.2)
    parser.add_argument("--advantage-eps", type=float, default=1e-8)

    parser.add_argument("--runner", type=Path, default=ROOT / "scripts/run_codex_cybergym_tasks.py")
    parser.add_argument("--executor-run-root", type=Path, default=ROOT / "runs/codex_cybergym")
    parser.add_argument("--executor-workers", type=int, default=2)
    parser.add_argument("--executor-model", default="gpt-5.5")
    parser.add_argument("--executor-reasoning-effort", default="medium")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)
    parser.add_argument("--submit-timeout-seconds", type=int, default=1800)
    parser.add_argument("--server", default="http://127.0.0.1:8666")
    parser.add_argument("--pocdb-path", type=Path, default=ROOT / "runs/cybergym_server/poc.db")
    parser.add_argument("--sandbox", default="workspace-write")
    parser.add_argument("--force-workspace", action="store_true")
    parser.add_argument("--max-auto-submit-candidates", type=int, default=5)
    parser.add_argument("--codex-provider", default=None)
    parser.add_argument("--codex-provider-base-url", default=None)
    parser.add_argument("--codex-provider-wire-api", default="responses")
    parser.add_argument("--codex-provider-env-key", default="LLM_GATEWAY_API_KEY")
    parser.add_argument("--codex-rate-limit-retries", type=int, default=3)
    parser.add_argument("--codex-rate-limit-stagger-seconds", type=float, default=5.0)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def safe_tinker_label(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return safe or "checkpoint"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def resolve_init_state(args: argparse.Namespace) -> str | None:
    if args.init_state_path:
        return args.init_state_path
    if args.init_result_json and args.init_result_json.exists():
        try:
            data = json.loads(args.init_result_json.read_text())
        except Exception:
            data = {}
        value = data.get("final_state_path")
        if isinstance(value, str) and value:
            return value
    return None


def selected_task_ids(args: argparse.Namespace) -> list[str]:
    manifest = load_manifest()
    if args.task_id:
        task_ids = list(args.task_id)
    elif args.task_ids_file:
        task_ids = [
            line.strip()
            for line in args.task_ids_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        task_ids = load_split_ids(manifest, args.split)
    task_ids = task_ids[args.task_offset :]
    if args.max_train_tasks is not None:
        task_ids = task_ids[: args.max_train_tasks]
    return task_ids


def build_task_context(
    task_meta: TaskMetadata | None,
    data_dir: Path | None,
    task_id: str,
    max_chars: int,
) -> str:
    parts: list[str] = []
    if task_meta:
        project = task_meta.project_name
        language = task_meta.project_language or ""
        vuln = task_meta.vulnerability_description or ""
        if project:
            parts.append(f"Project: {project} ({language})")
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


def build_strategy_prompt(
    *,
    task_id: str,
    difficulty: str,
    token_budget: int,
    task_context: str,
) -> str:
    context_block = f"\nTask context:\n{task_context}\n" if task_context else ""
    return f"""You are maintaining a task-local experience object for CyberGym task {task_id}.

Update the experience before the first executor attempt for this rollout.
{context_block}
Executor setting:
- The executor will run in CyberGym difficulty `{difficulty}`.
- The executor can inspect only the task workspace and must create a PoC under pocs/.
- The executor will submit candidate PoCs to the verifier; milestone 7 means vulnerable build crashes and fixed build is clean.

Constraints:
- Output only the updated experience text.
- Do not output hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Start directly with "1. Current best hypothesis".
- Keep it under {token_budget} tokens.
- Preserve concrete lessons that help the executor solve this same task.
- Include target files, likely bug mechanism, PoC shape, and concrete first commands when possible.
- Do not invent verifier results or previous attempts.
- Do not include generic CyberGym instructions.

Recommended structure:
1. Current best hypothesis
2. Evidence from task context
3. Failed approaches to avoid
4. Concrete next-trial plan

Previous experience:
```text
(No prior experience for this task in this rollout.)
```

Newest feedback:
```text
No executor attempt has run yet. Produce the initial strategy for the fixed executor to execute.
```"""


def render_tinker_prompt(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        tokens = tokenizer.encode(f"User: {prompt}\nAssistant:", add_special_tokens=False)
    return list(tokens) + tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)


def clean_strategy(text: str) -> str:
    text = text.replace("<|endoftext|>", "").replace("<|im_end|>", "")
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    heading = re.search(r"(?im)^\s*1\.\s+Current best hypothesis", text)
    if heading:
        text = text[heading.start() :]
    text = re.split(
        r"(?im)^\s*(?:User:|Assistant:|System:|<\|im_start\|>(?:user|assistant|system))",
        text,
        maxsplit=1,
    )[0]
    text = re.split(r"(?is)<think>", text, maxsplit=1)[0]
    return text.strip()


def invalid_strategy(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    return not stripped or "<think>" in lowered or "current best hypothesis" not in lowered


def milestone_value(row: dict[str, Any]) -> int:
    milestone = row.get("milestone")
    value = milestone.get("milestone") if isinstance(milestone, dict) else milestone
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(7, ivalue))


def compute_reward(
    row: dict[str, Any],
    *,
    invalid_format: bool,
    reward_compression: str,
    timeout_penalty: float,
    invalid_format_penalty: float,
) -> float:
    raw = MILESTONE_REWARD[milestone_value(row)]
    reward = math.log1p(raw) if reward_compression == "log1p" else raw
    metadata = row.get("metadata") or {}
    if row.get("status") == "TIMEOUT" or metadata.get("codex_timed_out"):
        reward -= timeout_penalty
    if invalid_format:
        reward -= invalid_format_penalty
    return float(reward)


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

    def kill_group(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

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
            kill_group(signal.SIGTERM)
            try:
                returncode = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                kill_group(signal.SIGKILL)
                returncode = proc.wait()
    return returncode, timed_out, time.monotonic() - started


def synthetic_missing_rollout(
    *,
    args: argparse.Namespace,
    sample: StrategySample,
    returncode: int | None,
    timed_out: bool,
    wall_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    status = "TIMEOUT" if timed_out else "AGENT_ERROR"
    return {
        "run_id": args.run_id,
        "task_id": sample.task_id,
        "agent": f"{args.run_id}-{safe_name(sample.task_id)}_grpo_{sample.step}_{sample.sample_index}",
        "model": args.executor_model,
        "executor": "codex.exec",
        "status": status,
        "milestone": {
            "milestone": 0,
            "reasoning": "executor runner did not produce a rollout row",
            "verified_fix": None,
            "raw": {},
        },
        "verification": {
            "status": "runner_missing_row",
            "passed": False,
            "submit_count": 0,
            "details": {
                "returncode": returncode,
                "timed_out": timed_out,
                "wall_seconds": wall_seconds,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
        },
        "trajectory_path": str(stdout_path),
        "wall_seconds": wall_seconds,
        "strategy_id": sample.sample_id,
        "strategy": sample.text,
        "started_at": now_iso(),
        "finished_at": now_iso(),
        "metadata": {
            "source": "train_tinker_strategy_grpo.py",
            "codex_returncode": returncode,
            "codex_timed_out": timed_out,
        },
    }


def augment_rollout_row(row: dict[str, Any], sample: StrategySample) -> dict[str, Any]:
    row = json.loads(json.dumps(row))
    row["strategy_id"] = sample.sample_id
    row["strategy"] = sample.text
    metadata = row.setdefault("metadata", {})
    metadata["grpo"] = {
        "source": "train_tinker_strategy_grpo.py",
        "sample_id": sample.sample_id,
        "step": sample.step,
        "task_index": sample.task_index,
        "sample_index": sample.sample_index,
        "group_index": sample.group_index,
        "group_sample_index": sample.group_sample_index,
        "strategy_path": sample.strategy_path,
        "invalid_format": sample.invalid_format,
        "sampled_tokens": sample.sampled_tokens,
        "prompt_tokens": sample.prompt_token_count,
    }
    verification = row.setdefault("verification", {})
    details = verification.setdefault("details", {})
    details["grpo"] = metadata["grpo"]
    return row


def run_executor_rollout(
    *,
    args: argparse.Namespace,
    sample: StrategySample,
    step_dir: Path,
    env: dict[str, str],
) -> RolloutResult:
    rollout_output = step_dir / "rollout_outputs" / f"{sample.sample_id}.jsonl"
    stdout_path = step_dir / "executor_logs" / f"{sample.sample_id}.stdout.log"
    stderr_path = step_dir / "executor_logs" / f"{sample.sample_id}.stderr.log"
    agent_suffix = f"_grpo_s{sample.step:04d}_k{sample.sample_index:02d}_{safe_name(sample.sample_id)[-12:]}"

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
        str(args.executor_run_root),
        "--output",
        str(rollout_output),
        "--task-id",
        sample.task_id,
        "--rerun-recorded",
        "--agent-suffix",
        agent_suffix,
        "--attempt-index",
        str(sample.step),
        "--strategy-file",
        sample.strategy_path,
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
    ]
    if args.force_workspace:
        command.append("--force-workspace")
    if args.executor_model:
        command.extend(["--model", args.executor_model])
    if args.executor_reasoning_effort:
        command.extend(["--model-reasoning-effort", args.executor_reasoning_effort])
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

    timeout_seconds = args.codex_timeout_seconds + args.submit_timeout_seconds + 600
    returncode, timed_out, wall_seconds = run_command(
        command,
        cwd=ROOT,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=timeout_seconds,
    )
    rows = [row for row in load_jsonl(rollout_output) if row.get("task_id") == sample.task_id]
    if rows:
        row = augment_rollout_row(rows[-1], sample)
    else:
        row = synthetic_missing_rollout(
            args=args,
            sample=sample,
            returncode=returncode,
            timed_out=timed_out,
            wall_seconds=wall_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        row = augment_rollout_row(row, sample)
    return RolloutResult(
        sample_id=sample.sample_id,
        task_id=sample.task_id,
        sample_index=sample.sample_index,
        row=row,
        returncode=returncode,
        timed_out=timed_out,
        wall_seconds=wall_seconds,
        output_path=str(rollout_output),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


async def maybe_compute_logprobs(
    *,
    sampling_client: Any,
    tinker: Any,
    prompt_tokens: list[int],
    generated_tokens: list[int],
) -> list[float]:
    full = prompt_tokens + generated_tokens
    logprobs = await sampling_client.compute_logprobs_async(tinker.ModelInput.from_ints(full))
    selected = [value for value in logprobs[len(prompt_tokens) :] if value is not None]
    return [float(value) for value in selected]


async def sample_strategies_for_task(
    *,
    args: argparse.Namespace,
    tinker: Any,
    sampling_client: Any,
    tokenizer: Any,
    task_id: str,
    task_index: int,
    step: int,
    task_meta: TaskMetadata | None,
    data_dir: Path,
    step_dir: Path,
) -> list[StrategySample]:
    task_context = build_task_context(
        task_meta,
        data_dir,
        task_id,
        max_chars=args.task_context_max_chars,
    )
    prompt = build_strategy_prompt(
        task_id=task_id,
        difficulty=args.difficulty,
        token_budget=args.max_strategy_tokens,
        task_context=task_context,
    )
    prompt_tokens = render_tinker_prompt(tokenizer, prompt)
    prompt_input = tinker.ModelInput.from_ints(prompt_tokens)
    params = tinker.SamplingParams(
        max_tokens=args.max_strategy_tokens,
        stop=["<|im_end|>", "\nUser:", "\n<|im_start|>user"],
        temperature=args.strategy_temperature,
        top_p=args.strategy_top_p,
    )
    result = await sampling_client.sample_async(
        prompt=prompt_input,
        num_samples=args.rollout_pool_per_task,
        sampling_params=params,
    )
    out: list[StrategySample] = []
    strategy_dir = step_dir / "strategies" / safe_name(task_id)
    strategy_dir.mkdir(parents=True, exist_ok=True)
    for sample_index, sequence in enumerate(result.sequences):
        group_index = sample_index // args.advantage_group_size
        group_sample_index = sample_index % args.advantage_group_size
        sample_id = (
            f"s{step:04d}__t{task_index:03d}__g{group_index:02d}"
            f"__k{group_sample_index:02d}__{safe_name(task_id)}"
        )
        tokens = list(sequence.tokens or [])
        logprobs = [float(value) for value in (sequence.logprobs or []) if value is not None]
        if tokens and len(logprobs) != len(tokens):
            try:
                logprobs = await maybe_compute_logprobs(
                    sampling_client=sampling_client,
                    tinker=tinker,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=tokens,
                )
            except Exception:
                logprobs = []
        decoded = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
        text = decoded if decoded.lstrip().startswith(ASSISTANT_PREFIX) else ASSISTANT_PREFIX + decoded
        text = clean_strategy(text)
        strategy_path = strategy_dir / f"{sample_id}.md"
        strategy_path.write_text(text.rstrip() + "\n")
        out.append(
            StrategySample(
                sample_id=sample_id,
                step=step,
                task_id=task_id,
                task_index=task_index,
                sample_index=sample_index,
                group_index=group_index,
                group_sample_index=group_sample_index,
                prompt_tokens=prompt_tokens,
                tokens=tokens,
                logprobs=logprobs,
                text=text,
                stop_reason=str(getattr(sequence, "stop_reason", "")),
                strategy_path=str(strategy_path),
                invalid_format=invalid_strategy(text),
                sampled_tokens=len(tokens),
                prompt_token_count=len(prompt_tokens),
            )
        )
    return out


def tensor_int(values: list[int], tinker: Any) -> Any:
    return tinker.TensorData(data=values, dtype="int64")


def tensor_float(values: list[float], tinker: Any) -> Any:
    return tinker.TensorData(data=values, dtype="float32")


def build_grpo_datum(tinker: Any, sample: StrategySample, advantage: float) -> Any | None:
    if not sample.tokens or not sample.logprobs or len(sample.tokens) != len(sample.logprobs):
        return None
    prompt = tinker.ModelInput.from_ints(sample.prompt_tokens)
    model_input = prompt.append(tinker.EncodedTextChunk(tokens=sample.tokens[:-1]))
    ob_len = prompt.length - 1
    target_tokens = [0] * ob_len + sample.tokens
    padded_logprobs = [0.0] * ob_len + sample.logprobs
    padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tensor_int(target_tokens, tinker),
            "logprobs": tensor_float(padded_logprobs, tinker),
            "advantages": tensor_float(padded_advantages, tinker),
        },
    )


def build_group_record_and_datums(
    *,
    args: argparse.Namespace,
    tinker: Any,
    step: int,
    task_id: str,
    group_index: int,
    group_samples: list[StrategySample],
    rows_by_sample: dict[str, dict[str, Any]],
    rewards_by_sample: dict[str, float],
) -> tuple[dict[str, Any], list[Any]]:
    group_samples = sorted(group_samples, key=lambda sample: sample.group_sample_index)
    complete_group = (
        len(group_samples) == args.advantage_group_size
        and all(sample.sample_id in rows_by_sample for sample in group_samples)
    )
    base_record = {
        "event": "group",
        "step": step,
        "task_id": task_id,
        "group_index": group_index,
        "advantage_group_size": args.advantage_group_size,
        "rollout_pool_per_task": args.rollout_pool_per_task,
        "sample_ids": [sample.sample_id for sample in group_samples],
        "time": now_iso(),
    }
    if not complete_group:
        return (
            {
                **base_record,
                "rewards": [],
                "advantages": [],
                "mean_reward": None,
                "reward_std": None,
                "uniform": None,
                "skipped_update": True,
                "skip_reason": "incomplete_group",
                "datums": 0,
                "milestones": [
                    milestone_value(rows_by_sample[sample.sample_id])
                    for sample in group_samples
                    if sample.sample_id in rows_by_sample
                ],
                "statuses": [
                    rows_by_sample[sample.sample_id].get("status")
                    for sample in group_samples
                    if sample.sample_id in rows_by_sample
                ],
            },
            [],
        )

    rewards = [rewards_by_sample[sample.sample_id] for sample in group_samples]
    mean_reward = statistics.mean(rewards) if rewards else 0.0
    reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    uniform = reward_std <= args.advantage_eps
    advantages = [
        0.0 if uniform else (reward - mean_reward) / (reward_std + args.advantage_eps)
        for reward in rewards
    ]
    datums: list[Any] = []
    if not (uniform and args.skip_uniform_groups):
        for sample, advantage in zip(group_samples, advantages):
            datum = build_grpo_datum(tinker, sample, advantage)
            if datum is not None:
                datums.append(datum)
    return (
        {
            **base_record,
            "rewards": rewards,
            "advantages": advantages,
            "mean_reward": mean_reward,
            "reward_std": reward_std,
            "uniform": uniform,
            "skipped_update": bool(uniform and args.skip_uniform_groups),
            "skip_reason": "uniform_group" if uniform and args.skip_uniform_groups else None,
            "datums": len(datums),
            "milestones": [
                milestone_value(rows_by_sample[sample.sample_id])
                for sample in group_samples
                if sample.sample_id in rows_by_sample
            ],
            "statuses": [
                rows_by_sample[sample.sample_id].get("status")
                for sample in group_samples
                if sample.sample_id in rows_by_sample
            ],
        },
        datums,
    )


def summarize_numbers(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None, "std": None}
    return {
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str((ROOT / "src").resolve()), str((ROOT / "cybergym/src").resolve())]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def interleaved_rollout_samples(args: argparse.Namespace, samples: list[StrategySample]) -> list[StrategySample]:
    if not samples:
        return []
    samples_by_task: dict[str, list[StrategySample]] = {}
    for sample in samples:
        samples_by_task.setdefault(sample.task_id, []).append(sample)
    interleaved_samples: list[StrategySample] = []
    for group in samples_by_task.values():
        group.sort(key=lambda sample: (sample.group_index, sample.group_sample_index))
    samples_by_position = {
        (sample.task_id, sample.group_index, sample.group_sample_index): sample
        for sample in samples
    }
    for group_index in range(args.rollout_groups_per_task):
        for group_sample_index in range(args.advantage_group_size):
            for task_id in sorted(samples_by_task):
                sample = samples_by_position.get((task_id, group_index, group_sample_index))
                if sample is not None:
                    interleaved_samples.append(sample)
    if len(interleaved_samples) != len(samples):
        seen = {sample.sample_id for sample in interleaved_samples}
        for task_id in sorted(samples_by_task):
            for sample in samples_by_task[task_id]:
                if sample.sample_id not in seen:
                    interleaved_samples.append(sample)
    return interleaved_samples


def execute_rollouts(
    *,
    args: argparse.Namespace,
    samples: list[StrategySample],
    step_dir: Path,
    env: dict[str, str],
) -> list[RolloutResult]:
    if not samples:
        return []
    interleaved_samples = interleaved_rollout_samples(args, samples)
    workers = max(1, min(args.executor_workers, len(samples)))
    out: list[RolloutResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_executor_rollout, args=args, sample=sample, step_dir=step_dir, env=env)
            for sample in interleaved_samples
        ]
        for future in as_completed(futures):
            out.append(future.result())
    out.sort(key=lambda r: (r.task_id, r.sample_index))
    return out


async def save_checkpoint(
    *,
    training_client: Any,
    out_dir: Path,
    run_id: str,
    step: int,
    kind: str,
) -> dict[str, Any]:
    name = safe_tinker_label(f"{run_id}-{kind}-step-{step:04d}")
    future = await training_client.save_state_async(name, overwrite=True)
    result = await future.result_async()
    record = {"event": "checkpoint", "kind": kind, "step": step, "path": result.path, "time": now_iso()}
    append_jsonl(out_dir / "checkpoints.jsonl", record)
    return record


async def save_sampler_weights(
    *,
    training_client: Any,
    out_dir: Path,
    run_id: str,
    step: int,
    kind: str,
) -> dict[str, Any]:
    name = safe_tinker_label(f"{run_id}-{kind}-sampler-step-{step:04d}")
    future = await training_client.save_weights_for_sampler_async(name)
    result = await future.result_async()
    record = {"event": "sampler_weights", "kind": kind, "step": step, "path": result.path, "time": now_iso()}
    append_jsonl(out_dir / "checkpoints.jsonl", record)
    return record


async def run_grpo_update(
    *,
    args: argparse.Namespace,
    training_client: Any,
    adam_params: Any,
    out_dir: Path,
    config: dict[str, float] | None,
    datums: list[Any],
    step: int,
    update_index: int,
    group_keys: list[tuple[str, int]],
) -> dict[str, Any]:
    update_rec: dict[str, Any] = {
        "event": "update_skipped",
        "step": step,
        "update_index": update_index,
        "group_keys": [{"task_id": task_id, "group_index": group_index} for task_id, group_index in group_keys],
        "reason": None,
        "datums": len(datums),
        "time": now_iso(),
    }
    if args.skip_grpo_update:
        update_rec["reason"] = "skip_grpo_update"
    elif not datums:
        update_rec["reason"] = "no_datums"
    else:
        t0 = time.monotonic()
        fwd_future = await training_client.forward_backward_async(datums, args.loss_fn, config)
        opt_future = await training_client.optim_step_async(adam_params)
        fwd_result = await fwd_future.result_async()
        opt_result = await opt_future.result_async()
        update_rec = {
            "event": "update",
            "step": step,
            "update_index": update_index,
            "group_keys": [{"task_id": task_id, "group_index": group_index} for task_id, group_index in group_keys],
            "loss_fn": args.loss_fn,
            "loss_fn_config": config,
            "datums": len(datums),
            "metrics": dict(fwd_result.metrics),
            "optim": getattr(opt_result, "model_dump", lambda: str(opt_result))(),
            "elapsed": time.monotonic() - t0,
            "time": now_iso(),
        }
    append_jsonl(out_dir / "metrics.jsonl", update_rec)
    print(json.dumps({k: v for k, v in update_rec.items() if k != "optim"}, sort_keys=True), flush=True)
    return update_rec


def loss_config(args: argparse.Namespace) -> dict[str, float] | None:
    if not args.loss_fn_config_json:
        return None
    raw = json.loads(args.loss_fn_config_json)
    if not isinstance(raw, dict):
        raise ValueError("--loss-fn-config-json must decode to a JSON object")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"loss config value for {key!r} must be numeric")
        out[str(key)] = float(value)
    return out


async def main_async() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(f"TINKER_API_KEY is not set; checked {args.env_file}")
    if args.group_size < 1:
        raise ValueError("--group-size must be >= 1")
    if args.advantage_group_size is None:
        args.advantage_group_size = args.group_size
    if args.rollout_pool_per_task is None:
        args.rollout_pool_per_task = args.group_size
    if args.advantage_group_size < 1:
        raise ValueError("--advantage-group-size must be >= 1")
    if args.rollout_pool_per_task < args.advantage_group_size:
        raise ValueError("--rollout-pool-per-task must be >= --advantage-group-size")
    if args.rollout_pool_per_task % args.advantage_group_size != 0:
        raise ValueError("--rollout-pool-per-task must be a multiple of --advantage-group-size")
    args.rollout_groups_per_task = args.rollout_pool_per_task // args.advantage_group_size
    if args.tasks_per_step < 1:
        raise ValueError("--tasks-per-step must be >= 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.executor_workers < 1:
        raise ValueError("--executor-workers must be >= 1")
    if args.groups_per_update < 1:
        raise ValueError("--groups-per-update must be >= 1")

    init_state_path = resolve_init_state(args)
    run_id = args.run_id or f"qwen36-strategy-grpo-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    args.run_id = run_id
    out_dir = args.out_dir or (args.out_root / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    task_metadata = load_task_metadata(manifest.benchmark.tasks_json)
    task_ids = selected_task_ids(args)
    if not task_ids:
        raise ValueError("No task ids selected")
    task_plan_path = out_dir / "task_ids.txt"
    task_plan_path.write_text("\n".join(task_ids) + "\n")

    config = loss_config(args)
    manifest_row = {
        "event": "start",
        "run_id": run_id,
        "time": now_iso(),
        "script": "train_tinker_strategy_grpo.py",
        "model": args.model,
        "init_state_path": init_state_path,
        "task_ids_path": str(task_plan_path),
        "task_count": len(task_ids),
        "hyperparameters": {
            "split": args.split,
            "difficulty": args.difficulty,
            "task_sampling": args.task_sampling,
            "group_size": args.advantage_group_size,
            "advantage_group_size": args.advantage_group_size,
            "rollout_pool_per_task": args.rollout_pool_per_task,
            "rollout_groups_per_task": args.rollout_groups_per_task,
            "tasks_per_step": args.tasks_per_step,
            "max_steps": args.max_steps,
            "max_strategy_tokens": args.max_strategy_tokens,
            "strategy_temperature": args.strategy_temperature,
            "strategy_top_p": args.strategy_top_p,
            "loss_fn": args.loss_fn,
            "loss_fn_config": config,
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "skip_grpo_update": args.skip_grpo_update,
            "skip_uniform_groups": args.skip_uniform_groups,
            "update_as_groups_complete": args.update_as_groups_complete,
            "groups_per_update": args.groups_per_update,
            "save_after_update": args.save_after_update,
            "reward_compression": args.reward_compression,
            "timeout_penalty": args.timeout_penalty,
            "invalid_format_penalty": args.invalid_format_penalty,
        },
        "executor": {
            "runner": str(args.runner),
            "model": args.executor_model,
            "workers": args.executor_workers,
            "codex_provider": args.codex_provider,
            "server": args.server,
            "pocdb_path": str(args.pocdb_path),
        },
    }
    write_json(out_dir / "manifest.json", manifest_row)
    print(json.dumps(manifest_row, sort_keys=True), flush=True)

    import tinker

    service = tinker.ServiceClient(
        user_metadata={"purpose": "cybergym_strategy_grpo", "run_id": run_id}
    )
    if init_state_path:
        training_client = await service.create_training_client_from_state_async(
            init_state_path,
            user_metadata={"run_id": run_id, "purpose": "strategy_grpo_from_sft"},
        )
    else:
        training_client = await service.create_lora_training_client_async(
            base_model=args.model,
            rank=args.lora_rank,
            seed=args.seed,
            train_mlp=True,
            train_attn=True,
            train_unembed=False,
            user_metadata={"run_id": run_id, "purpose": "strategy_grpo_from_base"},
        )
    tokenizer = training_client.get_tokenizer()
    adam_params = tinker.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
    )
    env = build_env(args)
    rng = random.Random(args.seed)
    train_order = list(task_ids)

    for step in range(1, args.max_steps + 1):
        step_started = time.monotonic()
        if args.task_sampling == "random":
            if len(train_order) >= args.tasks_per_step:
                step_task_ids = rng.sample(train_order, args.tasks_per_step)
            else:
                step_task_ids = [rng.choice(train_order) for _ in range(args.tasks_per_step)]
        elif (step - 1) * args.tasks_per_step >= len(train_order):
            rng.shuffle(train_order)
            start = ((step - 1) * args.tasks_per_step) % len(train_order)
            step_task_ids = train_order[start : start + args.tasks_per_step]
            if len(step_task_ids) < args.tasks_per_step:
                step_task_ids.extend(train_order[: args.tasks_per_step - len(step_task_ids)])
        else:
            start = ((step - 1) * args.tasks_per_step) % len(train_order)
            step_task_ids = train_order[start : start + args.tasks_per_step]
            if len(step_task_ids) < args.tasks_per_step:
                step_task_ids.extend(train_order[: args.tasks_per_step - len(step_task_ids)])
        step_dir = out_dir / f"step_{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        append_jsonl(
            out_dir / "metrics.jsonl",
            {"event": "step_start", "step": step, "task_ids": step_task_ids, "time": now_iso()},
        )
        print(json.dumps({"event": "step_start", "step": step, "task_ids": step_task_ids}, sort_keys=True), flush=True)

        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            safe_tinker_label(f"{run_id}-policy-step-{step:04d}")
        )
        sample_tasks = [
            sample_strategies_for_task(
                args=args,
                tinker=tinker,
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                task_id=task_id,
                task_index=task_index,
                step=step,
                task_meta=task_metadata.get(task_id),
                data_dir=manifest.benchmark.data_dir,
                step_dir=step_dir,
            )
            for task_index, task_id in enumerate(step_task_ids)
        ]
        nested_samples = await asyncio.gather(*sample_tasks)
        samples = [sample for group in nested_samples for sample in group]
        sample_records_path = step_dir / "sample_records.jsonl"
        for sample in samples:
            append_jsonl(sample_records_path, asdict(sample))
            append_jsonl(
                out_dir / "strategies.jsonl",
                {
                    "event": "strategy_sample",
                    "sample_id": sample.sample_id,
                    "step": step,
                    "task_id": sample.task_id,
                    "sample_index": sample.sample_index,
                    "group_index": sample.group_index,
                    "group_sample_index": sample.group_sample_index,
                    "strategy_path": sample.strategy_path,
                    "invalid_format": sample.invalid_format,
                    "sampled_tokens": sample.sampled_tokens,
                    "prompt_tokens": sample.prompt_token_count,
                    "stop_reason": sample.stop_reason,
                    "text": sample.text,
                    "time": now_iso(),
                },
            )

        append_jsonl(
            out_dir / "metrics.jsonl",
            {
                "event": "sample_done",
                "step": step,
                "samples": len(samples),
                "tasks": len(step_task_ids),
                "advantage_group_size": args.advantage_group_size,
                "rollout_pool_per_task": args.rollout_pool_per_task,
                "rollout_groups_per_task": args.rollout_groups_per_task,
                "invalid_format": sum(sample.invalid_format for sample in samples),
                "sampled_tokens": summarize_numbers([float(sample.sampled_tokens) for sample in samples]),
                "time": now_iso(),
            },
        )
        print(json.dumps({"event": "sample_done", "step": step, "samples": len(samples)}, sort_keys=True), flush=True)

        samples_by_id = {sample.sample_id: sample for sample in samples}
        task_positions = {task_id: idx for idx, task_id in enumerate(step_task_ids)}
        samples_by_group: dict[tuple[str, int], list[StrategySample]] = {}
        for sample in samples:
            samples_by_group.setdefault((sample.task_id, sample.group_index), []).append(sample)
        ordered_group_keys = [
            key
            for key, _ in sorted(
                samples_by_group.items(),
                key=lambda item: (task_positions.get(item[0][0], 10**9), item[0][1]),
            )
        ]
        rollout_results: list[RolloutResult] = []
        rewards_by_sample: dict[str, float] = {}
        rows_by_sample: dict[str, dict[str, Any]] = {}
        group_records: list[dict[str, Any]] = []
        completed_group_keys: set[tuple[str, int]] = set()
        pending_datums: list[Any] = []
        pending_group_keys: list[tuple[str, int]] = []
        update_records: list[dict[str, Any]] = []
        update_index = 0

        def add_rollout_result(rollout: RolloutResult) -> None:
            rollout_results.append(rollout)
            rows_by_sample[rollout.sample_id] = rollout.row
            append_jsonl(out_dir / "rollouts.jsonl", rollout.row)
            sample = samples_by_id[rollout.sample_id]
            rewards_by_sample[rollout.sample_id] = compute_reward(
                rollout.row,
                invalid_format=sample.invalid_format,
                reward_compression=args.reward_compression,
                timeout_penalty=args.timeout_penalty,
                invalid_format_penalty=args.invalid_format_penalty,
            )
            append_jsonl(
                out_dir / "metrics.jsonl",
                {
                    "event": "rollout_done",
                    "step": step,
                    "sample_id": rollout.sample_id,
                    "task_id": rollout.task_id,
                    "sample_index": rollout.sample_index,
                    "group_index": sample.group_index,
                    "group_sample_index": sample.group_sample_index,
                    "status": rollout.row.get("status"),
                    "milestone": milestone_value(rollout.row),
                    "reward": rewards_by_sample[rollout.sample_id],
                    "returncode": rollout.returncode,
                    "timed_out": rollout.timed_out,
                    "wall_seconds": rollout.wall_seconds,
                    "stdout_path": rollout.stdout_path,
                    "stderr_path": rollout.stderr_path,
                    "output_path": rollout.output_path,
                    "time": now_iso(),
                },
            )

        async def flush_update(group_keys: list[tuple[str, int]], datums: list[Any]) -> None:
            nonlocal update_index
            update_index += 1
            update_record = await run_grpo_update(
                args=args,
                training_client=training_client,
                adam_params=adam_params,
                out_dir=out_dir,
                config=config,
                datums=datums,
                step=step,
                update_index=update_index,
                group_keys=group_keys,
            )
            update_records.append(update_record)
            if args.save_after_update and update_record.get("event") == "update":
                checkpoint_records = [
                    await save_checkpoint(
                        training_client=training_client,
                        out_dir=out_dir,
                        run_id=run_id,
                        step=step,
                        kind=f"update-{update_index:04d}",
                    ),
                    await save_sampler_weights(
                        training_client=training_client,
                        out_dir=out_dir,
                        run_id=run_id,
                        step=step,
                        kind=f"update-{update_index:04d}",
                    ),
                ]
                append_jsonl(
                    out_dir / "metrics.jsonl",
                    {
                        "event": "update_checkpoint",
                        "step": step,
                        "update_index": update_index,
                        "checkpoint_records": checkpoint_records,
                        "time": now_iso(),
                    },
                )

        if args.update_as_groups_complete:
            interleaved_samples = interleaved_rollout_samples(args, samples)
            workers = max(1, min(args.executor_workers, len(samples)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(run_executor_rollout, args=args, sample=sample, step_dir=step_dir, env=env)
                    for sample in interleaved_samples
                ]
                for future in as_completed(futures):
                    try:
                        rollout = future.result()
                    except Exception as exc:
                        append_jsonl(
                            out_dir / "metrics.jsonl",
                            {
                                "event": "rollout_exception",
                                "step": step,
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                                "time": now_iso(),
                            },
                        )
                        raise
                    add_rollout_result(rollout)
                    sample = samples_by_id[rollout.sample_id]
                    key = (sample.task_id, sample.group_index)
                    group_samples = samples_by_group[key]
                    if key in completed_group_keys:
                        continue
                    if not all(group_sample.sample_id in rows_by_sample for group_sample in group_samples):
                        continue
                    completed_group_keys.add(key)
                    group_record, group_datums = build_group_record_and_datums(
                        args=args,
                        tinker=tinker,
                        step=step,
                        task_id=sample.task_id,
                        group_index=sample.group_index,
                        group_samples=group_samples,
                        rows_by_sample=rows_by_sample,
                        rewards_by_sample=rewards_by_sample,
                    )
                    group_records.append(group_record)
                    append_jsonl(out_dir / "groups.jsonl", group_record)
                    if group_datums:
                        pending_datums.extend(group_datums)
                        pending_group_keys.append(key)
                        if len(pending_group_keys) >= args.groups_per_update:
                            await flush_update(pending_group_keys, pending_datums)
                            pending_datums = []
                            pending_group_keys = []
            if pending_group_keys or (not update_records and not args.skip_grpo_update):
                await flush_update(pending_group_keys, pending_datums)
        else:
            for rollout in execute_rollouts(args=args, samples=samples, step_dir=step_dir, env=env):
                add_rollout_result(rollout)
            all_datums: list[Any] = []
            for task_id, group_index in ordered_group_keys:
                group_record, group_datums = build_group_record_and_datums(
                    args=args,
                    tinker=tinker,
                    step=step,
                    task_id=task_id,
                    group_index=group_index,
                    group_samples=samples_by_group[(task_id, group_index)],
                    rows_by_sample=rows_by_sample,
                    rewards_by_sample=rewards_by_sample,
                )
                group_records.append(group_record)
                append_jsonl(out_dir / "groups.jsonl", group_record)
                all_datums.extend(group_datums)
            await flush_update(ordered_group_keys, all_datums)

        reward_values = list(rewards_by_sample.values())
        milestone_counts = Counter(str(milestone_value(rollout.row)) for rollout in rollout_results)
        status_counts = Counter(str(rollout.row.get("status") or "UNKNOWN") for rollout in rollout_results)
        datums_count = sum(int(record.get("datums") or 0) for record in group_records)

        checkpoint_records: list[dict[str, Any]] = []
        if args.save_every and step % args.save_every == 0:
            checkpoint_records.append(
                await save_checkpoint(
                    training_client=training_client,
                    out_dir=out_dir,
                    run_id=run_id,
                    step=step,
                    kind="periodic",
                )
            )
            checkpoint_records.append(
                await save_sampler_weights(
                    training_client=training_client,
                    out_dir=out_dir,
                    run_id=run_id,
                    step=step,
                    kind="periodic",
                )
            )

        step_rec = {
            "event": "step_complete",
            "step": step,
            "tasks": len(step_task_ids),
            "samples": len(samples),
            "rollouts": len(rollout_results),
            "groups": len(group_records),
            "datums": datums_count,
            "updates": len([record for record in update_records if record.get("event") == "update"]),
            "skipped_updates": len(
                [record for record in update_records if record.get("event") == "update_skipped"]
            ),
            "mean_reward": statistics.mean(reward_values) if reward_values else None,
            "reward": summarize_numbers(reward_values),
            "uniform_groups": sum(bool(record.get("uniform")) for record in group_records),
            "skipped_groups": sum(bool(record.get("skipped_update")) for record in group_records),
            "milestone_counts": dict(sorted(milestone_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "checkpoint_records": checkpoint_records,
            "elapsed": time.monotonic() - step_started,
            "time": now_iso(),
        }
        append_jsonl(out_dir / "metrics.jsonl", step_rec)
        print(json.dumps(step_rec, sort_keys=True), flush=True)

    final_state = await save_checkpoint(
        training_client=training_client,
        out_dir=out_dir,
        run_id=run_id,
        step=args.max_steps,
        kind="final",
    )
    final_sampler = await save_sampler_weights(
        training_client=training_client,
        out_dir=out_dir,
        run_id=run_id,
        step=args.max_steps,
        kind="final",
    )
    result = {
        "event": "complete",
        "run_id": run_id,
        "final_state_path": final_state["path"],
        "final_sampler_path": final_sampler["path"],
        "steps": args.max_steps,
        "time": now_iso(),
    }
    write_json(out_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
