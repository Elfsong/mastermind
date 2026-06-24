#!/usr/bin/env python3
"""Train a Qwen3.6 one-step CyberGym strategy generator with Tinker SFT.

This intentionally uses the low-level Tinker SDK instead of tinker_cookbook,
because this repo's current environment has tinker installed but not the
cookbook package. The script masks loss to assistant completion tokens only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "runs/strategy_sft/qwen36_strategy_sft_data"
DEFAULT_OUT_ROOT = ROOT / "runs/strategy_sft"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-length", type=int, default=12_288)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--train-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-unembed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default false to reduce overfit and adapter size for this small SFT set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only tokenize/build batches locally; do not call Tinker.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def safe_tinker_label(value: str) -> str:
    """Tinker weight labels may only contain alnum, hyphen, underscore, dot."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return safe or "checkpoint"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def manual_message_tokens(tokenizer: Any, role: str, content: str) -> list[int]:
    return tokenizer.encode(f"<|im_start|>{role}\n{content}<|im_end|>\n", add_special_tokens=False)


def manual_chat_prompt_tokens(
    tokenizer: Any,
    *,
    prompt_messages: list[dict[str, str]],
    assistant_prefix: str,
    target_len: int,
    max_length: int,
) -> list[int]:
    """Render prompt, truncating the last prompt message body from the front if needed."""
    assistant_start = tokenizer.encode(
        f"<|im_start|>assistant\n{assistant_prefix}", add_special_tokens=False
    )
    budget = max_length - target_len
    if budget <= len(assistant_start) + 16:
        raise ValueError(
            f"target is too long for max_length={max_length}: target_len={target_len}"
        )
    if not prompt_messages:
        return assistant_start

    fixed: list[int] = []
    for msg in prompt_messages[:-1]:
        fixed.extend(manual_message_tokens(tokenizer, msg["role"], msg["content"]))

    last = prompt_messages[-1]
    header = tokenizer.encode(f"<|im_start|>{last['role']}\n", add_special_tokens=False)
    body = tokenizer.encode(str(last["content"]), add_special_tokens=False)
    end = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    fixed_len = len(fixed) + len(header) + len(end) + len(assistant_start)
    body_budget = budget - fixed_len
    if body_budget < len(body):
        marker = tokenizer.encode(
            "\n[...prompt truncated from the front to fit context...]\n",
            add_special_tokens=False,
        )
        body_budget = max(0, body_budget - len(marker))
        body = marker + body[-body_budget:] if body_budget else marker
    prompt = fixed + header + body + end + assistant_start
    if len(prompt) > budget:
        overflow = len(prompt) - budget
        prompt = prompt[: len(fixed) + len(header)] + prompt[len(fixed) + len(header) + overflow :]
    return prompt


@dataclass
class TokenizedExample:
    row_id: str
    task_id: str
    model_input_tokens: list[int]
    target_tokens: list[int]
    weights: list[float]
    prompt_len: int
    full_len: int
    target_len: int
    truncated: bool


def tokenize_example(tokenizer: Any, row: dict[str, Any], max_length: int) -> TokenizedExample:
    messages = row["messages"]
    prompt_messages = messages[:-1]
    assistant_prefix = str(row.get("prompt_assistant_prefix") or "")
    target_text = str(row.get("target_completion") or messages[-1]["content"])
    target_tokens = tokenizer.encode(target_text + "<|im_end|>\n", add_special_tokens=False)
    prompt_tokens = manual_chat_prompt_tokens(
        tokenizer,
        prompt_messages=prompt_messages,
        assistant_prefix=assistant_prefix,
        target_len=len(target_tokens),
        max_length=max_length,
    )
    full = prompt_tokens + target_tokens
    truncated = False
    if len(full) > max_length:
        # This should be rare because manual_chat_prompt_tokens budgets for target_len.
        full = full[-max_length:]
        truncated = True
        prompt_tokens = full[: max(1, len(full) - len(target_tokens))]
    model_input = full[:-1]
    shifted_targets = full[1:]
    # Assistant completion loss starts when predicting the first target token.
    loss_start = max(0, len(prompt_tokens) - 1)
    weights = [0.0] * loss_start + [1.0] * (len(model_input) - loss_start)
    return TokenizedExample(
        row_id=str(row.get("id")),
        task_id=str(row.get("task_id")),
        model_input_tokens=model_input,
        target_tokens=shifted_targets,
        weights=weights,
        prompt_len=len(prompt_tokens),
        full_len=len(full),
        target_len=len(target_tokens),
        truncated=truncated,
    )


def build_datum(tinker_types: Any, tokenized: TokenizedExample) -> Any:
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(tokenized.model_input_tokens),
        loss_fn_inputs={
            "target_tokens": tokenized.target_tokens,
            "weights": tokenized.weights,
        },
    )


def batches(rows: list[Any], batch_size: int) -> list[list[Any]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def summarize_token_lengths(tokenized: list[TokenizedExample]) -> dict[str, Any]:
    if not tokenized:
        return {}
    lengths = sorted(t.full_len for t in tokenized)
    targets = sorted(t.target_len for t in tokenized)

    def q(xs: list[int], frac: float) -> int:
        return xs[min(len(xs) - 1, int(len(xs) * frac))]

    return {
        "n": len(tokenized),
        "full": {
            "min": lengths[0],
            "p50": q(lengths, 0.50),
            "p90": q(lengths, 0.90),
            "p95": q(lengths, 0.95),
            "p99": q(lengths, 0.99),
            "max": lengths[-1],
        },
        "target": {
            "min": targets[0],
            "p50": q(targets, 0.50),
            "p90": q(targets, 0.90),
            "p95": q(targets, 0.95),
            "p99": q(targets, 0.99),
            "max": targets[-1],
        },
        "truncated": sum(t.truncated for t in tokenized),
    }


def metric_loss(metrics: dict[str, float]) -> float | None:
    for key in ("loss", "cross_entropy", "ce_loss", "mean_loss", "nll"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for key, value in metrics.items():
        if "loss" in key and isinstance(value, (int, float)):
            return float(value)
    return None


async def evaluate_loss(
    *,
    training_client: Any,
    tinker_types: Any,
    tokenized_eval: list[TokenizedExample],
    batch_size: int,
    limit: int,
) -> dict[str, Any]:
    selected = tokenized_eval[:limit] if limit > 0 else tokenized_eval
    out: list[dict[str, Any]] = []
    losses: list[float] = []
    for batch in batches(selected, batch_size):
        datums = [build_datum(tinker_types, item) for item in batch]
        future = await training_client.forward_async(datums, "cross_entropy")
        result = await future.result_async()
        loss = metric_loss(dict(result.metrics))
        if loss is not None and math.isfinite(loss):
            losses.append(loss)
        out.append({"metrics": dict(result.metrics), "rows": len(batch)})
    return {
        "rows": len(selected),
        "batches": len(out),
        "loss_mean_unweighted": sum(losses) / len(losses) if losses else None,
        "raw": out,
    }


async def save_checkpoint(
    *,
    training_client: Any,
    name: str,
    out_dir: Path,
    step: int,
    kind: str,
) -> dict[str, Any]:
    future = await training_client.save_state_async(name, overwrite=True)
    result = await future.result_async()
    record = {"event": "checkpoint", "kind": kind, "step": step, "path": result.path, "time": now_iso()}
    with (out_dir / "checkpoints.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


async def save_sampler_weights(
    *,
    training_client: Any,
    name: str,
    out_dir: Path,
    step: int,
) -> dict[str, Any]:
    future = await training_client.save_weights_for_sampler_async(name)
    result = await future.result_async()
    record = {"event": "sampler_weights", "step": step, "path": result.path, "time": now_iso()}
    with (out_dir / "checkpoints.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


async def main_async() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    if not os.environ.get("TINKER_API_KEY") and not args.dry_run:
        raise RuntimeError(f"TINKER_API_KEY is not set; checked {args.env_file}")

    run_id = args.run_id or f"qwen36-strategy-sft-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    val_rows = load_jsonl(args.data_dir / "val.jsonl")

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=True
        )
        tinker = None
        training_client = None
    else:
        import tinker

        service = tinker.ServiceClient(
            user_metadata={
                "purpose": "cybergym_strategy_sft",
                "run_id": run_id,
                "script": "train_tinker_strategy_sft.py",
            }
        )
        training_client = await service.create_lora_training_client_async(
            base_model=args.model,
            rank=args.lora_rank,
            seed=args.seed,
            train_mlp=args.train_mlp,
            train_attn=args.train_attn,
            train_unembed=args.train_unembed,
            user_metadata={"run_id": run_id, "purpose": "strategy_generator_sft"},
        )
        tokenizer = training_client.get_tokenizer()

    tokenized_train = [tokenize_example(tokenizer, row, args.max_length) for row in train_rows]
    tokenized_val = [tokenize_example(tokenizer, row, args.max_length) for row in val_rows]

    rng = random.Random(args.seed)
    steps_per_epoch = math.ceil(len(tokenized_train) / args.batch_size)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    manifest = {
        "event": "start",
        "run_id": run_id,
        "time": now_iso(),
        "model": args.model,
        "data_dir": str(args.data_dir),
        "out_dir": str(out_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "token_lengths_train": summarize_token_lengths(tokenized_train),
        "token_lengths_val": summarize_token_lengths(tokenized_val),
        "hyperparameters": {
            "lora_rank": args.lora_rank,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_steps": total_steps,
            "max_length": args.max_length,
            "train_attn": args.train_attn,
            "train_mlp": args.train_mlp,
            "train_unembed": args.train_unembed,
        },
        "dry_run": args.dry_run,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    assert tinker is not None and training_client is not None
    metrics_path = out_dir / "metrics.jsonl"
    adam_params = tinker.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
    )

    step = 0
    if tokenized_val:
        t0 = time.time()
        val_metrics = await evaluate_loss(
            training_client=training_client,
            tinker_types=tinker.types,
            tokenized_eval=tokenized_val,
            batch_size=args.batch_size,
            limit=args.eval_limit,
        )
        rec = {"event": "eval", "step": 0, "split": "val", "elapsed": time.time() - t0, **val_metrics}
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True), flush=True)

    train_order = list(tokenized_train)
    epoch = 0
    while step < total_steps:
        rng.shuffle(train_order)
        for batch in batches(train_order, args.batch_size):
            if step >= total_steps:
                break
            step += 1
            t0 = time.time()
            datums = [build_datum(tinker.types, item) for item in batch]
            fwd_future = await training_client.forward_backward_async(datums, "cross_entropy")
            opt_future = await training_client.optim_step_async(adam_params)
            fwd_result = await fwd_future.result_async()
            opt_result = await opt_future.result_async()
            rec = {
                "event": "train",
                "step": step,
                "epoch_float": epoch + (step % max(1, steps_per_epoch)) / max(1, steps_per_epoch),
                "rows": len(batch),
                "tokens_full": sum(item.full_len for item in batch),
                "tokens_target": sum(item.target_len for item in batch),
                "metrics": dict(fwd_result.metrics),
                "optim": getattr(opt_result, "model_dump", lambda: str(opt_result))(),
                "elapsed": time.time() - t0,
                "time": now_iso(),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            print(json.dumps(rec, sort_keys=True), flush=True)

            if args.eval_every and step % args.eval_every == 0 and tokenized_val:
                t_eval = time.time()
                val_metrics = await evaluate_loss(
                    training_client=training_client,
                    tinker_types=tinker.types,
                    tokenized_eval=tokenized_val,
                    batch_size=args.batch_size,
                    limit=args.eval_limit,
                )
                eval_rec = {
                    "event": "eval",
                    "step": step,
                    "split": "val",
                    "elapsed": time.time() - t_eval,
                    **val_metrics,
                }
                with metrics_path.open("a") as f:
                    f.write(json.dumps(eval_rec, sort_keys=True) + "\n")
                print(json.dumps(eval_rec, sort_keys=True), flush=True)

            if args.save_every and step % args.save_every == 0:
                ckpt = await save_checkpoint(
                    training_client=training_client,
                    name=safe_tinker_label(f"{run_id}-step-{step:04d}"),
                    out_dir=out_dir,
                    step=step,
                    kind="periodic",
                )
                print(json.dumps(ckpt, sort_keys=True), flush=True)
        epoch += 1

    final_state = await save_checkpoint(
        training_client=training_client,
        name=safe_tinker_label(f"{run_id}-final"),
        out_dir=out_dir,
        step=step,
        kind="final",
    )
    final_sampler = await save_sampler_weights(
        training_client=training_client,
        name=safe_tinker_label(f"{run_id}-final-sampler"),
        out_dir=out_dir,
        step=step,
    )
    done = {
        "event": "complete",
        "run_id": run_id,
        "step": step,
        "final_state_path": final_state["path"],
        "final_sampler_path": final_sampler["path"],
        "time": now_iso(),
    }
    (out_dir / "result.json").write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
    print(json.dumps(done, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
