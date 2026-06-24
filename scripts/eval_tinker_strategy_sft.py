#!/usr/bin/env python3
"""Offline evaluation for a Tinker strategy-generator SFT checkpoint.

This compares base Qwen3.6 and an optional trained Tinker checkpoint on held-out
one-step examples. It reports:
- teacher-forced target NLL from Tinker prompt logprobs;
- deterministic generation surface metrics against the GPT-5.5 target strategy;
- basic format validity checks.

This is not a replacement for end-to-end CyberGym execution, but it is the fast
gate that should improve if SFT learned the one-step generator mapping.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_DATA = ROOT / "runs/strategy_sft/qwen36_strategy_sft_data/test.jsonl"
ASSISTANT_PREFIX = "1. Current best hypothesis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path", help="Tinker checkpoint/sampler path for the SFT model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=12_288)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-logprobs", action="store_true")
    parser.add_argument(
        "--clients",
        choices=["base", "sft", "both"],
        default="both",
        help="Which sampler(s) to evaluate. `sft` requires --model-path.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manual_message_tokens(tokenizer: Any, role: str, content: str) -> list[int]:
    return tokenizer.encode(f"<|im_start|>{role}\n{content}<|im_end|>\n", add_special_tokens=False)


def render_prompt_and_full(
    tokenizer: Any, row: dict[str, Any], max_length: int
) -> tuple[list[int], list[int], str]:
    messages = row["messages"]
    prompt_messages = messages[:-1]
    assistant_prefix = str(row.get("prompt_assistant_prefix") or ASSISTANT_PREFIX)
    target = str(row.get("target_completion") or messages[-1]["content"])
    target_tokens = tokenizer.encode(target + "<|im_end|>\n", add_special_tokens=False)
    assistant_start = tokenizer.encode(
        f"<|im_start|>assistant\n{assistant_prefix}", add_special_tokens=False
    )
    prefix: list[int] = []
    for msg in prompt_messages[:-1]:
        prefix.extend(manual_message_tokens(tokenizer, msg["role"], msg["content"]))
    last = prompt_messages[-1]
    header = tokenizer.encode(f"<|im_start|>{last['role']}\n", add_special_tokens=False)
    body = tokenizer.encode(str(last["content"]), add_special_tokens=False)
    end = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    budget = max_length - len(target_tokens) - len(prefix) - len(header) - len(end) - len(assistant_start)
    if budget < len(body):
        marker = tokenizer.encode(
            "\n[...prompt truncated from the front to fit context...]\n",
            add_special_tokens=False,
        )
        tail_budget = max(0, budget - len(marker))
        body = marker + (body[-tail_budget:] if tail_budget else [])
    prompt = prefix + header + body + end + assistant_start
    full = prompt + target_tokens
    if len(full) > max_length:
        # Last-resort guard. Keep prompt/target alignment by trimming prompt body more.
        overflow = len(full) - max_length
        keep_prefix = prefix + header
        body_plus_end_start = len(keep_prefix)
        body_len = len(body)
        drop = min(overflow, body_len)
        body = body[drop:]
        prompt = keep_prefix + body + end + assistant_start
        full = prompt + target_tokens
    return prompt, full, target


def clean(text: str) -> str:
    text = text.replace("<|endoftext|>", "").replace("<|im_end|>", "")
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    return text.strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_./:-]+", text.lower())


def f1(pred: str, target: str) -> float:
    p = Counter(word_tokens(pred))
    t = Counter(word_tokens(target))
    if not p or not t:
        return 0.0
    overlap = sum((p & t).values())
    precision = overlap / sum(p.values())
    recall = overlap / sum(t.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def format_valid(text: str) -> bool:
    if not text.strip():
        return False
    lowered = text.lower()
    if "<think>" in lowered or "analysis:" in lowered[:200]:
        return False
    return "current best hypothesis" in lowered and "concrete next" in lowered


async def eval_one_client(
    *,
    name: str,
    sampler: Any,
    tokenizer: Any,
    tinker_types: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    skip_generation: bool,
    skip_logprobs: bool,
) -> dict[str, Any]:
    import tinker

    row_metrics: list[dict[str, Any]] = []
    sampling_params = tinker.SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|im_end|>", "\n<|im_start|>user", "\nUser:"],
    )
    for idx, row in enumerate(rows):
        prompt, full, target_suffix = render_prompt_and_full(tokenizer, row, max_length)
        target_full = ASSISTANT_PREFIX + target_suffix
        rec: dict[str, Any] = {
            "id": row.get("id"),
            "task_id": row.get("task_id"),
            "prompt_tokens": len(prompt),
            "full_tokens": len(full),
            "target_tokens": max(0, len(full) - len(prompt)),
        }
        if not skip_logprobs:
            logprobs = await sampler.compute_logprobs_async(tinker_types.ModelInput.from_ints(full))
            target_logprobs = [
                lp for lp in logprobs[len(prompt) :] if lp is not None and math.isfinite(lp)
            ]
            rec["target_nll"] = (
                -sum(target_logprobs) / len(target_logprobs) if target_logprobs else None
            )
        if not skip_generation:
            result = await sampler.sample_async(
                prompt=tinker_types.ModelInput.from_ints(prompt),
                num_samples=1,
                sampling_params=sampling_params,
            )
            seq = result.sequences[0]
            decoded = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            pred = clean(ASSISTANT_PREFIX + decoded)
            rec.update(
                {
                    "prediction": pred,
                    "target": target_full,
                    "sampled_tokens": len(seq.tokens),
                    "pred_tokens": len(tokenizer.encode(pred, add_special_tokens=False)),
                    "target_output_tokens": len(
                        tokenizer.encode(target_full, add_special_tokens=False)
                    ),
                    "word_f1": f1(pred, target_full),
                    "format_valid": format_valid(pred),
                    "pred_chars": len(pred),
                    "target_chars": len(target_full),
                    "stop_reason": str(getattr(seq, "stop_reason", "")),
                }
            )
        row_metrics.append(rec)
        print(json.dumps({"event": "eval_row", "client": name, "i": idx, **{k: rec[k] for k in rec if k not in {"prediction", "target"}}}, sort_keys=True), flush=True)

    nlls = [r["target_nll"] for r in row_metrics if isinstance(r.get("target_nll"), (int, float))]
    f1s = [r["word_f1"] for r in row_metrics if isinstance(r.get("word_f1"), (int, float))]
    valid = [r["format_valid"] for r in row_metrics if "format_valid" in r]
    return {
        "client": name,
        "rows": len(row_metrics),
        "target_nll_mean": statistics.mean(nlls) if nlls else None,
        "target_nll_median": statistics.median(nlls) if nlls else None,
        "word_f1_mean": statistics.mean(f1s) if f1s else None,
        "word_f1_median": statistics.median(f1s) if f1s else None,
        "format_valid_rate": sum(valid) / len(valid) if valid else None,
        "rows_detail": row_metrics,
    }


async def main_async() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(f"TINKER_API_KEY is not set; checked {args.env_file}")
    import tinker

    rows = load_jsonl(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]
    service = tinker.ServiceClient(
        user_metadata={"purpose": "cybergym_strategy_sft_eval", "script": "eval_tinker_strategy_sft.py"}
    )
    clients: list[tuple[str, Any]] = []
    base_sampler = None
    if args.clients in {"base", "both"}:
        base_sampler = await service.create_sampling_client_async(base_model=args.model)
        clients.append(("base", base_sampler))
    if args.clients in {"sft", "both"} and args.model_path:
        tuned_sampler = await service.create_sampling_client_async(model_path=args.model_path)
        clients.append(("sft", tuned_sampler))
    if args.clients in {"sft", "both"} and not args.model_path:
        raise RuntimeError("--clients=sft/both requires --model-path")
    if not clients:
        raise RuntimeError("No clients selected for evaluation")

    # Use the base tokenizer; Tinker checkpoints preserve the same base model.
    if base_sampler is None:
        base_sampler = await service.create_sampling_client_async(base_model=args.model)
    tokenizer = base_sampler.get_tokenizer()
    summaries = []
    for name, sampler in clients:
        summaries.append(
            await eval_one_client(
                name=name,
                sampler=sampler,
                tokenizer=tokenizer,
                tinker_types=tinker.types,
                rows=rows,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                skip_generation=args.skip_generation,
                skip_logprobs=args.skip_logprobs,
            )
        )
    result = {
        "event": "eval_complete",
        "time": now_iso(),
        "model": args.model,
        "model_path": args.model_path,
        "data": str(args.data),
        "rows": len(rows),
        "summaries": [{k: v for k, v in s.items() if k != "rows_detail"} for s in summaries],
        "details": summaries,
    }
    out = args.out or (
        ROOT
        / "runs/strategy_sft"
        / f"eval_strategy_sft_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
