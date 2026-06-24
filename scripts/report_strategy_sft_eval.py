#!/usr/bin/env python3
"""Generate a final/point-in-time report for the strategy-generator SFT.

The script is read-only: it does not launch training, evaluation, or polling.
It combines three evidence sources:

1. Tinker SFT completion/checkpoint metadata.
2. Offline one-step held-out NLL evaluation.
3. End-to-end CyberGym base-vs-SFT rollout comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare_cybergym_strategy_eval import (
    latest_by_task,
    matched_report,
)
from summarize_cybergym_pass_rates import (
    attempt_index,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=Path("runs/strategy_sft/qwen36-strategy-sft-20260601T1935Z"),
    )
    parser.add_argument(
        "--offline-eval",
        type=Path,
        default=None,
        help="Defaults to <sft-dir>/offline_eval_test84_nll_only.json.",
    )
    parser.add_argument(
        "--base-rollouts",
        type=Path,
        default=Path("runs/codex_gpt55_eval100_tinker_qwen36_phase0_rollouts.jsonl"),
    )
    parser.add_argument(
        "--sft-rollouts",
        type=Path,
        default=Path("runs/codex_gpt55_eval100_tinker_qwen36_sft_parallel_rollouts.jsonl"),
    )
    parser.add_argument(
        "--base-log",
        type=Path,
        default=Path(
            "runs/codex_cybergym/sequential_self_logs/"
            "codex-gpt55-eval100-tinker-qwen36-phase0-20260531T0715Z.log"
        ),
    )
    parser.add_argument(
        "--sft-log",
        type=Path,
        default=Path(
            "runs/codex_cybergym/sequential_self_logs/"
            "codex-gpt55-eval100-tinker-qwen36-sft-parallel-20260601T2115Z.log"
        ),
    )
    parser.add_argument("--base-label", default="base_qwen36")
    parser.add_argument("--sft-label", default="sft_qwen36")
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("runs/strategy_sft/qwen36-strategy-sft-20260601T1935Z/final_eval_report.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("runs/strategy_sft/qwen36-strategy-sft-20260601T1935Z/final_eval_report.json"),
    )
    parser.add_argument(
        "--require-final",
        action="store_true",
        help=(
            "Exit non-zero unless the end-to-end eval has enough evidence to be "
            "treated as final. This is intended as the completion gate."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def read_event_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            event["_line_no"] = line_no
            events.append(event)
    return events


def extract_eval_log(log_path: Path) -> dict[str, Any]:
    events = read_event_log(log_path)
    starts = [
        event
        for event in events
        if str(event.get("event") or "").endswith("_matched_start")
        or event.get("event") == "seq_self_eval100_tinker_qwen36_phase0_start"
    ]
    completes = [
        event
        for event in events
        if str(event.get("event") or "").endswith("_matched_complete")
    ]
    sequential_starts = [event for event in events if event.get("event") == "sequential_start"]
    sequential_completes = [event for event in events if event.get("event") == "sequential_complete"]
    latest_start = starts[-1] if starts else None
    latest_complete = completes[-1] if completes else None
    latest_sequential_start = sequential_starts[-1] if sequential_starts else None
    latest_sequential_complete = sequential_completes[-1] if sequential_completes else None
    latest_start_line = latest_start.get("_line_no") if latest_start else -1
    latest_complete_line = latest_complete.get("_line_no") if latest_complete else -1
    latest_seq_complete_line = latest_sequential_complete.get("_line_no") if latest_sequential_complete else -1
    latest_start_completed = latest_complete_line > latest_start_line
    latest_sequential_completed = latest_seq_complete_line > latest_start_line
    return {
        "path": str(log_path),
        "events": len(events),
        "matched_starts": len(starts),
        "matched_completes": len(completes),
        "sequential_starts": len(sequential_starts),
        "sequential_completes": len(sequential_completes),
        "latest_start": latest_start,
        "latest_complete": latest_complete,
        "latest_sequential_start": latest_sequential_start,
        "latest_sequential_complete": latest_sequential_complete,
        "latest_start_completed": latest_start_completed,
        "latest_sequential_completed": latest_sequential_completed,
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def num(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def extract_training(sft_dir: Path) -> dict[str, Any]:
    result = read_json(sft_dir / "result.json")
    metrics = read_jsonl(sft_dir / "metrics.jsonl")
    evals = [
        {
            "step": row.get("step"),
            "split": row.get("split"),
            "loss_mean_unweighted": row.get("loss_mean_unweighted"),
            "rows": row.get("rows"),
        }
        for row in metrics
        if row.get("event") == "eval"
    ]
    val_evals = [row for row in evals if row.get("split") == "val"]
    loss_delta = None
    loss_relative_drop = None
    if len(val_evals) >= 2:
        first = val_evals[0].get("loss_mean_unweighted")
        last = val_evals[-1].get("loss_mean_unweighted")
        if isinstance(first, (int, float)) and isinstance(last, (int, float)):
            loss_delta = last - first
            loss_relative_drop = (first - last) / first if first else None
    return {
        "result": result,
        "evals": evals,
        "val_evals": val_evals,
        "val_loss_first": val_evals[0] if val_evals else None,
        "val_loss_last": val_evals[-1] if val_evals else None,
        "val_loss_delta_last_minus_first": loss_delta,
        "val_loss_relative_drop": loss_relative_drop,
    }


def extract_offline(offline_eval: Path) -> dict[str, Any]:
    data = read_json(offline_eval)
    summaries = data.get("summaries") or []
    by_client = {
        row.get("client"): row
        for row in summaries
        if isinstance(row, dict) and row.get("client")
    }
    base = by_client.get("base") or {}
    sft = by_client.get("sft") or {}
    base_nll = base.get("target_nll_mean")
    sft_nll = sft.get("target_nll_mean")
    nll_delta = None
    nll_relative_drop = None
    if isinstance(base_nll, (int, float)) and isinstance(sft_nll, (int, float)):
        nll_delta = sft_nll - base_nll
        nll_relative_drop = (base_nll - sft_nll) / base_nll if base_nll else None
    return {
        "path": str(offline_eval),
        "rows": data.get("rows"),
        "summaries": summaries,
        "target_nll_delta_sft_minus_base": nll_delta,
        "target_nll_relative_drop": nll_relative_drop,
    }


def extract_end_to_end(args: argparse.Namespace) -> dict[str, Any]:
    base_latest, base_rows, base_infra = latest_by_task(args.base_rollouts, include_infra=False)
    sft_latest, sft_rows, sft_infra = latest_by_task(args.sft_rollouts, include_infra=False)
    common = sorted(set(base_latest) & set(sft_latest))
    both_terminal = [
        task_id
        for task_id in common
        if (
            base_latest[task_id].get("status") == "PASSED"
            or attempt_index(base_latest[task_id]) >= args.max_attempts
        )
        and (
            sft_latest[task_id].get("status") == "PASSED"
            or attempt_index(sft_latest[task_id]) >= args.max_attempts
        )
    ]
    base_log = extract_eval_log(args.base_log)
    sft_log = extract_eval_log(args.sft_log)
    base_summary = summarize(args.base_rollouts, include_infra=False)
    sft_summary = summarize(args.sft_rollouts, include_infra=False)
    matched = matched_report(
        "both_terminal_common_tasks",
        both_terminal,
        base_latest,
        sft_latest,
        base_label=args.base_label,
        sft_label=args.sft_label,
        max_attempts=args.max_attempts,
        include_details=False,
    )
    matched_all = matched_report(
        "all_seen_common_tasks",
        common,
        base_latest,
        sft_latest,
        base_label=args.base_label,
        sft_label=args.sft_label,
        max_attempts=args.max_attempts,
        include_details=False,
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not base_log.get("latest_start_completed"):
        blockers.append(f"{args.base_label} latest eval start has no matching complete marker.")
    if not sft_log.get("latest_start_completed"):
        blockers.append(f"{args.sft_label} latest eval start has no matching complete marker.")
    if len(sft_latest) < min(args.max_tasks, len(base_latest)):
        blockers.append(
            f"{args.sft_label} has {len(sft_latest)} latest tasks, fewer than "
            f"{args.base_label}'s {len(base_latest)} clean latest tasks."
        )
    if len(common) < min(len(base_latest), len(sft_latest)):
        blockers.append(
            f"Only {len(common)} common latest tasks are shared between variants; "
            "matched pass-rate would not cover the smaller eval set."
        )
    if matched["tasks"] < len(common):
        warnings.append(
            f"Only {matched['tasks']} of {len(common)} common tasks are terminal for both variants. "
            "This is diagnostic only; the primary final comparison uses matched latest-task outcomes."
        )
    ready_for_final_conclusion = not blockers
    return {
        "inputs": {
            args.base_label: {
                "path": str(args.base_rollouts),
                "rows": base_rows,
                "infra_rows_detected": base_infra,
            },
            args.sft_label: {
                "path": str(args.sft_rollouts),
                "rows": sft_rows,
                "infra_rows_detected": sft_infra,
            },
        },
        "logs": {
            args.base_label: base_log,
            args.sft_label: sft_log,
        },
        "individual": {
            args.base_label: base_summary,
            args.sft_label: sft_summary,
        },
        "progress": {
            args.base_label: {
                "latest_tasks": len(base_latest),
                "estimated_remaining_latest_tasks": max(args.max_tasks - len(base_latest), 0),
            },
            args.sft_label: {
                "latest_tasks": len(sft_latest),
                "estimated_remaining_latest_tasks": max(args.max_tasks - len(sft_latest), 0),
            },
        },
        "matched_both_terminal": matched,
        "matched_all_common": matched_all,
        "ready_for_final_conclusion": ready_for_final_conclusion,
        "finality_blockers": blockers,
        "finality_warnings": warnings,
    }


def render_markdown(report: dict[str, Any], *, base_label: str, sft_label: str) -> str:
    conclusion = report.get("conclusion") or {}
    training = report["training"]
    offline = report["offline_eval"]
    e2e = report["end_to_end"]
    base_ind = e2e["individual"][base_label]
    sft_ind = e2e["individual"][sft_label]
    base_log = e2e["logs"][base_label]
    sft_log = e2e["logs"][sft_label]
    matched_all = e2e["matched_all_common"]
    base_matched_all = matched_all[base_label]
    sft_matched_all = matched_all[sft_label]
    matched = e2e["matched_both_terminal"]
    base_matched = matched[base_label]
    sft_matched = matched[sft_label]

    val_first = training.get("val_loss_first") or {}
    val_last = training.get("val_loss_last") or {}
    result = training.get("result") or {}

    lines = [
        "# Strategy Generator SFT Evaluation Report",
        "",
        "## Conclusion",
        "",
        f"- Training validation loss decreased: {conclusion.get('training_loss_decreased')}",
        f"- Offline one-step NLL improved: {conclusion.get('offline_nll_improved')}",
        f"- End-to-end result status: {conclusion.get('end_to_end_status')}",
        f"- End-to-end direction: {conclusion.get('end_to_end_direction')}",
        f"- End-to-end summary: {conclusion.get('end_to_end_summary')}",
        "",
        "## Training",
        "",
        f"- Run id: `{result.get('run_id', 'n/a')}`",
        f"- Final step: {result.get('step', 'n/a')}",
        f"- Final sampler: `{result.get('final_sampler_path', 'n/a')}`",
        f"- Validation loss: step {val_first.get('step', 'n/a')} = {num(val_first.get('loss_mean_unweighted'))}; "
        f"step {val_last.get('step', 'n/a')} = {num(val_last.get('loss_mean_unweighted'))}; "
        f"relative drop = {pct(training.get('val_loss_relative_drop'))}",
        "",
        "## Offline one-step held-out NLL",
        "",
    ]
    for row in offline.get("summaries") or []:
        lines.append(
            f"- {row.get('client')}: rows={row.get('rows')}, "
            f"target_nll_mean={num(row.get('target_nll_mean'))}, "
            f"target_nll_median={num(row.get('target_nll_median'))}"
        )
    lines.extend(
        [
            f"- SFT vs base mean NLL delta: {num(offline.get('target_nll_delta_sft_minus_base'))}",
            f"- SFT mean NLL relative drop: {pct(offline.get('target_nll_relative_drop'))}",
            "",
            "## End-to-end CyberGym eval",
            "",
            "Individual latest-task summary, infra/API failures excluded:",
            "",
            "| Variant | Tasks | Passed | Pass rate | M6/M7 rate | Infra rows ignored |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {base_label} | {base_ind['tasks']} | {base_ind['passed']} | {pct(base_ind['pass_rate'])} | "
            f"{pct(base_ind['milestone_6_or_7_rate'])} | {base_ind['infra_rows_ignored']} |",
            f"| {sft_label} | {sft_ind['tasks']} | {sft_ind['passed']} | {pct(sft_ind['pass_rate'])} | "
            f"{pct(sft_ind['milestone_6_or_7_rate'])} | {sft_ind['infra_rows_ignored']} |",
            "",
            "Matched latest-task comparison (primary end-to-end comparison):",
            "",
            "| Variant | Tasks | Passed | Pass rate | M6/M7 rate |",
            "|---|---:|---:|---:|---:|",
            f"| {base_label} | {base_matched_all['tasks']} | {base_matched_all['passed']} | "
            f"{pct(base_matched_all['pass_rate'])} | {pct(base_matched_all['milestone_6_or_7_rate'])} |",
            f"| {sft_label} | {sft_matched_all['tasks']} | {sft_matched_all['passed']} | "
            f"{pct(sft_matched_all['pass_rate'])} | {pct(sft_matched_all['milestone_6_or_7_rate'])} |",
            "",
            f"- Matched latest-task pass-rate delta, SFT - base: {pct(matched_all.get('pass_rate_delta_sft_minus_base'))}",
            "",
            "Matched both-terminal task comparison (diagnostic subset):",
            "",
            "| Variant | Tasks | Passed | Pass rate | M6/M7 rate |",
            "|---|---:|---:|---:|---:|",
            f"| {base_label} | {base_matched['tasks']} | {base_matched['passed']} | "
            f"{pct(base_matched['pass_rate'])} | {pct(base_matched['milestone_6_or_7_rate'])} |",
            f"| {sft_label} | {sft_matched['tasks']} | {sft_matched['passed']} | "
            f"{pct(sft_matched['pass_rate'])} | {pct(sft_matched['milestone_6_or_7_rate'])} |",
            "",
            f"- Matched both-terminal pass-rate delta, SFT - base: {pct(matched.get('pass_rate_delta_sft_minus_base'))}",
            "",
            "## Current completion status",
            "",
            f"- {base_label} latest eval complete marker present: {base_log.get('latest_start_completed')}",
            f"- {sft_label} latest eval complete marker present: {sft_log.get('latest_start_completed')}",
            f"- {base_label}: {e2e['progress'][base_label]['latest_tasks']} latest tasks; "
            f"estimated remaining latest tasks: {e2e['progress'][base_label]['estimated_remaining_latest_tasks']}",
            f"- {sft_label}: {e2e['progress'][sft_label]['latest_tasks']} latest tasks; "
            f"estimated remaining latest tasks: {e2e['progress'][sft_label]['estimated_remaining_latest_tasks']}",
            f"- Ready for final end-to-end conclusion: {e2e.get('ready_for_final_conclusion')}",
            "",
            "Finality blockers:",
            "",
            *[f"- {blocker}" for blocker in (e2e.get("finality_blockers") or ["None"])],
            "",
            "Diagnostic warnings:",
            "",
            *[f"- {warning}" for warning in (e2e.get("finality_warnings") or ["None"])],
            "",
            "Interpretation: offline NLL shows whether the one-step supervised target was learned; "
            "the end-to-end matched comparison is the decisive pass-rate evaluation and should be read as final only after SFT has enough clean terminal tasks.",
            "",
        ]
    )
    return "\n".join(lines)


def build_conclusion(report: dict[str, Any], *, base_label: str, sft_label: str) -> dict[str, Any]:
    training = report["training"]
    offline = report["offline_eval"]
    e2e = report["end_to_end"]
    training_drop = training.get("val_loss_relative_drop")
    nll_drop = offline.get("target_nll_relative_drop")
    matched = e2e.get("matched_all_common") or {}
    delta = matched.get("pass_rate_delta_sft_minus_base")
    ready = bool(e2e.get("ready_for_final_conclusion"))

    training_loss_decreased = (
        isinstance(training_drop, (int, float)) and training_drop > 0
    )
    offline_nll_improved = isinstance(nll_drop, (int, float)) and nll_drop > 0

    if not ready:
        return {
            "training_loss_decreased": training_loss_decreased,
            "offline_nll_improved": offline_nll_improved,
            "end_to_end_status": "pending",
            "end_to_end_direction": "pending",
            "end_to_end_delta_sft_minus_base": delta,
            "end_to_end_summary": (
                "End-to-end eval is not final yet; inspect finality_blockers before "
                "claiming whether SFT improves pass rate."
            ),
        }

    base = matched[base_label]
    sft = matched[sft_label]
    if not isinstance(delta, (int, float)):
        direction = "unknown"
    elif delta > 0:
        direction = "sft_higher"
    elif delta < 0:
        direction = "sft_lower"
    else:
        direction = "tie"
    return {
        "training_loss_decreased": training_loss_decreased,
        "offline_nll_improved": offline_nll_improved,
        "end_to_end_status": "final",
        "end_to_end_direction": direction,
        "end_to_end_delta_sft_minus_base": delta,
        "end_to_end_summary": (
            f"Matched latest-task pass rate: {sft_label}={pct(sft.get('pass_rate'))} "
            f"({sft.get('passed')}/{sft.get('tasks')}), "
            f"{base_label}={pct(base.get('pass_rate'))} "
            f"({base.get('passed')}/{base.get('tasks')}), "
            f"delta={pct(delta)}."
        ),
    }


def main() -> int:
    args = parse_args()
    offline_eval = args.offline_eval or (args.sft_dir / "offline_eval_test84_nll_only.json")
    report = {
        "training": extract_training(args.sft_dir),
        "offline_eval": extract_offline(offline_eval),
        "end_to_end": extract_end_to_end(args),
    }
    report["conclusion"] = build_conclusion(
        report, base_label=args.base_label, sft_label=args.sft_label
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md = render_markdown(report, base_label=args.base_label, sft_label=args.sft_label)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md)
    ready = bool(report["end_to_end"].get("ready_for_final_conclusion"))
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "ready_for_final_conclusion": ready,
                "finality_blockers": report["end_to_end"].get("finality_blockers") or [],
                "finality_warnings": report["end_to_end"].get("finality_warnings") or [],
            },
            indent=2,
        )
    )
    if args.require_final and not ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
