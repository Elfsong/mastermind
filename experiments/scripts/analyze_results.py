#!/usr/bin/env python3
"""
Aggregate statistics, tables, and go/no-go decisions across all experiments.

Produces summary tables matching the experiment plan format, plus
the final decision matrix.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Experiment 1: Homogeneity Analysis
# ---------------------------------------------------------------------------

def format_exp1_table(results: Dict) -> str:
    """Format Experiment 1 results as a markdown table."""
    lines = []
    header = (
        "| Challenge | Category | Avg Tool Jaccard | Avg Embedding Sim | "
        "Mode Freq | Complement Ratio | Solved (x/n) |"
    )
    sep = "|" + "|".join(["---"] * 7) + "|"
    lines.append(header)
    lines.append(sep)

    for r in results.get("per_challenge", []):
        emb_sim = f"{r['avg_embedding_similarity']:.2f}" if r['avg_embedding_similarity'] else "N/A"
        line = (
            f"| {r['challenge_id'][:20]} | - | {r['avg_tool_jaccard']:.2f} | "
            f"{emb_sim} | {r['mode_frequency']:.0%} {r['mode_attack_vector']} | "
            f"{r['complement_ratio']:.1f} | {r['n_solved']}/{r['n_runs']} |"
        )
        lines.append(line)

    agg = results.get("aggregate", {})
    lines.append("")
    lines.append(f"**Aggregate**: Jaccard={agg.get('avg_tool_jaccard', 'N/A')}, "
                 f"Embedding Sim={agg.get('avg_embedding_similarity', 'N/A')}, "
                 f"Mode Freq={agg.get('avg_mode_frequency', 'N/A')}, "
                 f"Complement Ratio={agg.get('avg_complement_ratio', 'N/A')}")

    go = results.get("go_nogo", {})
    lines.append(f"\n**Decision**: {go.get('decision', 'UNKNOWN')} — {go.get('reason', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment 2: Continuation Effectiveness
# ---------------------------------------------------------------------------

def analyze_exp2(trajectory_dir: str) -> Dict:
    """Analyze Experiment 2 results by condition."""
    condition_results = defaultdict(lambda: {"solved": 0, "total": 0, "steps": []})

    for filepath in Path(trajectory_dir).glob("*.json"):
        with open(filepath) as f:
            data = json.load(f)
        condition = data.get("condition", "unknown")
        solved = data.get("outcome", {}).get("solved", False)
        steps = data.get("outcome", {}).get("steps_to_flag")

        condition_results[condition]["total"] += 1
        if solved:
            condition_results[condition]["solved"] += 1
            if steps is not None:
                condition_results[condition]["steps"].append(steps)

    results = {}
    for cond, stats in condition_results.items():
        success_rate = stats["solved"] / stats["total"] if stats["total"] > 0 else 0
        avg_steps = np.mean(stats["steps"]) if stats["steps"] else None
        median_steps = np.median(stats["steps"]) if stats["steps"] else None
        results[cond] = {
            "success_rate": round(success_rate, 3),
            "solved": stats["solved"],
            "total": stats["total"],
            "avg_steps_to_flag": round(float(avg_steps), 1) if avg_steps else None,
            "median_steps_to_flag": round(float(median_steps), 1) if median_steps else None,
        }

    # Go/No-Go
    cond_a = results.get("full_context", {}).get("success_rate", 0)
    cond_b = results.get("summary_only", {}).get("success_rate", 0)
    cond_c = results.get("fresh_start", {}).get("success_rate", 0)

    if (cond_a > cond_c or cond_b > cond_c):
        go = "GO"
        reason = f"Continuation improves over fresh start (A={cond_a:.0%}, B={cond_b:.0%} vs C={cond_c:.0%})"
    elif abs(cond_b - cond_c) < 0.05:
        if cond_a > cond_c + 0.1:
            go = "PARTIAL-GO"
            reason = "Full history helps but summaries are too lossy"
        else:
            go = "NO-GO"
            reason = "State summaries do not transfer useful information"
    else:
        go = "BORDERLINE"
        reason = "Results are mixed; need significance testing"

    return {
        "per_condition": results,
        "go_nogo": {"decision": go, "reason": reason},
    }


def format_exp2_table(results: Dict) -> str:
    """Format Experiment 2 results."""
    lines = ["## Experiment 2: State-Based Continuation", ""]
    lines.append("| Condition | Success Rate | Solved/Total | Avg Steps | Median Steps |")
    lines.append("|-----------|-------------|-------------|-----------|-------------|")

    for cond_name, label in [("full_context", "A: Full Context"),
                              ("summary_only", "B: Summary Only"),
                              ("fresh_start", "C: Fresh Start")]:
        cond = results.get("per_condition", {}).get(cond_name, {})
        sr = f"{cond.get('success_rate', 0):.0%}"
        solved_total = f"{cond.get('solved', 0)}/{cond.get('total', 0)}"
        avg_s = f"{cond.get('avg_steps_to_flag', 'N/A')}"
        med_s = f"{cond.get('median_steps_to_flag', 'N/A')}"
        lines.append(f"| {label} | {sr} | {solved_total} | {avg_s} | {med_s} |")

    go = results.get("go_nogo", {})
    lines.append(f"\n**Decision**: {go.get('decision', 'UNKNOWN')} — {go.get('reason', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment 3: Diversity Guidance
# ---------------------------------------------------------------------------

def analyze_exp3(trajectory_dir: str) -> Dict:
    """Analyze Experiment 3 results by condition."""
    condition_results = defaultdict(lambda: {"solved": 0, "total": 0, "steps": []})

    for filepath in Path(trajectory_dir).glob("*.json"):
        with open(filepath) as f:
            data = json.load(f)
        condition = data.get("condition", "unknown")
        solved = data.get("outcome", {}).get("solved", False)
        steps = data.get("outcome", {}).get("steps_to_flag")

        condition_results[condition]["total"] += 1
        if solved:
            condition_results[condition]["solved"] += 1
            if steps is not None:
                condition_results[condition]["steps"].append(steps)

    results = {}
    for cond, stats in condition_results.items():
        success_rate = stats["solved"] / stats["total"] if stats["total"] > 0 else 0
        results[cond] = {
            "success_rate": round(success_rate, 3),
            "solved": stats["solved"],
            "total": stats["total"],
        }

    return {"per_condition": results}


# ---------------------------------------------------------------------------
# Decision Matrix
# ---------------------------------------------------------------------------

def decision_matrix(exp1_go: str, exp2_go: str, exp3_go: str) -> Dict:
    """Evaluate the overall decision matrix."""
    key = (exp1_go, exp2_go, exp3_go)
    decisions = {
        ("GO", "GO", "GO"): "Full proceed — implement AGE-CTF",
        ("GO", "PARTIAL-GO", "GO"): "Proceed; increase context budget for state transfer",
        ("GO", "GO", "PARTIAL-GO"): "Proceed; expect modest gains, invest in analysis depth",
        ("GO", "GO", "NO-GO"): "Pivot — archive works but LLM can't be guided; try constrained action spaces",
        ("GO", "NO-GO", "GO"): "Pivot — archive concept fails; try BoN + diverse prompting",
        ("GO", "NO-GO", "NO-GO"): "Pivot — archive concept fails; try BoN + diverse prompting",
        ("NO-GO", "GO", "GO"): "Stop — diversity is not the bottleneck",
        ("NO-GO", "NO-GO", "NO-GO"): "Stop — diversity is not the bottleneck",
    }

    if key in decisions:
        return {"decision": decisions[key], "experiments": {"exp1": exp1_go, "exp2": exp2_go, "exp3": exp3_go}}

    # Default for combinations not explicitly listed
    if exp1_go == "NO-GO":
        return {"decision": "Stop — diversity is not the bottleneck",
                "experiments": {"exp1": exp1_go, "exp2": exp2_go, "exp3": exp3_go}}

    return {
        "decision": f"Reassess — mixed results ({exp1_go}/{exp2_go}/{exp3_go})",
        "experiments": {"exp1": exp1_go, "exp2": exp2_go, "exp3": exp3_go},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_full_report(exp1_file: str, exp2_dir: str, exp3_dir: str,
                         output_file: str):
    """Generate complete experiment report."""
    report = {"experiments": {}}

    # Experiment 1
    if Path(exp1_file).exists():
        with open(exp1_file) as f:
            exp1_data = json.load(f)
        report["experiments"]["exp1"] = exp1_data
        exp1_go = exp1_data.get("go_nogo", {}).get("decision", "UNKNOWN")
    else:
        exp1_go = "UNKNOWN"

    # Experiment 2
    if Path(exp2_dir).exists():
        exp2_data = analyze_exp2(exp2_dir)
        report["experiments"]["exp2"] = exp2_data
        exp2_go = exp2_data.get("go_nogo", {}).get("decision", "UNKNOWN")
    else:
        exp2_go = "UNKNOWN"

    # Experiment 3
    if Path(exp3_dir).exists():
        exp3_data = analyze_exp3(exp3_dir)
        report["experiments"]["exp3"] = exp3_data
        exp3_go = "UNKNOWN"  # requires manual annotation
    else:
        exp3_go = "UNKNOWN"

    # Decision matrix
    report["decision_matrix"] = decision_matrix(exp1_go, exp2_go, exp3_go)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("=" * 60)
    print("AGE-CTF Experiment Results Summary")
    print("=" * 60)

    if "exp1" in report["experiments"]:
        print("\n" + format_exp1_table(report["experiments"]["exp1"]))

    if "exp2" in report["experiments"]:
        print("\n" + format_exp2_table(report["experiments"]["exp2"]))

    print(f"\n## Final Decision Matrix")
    dm = report["decision_matrix"]
    print(f"Exp1={dm['experiments'].get('exp1', '?')}, "
          f"Exp2={dm['experiments'].get('exp2', '?')}, "
          f"Exp3={dm['experiments'].get('exp3', '?')}")
    print(f"Decision: {dm['decision']}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--exp1-results", default="exp1_homogeneity/results.json")
    parser.add_argument("--exp2-trajectory-dir", default="exp2_continuation/trajectories")
    parser.add_argument("--exp3-trajectory-dir", default="exp3_guidance/trajectories")
    parser.add_argument("--output", default="full_report.json")
    args = parser.parse_args()

    generate_full_report(
        args.exp1_results,
        args.exp2_trajectory_dir,
        args.exp3_trajectory_dir,
        args.output,
    )
