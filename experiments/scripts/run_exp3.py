#!/usr/bin/env python3
"""
Experiment 3: Diversity Guidance Effectiveness

Research Question: Does explicitly telling the agent about other explored
directions cause it to pursue meaningfully different — and productive —
attack strategies?

Setup: Select top 5 most homogeneous challenges from Experiment 1.
  3 conditions x 5 runs per challenge = 75 runs total.
  A) No hint (control)
  B) Negative hint (tells what others are doing, asks to diverge)
  C) Positive hint (suggests specific alternative approach)
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

from dotenv import load_dotenv

from run_agent import run_single, run_single_simulated
from extract_tools import classify_attack_vector, ATTACK_VECTORS


N_RUNS_PER_CONDITION = 5
MAX_STEPS = 30
MAX_TIME = 900

# Alternative attack suggestions per category
ALTERNATIVE_APPROACHES = {
    "web": [
        "Server-Side Request Forgery (SSRF) via file upload or URL parameters",
        "XML External Entity (XXE) injection via XML parsing endpoints",
        "Server-Side Template Injection (SSTI) in templating engine",
        "Insecure Direct Object Reference (IDOR) in API endpoints",
        "Authentication bypass via JWT token manipulation",
    ],
    "reverse": [
        "Dynamic analysis with gdb breakpoints on comparison functions",
        "Symbolic execution using angr or z3 to solve constraints",
        "Side-channel analysis (timing, memory access patterns)",
        "Binary patching to bypass checks and extract flag",
        "Anti-debugging detection and bypass techniques",
    ],
    "crypto": [
        "Padding oracle attack on CBC mode encryption",
        "RSA attacks: small exponent, Wiener, Hastad, common modulus",
        "Known-plaintext or chosen-plaintext attack patterns",
        "Hash length extension attack",
        "Lattice-based attacks (LLL algorithm)",
    ],
}


def select_most_homogeneous(exp1_results_file: str, n: int = 5) -> List[Dict]:
    """Select the N most homogeneous challenges from Experiment 1 results."""
    with open(exp1_results_file) as f:
        results = json.load(f)

    per_challenge = results.get("per_challenge", [])

    # Sort by embedding similarity (descending), fall back to Jaccard
    per_challenge.sort(
        key=lambda r: (r.get("avg_embedding_similarity") or r.get("avg_tool_jaccard", 0)),
        reverse=True,
    )

    return per_challenge[:n]


def get_dominant_approach(exp1_dir: str, challenge_id: str) -> str:
    """Get the most common attack approach for a challenge from Exp 1 data."""
    trajectories = []
    for filepath in Path(exp1_dir).glob(f"*{challenge_id}*.json"):
        with open(filepath) as f:
            data = json.load(f)
        trajectories.append(data.get("trajectory", []))

    if not trajectories:
        return "standard reconnaissance and enumeration"

    vectors = [classify_attack_vector(t) for t in trajectories]
    counter = Counter(vectors)
    dominant = counter.most_common(1)[0][0]

    # Map to human-readable description
    descriptions = {
        "sqli": "SQL injection on form parameters",
        "xss": "Cross-Site Scripting (XSS) attacks",
        "ssrf": "Server-Side Request Forgery",
        "lfi": "Local File Inclusion via path traversal",
        "rce": "Remote Code Execution via command injection",
        "bof": "Buffer overflow exploitation",
        "format_string": "Format string vulnerability exploitation",
        "static_rev": "Static reverse engineering (disassembly/decompilation)",
        "dynamic_rev": "Dynamic analysis with debugger",
        "known_cipher": "Known cipher/encryption attacks",
        "frequency_analysis": "Frequency analysis on ciphertext",
        "brute_force": "Brute force / dictionary attacks",
        "steganography": "Steganography extraction",
    }
    return descriptions.get(dominant, dominant)


def get_alternative_approach(category: str, dominant: str, idx: int = 0) -> str:
    """Get an alternative approach suggestion different from the dominant one."""
    alternatives = ALTERNATIVE_APPROACHES.get(category, ALTERNATIVE_APPROACHES["web"])
    # Try to pick one that doesn't overlap with dominant
    for alt in alternatives:
        if dominant.lower() not in alt.lower():
            return alt
    return alternatives[idx % len(alternatives)]


# ---------------------------------------------------------------------------
# Prompt Conditions
# ---------------------------------------------------------------------------

def build_negative_hint(dominant_approach: str) -> str:
    """Condition B: Tell agent what others are doing, ask to diverge."""
    return (
        f"NOTE: Other security analysts working on this challenge are already pursuing "
        f"the following approach: {dominant_approach}. To maximize coverage, please explore a "
        f"DIFFERENT attack vector. Think about what other vulnerabilities might exist "
        f"in this system."
    )


def build_positive_hint(alternative_approach: str) -> str:
    """Condition C: Suggest a specific alternative direction."""
    return (
        f"Consider exploring the following attack vector: {alternative_approach}. "
        f"This approach has not been extensively tested yet."
    )


# ---------------------------------------------------------------------------
# Run Experiment
# ---------------------------------------------------------------------------

def run_experiment(challenges: List[Dict], exp1_dir: str,
                   challenges_config: str, output_dir: str,
                   model_config: Dict, simulated: bool = False):
    """Run Experiment 3: diversity guidance on most homogeneous challenges."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load challenge descriptions
    with open(challenges_config) as f:
        ch_config = json.load(f)
    ch_descs = {ch["id"]: ch for ch in ch_config.get("challenges", [])}

    for ch_idx, challenge in enumerate(challenges):
        ch_id = challenge["challenge_id"]
        ch_info = ch_descs.get(ch_id, {})
        ch_desc = ch_info.get("description", "")
        ch_dir = ch_info.get("path", ".")
        ch_cat = ch_info.get("category", "web")

        # Get dominant approach from Exp 1
        dominant = get_dominant_approach(exp1_dir, ch_id)
        alternative = get_alternative_approach(ch_cat, dominant)

        print(f"\n{'='*60}")
        print(f"Challenge {ch_idx+1}/{len(challenges)}: {ch_id}")
        print(f"Dominant approach: {dominant}")
        print(f"Alternative suggestion: {alternative}")
        print(f"{'='*60}")

        conditions = [
            ("no_hint", "", ""),
            ("negative_hint", "", build_negative_hint(dominant)),
            ("positive_hint", "", build_positive_hint(alternative)),
        ]

        for cond_name, prefix, suffix in conditions:
            for run_idx in range(N_RUNS_PER_CONDITION):
                seed = 300 + ch_idx * 100 + run_idx * 17
                run_id = f"exp3_{ch_id}_{cond_name}_run{run_idx:02d}"

                result_file = output_path / f"{run_id}.json"
                if result_file.exists():
                    print(f"  {cond_name} run {run_idx+1}: SKIPPED (exists)")
                    continue

                print(f"  {cond_name} run {run_idx+1}/{N_RUNS_PER_CONDITION}...", end=" ", flush=True)
                start = time.time()

                try:
                    if simulated:
                        result = run_single_simulated(
                            challenge_id=ch_id,
                            challenge_desc=ch_desc,
                            experiment="exp3_guidance",
                            condition=cond_name,
                            seed=seed,
                            max_steps=MAX_STEPS,
                            max_time=MAX_TIME,
                            output_dir=str(output_path),
                            prompt_prefix=prefix,
                            prompt_suffix=suffix,
                            model_config=model_config,
                        )
                    else:
                        result = run_single(
                            challenge_id=ch_id,
                            challenge_desc=ch_desc,
                            challenge_dir=ch_dir,
                            experiment="exp3_guidance",
                            condition=cond_name,
                            seed=seed,
                            max_steps=MAX_STEPS,
                            max_time=MAX_TIME,
                            output_dir=str(output_path),
                            prompt_prefix=prefix,
                            prompt_suffix=suffix,
                            model_config=model_config,
                        )

                    elapsed = time.time() - start
                    solved = result.outcome.get("solved", False)
                    print(f"{'SOLVED' if solved else 'FAILED'} ({elapsed:.0f}s)")

                except Exception as e:
                    print(f"ERROR: {e}")


def analyze_and_report(trajectory_dir: str, exp1_dir: str, output_file: str):
    """Analyze Experiment 3 results."""
    from extract_tools import extract_tool_set
    from compute_similarity import tool_jaccard as jaccard_fn

    results_by_challenge = {}

    for filepath in Path(trajectory_dir).glob("*.json"):
        with open(filepath) as f:
            data = json.load(f)

        ch_id = data.get("challenge_id", "unknown")
        condition = data.get("condition", "unknown")
        solved = data.get("outcome", {}).get("solved", False)
        trajectory = data.get("trajectory", [])

        if ch_id not in results_by_challenge:
            results_by_challenge[ch_id] = {}
        if condition not in results_by_challenge[ch_id]:
            results_by_challenge[ch_id][condition] = []

        results_by_challenge[ch_id][condition].append({
            "solved": solved,
            "trajectory": trajectory,
            "attack_vector": classify_attack_vector(trajectory),
        })

    # Compute per-challenge metrics
    summary = []
    for ch_id, cond_data in results_by_challenge.items():
        baseline_runs = cond_data.get("no_hint", [])
        neg_hint_runs = cond_data.get("negative_hint", [])
        pos_hint_runs = cond_data.get("positive_hint", [])

        # Divergence: avg Jaccard distance between hinted and baseline
        neg_divergence = _compute_divergence(neg_hint_runs, baseline_runs) if neg_hint_runs and baseline_runs else None
        pos_divergence = _compute_divergence(pos_hint_runs, baseline_runs) if pos_hint_runs and baseline_runs else None

        # Success rates
        a_solved = sum(1 for r in baseline_runs if r["solved"]) if baseline_runs else 0
        b_solved = sum(1 for r in neg_hint_runs if r["solved"]) if neg_hint_runs else 0
        c_solved = sum(1 for r in pos_hint_runs if r["solved"]) if pos_hint_runs else 0

        # Dominant vector
        baseline_vectors = [r["attack_vector"] for r in baseline_runs]
        baseline_mode = Counter(baseline_vectors).most_common(1)[0][0] if baseline_vectors else "unknown"

        summary.append({
            "challenge_id": ch_id,
            "baseline_mode": baseline_mode,
            "neg_divergence": round(neg_divergence, 3) if neg_divergence else None,
            "pos_divergence": round(pos_divergence, 3) if pos_divergence else None,
            "a_solved": f"{a_solved}/{len(baseline_runs)}",
            "b_solved": f"{b_solved}/{len(neg_hint_runs)}",
            "c_solved": f"{c_solved}/{len(pos_hint_runs)}",
        })

    # Go/No-Go
    avg_divergence = _avg([s["neg_divergence"] for s in summary if s["neg_divergence"]] +
                          [s["pos_divergence"] for s in summary if s["pos_divergence"]])

    if avg_divergence and avg_divergence > 0.5:
        go = "GO"
        reason = f"Avg divergence {avg_divergence:.2f} > 0.5; diversity guidance is effective"
    elif avg_divergence and avg_divergence < 0.2:
        go = "NO-GO"
        reason = f"Avg divergence {avg_divergence:.2f} < 0.2; agent ignores hints"
    else:
        go = "BORDERLINE"
        div_str = f"{avg_divergence:.2f}" if avg_divergence else "N/A"
        reason = f"Avg divergence {div_str}; mixed results"

    output = {
        "per_challenge": summary,
        "go_nogo": {"decision": go, "reason": reason},
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print table
    print("\n## Experiment 3: Diversity Guidance Results\n")
    print("| Challenge | Baseline Mode | Neg Divergence | Pos Divergence | A Solved | B Solved | C Solved |")
    print("|-----------|--------------|----------------|----------------|----------|----------|----------|")
    for s in summary:
        print(f"| {s['challenge_id'][:15]} | {s['baseline_mode'][:12]} | "
              f"{s['neg_divergence'] or 'N/A':>14} | {s['pos_divergence'] or 'N/A':>14} | "
              f"{s['a_solved']:>8} | {s['b_solved']:>8} | {s['c_solved']:>8} |")

    print(f"\n**Decision**: {go} — {reason}")
    return output


def _compute_divergence(hint_runs, baseline_runs):
    """Compute average Jaccard distance between hint and baseline runs."""
    from extract_tools import extract_tool_set

    distances = []
    for hr in hint_runs:
        ht = extract_tool_set(hr["trajectory"])
        for br in baseline_runs:
            bt = extract_tool_set(br["trajectory"])
            union = ht | bt
            if union:
                distances.append(1 - len(ht & bt) / len(union))
            else:
                distances.append(0)
    return sum(distances) / len(distances) if distances else 0


def _avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 3: Diversity Guidance")
    parser.add_argument("--exp1-results", default="../exp1_homogeneity/results.json")
    parser.add_argument("--exp1-dir", default="../exp1_homogeneity/trajectories")
    parser.add_argument("--challenges", default="../config/challenges.json")
    parser.add_argument("--output-dir", default="../exp3_guidance/trajectories")
    parser.add_argument("--results-file", default="../exp3_guidance/results.json")
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--model-config", default="../config/models.json")
    args = parser.parse_args()

    load_dotenv()

    with open(args.model_config) as f:
        models = json.load(f)
    model_config = models.get("primary", {})

    if not args.analyze_only:
        # Select most homogeneous challenges
        print("Selecting most homogeneous challenges from Experiment 1...")
        if Path(args.exp1_results).exists():
            challenges = select_most_homogeneous(args.exp1_results, n=5)
        else:
            print("Experiment 1 results not found. Using all challenges.")
            with open(args.challenges) as f:
                ch_config = json.load(f)
            challenges = [{"challenge_id": ch["id"]} for ch in ch_config.get("challenges", [])[:5]]

        print(f"Selected {len(challenges)} challenges")

        run_experiment(challenges, args.exp1_dir, args.challenges,
                       args.output_dir, model_config, simulated=args.simulated)

    # Analyze
    print("\n\nRunning analysis...")
    results = analyze_and_report(args.output_dir, args.exp1_dir, args.results_file)

    go_decision = results.get("go_nogo", {}).get("decision", "UNKNOWN")
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 3 DECISION: {go_decision}")
    print(f"{'='*60}")
