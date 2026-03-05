#!/usr/bin/env python3
"""
Experiment 1: Trajectory Homogeneity Analysis

Research Question: Do multiple independent agent runs on the same CTF
challenge converge to similar attack strategies, wasting parallel compute?

Setup: For each of 10 pilot challenges, run baseline agent 8 times independently.
  - Temperature = 1.0, 30 steps or 15 min per run
  - Fresh Docker container and independent seed per run
  - Total: 80 runs

Measurements: M1-M4 (tool Jaccard, embedding similarity, mode frequency,
complementary coverage).
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from run_agent import run_single, run_single_simulated, RunResult


N_RUNS_PER_CHALLENGE = 8
MAX_STEPS = 30
MAX_TIME = 900  # 15 minutes


def load_challenges(config_file: str) -> List[Dict]:
    """Load pilot challenges from config."""
    with open(config_file) as f:
        config = json.load(f)
    return config.get("challenges", [])


def run_experiment(challenges: List[Dict], output_dir: str,
                   model_config: Dict, simulated: bool = False):
    """Run Experiment 1: 8 independent runs per challenge."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ch_idx, challenge in enumerate(challenges):
        ch_id = challenge["id"]
        ch_desc = challenge.get("description", "")
        ch_dir = challenge.get("path", ".")
        ch_cat = challenge.get("category", "unknown")

        print(f"\n{'='*60}")
        print(f"Challenge {ch_idx+1}/{len(challenges)}: {ch_id} ({ch_cat})")
        print(f"{'='*60}")

        for run_idx in range(N_RUNS_PER_CHALLENGE):
            seed = 100 + run_idx * 31  # spread seeds
            run_id = f"exp1_{ch_id}_run{run_idx:02d}"

            # Skip if already completed
            result_file = output_path / f"{run_id}.json"
            if result_file.exists():
                print(f"  Run {run_idx+1}/{N_RUNS_PER_CHALLENGE}: SKIPPED (exists)")
                with open(result_file) as f:
                    existing = json.load(f)
                all_results.append(existing)
                continue

            print(f"  Run {run_idx+1}/{N_RUNS_PER_CHALLENGE} (seed={seed})...", end=" ", flush=True)
            start = time.time()

            try:
                if simulated:
                    result = run_single_simulated(
                        challenge_id=ch_id,
                        challenge_desc=ch_desc,
                        experiment="exp1_homogeneity",
                        condition="baseline",
                        seed=seed,
                        max_steps=MAX_STEPS,
                        max_time=MAX_TIME,
                        output_dir=str(output_path),
                        model_config=model_config,
                    )
                else:
                    result = run_single(
                        challenge_id=ch_id,
                        challenge_desc=ch_desc,
                        challenge_dir=ch_dir,
                        experiment="exp1_homogeneity",
                        condition="baseline",
                        seed=seed,
                        max_steps=MAX_STEPS,
                        max_time=MAX_TIME,
                        output_dir=str(output_path),
                        model_config=model_config,
                    )

                elapsed = time.time() - start
                solved = result.outcome.get("solved", False)
                steps = len(result.trajectory)
                print(f"{'SOLVED' if solved else 'FAILED'} ({steps} steps, {elapsed:.0f}s)")
                all_results.append(json.loads(json.dumps(
                    {"run_id": result.run_id, **result.outcome},
                    default=str
                )))

            except Exception as e:
                print(f"ERROR: {e}")

    # Save summary
    summary_file = output_path.parent / "results_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"runs": all_results, "n_challenges": len(challenges),
                   "n_runs_per_challenge": N_RUNS_PER_CHALLENGE}, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


def analyze_and_report(trajectory_dir: str, output_file: str):
    """Run the full analysis pipeline after all runs complete."""
    from embed_trajectories import process_trajectory_dir
    from compute_similarity import analyze_all

    embeddings_file = str(Path(trajectory_dir).parent / "embeddings" / "embeddings.json")

    # Step 1: Compute embeddings
    print("\nComputing trajectory embeddings...")
    try:
        process_trajectory_dir(trajectory_dir, embeddings_file)
    except Exception as e:
        print(f"Embedding computation failed: {e}")
        embeddings_file = None

    # Step 2: Compute similarity metrics
    print("\nComputing similarity metrics...")
    results = analyze_all(trajectory_dir, embeddings_file)

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    from analyze_results import format_exp1_table
    print("\n" + format_exp1_table(results))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1: Homogeneity Analysis")
    parser.add_argument("--challenges", default="../config/challenges.json",
                        help="Path to challenge config")
    parser.add_argument("--output-dir", default="../exp1_homogeneity/trajectories",
                        help="Output directory for trajectories")
    parser.add_argument("--results-file", default="../exp1_homogeneity/results.json",
                        help="Output file for analysis results")
    parser.add_argument("--simulated", action="store_true",
                        help="Use simulated environment (no Docker)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip runs, only analyze existing trajectories")
    parser.add_argument("--model-config", default="../config/models.json",
                        help="Model configuration file")
    args = parser.parse_args()

    load_dotenv()

    # Load model config
    with open(args.model_config) as f:
        models = json.load(f)
    model_config = models.get("primary", {
        "provider": "google",
        "model": "gemini-2.5-pro-preview-05-06",
        "temperature": 1.0,
    })

    if not args.analyze_only:
        challenges = load_challenges(args.challenges)
        if not challenges:
            print("No challenges found. Run select_challenges.py first.")
            exit(1)
        run_experiment(challenges, args.output_dir, model_config,
                       simulated=args.simulated)

    # Analyze
    print("\n\nRunning analysis...")
    results = analyze_and_report(args.output_dir, args.results_file)

    go_decision = results.get("go_nogo", {}).get("decision", "UNKNOWN")
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1 DECISION: {go_decision}")
    print(f"{'='*60}")
