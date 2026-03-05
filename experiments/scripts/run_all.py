#!/usr/bin/env python3
"""
AGE-CTF: Main experiment orchestrator.

Runs all three hypothesis validation experiments in sequence:
  1. Trajectory Homogeneity Analysis (80 runs)
  2. State-Based Continuation Effectiveness (135 runs)
  3. Diversity Guidance Effectiveness (75 runs)

Usage:
  python run_all.py                        # Full pipeline with Docker
  python run_all.py --simulated            # Simulated mode (no Docker)
  python run_all.py --start-from exp2      # Resume from Experiment 2
  python run_all.py --analyze-only         # Only analyze existing data
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv


EXPERIMENTS_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent


def check_prerequisites():
    """Verify all required dependencies and configuration."""
    errors = []

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        errors.append("GOOGLE_API_KEY not set. Add it to .env file.")

    # Check challenge config
    config_file = EXPERIMENTS_ROOT / "config" / "challenges.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        if not config.get("challenges"):
            errors.append("No challenges configured. Run select_challenges.py first (or use --skip-selection).")
    else:
        errors.append(f"Challenge config not found: {config_file}")

    # Check Docker
    import subprocess
    result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    if result.returncode != 0:
        errors.append("Docker not available. Use --simulated flag for Docker-free mode.")

    return errors


def run_challenge_selection(bench_dir: str, skip_screening: bool = False):
    """Step 0: Select pilot challenges."""
    print("\n" + "=" * 70)
    print("STEP 0: Challenge Selection")
    print("=" * 70)

    sys.path.insert(0, str(SCRIPTS_DIR))
    from select_challenges import (
        load_nyu_challenges, filter_challenges,
        run_baseline_screening, select_pilot_challenges,
        save_selected_challenges,
    )

    challenges = load_nyu_challenges(bench_dir)
    print(f"Found {len(challenges)} challenges in NYU CTF Bench")

    filtered = filter_challenges(challenges,
                                  categories=["web", "reverse", "crypto"],
                                  min_difficulty="medium")
    print(f"After filtering: {len(filtered)} candidates")

    if skip_screening:
        import random
        random.shuffle(filtered)
        selected = select_pilot_challenges(
            [{**ch, "_screening_rate": 0.5} for ch in filtered]
        )
    else:
        print("Running baseline screening (3 runs per challenge)...")
        screened = run_baseline_screening(filtered)
        selected = select_pilot_challenges(screened)

    output = str(EXPERIMENTS_ROOT / "config" / "challenges.json")
    save_selected_challenges(selected, output)
    return selected


def run_experiment_1(simulated: bool = False, analyze_only: bool = False):
    """Step 1: Trajectory Homogeneity Analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Trajectory Homogeneity Analysis")
    print("=" * 70)

    sys.path.insert(0, str(SCRIPTS_DIR))
    from run_exp1 import load_challenges, run_experiment, analyze_and_report

    config_file = str(EXPERIMENTS_ROOT / "config" / "models.json")
    with open(config_file) as f:
        models = json.load(f)
    model_config = models.get("primary", {})

    challenges_file = str(EXPERIMENTS_ROOT / "config" / "challenges.json")
    traj_dir = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "trajectories")
    results_file = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "results.json")

    if not analyze_only:
        challenges = load_challenges(challenges_file)
        if not challenges:
            print("ERROR: No challenges configured.")
            return None
        run_experiment(challenges, traj_dir, model_config, simulated=simulated)

    results = analyze_and_report(traj_dir, results_file)
    return results


def run_experiment_2(simulated: bool = False, analyze_only: bool = False):
    """Step 2: State-Based Continuation Effectiveness."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: State-Based Continuation Effectiveness")
    print("=" * 70)

    sys.path.insert(0, str(SCRIPTS_DIR))
    from run_exp2 import extract_intermediate_states, run_experiment, analyze_and_report
    from dataclasses import asdict

    config_file = str(EXPERIMENTS_ROOT / "config" / "models.json")
    with open(config_file) as f:
        models = json.load(f)
    model_config = models.get("primary", {})

    exp1_dir = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "trajectories")
    challenges_file = str(EXPERIMENTS_ROOT / "config" / "challenges.json")
    traj_dir = str(EXPERIMENTS_ROOT / "exp2_continuation" / "trajectories")
    states_file = str(EXPERIMENTS_ROOT / "exp2_continuation" / "intermediate_states" / "states.json")
    results_file = str(EXPERIMENTS_ROOT / "exp2_continuation" / "results.json")

    if not analyze_only:
        states = extract_intermediate_states(exp1_dir, n_states=15, model_config=model_config)
        Path(states_file).parent.mkdir(parents=True, exist_ok=True)
        with open(states_file, "w") as f:
            json.dump([asdict(s) for s in states], f, indent=2, default=str)
        print(f"Extracted {len(states)} intermediate states")

        if not states:
            print("No solved trajectories from Experiment 1. Skipping Experiment 2.")
            return None

        run_experiment(states, traj_dir, model_config, challenges_file, simulated=simulated)

    results = analyze_and_report(traj_dir, states_file, results_file)
    return results


def run_experiment_3(simulated: bool = False, analyze_only: bool = False):
    """Step 3: Diversity Guidance Effectiveness."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Diversity Guidance Effectiveness")
    print("=" * 70)

    sys.path.insert(0, str(SCRIPTS_DIR))
    from run_exp3 import select_most_homogeneous, run_experiment, analyze_and_report

    config_file = str(EXPERIMENTS_ROOT / "config" / "models.json")
    with open(config_file) as f:
        models = json.load(f)
    model_config = models.get("primary", {})

    exp1_results = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "results.json")
    exp1_dir = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "trajectories")
    challenges_file = str(EXPERIMENTS_ROOT / "config" / "challenges.json")
    traj_dir = str(EXPERIMENTS_ROOT / "exp3_guidance" / "trajectories")
    results_file = str(EXPERIMENTS_ROOT / "exp3_guidance" / "results.json")

    if not analyze_only:
        if Path(exp1_results).exists():
            challenges = select_most_homogeneous(exp1_results, n=5)
        else:
            with open(challenges_file) as f:
                ch_config = json.load(f)
            challenges = [{"challenge_id": ch["id"]} for ch in ch_config.get("challenges", [])[:5]]

        run_experiment(challenges, exp1_dir, challenges_file, traj_dir,
                       model_config, simulated=simulated)

    results = analyze_and_report(traj_dir, exp1_dir, results_file)
    return results


def print_decision_matrix(exp1_go, exp2_go, exp3_go):
    """Print the final decision matrix."""
    from analyze_results import decision_matrix

    result = decision_matrix(exp1_go, exp2_go, exp3_go)

    print("\n" + "=" * 70)
    print("FINAL DECISION MATRIX")
    print("=" * 70)
    print(f"  Experiment 1 (Homogeneity):  {exp1_go}")
    print(f"  Experiment 2 (Continuation): {exp2_go}")
    print(f"  Experiment 3 (Guidance):     {exp3_go}")
    print(f"\n  DECISION: {result['decision']}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGE-CTF Experiment Orchestrator")
    parser.add_argument("--simulated", action="store_true",
                        help="Use simulated environment (no Docker)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing data")
    parser.add_argument("--start-from", choices=["selection", "exp1", "exp2", "exp3", "analysis"],
                        default="selection",
                        help="Resume from a specific stage")
    parser.add_argument("--bench-dir", default=str(EXPERIMENTS_ROOT.parent / "NYU_CTF_Bench"),
                        help="NYU CTF Bench directory")
    parser.add_argument("--skip-screening", action="store_true",
                        help="Skip baseline screening for challenge selection")
    args = parser.parse_args()

    load_dotenv(EXPERIMENTS_ROOT.parent / ".env")

    start_time = time.time()
    stages = ["selection", "exp1", "exp2", "exp3", "analysis"]
    start_idx = stages.index(args.start_from)

    exp1_results = exp2_results = exp3_results = None

    # Stage 0: Challenge Selection
    if start_idx <= 0 and not args.analyze_only:
        run_challenge_selection(args.bench_dir, skip_screening=args.skip_screening)

    # Stage 1: Experiment 1
    if start_idx <= 1:
        exp1_results = run_experiment_1(simulated=args.simulated,
                                         analyze_only=args.analyze_only)

    # Stage 2: Experiment 2
    if start_idx <= 2:
        exp2_results = run_experiment_2(simulated=args.simulated,
                                         analyze_only=args.analyze_only)

    # Stage 3: Experiment 3
    if start_idx <= 3:
        exp3_results = run_experiment_3(simulated=args.simulated,
                                         analyze_only=args.analyze_only)

    # Stage 4: Final Analysis
    exp1_go = (exp1_results or {}).get("go_nogo", {}).get("decision", "UNKNOWN")
    exp2_go = (exp2_results or {}).get("go_nogo", {}).get("decision", "UNKNOWN")
    exp3_go = (exp3_results or {}).get("go_nogo", {}).get("decision", "UNKNOWN")

    final = print_decision_matrix(exp1_go, exp2_go, exp3_go)

    # Save final report
    report_file = str(EXPERIMENTS_ROOT / "final_report.json")
    with open(report_file, "w") as f:
        json.dump({
            "exp1": exp1_results,
            "exp2": exp2_results,
            "exp3": exp3_results,
            "decision_matrix": final,
            "total_time_seconds": time.time() - start_time,
        }, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Report saved to: {report_file}")
