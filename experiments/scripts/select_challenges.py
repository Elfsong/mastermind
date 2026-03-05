#!/usr/bin/env python3
"""
Challenge selection for AGE-CTF pilot experiments.

Identifies challenges from NYU CTF Bench that existing agents solve
inconsistently (1/3 or 2/3 success rate) — the "sweet spot" where
exploration strategy plausibly matters.

Steps:
  1. Load challenge metadata from NYU CTF Bench
  2. Filter by category (web, reverse, crypto) and difficulty
  3. Run baseline agent 3 times per candidate
  4. Keep challenges with 1/3 or 2/3 success rate
  5. Select final 10: 4 web, 3 reverse, 3 crypto
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv


def load_nyu_challenges(bench_dir: str) -> List[Dict]:
    """Load challenge metadata from NYU CTF Bench.

    Supports two loading methods:
    1. development_dataset.json (main index with id -> metadata mapping)
    2. Individual challenge.json files found recursively
    """
    challenges = []
    bench_path = Path(bench_dir)

    # Method 1: Load from development_dataset.json (preferred)
    for dataset_file in ["development_dataset.json", "test_dataset.json"]:
        ds_path = bench_path / dataset_file
        if ds_path.exists():
            with open(ds_path) as f:
                dataset = json.load(f)
            if isinstance(dataset, dict):
                for ch_id, meta in dataset.items():
                    meta["id"] = ch_id
                    ch_path = bench_path / meta.get("path", "")
                    meta["_path"] = str(ch_path)
                    # Load challenge.json for description if available
                    ch_json = ch_path / "challenge.json"
                    if ch_json.exists():
                        try:
                            with open(ch_json) as f:
                                ch_meta = json.load(f)
                            meta["description"] = ch_meta.get("description", "")
                            meta["flag"] = ch_meta.get("flag", "")
                            meta["points"] = ch_meta.get("points", 0)
                            meta["files"] = ch_meta.get("files", [])
                        except (json.JSONDecodeError, IOError):
                            pass
                    challenges.append(meta)

    # Method 2: Fallback — load individual challenge.json files
    if not challenges:
        for metadata_file in bench_path.rglob("challenge.json"):
            try:
                with open(metadata_file) as f:
                    meta = json.load(f)
                meta["_path"] = str(metadata_file.parent)
                meta["id"] = meta.get("name", metadata_file.parent.name)
                challenges.append(meta)
            except (json.JSONDecodeError, IOError):
                continue

    return challenges


def filter_challenges(challenges: List[Dict],
                      categories: Optional[List[str]] = None,
                      min_difficulty: str = "medium") -> List[Dict]:
    """Filter challenges by category and difficulty."""
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    min_diff_val = difficulty_order.get(min_difficulty, 0)

    filtered = []
    for ch in challenges:
        # Get category - try various field names
        cat = (ch.get("category", "") or ch.get("type", "") or "").lower()

        # Map subcategories to normalized names
        if any(w in cat for w in ["web", "http", "webapp"]):
            cat = "web"
        elif any(w in cat for w in ["reverse", "rev", "binary", "pwn"]):
            cat = "reverse"
        elif any(w in cat for w in ["crypto", "cipher"]):
            cat = "crypto"
        elif any(w in cat for w in ["forensic", "stego"]):
            cat = "forensics"
        elif any(w in cat for w in ["misc"]):
            cat = "misc"

        if categories and cat not in categories:
            continue

        # Check difficulty (by label or points threshold)
        diff = (ch.get("difficulty", "") or "").lower()
        points = ch.get("points", 0)
        if isinstance(points, str):
            try:
                points = int(points)
            except ValueError:
                points = 0

        if diff in difficulty_order:
            if difficulty_order[diff] < min_diff_val:
                continue
        elif points > 0 and min_difficulty == "medium" and points < 200:
            continue  # skip easy challenges (< 200 points)

        ch["_normalized_category"] = cat
        filtered.append(ch)

    return filtered


def run_baseline_screening(challenges: List[Dict], n_runs: int = 3,
                           max_steps: int = 15,
                           output_dir: str = "screening") -> List[Dict]:
    """Run baseline agent on each challenge n_runs times to identify sweet spot."""
    from run_agent import run_single, run_single_simulated

    results = []
    for ch in challenges:
        ch_id = ch.get("id", ch.get("name", "unknown"))
        ch_desc = ch.get("description", "")
        ch_dir = ch.get("_path", ".")

        successes = 0
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 17
            try:
                result = run_single(
                    challenge_id=ch_id,
                    challenge_desc=ch_desc,
                    challenge_dir=ch_dir,
                    experiment="screening",
                    condition="baseline",
                    seed=seed,
                    max_steps=max_steps,
                    max_time=300,
                    output_dir=output_dir,
                )
                if result.outcome.get("solved"):
                    successes += 1
            except Exception as e:
                print(f"  Run {run_idx} failed for {ch_id}: {e}")

        ch["_screening_successes"] = successes
        ch["_screening_runs"] = n_runs
        ch["_screening_rate"] = successes / n_runs
        results.append(ch)

        print(f"  {ch_id}: {successes}/{n_runs}")

    return results


def select_pilot_challenges(screened: List[Dict],
                            target: Dict[str, int] = None) -> List[Dict]:
    """Select pilot challenges with 1/3 or 2/3 success rate.

    Target distribution: 4 web, 3 reverse, 3 crypto.
    """
    target = target or {"web": 4, "reverse": 3, "crypto": 3}

    # Filter to sweet spot
    sweet_spot = [ch for ch in screened
                  if ch.get("_screening_rate", 0) in (1/3, 2/3)]

    # If not enough in sweet spot, relax to include any non-0/0 and non-3/3
    if len(sweet_spot) < sum(target.values()):
        sweet_spot = [ch for ch in screened
                      if 0 < ch.get("_screening_rate", 0) < 1]

    selected = []
    for category, count in target.items():
        candidates = [ch for ch in sweet_spot
                      if ch.get("_normalized_category") == category]
        # Sort by how close to 50% success rate
        candidates.sort(key=lambda c: abs(c.get("_screening_rate", 0) - 0.5))
        selected.extend(candidates[:count])

    return selected


def save_selected_challenges(selected: List[Dict], output_file: str):
    """Save selected challenges to config."""
    config = {
        "description": "Pilot challenge selection for AGE-CTF experiments",
        "selection_criteria": "Challenges solved inconsistently (1/3 or 2/3) by baseline agent",
        "target_distribution": {"web": 4, "reverse": 3, "crypto": 3},
        "challenges": [],
    }

    for ch in selected:
        config["challenges"].append({
            "id": ch.get("id", ch.get("name", "unknown")),
            "category": ch.get("_normalized_category", "unknown"),
            "difficulty": ch.get("difficulty", "unknown"),
            "description": ch.get("description", "")[:500],
            "path": ch.get("_path", ""),
            "screening_rate": ch.get("_screening_rate", 0),
        })

    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Selected {len(selected)} challenges -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select pilot challenges")
    parser.add_argument("--bench-dir", default="NYU_CTF_Bench",
                        help="Path to NYU CTF Bench directory")
    parser.add_argument("--output", default="config/challenges.json")
    parser.add_argument("--skip-screening", action="store_true",
                        help="Skip baseline screening (use all matching challenges)")
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    load_dotenv()

    print("Loading challenges from NYU CTF Bench...")
    all_challenges = load_nyu_challenges(args.bench_dir)
    print(f"Found {len(all_challenges)} total challenges")

    print("Filtering by category and difficulty...")
    filtered = filter_challenges(all_challenges,
                                  categories=["web", "reverse", "crypto"],
                                  min_difficulty="medium")
    print(f"After filtering: {len(filtered)} challenges")

    if args.skip_screening:
        # Take a sample without screening
        import random
        random.shuffle(filtered)
        selected = select_pilot_challenges(
            [{**ch, "_screening_rate": 0.5} for ch in filtered]
        )
    else:
        print(f"\nRunning baseline screening ({args.n_runs} runs per challenge)...")
        screened = run_baseline_screening(filtered, n_runs=args.n_runs)
        selected = select_pilot_challenges(screened)

    save_selected_challenges(selected, args.output)
