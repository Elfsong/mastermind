#!/usr/bin/env python3
"""
Compute pairwise trajectory similarity metrics for Experiment 1.

Metrics:
  M1 - Tool-use Jaccard similarity
  M2 - Trajectory embedding cosine similarity
  M3 - Attack vector mode frequency
  M4 - Complementary coverage ratio
"""

import json
import argparse
import itertools
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

from extract_tools import (
    extract_tool_set,
    extract_command_set,
    classify_attack_vector,
)


# ---------------------------------------------------------------------------
# M1: Tool-use Jaccard similarity
# ---------------------------------------------------------------------------

def tool_jaccard(traj_a: List[Dict], traj_b: List[Dict]) -> float:
    """Compute Jaccard similarity of tool-use sets between two trajectories."""
    tools_a = extract_tool_set(traj_a)
    tools_b = extract_tool_set(traj_b)
    if not tools_a and not tools_b:
        return 1.0
    intersection = tools_a & tools_b
    union = tools_a | tools_b
    return len(intersection) / len(union)


def pairwise_jaccard(trajectories: List[List[Dict]]) -> List[float]:
    """Compute all pairwise Jaccard similarities."""
    sims = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            sims.append(tool_jaccard(trajectories[i], trajectories[j]))
    return sims


# ---------------------------------------------------------------------------
# M2: Trajectory embedding similarity (uses pre-computed embeddings)
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def pairwise_embedding_similarity(embeddings: List[List[float]]) -> List[float]:
    """Compute all pairwise cosine similarities from embeddings."""
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))
    return sims


# ---------------------------------------------------------------------------
# M3: Attack vector mode frequency
# ---------------------------------------------------------------------------

def mode_frequency(trajectories: List[List[Dict]]) -> Tuple[str, float]:
    """Compute the mode attack vector and its frequency."""
    vectors = [classify_attack_vector(t) for t in trajectories]
    if not vectors:
        return "unknown", 0.0
    counter = Counter(vectors)
    mode_vec, mode_count = counter.most_common(1)[0]
    return mode_vec, mode_count / len(vectors)


# ---------------------------------------------------------------------------
# M4: Complementary coverage ratio
# ---------------------------------------------------------------------------

def complementary_ratio(trajectories: List[List[Dict]]) -> float:
    """Compute ratio of union-of-commands to avg-commands-per-run."""
    all_commands: Set[str] = set()
    per_run_sizes = []
    for traj in trajectories:
        run_commands = extract_command_set(traj)
        per_run_sizes.append(len(run_commands))
        all_commands.update(run_commands)
    avg_size = np.mean(per_run_sizes) if per_run_sizes else 1
    if avg_size == 0:
        return 0.0
    return len(all_commands) / avg_size


# ---------------------------------------------------------------------------
# Full analysis per challenge
# ---------------------------------------------------------------------------

def analyze_challenge(challenge_id: str, run_files: List[str],
                      embeddings: Optional[Dict] = None) -> Dict:
    """Compute all metrics for a single challenge's runs."""
    trajectories = []
    outcomes = []

    for filepath in run_files:
        with open(filepath) as f:
            data = json.load(f)
        trajectories.append(data.get("trajectory", []))
        outcomes.append(data.get("outcome", {}).get("solved", False))

    n_runs = len(trajectories)
    n_solved = sum(outcomes)

    # M1: Jaccard
    jaccard_sims = pairwise_jaccard(trajectories)
    avg_jaccard = float(np.mean(jaccard_sims)) if jaccard_sims else 0.0

    # M2: Embedding similarity (if embeddings provided)
    avg_embedding_sim = None
    if embeddings:
        run_embeddings = []
        for filepath in run_files:
            run_id = Path(filepath).stem
            if run_id in embeddings:
                run_embeddings.append(embeddings[run_id])
        if len(run_embeddings) >= 2:
            embedding_sims = pairwise_embedding_similarity(run_embeddings)
            avg_embedding_sim = float(np.mean(embedding_sims))

    # M3: Mode frequency
    mode_vec, mode_freq = mode_frequency(trajectories)

    # M4: Complementary ratio
    comp_ratio = complementary_ratio(trajectories)

    return {
        "challenge_id": challenge_id,
        "n_runs": n_runs,
        "n_solved": n_solved,
        "avg_tool_jaccard": round(avg_jaccard, 3),
        "avg_embedding_similarity": round(avg_embedding_sim, 3) if avg_embedding_sim else None,
        "mode_attack_vector": mode_vec,
        "mode_frequency": round(mode_freq, 3),
        "complement_ratio": round(comp_ratio, 2),
    }


def analyze_all(trajectory_dir: str, embeddings_file: Optional[str] = None) -> Dict:
    """Analyze all challenges in a directory."""
    # Group trajectory files by challenge_id
    challenge_runs: Dict[str, List[str]] = {}
    for filepath in Path(trajectory_dir).glob("*.json"):
        with open(filepath) as f:
            data = json.load(f)
        cid = data.get("challenge_id", "unknown")
        challenge_runs.setdefault(cid, []).append(str(filepath))

    # Load embeddings if available
    embeddings = None
    if embeddings_file and Path(embeddings_file).exists():
        with open(embeddings_file) as f:
            embeddings = json.load(f)

    results = []
    for cid, files in sorted(challenge_runs.items()):
        result = analyze_challenge(cid, files, embeddings)
        results.append(result)

    # Aggregate statistics
    if results:
        avg_jaccard_all = np.mean([r["avg_tool_jaccard"] for r in results])
        emb_sims = [r["avg_embedding_similarity"] for r in results
                     if r["avg_embedding_similarity"] is not None]
        avg_emb_all = np.mean(emb_sims) if emb_sims else None
        avg_mode_freq = np.mean([r["mode_frequency"] for r in results])
        avg_comp_ratio = np.mean([r["complement_ratio"] for r in results])
    else:
        avg_jaccard_all = avg_emb_all = avg_mode_freq = avg_comp_ratio = 0

    return {
        "per_challenge": results,
        "aggregate": {
            "n_challenges": len(results),
            "avg_tool_jaccard": round(float(avg_jaccard_all), 3),
            "avg_embedding_similarity": round(float(avg_emb_all), 3) if avg_emb_all else None,
            "avg_mode_frequency": round(float(avg_mode_freq), 3),
            "avg_complement_ratio": round(float(avg_comp_ratio), 2),
        },
        "go_nogo": _evaluate_go_nogo(float(avg_emb_all) if avg_emb_all else 0,
                                      float(avg_mode_freq)),
    }


def _evaluate_go_nogo(avg_embedding_sim: float, avg_mode_freq: float) -> Dict:
    """Evaluate go/no-go criteria for Experiment 1."""
    if avg_embedding_sim > 0.6 and avg_mode_freq > 0.6:
        decision = "GO"
        reason = "High homogeneity confirmed; diversity guidance has clear target"
    elif avg_embedding_sim < 0.4 and avg_mode_freq < 0.5:
        decision = "NO-GO"
        reason = "Runs already naturally diverse; core motivation does not hold"
    else:
        decision = "BORDERLINE"
        reason = f"Embedding sim={avg_embedding_sim:.2f}, mode freq={avg_mode_freq:.2f}; examine per-category breakdown"
    return {"decision": decision, "reason": reason}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute trajectory similarity metrics")
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--embeddings-file", default=None)
    parser.add_argument("--output", default="similarity_results.json")
    args = parser.parse_args()

    results = analyze_all(args.trajectory_dir, args.embeddings_file)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment 1 Results Summary")
    print(f"{'='*50}")
    agg = results["aggregate"]
    print(f"Challenges analyzed: {agg['n_challenges']}")
    print(f"Avg Tool Jaccard:    {agg['avg_tool_jaccard']}")
    print(f"Avg Embedding Sim:   {agg['avg_embedding_similarity']}")
    print(f"Avg Mode Frequency:  {agg['avg_mode_frequency']}")
    print(f"Avg Complement Ratio:{agg['avg_complement_ratio']}")
    print(f"\nGo/No-Go: {results['go_nogo']['decision']}")
    print(f"Reason: {results['go_nogo']['reason']}")
