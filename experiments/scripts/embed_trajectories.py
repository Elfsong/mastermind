#!/usr/bin/env python3
"""
Batch compute trajectory embeddings using Google's text-embedding model.

Reads trajectory JSON files, concatenates action+observation pairs into
a single text per trajectory, and embeds them for similarity analysis.
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv


def trajectory_to_text(trajectory: List[Dict], max_obs_len: int = 500) -> str:
    """Convert a trajectory to a single text string for embedding."""
    parts = []
    for step in trajectory:
        action = step.get("action", "")
        observation = step.get("observation", "")[:max_obs_len]
        parts.append(f"Action: {action}\nObs: {observation}")
    return "\n".join(parts)


def embed_with_google(texts: List[str], model: str = "text-embedding-004") -> List[List[float]]:
    """Embed texts using Google's embedding API via Gemini."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    embeddings = []

    # Process in batches to avoid rate limits
    batch_size = 5
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for text in batch:
            # Truncate to model's max input
            truncated = text[:8000]
            try:
                result = genai.embed_content(
                    model=f"models/{model}",
                    content=truncated,
                    task_type="SEMANTIC_SIMILARITY",
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"  Embedding error: {e}")
                embeddings.append([0.0] * 768)  # fallback zero vector
            time.sleep(0.5)  # rate limit

    return embeddings


def process_trajectory_dir(trajectory_dir: str, output_file: str,
                           model: str = "text-embedding-004"):
    """Process all trajectory files and compute embeddings."""
    load_dotenv()

    trajectory_files = sorted(Path(trajectory_dir).glob("*.json"))
    if not trajectory_files:
        print(f"No trajectory files found in {trajectory_dir}")
        return

    print(f"Processing {len(trajectory_files)} trajectory files...")

    results = {}
    texts = []
    run_ids = []

    for filepath in trajectory_files:
        with open(filepath) as f:
            data = json.load(f)
        run_id = data.get("run_id", filepath.stem)
        trajectory = data.get("trajectory", [])
        text = trajectory_to_text(trajectory)
        texts.append(text)
        run_ids.append(run_id)

    print(f"Computing embeddings for {len(texts)} trajectories...")
    embeddings = embed_with_google(texts, model=model)

    for run_id, embedding in zip(run_ids, embeddings):
        results[run_id] = embedding

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Embeddings saved to {output_file}")
    print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute trajectory embeddings")
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--output", default="embeddings.json")
    parser.add_argument("--model", default="text-embedding-004")
    args = parser.parse_args()

    process_trajectory_dir(args.trajectory_dir, args.output, args.model)
