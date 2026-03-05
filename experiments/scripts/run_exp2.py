#!/usr/bin/env python3
"""
Experiment 2: State-Based Continuation Effectiveness

Research Question: Can an agent make faster progress by resuming from a
previously discovered intermediate state compared to starting from scratch?

Steps:
  1. Extract 15 "good intermediate states" from Experiment 1 trajectories
  2. Run agent under 3 conditions per state:
     A) Full Context  — complete prior trajectory
     B) Summary Only  — structured state summary (~300 tokens)
     C) Fresh Start   — no prior information
  3. 3 runs per condition per state = 135 runs total
  4. Compare steps-to-flag and success rates
"""

import os
import re
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv

from run_agent import run_single, run_single_simulated, RunResult, LLMClient


# ---------------------------------------------------------------------------
# Intermediate State Extraction
# ---------------------------------------------------------------------------

@dataclass
class IntermediateState:
    challenge_id: str
    category: str
    truncation_step: int
    full_history: List[Dict]
    summary: str
    discovered_files: List[str]
    discovered_services: List[str]
    vulnerability_hypothesis: str
    tools_used: List[str]
    failed_approaches: List[str]


def generate_state_summary(trajectory: List[Dict], challenge_desc: str,
                           truncation_step: int, llm: Optional[LLMClient] = None) -> Dict:
    """Generate a structured state summary from a trajectory up to truncation_step."""
    history = trajectory[:truncation_step + 1]

    # Extract information from trajectory
    tools_used = set()
    discovered_files = set()
    discovered_services = set()
    failed = []

    for step in history:
        action = step.get("action", "")
        obs = step.get("observation", "")

        # Extract tool names
        cmd_match = re.search(r'COMMAND:\s*(\S+)', action)
        if cmd_match:
            tools_used.add(cmd_match.group(1))

        # Extract discovered files
        for m in re.finditer(r'(/[\w/.-]+\.\w+)', obs):
            discovered_files.add(m.group(1))

        # Extract services/ports
        for m in re.finditer(r'(\d+)/tcp\s+open\s+(\S+)', obs):
            discovered_services.add(f"Port {m.group(1)}: {m.group(2)}")

        # Track failures
        if any(kw in obs.lower() for kw in ["error", "failed", "denied", "timeout", "not found"]):
            cmd = re.search(r'COMMAND:\s*(.+?)(?:\n|$)', action)
            if cmd:
                failed.append(cmd.group(1).strip()[:100])

    # Use LLM to generate summary if available
    if llm:
        summary_prompt = f"""Analyze the following CTF challenge trajectory and create a concise progress summary.

Challenge: {challenge_desc}

Trajectory ({len(history)} steps):
{_format_trajectory(history)}

Create a structured summary in this format:
## Progress Summary
- **Challenge type**: [type of challenge]
- **Discovered**: [key findings]
- **Vulnerability identified**: [the vulnerability, if found]
- **Evidence**: [supporting evidence]
- **Tools used**: [tools used so far]
- **Not yet attempted**: [approaches not yet tried]
"""
        try:
            summary = llm.generate(summary_prompt)
        except Exception as e:
            summary = _generate_fallback_summary(history, tools_used, discovered_files,
                                                  discovered_services)
    else:
        summary = _generate_fallback_summary(history, tools_used, discovered_files,
                                              discovered_services)

    return {
        "summary": summary,
        "discovered_files": sorted(discovered_files),
        "discovered_services": sorted(discovered_services),
        "tools_used": sorted(tools_used),
        "failed_approaches": failed[:10],
    }


def _format_trajectory(trajectory: List[Dict]) -> str:
    """Format trajectory for LLM consumption."""
    parts = []
    for step in trajectory:
        action = step.get("action", "")[:500]
        obs = step.get("observation", "")[:300]
        parts.append(f"Step {step.get('step', '?')}:\n  Action: {action}\n  Result: {obs}")
    return "\n".join(parts)


def _generate_fallback_summary(history, tools_used, files, services):
    """Generate a summary without LLM."""
    parts = ["## Progress Summary"]
    parts.append(f"- **Steps completed**: {len(history)}")
    if services:
        parts.append(f"- **Discovered services**: {', '.join(list(services)[:5])}")
    if files:
        parts.append(f"- **Discovered files**: {', '.join(list(files)[:5])}")
    parts.append(f"- **Tools used**: {', '.join(list(tools_used)[:10])}")
    return "\n".join(parts)


def find_aha_moment(trajectory: List[Dict]) -> int:
    """Heuristic to find the step where vulnerability was first identified."""
    vuln_keywords = [
        "vulnerability", "vulnerable", "exploit", "injection", "overflow",
        "bypass", "weakness", "flaw", "insecure", "misconfigur",
        "sql injection", "xss", "ssrf", "lfi", "rce", "buffer overflow",
        "format string", "race condition", "deserialization",
    ]

    for i, step in enumerate(trajectory):
        text = (step.get("action", "") + " " + step.get("observation", "")).lower()
        matches = sum(1 for kw in vuln_keywords if kw in text)
        if matches >= 2:  # At least 2 vulnerability indicators
            return i

    # If no clear aha moment, use ~40% through the trajectory
    return max(1, len(trajectory) * 2 // 5)


def extract_intermediate_states(exp1_dir: str, n_states: int = 15,
                                model_config: Optional[Dict] = None) -> List[IntermediateState]:
    """Extract intermediate states from Experiment 1's solved trajectories."""
    states = []
    solved_by_category = {"web": [], "reverse": [], "crypto": []}

    # Load all solved trajectories
    for filepath in Path(exp1_dir).glob("*.json"):
        with open(filepath) as f:
            data = json.load(f)
        if not data.get("outcome", {}).get("solved"):
            continue

        ch_id = data.get("challenge_id", "")
        # Infer category from challenge_id
        cat = "web"
        if any(w in ch_id.lower() for w in ["rev", "binary", "pwn"]):
            cat = "reverse"
        elif any(w in ch_id.lower() for w in ["crypto", "cipher"]):
            cat = "crypto"

        solved_by_category.get(cat, []).append(data)

    # Initialize LLM for summary generation
    llm = None
    if model_config:
        try:
            llm = LLMClient(
                model=model_config.get("model", "gemini-2.5-pro-preview-05-06"),
                temperature=0.3,  # lower temperature for summaries
                provider=model_config.get("provider", "google"),
            )
        except Exception:
            pass

    # Select states across categories
    target = {"web": 5, "reverse": 4, "crypto": 3}
    for cat, target_n in target.items():
        candidates = solved_by_category.get(cat, [])
        for data in candidates[:target_n]:
            trajectory = data.get("trajectory", [])
            truncation = find_aha_moment(trajectory)
            ch_desc = data.get("challenge_desc", "")

            summary_info = generate_state_summary(trajectory, ch_desc, truncation, llm)

            state = IntermediateState(
                challenge_id=data.get("challenge_id", "unknown"),
                category=cat,
                truncation_step=truncation,
                full_history=trajectory[:truncation + 1],
                summary=summary_info["summary"],
                discovered_files=summary_info["discovered_files"],
                discovered_services=summary_info["discovered_services"],
                vulnerability_hypothesis="See summary",
                tools_used=summary_info["tools_used"],
                failed_approaches=summary_info["failed_approaches"],
            )
            states.append(state)

    return states[:n_states]


# ---------------------------------------------------------------------------
# Experiment Conditions
# ---------------------------------------------------------------------------

def build_condition_a_prefix(state: IntermediateState) -> str:
    """Condition A: Full Context - complete prior trajectory."""
    parts = ["Here is your work so far:\n"]
    for step in state.full_history:
        parts.append(f"Step {step.get('step', '?')}:")
        parts.append(f"  Action: {step.get('action', '')[:500]}")
        parts.append(f"  Result: {step.get('observation', '')[:300]}")
        parts.append("")
    parts.append("Continue from where you left off.")
    return "\n".join(parts)


def build_condition_b_prefix(state: IntermediateState) -> str:
    """Condition B: Summary Only - structured state summary."""
    parts = [
        "Previous analysis has established the following:\n",
        state.summary,
        f"\nFiles discovered: {', '.join(state.discovered_files) or 'None'}",
        f"Services found: {', '.join(state.discovered_services) or 'None'}",
        f"Vulnerability hypothesis: {state.vulnerability_hypothesis}",
        f"Tools already used: {', '.join(state.tools_used) or 'None'}",
        f"Approaches that failed: {', '.join(state.failed_approaches) or 'None'}",
        "\nBuild on these findings.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Run Experiment
# ---------------------------------------------------------------------------

N_RUNS_PER_CONDITION = 3
MAX_STEPS = 20
MAX_TIME = 600


def run_experiment(states: List[IntermediateState], output_dir: str,
                   model_config: Dict, challenges_config: str,
                   simulated: bool = False):
    """Run Experiment 2: 3 conditions x 3 runs per state."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load challenge descriptions
    with open(challenges_config) as f:
        ch_config = json.load(f)
    ch_descs = {ch["id"]: ch.get("description", "") for ch in ch_config.get("challenges", [])}

    conditions = [
        ("full_context", lambda s: build_condition_a_prefix(s)),
        ("summary_only", lambda s: build_condition_b_prefix(s)),
        ("fresh_start", lambda s: ""),
    ]

    for state_idx, state in enumerate(states):
        ch_desc = ch_descs.get(state.challenge_id, "")
        ch_dir = ""  # would come from challenge config

        print(f"\n{'='*60}")
        print(f"State {state_idx+1}/{len(states)}: {state.challenge_id} ({state.category})")
        print(f"Truncated at step {state.truncation_step}")
        print(f"{'='*60}")

        for cond_name, prefix_fn in conditions:
            prefix = prefix_fn(state)

            for run_idx in range(N_RUNS_PER_CONDITION):
                seed = 200 + state_idx * 100 + run_idx * 13
                run_id = f"exp2_{state.challenge_id}_{cond_name}_run{run_idx:02d}"

                result_file = output_path / f"{run_id}.json"
                if result_file.exists():
                    print(f"  {cond_name} run {run_idx+1}: SKIPPED (exists)")
                    continue

                print(f"  {cond_name} run {run_idx+1}/{N_RUNS_PER_CONDITION}...", end=" ", flush=True)
                start = time.time()

                try:
                    if simulated:
                        result = run_single_simulated(
                            challenge_id=state.challenge_id,
                            challenge_desc=ch_desc,
                            experiment="exp2_continuation",
                            condition=cond_name,
                            seed=seed,
                            max_steps=MAX_STEPS,
                            max_time=MAX_TIME,
                            output_dir=str(output_path),
                            prompt_prefix=prefix,
                            model_config=model_config,
                        )
                    else:
                        result = run_single(
                            challenge_id=state.challenge_id,
                            challenge_desc=ch_desc,
                            challenge_dir=ch_dir,
                            experiment="exp2_continuation",
                            condition=cond_name,
                            seed=seed,
                            max_steps=MAX_STEPS,
                            max_time=MAX_TIME,
                            output_dir=str(output_path),
                            prompt_prefix=prefix,
                            model_config=model_config,
                        )

                    elapsed = time.time() - start
                    solved = result.outcome.get("solved", False)
                    print(f"{'SOLVED' if solved else 'FAILED'} ({elapsed:.0f}s)")

                except Exception as e:
                    print(f"ERROR: {e}")


def analyze_and_report(trajectory_dir: str, states_file: str, output_file: str):
    """Analyze Experiment 2 results."""
    from analyze_results import analyze_exp2, format_exp2_table

    results = analyze_exp2(trajectory_dir)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + format_exp2_table(results))
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 2: Continuation Effectiveness")
    parser.add_argument("--exp1-dir", default="../exp1_homogeneity/trajectories",
                        help="Directory with Experiment 1 trajectories")
    parser.add_argument("--challenges", default="../config/challenges.json")
    parser.add_argument("--output-dir", default="../exp2_continuation/trajectories")
    parser.add_argument("--states-file", default="../exp2_continuation/intermediate_states/states.json")
    parser.add_argument("--results-file", default="../exp2_continuation/results.json")
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--model-config", default="../config/models.json")
    args = parser.parse_args()

    load_dotenv()

    with open(args.model_config) as f:
        models = json.load(f)
    model_config = models.get("primary", {})

    if not args.analyze_only:
        # Extract intermediate states
        print("Extracting intermediate states from Experiment 1...")
        states = extract_intermediate_states(args.exp1_dir, n_states=15,
                                              model_config=model_config)

        # Save states
        states_dir = Path(args.states_file).parent
        states_dir.mkdir(parents=True, exist_ok=True)
        with open(args.states_file, "w") as f:
            json.dump([asdict(s) for s in states], f, indent=2, default=str)
        print(f"Extracted {len(states)} intermediate states")

        if not states:
            print("No solved trajectories found in Experiment 1. Cannot run Experiment 2.")
            exit(1)

        # Run experiment
        run_experiment(states, args.output_dir, model_config, args.challenges,
                       simulated=args.simulated)

    # Analyze
    print("\n\nRunning analysis...")
    results = analyze_and_report(args.output_dir, args.states_file, args.results_file)

    go_decision = results.get("go_nogo", {}).get("decision", "UNKNOWN")
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2 DECISION: {go_decision}")
    print(f"{'='*60}")
