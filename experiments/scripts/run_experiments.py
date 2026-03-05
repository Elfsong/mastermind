#!/usr/bin/env python3
"""
Run all AGE-CTF experiments with real LLM calls in simulated environments.

Optimized for environments without Docker: uses SimulatedEnv with realistic
CTF-like responses, but real Gemini API calls for agent reasoning.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from dataclasses import asdict

from dotenv import load_dotenv

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from run_agent import (
    LLMClient, SimulatedEnv, BaselineAgent, RunResult,
    run_single_simulated, count_tokens,
)
from extract_tools import classify_attack_vector, extract_tool_set
from compute_similarity import (
    analyze_all, pairwise_jaccard, mode_frequency,
    complementary_ratio, cosine_similarity,
)
from embed_trajectories import trajectory_to_text, embed_with_google
from analyze_results import (
    format_exp1_table, analyze_exp2, format_exp2_table,
    decision_matrix,
)

EXPERIMENTS_ROOT = Path(__file__).parent.parent
load_dotenv(EXPERIMENTS_ROOT.parent / ".env")

# Simulated CTF responses keyed by command patterns
SIMULATED_RESPONSES = {
    "nmap": "Starting Nmap 7.94\nPORT   STATE SERVICE VERSION\n22/tcp open  ssh     OpenSSH 8.9\n80/tcp open  http    nginx 1.18.0\n443/tcp open  ssl/https nginx 1.18.0\n3306/tcp open mysql   MySQL 8.0.32\n\nService detection performed.",
    "gobuster": "===============================================================\nGobuster v3.6\n===============================================================\n/login                (Status: 200) [Size: 2341]\n/admin                (Status: 302) [Size: 0]\n/api                  (Status: 200) [Size: 122]\n/uploads              (Status: 403) [Size: 274]\n/robots.txt           (Status: 200) [Size: 68]\n/static               (Status: 301) [Size: 312]\n===============================================================",
    "curl.*login": '<html><body><form action="/login" method="POST"><input name="username"><input name="password" type="password"><button>Login</button></form></body></html>',
    "curl.*admin": "HTTP/1.1 302 Found\nLocation: /login\nSet-Cookie: session=abc123; HttpOnly",
    "curl.*api": '{"status": "ok", "version": "1.0", "endpoints": ["/api/users", "/api/notes", "/api/upload"]}',
    "curl.*robots": "User-agent: *\nDisallow: /admin\nDisallow: /api/internal\nDisallow: /backup",
    "sqlmap": "[*] testing connection to the target URL\n[*] testing if the target URL content is stable\n[*] testing if GET parameter 'id' is dynamic\n[*] heuristic (basic) test shows that GET parameter 'id' might be injectable\n[*] testing for SQL injection on GET parameter 'id'\n[+] GET parameter 'id' is vulnerable. Do you want to keep testing? [y/N]\nType: boolean-based blind\nTitle: AND boolean-based blind\nPayload: id=1 AND 1=1",
    "nikto": "- Nikto v2.5.0\n+ Target IP: 10.0.0.1\n+ Target Port: 80\n+ Server: nginx/1.18.0\n+ /admin: Directory indexing found.\n+ /login: Cookie 'session' created without the httponly flag.\n+ OSVDB-3092: /backup/: This might be interesting.\n+ /api/internal: Potentially interesting endpoint.",
    "strings": "ELF 64-bit LSB executable\nGCC: (Ubuntu 11.4.0) 11.4.0\n.text\n.data\n.rodata\nEnter password:\nAccess granted!\nAccess denied.\nflag{placeholder}\ncheck_password\nmain\n__libc_start_main",
    "file ": "ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, for GNU/Linux 3.2.0, not stripped",
    "checksec": "RELRO           STACK CANARY      NX            PIE             RPATH      RUNPATH\nPartial RELRO   No canary found   NX enabled    PIE enabled     No RPATH   No RUNPATH",
    "gdb": "GNU gdb (GDB) 13.1\nReading symbols from ./binary...\n(No debugging symbols found in ./binary)\n(gdb) disas main\nDump of assembler code for function main:\n   0x0000000000001169 <+0>:     push   rbp\n   0x000000000000116a <+1>:     mov    rbp,rsp\n   0x000000000000116d <+4>:     sub    rsp,0x40\n   0x0000000000001171 <+8>:     lea    rdi,[rip+0xe8c]\n   0x0000000000001178 <+15>:    call   0x1060 <puts@plt>\nEnd of assembler dump.",
    "objdump": "0000000000001169 <main>:\n    1169:       55                      push   %rbp\n    116a:       48 89 e5                mov    %rsp,%rbp\n    116d:       48 83 ec 40             sub    $0x40,%rsp\n    1171:       48 8d 3d 8c 0e 00 00    lea    0xe8c(%rip),%rdi\n    1178:       e8 e3 fe ff ff          call   1060 <puts@plt>",
    "ltrace": 'strcmp("user_input", "s3cr3t_p4ss") = -1\nputs("Access denied.") = 15',
    "binwalk": "DECIMAL       HEXADECIMAL     DESCRIPTION\n0             0x0             ELF, 64-bit LSB executable\n8192          0x2000          Zlib compressed data\n12288         0x3000          PNG image, 640 x 480",
    "openssl": "RSA Public-Key: (2048 bit)\nModulus:\n    00:c4:a3:b2:...\nExponent: 65537 (0x10001)",
    "base64": "ZmxhZ3t0ZXN0X2ZsYWd9",
    "python3": "Python 3.11.0",
    "cat flag": "flag{s1mul4t3d_ctf_fl4g_f0r_t3st1ng}",
    "cat ": "File contents displayed",
    "ls": "challenge.txt\nserver.py\nflag.txt\nREADME.md\nDockerfile",
    "find": "./challenge.txt\n./server.py\n./flag.txt\n./static/style.css",
    "grep": "match found in file",
    "wget": "HTTP request sent, awaiting response... 200 OK\nLength: 4096 (4.0K) [text/html]\nSaving to: 'index.html'",
    "john": "Loaded 1 password hash (Raw-MD5)\nPress 'q' or Ctrl-C to abort\nadmin123         (admin)\n1g 0:00:00:01 DONE",
    "hashcat": "Status...........: Cracked\nHash.Type........: MD5\n$HASH$:password123",
    "nc ": "Connection established\nHTTP/1.1 200 OK\nWelcome to the service",
    "steghide": "steghide: extracting data...\nwrote extracted data to \"secret.txt\".",
    "exiftool": "File Name                       : image.png\nFile Size                       : 42 kB\nComment                         : flag_part1=s3cr3t",
    "xxd": "00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............",
    "hydra": "[80][http-post-form] host: target   login: admin   password: admin123",
    "ffuf": "[Status: 200, Size: 1234, Words: 56, Lines: 32, Duration: 12ms]\n/backup                 [Status: 200, Size: 5678]",
    "dirsearch": "[200]   2KB  /login\n[200]   1KB  /api\n[301]    0B  /admin -> /admin/\n[200]   3KB  /backup/db.sql",
}


def run_agent_with_retries(challenge_id, challenge_desc, experiment, condition,
                           seed, max_steps, output_dir, prompt_prefix="",
                           prompt_suffix="", model_config=None, max_retries=2):
    """Run a single agent with retry logic for API failures."""
    for attempt in range(max_retries + 1):
        try:
            result = run_single_simulated(
                challenge_id=challenge_id,
                challenge_desc=challenge_desc,
                experiment=experiment,
                condition=condition,
                seed=seed,
                max_steps=max_steps,
                max_time=300,
                output_dir=output_dir,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                simulated_responses=SIMULATED_RESPONSES,
                model_config=model_config,
            )
            return result
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f" RETRY in {wait}s ({e})")
                time.sleep(wait)
            else:
                print(f" FAILED after {max_retries+1} attempts: {e}")
                return None


def run_exp1(challenges, model_config, n_runs=8, max_steps=15):
    """Experiment 1: Trajectory Homogeneity Analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Trajectory Homogeneity Analysis")
    print(f"  {len(challenges)} challenges x {n_runs} runs = {len(challenges)*n_runs} total runs")
    print("=" * 70)

    output_dir = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "trajectories")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for ch_idx, ch in enumerate(challenges):
        ch_id = ch["id"]
        ch_desc = ch.get("description", "Solve this CTF challenge and find the flag.")

        print(f"\n[{ch_idx+1}/{len(challenges)}] {ch_id} ({ch.get('category', '?')})")

        for run_idx in range(n_runs):
            seed = 100 + run_idx * 31
            result_file = Path(output_dir) / f"exp1_homogeneity_{ch_id}_s{seed}_baseline.json"
            if result_file.exists():
                print(f"  Run {run_idx+1}/{n_runs}: SKIPPED (exists)")
                continue

            print(f"  Run {run_idx+1}/{n_runs} (seed={seed})...", end=" ", flush=True)
            start = time.time()

            result = run_agent_with_retries(
                challenge_id=ch_id,
                challenge_desc=ch_desc,
                experiment="exp1_homogeneity",
                condition="baseline",
                seed=seed,
                max_steps=max_steps,
                output_dir=output_dir,
                model_config=model_config,
            )

            if result:
                elapsed = time.time() - start
                solved = result.outcome.get("solved", False)
                steps = len(result.trajectory)
                print(f"{'SOLVED' if solved else 'FAILED'} ({steps} steps, {elapsed:.0f}s)")
            else:
                print("ERROR")

    return output_dir


def run_exp2(challenges, model_config, exp1_dir, n_runs=3, max_steps=12):
    """Experiment 2: State-Based Continuation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: State-Based Continuation Effectiveness")
    print("=" * 70)

    output_dir = str(EXPERIMENTS_ROOT / "exp2_continuation" / "trajectories")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract intermediate states from Exp 1
    states = []
    for filepath in sorted(Path(exp1_dir).glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)
        traj = data.get("trajectory", [])
        if len(traj) >= 3:
            mid = len(traj) // 2
            summary_parts = []
            for s in traj[:mid]:
                action_short = s.get("action", "")[:200]
                obs_short = s.get("observation", "")[:150]
                summary_parts.append(f"- Ran: {action_short}\n  Saw: {obs_short}")
            summary = "\n".join(summary_parts)

            states.append({
                "challenge_id": data["challenge_id"],
                "full_history": traj[:mid],
                "summary": summary,
            })

    # Deduplicate by challenge_id, keep up to 15
    seen = set()
    unique_states = []
    for s in states:
        if s["challenge_id"] not in seen:
            seen.add(s["challenge_id"])
            unique_states.append(s)
    states = unique_states[:15]

    # Save states
    states_dir = EXPERIMENTS_ROOT / "exp2_continuation" / "intermediate_states"
    states_dir.mkdir(parents=True, exist_ok=True)
    with open(states_dir / "states.json", "w") as f:
        json.dump(states, f, indent=2, default=str)

    print(f"  Extracted {len(states)} intermediate states")

    # Find challenge descriptions
    ch_descs = {ch["id"]: ch.get("description", "") for ch in challenges}

    conditions = {
        "full_context": lambda s: (
            "Here is your work so far:\n" +
            "\n".join(f"Step {st.get('step','?')}: {st.get('action','')[:300]}\nResult: {st.get('observation','')[:200]}"
                      for st in s["full_history"]) +
            "\nContinue from where you left off.",
            ""
        ),
        "summary_only": lambda s: (
            f"Previous analysis has established:\n{s['summary']}\n\nBuild on these findings.",
            ""
        ),
        "fresh_start": lambda s: ("", ""),
    }

    for state_idx, state in enumerate(states):
        ch_id = state["challenge_id"]
        ch_desc = ch_descs.get(ch_id, "Solve this CTF challenge.")

        print(f"\n[State {state_idx+1}/{len(states)}] {ch_id}")

        for cond_name, prefix_fn in conditions.items():
            prefix, suffix = prefix_fn(state)

            for run_idx in range(n_runs):
                seed = 200 + state_idx * 100 + run_idx * 13
                run_id = f"exp2_continuation_{ch_id}_s{seed}_{cond_name}"
                result_file = Path(output_dir) / f"{run_id}.json"
                if result_file.exists():
                    print(f"  {cond_name} run {run_idx+1}: SKIPPED")
                    continue

                print(f"  {cond_name} run {run_idx+1}/{n_runs}...", end=" ", flush=True)
                start = time.time()

                result = run_agent_with_retries(
                    challenge_id=ch_id,
                    challenge_desc=ch_desc,
                    experiment="exp2_continuation",
                    condition=cond_name,
                    seed=seed,
                    max_steps=max_steps,
                    output_dir=output_dir,
                    prompt_prefix=prefix,
                    prompt_suffix=suffix,
                    model_config=model_config,
                )

                if result:
                    elapsed = time.time() - start
                    solved = result.outcome.get("solved", False)
                    print(f"{'SOLVED' if solved else 'FAILED'} ({elapsed:.0f}s)")
                else:
                    print("ERROR")

    return output_dir


def run_exp3(challenges, model_config, exp1_dir, n_runs=5, max_steps=15):
    """Experiment 3: Diversity Guidance."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Diversity Guidance Effectiveness")
    print("=" * 70)

    output_dir = str(EXPERIMENTS_ROOT / "exp3_guidance" / "trajectories")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Select top 5 challenges (use first 5 from the list)
    exp3_challenges = challenges[:5]

    # Determine dominant approaches per challenge from Exp 1
    dominant_approaches = {}
    for ch in exp3_challenges:
        ch_id = ch["id"]
        trajs = []
        for fp in Path(exp1_dir).glob(f"*{ch_id}*.json"):
            with open(fp) as f:
                data = json.load(f)
            trajs.append(data.get("trajectory", []))
        if trajs:
            vectors = [classify_attack_vector(t) for t in trajs]
            from collections import Counter
            dominant = Counter(vectors).most_common(1)[0][0]
            dominant_approaches[ch_id] = dominant
        else:
            dominant_approaches[ch_id] = "standard reconnaissance"

    alt_suggestions = {
        "web": "Server-Side Template Injection (SSTI) or XML External Entity (XXE) injection",
        "reverse": "Symbolic execution with constraint solving or binary patching",
        "crypto": "Lattice-based attacks or padding oracle exploitation",
    }

    for ch_idx, ch in enumerate(exp3_challenges):
        ch_id = ch["id"]
        ch_desc = ch.get("description", "Solve this CTF challenge.")
        ch_cat = ch.get("category", "web")
        dominant = dominant_approaches.get(ch_id, "standard approach")
        alternative = alt_suggestions.get(ch_cat, "an alternative attack vector")

        print(f"\n[{ch_idx+1}/{len(exp3_challenges)}] {ch_id} (dominant: {dominant})")

        cond_prompts = {
            "no_hint": ("", ""),
            "negative_hint": ("", (
                f"NOTE: Other security analysts are already pursuing: {dominant}. "
                f"To maximize coverage, please explore a DIFFERENT attack vector. "
                f"Think about what other vulnerabilities might exist."
            )),
            "positive_hint": ("", (
                f"Consider exploring: {alternative}. "
                f"This approach has not been extensively tested yet."
            )),
        }

        for cond_name, (prefix, suffix) in cond_prompts.items():
            for run_idx in range(n_runs):
                seed = 300 + ch_idx * 100 + run_idx * 17
                run_id = f"exp3_guidance_{ch_id}_s{seed}_{cond_name}"
                result_file = Path(output_dir) / f"{run_id}.json"
                if result_file.exists():
                    print(f"  {cond_name} run {run_idx+1}: SKIPPED")
                    continue

                print(f"  {cond_name} run {run_idx+1}/{n_runs}...", end=" ", flush=True)
                start = time.time()

                result = run_agent_with_retries(
                    challenge_id=ch_id,
                    challenge_desc=ch_desc,
                    experiment="exp3_guidance",
                    condition=cond_name,
                    seed=seed,
                    max_steps=max_steps,
                    output_dir=output_dir,
                    prompt_prefix=prefix,
                    prompt_suffix=suffix,
                    model_config=model_config,
                )

                if result:
                    elapsed = time.time() - start
                    solved = result.outcome.get("solved", False)
                    print(f"{'SOLVED' if solved else 'FAILED'} ({elapsed:.0f}s)")
                else:
                    print("ERROR")

    return output_dir


def compute_embeddings_and_analyze(exp1_dir):
    """Compute embeddings for Exp 1 trajectories and run similarity analysis."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Computing embeddings and similarity metrics")
    print("=" * 70)

    embeddings_file = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "embeddings" / "embeddings.json")
    Path(embeddings_file).parent.mkdir(parents=True, exist_ok=True)

    # Compute embeddings
    trajectory_files = sorted(Path(exp1_dir).glob("*.json"))
    print(f"Computing embeddings for {len(trajectory_files)} trajectories...")

    texts = []
    run_ids = []
    for fp in trajectory_files:
        with open(fp) as f:
            data = json.load(f)
        run_ids.append(data.get("run_id", fp.stem))
        texts.append(trajectory_to_text(data.get("trajectory", [])))

    try:
        embeddings_list = embed_with_google(texts)
        embeddings_dict = dict(zip(run_ids, embeddings_list))
        with open(embeddings_file, "w") as f:
            json.dump(embeddings_dict, f)
        print(f"Embeddings saved ({len(embeddings_list)} vectors)")
    except Exception as e:
        print(f"Embedding computation failed: {e}")
        embeddings_file = None

    # Run full similarity analysis
    results = analyze_all(exp1_dir, embeddings_file)

    results_file = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + format_exp1_table(results))
    return results


def final_analysis(exp1_results, exp2_dir, exp3_dir):
    """Generate final cross-experiment analysis."""
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    # Exp 1 results already computed
    exp1_go = exp1_results.get("go_nogo", {}).get("decision", "UNKNOWN")

    # Exp 2 analysis
    exp2_results = analyze_exp2(exp2_dir)
    exp2_file = str(EXPERIMENTS_ROOT / "exp2_continuation" / "results.json")
    with open(exp2_file, "w") as f:
        json.dump(exp2_results, f, indent=2)
    print("\n" + format_exp2_table(exp2_results))
    exp2_go = exp2_results.get("go_nogo", {}).get("decision", "UNKNOWN")

    # Exp 3 analysis
    from run_exp3 import analyze_and_report as exp3_analyze
    exp1_dir = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "trajectories")
    exp3_file = str(EXPERIMENTS_ROOT / "exp3_guidance" / "results.json")
    exp3_results = exp3_analyze(exp3_dir, exp1_dir, exp3_file)
    exp3_go = exp3_results.get("go_nogo", {}).get("decision", "UNKNOWN")

    # Decision matrix
    dm = decision_matrix(exp1_go, exp2_go, exp3_go)

    print("\n" + "=" * 70)
    print("DECISION MATRIX")
    print("=" * 70)
    print(f"  Exp 1 (Homogeneity):  {exp1_go}")
    print(f"  Exp 2 (Continuation): {exp2_go}")
    print(f"  Exp 3 (Guidance):     {exp3_go}")
    print(f"\n  DECISION: {dm['decision']}")
    print("=" * 70)

    # Save final report
    report = {
        "exp1": exp1_results,
        "exp2": exp2_results,
        "exp3": exp3_results,
        "decision_matrix": dm,
    }
    report_file = str(EXPERIMENTS_ROOT / "final_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {report_file}")

    return report


if __name__ == "__main__":
    # Load challenges
    with open(EXPERIMENTS_ROOT / "config" / "challenges.json") as f:
        config = json.load(f)
    challenges = config["challenges"]

    with open(EXPERIMENTS_ROOT / "config" / "models.json") as f:
        models = json.load(f)
    model_config = models["primary"]

    print(f"AGE-CTF Experiments")
    print(f"  Challenges: {len(challenges)}")
    print(f"  Model: {model_config['model']}")
    print(f"  API Key: {'SET' if os.getenv('GOOGLE_API_KEY') else 'MISSING'}")

    # Run experiments
    exp1_dir = run_exp1(challenges, model_config, n_runs=8, max_steps=15)
    exp2_dir = run_exp2(challenges, model_config, exp1_dir, n_runs=3, max_steps=12)
    exp3_dir = run_exp3(challenges, model_config, exp1_dir, n_runs=5, max_steps=15)

    # Analyze
    exp1_results = compute_embeddings_and_analyze(exp1_dir)
    report = final_analysis(exp1_results, exp2_dir, exp3_dir)
