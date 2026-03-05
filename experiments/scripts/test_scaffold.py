#!/usr/bin/env python3
"""
Validation test for the AGE-CTF experiment scaffold.

Tests:
  1. Challenge loading from NYU CTF Bench
  2. Agent scaffold produces structured trajectory logs
  3. Tool extraction and classification works
  4. Similarity metrics compute correctly
  5. End-to-end simulated run
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract_tools import (
    extract_command, extract_tool_name, classify_tool,
    classify_attack_vector, extract_tool_set, extract_command_set,
    extract_tool_categories,
)
from compute_similarity import (
    tool_jaccard, pairwise_jaccard, cosine_similarity,
    mode_frequency, complementary_ratio,
)
from run_agent import (
    BaselineAgent, SimulatedEnv, LLMClient, RunResult, count_tokens,
)


class TestTokenCounting(unittest.TestCase):
    def test_count_tokens(self):
        self.assertEqual(count_tokens("hello world"), 2)  # ~11 chars / 4
        self.assertEqual(count_tokens(""), 1)  # minimum 1
        self.assertGreater(count_tokens("a" * 1000), 200)


class TestCommandExtraction(unittest.TestCase):
    def test_command_format(self):
        action = "THOUGHT: Let me scan the target\nCOMMAND: nmap -sV 10.0.0.1"
        self.assertEqual(extract_command(action), "nmap -sV 10.0.0.1")

    def test_code_block_format(self):
        action = "Let me run this:\n```bash\ncurl http://target/login\n```"
        self.assertEqual(extract_command(action), "curl http://target/login")

    def test_no_command(self):
        self.assertIsNone(extract_command("Just thinking..."))

    def test_tool_name_extraction(self):
        self.assertEqual(extract_tool_name("nmap -sV 10.0.0.1"), "nmap")
        self.assertEqual(extract_tool_name("sudo gobuster dir"), "gobuster")
        self.assertEqual(extract_tool_name(""), "unknown")


class TestToolClassification(unittest.TestCase):
    def test_classify_tools(self):
        self.assertEqual(classify_tool("nmap -sV target"), "recon")
        self.assertEqual(classify_tool("sqlmap -u http://target"), "web_exploit")
        self.assertEqual(classify_tool("gdb ./binary"), "binary_analysis")
        self.assertEqual(classify_tool("binwalk firmware.bin"), "forensics")
        self.assertEqual(classify_tool("openssl enc -d"), "crypto")

    def test_attack_vector_classification(self):
        trajectory = [
            {"action": "COMMAND: sqlmap -u http://target/login",
             "observation": "SQL injection vulnerability found"},
            {"action": "COMMAND: sqlmap --dump",
             "observation": "UNION SELECT flag from secrets"},
        ]
        self.assertEqual(classify_attack_vector(trajectory), "sqli")

    def test_tool_set_extraction(self):
        trajectory = [
            {"action": "THOUGHT: scan\nCOMMAND: nmap target", "observation": "open ports"},
            {"action": "THOUGHT: enum\nCOMMAND: gobuster dir -u http://t", "observation": "/admin"},
            {"action": "THOUGHT: exploit\nCOMMAND: sqlmap -u http://t", "observation": "vuln"},
        ]
        tools = extract_tool_set(trajectory)
        self.assertEqual(tools, {"nmap", "gobuster", "sqlmap"})


class TestSimilarityMetrics(unittest.TestCase):
    def test_jaccard_identical(self):
        traj = [
            {"action": "COMMAND: nmap target", "observation": ""},
            {"action": "COMMAND: gobuster dir", "observation": ""},
        ]
        self.assertEqual(tool_jaccard(traj, traj), 1.0)

    def test_jaccard_disjoint(self):
        traj_a = [{"action": "COMMAND: nmap target", "observation": ""}]
        traj_b = [{"action": "COMMAND: sqlmap -u url", "observation": ""}]
        self.assertLess(tool_jaccard(traj_a, traj_b), 1.0)

    def test_jaccard_empty(self):
        self.assertEqual(tool_jaccard([], []), 1.0)

    def test_cosine_similarity(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0)

        c = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, c), 0.0)

    def test_mode_frequency(self):
        # All same attack vector
        trajs = [
            [{"action": "sqlmap", "observation": "SQL injection"}],
            [{"action": "sqlmap", "observation": "SQL injection"}],
            [{"action": "sqlmap", "observation": "SQL injection"}],
        ]
        vec, freq = mode_frequency(trajs)
        self.assertEqual(freq, 1.0)

    def test_complementary_ratio(self):
        # All identical
        traj = [{"action": "COMMAND: nmap target", "observation": ""}]
        ratio = complementary_ratio([traj, traj, traj])
        self.assertEqual(ratio, 1.0)

        # All different
        t1 = [{"action": "COMMAND: nmap target", "observation": ""}]
        t2 = [{"action": "COMMAND: gobuster dir", "observation": ""}]
        t3 = [{"action": "COMMAND: sqlmap -u url", "observation": ""}]
        ratio = complementary_ratio([t1, t2, t3])
        self.assertEqual(ratio, 3.0)


class TestSimulatedEnv(unittest.TestCase):
    def test_simulated_env(self):
        responses = {
            "nmap": "80/tcp open http\n443/tcp open https",
            "cat": "flag{test_flag_123}",
        }
        env = SimulatedEnv(responses=responses)
        result = env.execute("nmap -sV target")
        self.assertIn("80/tcp", result)

        result = env.execute("cat flag.txt")
        self.assertIn("flag{test_flag_123}", result)

        result = env.execute("unknown_cmd")
        self.assertIn("SIMULATED", result)


class TestRunResult(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = RunResult(
                run_id="test_run",
                challenge_id="test_challenge",
                experiment="test",
                condition="baseline",
                model="test-model",
                temperature=1.0,
                seed=42,
                max_steps=10,
                trajectory=[{"step": 0, "action": "test", "observation": "test",
                             "tokens_action": 1, "tokens_observation": 1,
                             "timestamp": 0}],
                outcome={"solved": True, "flag": "flag{test}"},
            )
            path = result.save(tmpdir)
            self.assertTrue(path.exists())

            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["run_id"], "test_run")
            self.assertTrue(loaded["outcome"]["solved"])


class TestChallengeLoading(unittest.TestCase):
    def test_load_nyu_bench(self):
        bench_path = Path(__file__).parent.parent.parent / "NYU_CTF_Bench"
        dataset_file = bench_path / "development_dataset.json"
        if not dataset_file.exists():
            self.skipTest("NYU CTF Bench not available")

        with open(dataset_file) as f:
            dataset = json.load(f)

        self.assertGreater(len(dataset), 0)

        # Check required fields
        for key, entry in list(dataset.items())[:5]:
            self.assertIn("category", entry)
            self.assertIn("path", entry)

        # Count categories
        categories = {}
        for entry in dataset.values():
            cat = entry["category"]
            categories[cat] = categories.get(cat, 0) + 1

        self.assertIn("crypto", categories)
        self.assertIn("web", categories)
        print(f"\nNYU CTF Bench: {len(dataset)} challenges, categories: {categories}")


class TestEndToEnd(unittest.TestCase):
    """Test the full pipeline with simulated components."""

    def test_simulated_run_produces_trajectory(self):
        """Verify scaffold produces structured trajectory logs."""
        env = SimulatedEnv(responses={
            "nmap": "80/tcp open http nginx 1.18.0",
            "gobuster": "/admin /login /api",
            "curl": '<form action="/login" method="POST">',
            "flag": "flag{simulated_test_flag}",
        })

        # Use a mock LLM that cycles through commands
        class MockLLM:
            def __init__(self):
                self.call_count = 0
                self.commands = [
                    "THOUGHT: Scan target\nCOMMAND: nmap -sV target",
                    "THOUGHT: Enumerate web\nCOMMAND: gobuster dir -u http://target",
                    "THOUGHT: Check login\nCOMMAND: curl http://target/login",
                    "THOUGHT: Get flag\nCOMMAND: cat flag.txt",
                ]

            def generate(self, prompt):
                cmd = self.commands[min(self.call_count, len(self.commands) - 1)]
                self.call_count += 1
                return cmd

        agent = BaselineAgent(MockLLM(), env, max_steps=5, max_time=60)
        flag = agent.run("Test challenge: find the flag on the web server")

        # Check trajectory structure
        self.assertGreater(len(agent.trajectory), 0)
        for step in agent.trajectory:
            self.assertIn("step", step)
            self.assertIn("action", step)
            self.assertIn("observation", step)
            self.assertIn("tokens_action", step)
            self.assertIn("tokens_observation", step)
            self.assertIn("timestamp", step)

        # Check flag detection
        self.assertEqual(flag, "flag{simulated_test_flag}")
        print(f"\nSimulated run: {len(agent.trajectory)} steps, flag={'found' if flag else 'not found'}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
