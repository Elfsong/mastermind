#!/usr/bin/env python3
"""
Extract tool-use information from agent trajectory JSON files.

Parses commands from trajectory actions and classifies them into
tool categories for the homogeneity analysis (Experiment 1).
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional


# Tool classification taxonomy
TOOL_CATEGORIES = {
    "recon": ["nmap", "masscan", "ping", "traceroute", "dig", "host", "whois"],
    "web_enum": ["gobuster", "dirb", "dirsearch", "nikto", "wfuzz", "ffuf"],
    "web_exploit": ["sqlmap", "curl", "wget", "hydra", "burpsuite"],
    "binary_analysis": ["gdb", "objdump", "readelf", "checksec", "ltrace", "strace",
                        "radare2", "r2", "ghidra"],
    "binary_exploit": ["pwntools", "python3 -c", "python -c", "rop"],
    "forensics": ["binwalk", "foremost", "steghide", "exiftool", "strings", "file",
                  "xxd", "hexdump"],
    "crypto": ["openssl", "john", "hashcat", "base64", "cyberchef"],
    "network": ["netcat", "nc", "socat", "tcpdump", "wireshark"],
    "general": ["cat", "ls", "find", "grep", "awk", "sed", "head", "tail",
                "chmod", "cd", "pwd", "echo", "python3", "python"],
}

# Attack vector classification
ATTACK_VECTORS = {
    "sqli": ["sqlmap", "sql injection", "' OR ", "UNION SELECT", "-- -"],
    "xss": ["<script>", "alert(", "XSS", "document.cookie"],
    "ssrf": ["ssrf", "server-side request", "internal", "127.0.0.1", "localhost"],
    "lfi": ["../", "file://", "etc/passwd", "local file inclusion", "LFI"],
    "rce": ["reverse shell", "bash -i", "nc -e", "remote code execution"],
    "bof": ["buffer overflow", "segfault", "SIGSEGV", "overflow", "shellcode",
            "pwntools", "EIP", "RIP"],
    "format_string": ["format string", "%x", "%n", "%p"],
    "static_rev": ["strings", "objdump", "ghidra", "disassembl", "decompil"],
    "dynamic_rev": ["gdb", "ltrace", "strace", "breakpoint", "debug"],
    "known_cipher": ["caesar", "vigenere", "RSA", "AES", "DES", "rot13"],
    "frequency_analysis": ["frequency", "letter count", "histogram", "statistical"],
    "brute_force": ["brute", "wordlist", "dictionary", "rockyou", "john", "hashcat"],
    "steganography": ["steghide", "binwalk", "exiftool", "hidden", "stego"],
}


def extract_command(action: str) -> Optional[str]:
    """Extract the bash command from an action string."""
    # Try COMMAND: format
    match = re.search(r'COMMAND:\s*(.+?)(?:\n|$)', action, re.DOTALL)
    if match:
        cmd = match.group(1).strip()
        cmd = re.sub(r'^```\w*\s*', '', cmd)
        cmd = re.sub(r'\s*```$', '', cmd)
        return cmd
    # Try code block
    match = re.search(r'```(?:bash|sh)?\s*\n(.+?)\n```', action, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_tool_name(command: str) -> str:
    """Extract the primary tool name from a command string."""
    if not command:
        return "unknown"
    # Get the first word (the tool/binary)
    parts = command.strip().split()
    if not parts:
        return "unknown"
    tool = parts[0]
    # Handle sudo, env, etc.
    if tool in ("sudo", "env", "timeout", "time"):
        return parts[1] if len(parts) > 1 else tool
    return tool


def classify_tool(command: str) -> str:
    """Classify a command into a tool category."""
    if not command:
        return "unknown"
    cmd_lower = command.lower()
    for category, tools in TOOL_CATEGORIES.items():
        for tool in tools:
            if tool in cmd_lower:
                return category
    return "general"


def classify_attack_vector(trajectory: List[Dict]) -> str:
    """Classify the primary attack vector from a full trajectory."""
    scores = {vec: 0 for vec in ATTACK_VECTORS}
    for step in trajectory:
        text = (step.get("action", "") + " " + step.get("observation", "")).lower()
        for vector, indicators in ATTACK_VECTORS.items():
            for indicator in indicators:
                if indicator.lower() in text:
                    scores[vector] += 1

    if max(scores.values()) == 0:
        return "unknown"
    return max(scores, key=scores.get)


def extract_tool_set(trajectory: List[Dict]) -> Set[str]:
    """Extract the set of unique tools used in a trajectory."""
    tools = set()
    for step in trajectory:
        cmd = extract_command(step.get("action", ""))
        if cmd:
            tools.add(extract_tool_name(cmd))
    return tools


def extract_command_set(trajectory: List[Dict]) -> Set[str]:
    """Extract the set of unique full commands from a trajectory."""
    commands = set()
    for step in trajectory:
        cmd = extract_command(step.get("action", ""))
        if cmd:
            commands.add(cmd)
    return commands


def extract_tool_categories(trajectory: List[Dict]) -> Dict[str, int]:
    """Count tool category usage across a trajectory."""
    categories = {}
    for step in trajectory:
        cmd = extract_command(step.get("action", ""))
        if cmd:
            cat = classify_tool(cmd)
            categories[cat] = categories.get(cat, 0) + 1
    return categories


def process_trajectory_file(filepath: str) -> Dict:
    """Process a single trajectory JSON file and extract tool information."""
    with open(filepath) as f:
        data = json.load(f)

    trajectory = data.get("trajectory", [])
    tools = extract_tool_set(trajectory)
    commands = extract_command_set(trajectory)
    categories = extract_tool_categories(trajectory)
    attack_vector = classify_attack_vector(trajectory)

    return {
        "run_id": data.get("run_id"),
        "challenge_id": data.get("challenge_id"),
        "tools_used": sorted(tools),
        "commands": sorted(commands),
        "tool_categories": categories,
        "attack_vector": attack_vector,
        "num_steps": len(trajectory),
        "solved": data.get("outcome", {}).get("solved", False),
    }


def process_directory(directory: str) -> List[Dict]:
    """Process all trajectory files in a directory."""
    results = []
    for filepath in sorted(Path(directory).glob("*.json")):
        try:
            result = process_trajectory_file(str(filepath))
            results.append(result)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tool-use from trajectories")
    parser.add_argument("--input-dir", required=True, help="Directory with trajectory JSONs")
    parser.add_argument("--output", default="tool_extraction.json")
    args = parser.parse_args()

    results = process_directory(args.input_dir)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Processed {len(results)} trajectories -> {args.output}")
