# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-12-18

import os
import uuid
import argparse
import readline
import subprocess
from pathlib import Path

import importlib.resources

def get_system_prompt(agent_name: str):
    try:
        # Try finding the file relative to this file's location
        # This works for both local dev (if structure is preserved) and installed package
        # if package_data is correctly configured.
        prompt_path = Path(__file__).parent / "prompts" / f"{agent_name}.md"
        if prompt_path.exists():
            return prompt_path.read_text()
            
        raise FileNotFoundError(f"Could not find prompt file at {prompt_path}")

    except Exception as e:
        raise FileNotFoundError(f"Error loading prompt: {e}")