# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-12-18

import os
import uuid
import argparse
import readline
import subprocess
from pathlib import Path

def get_system_prompt(agent_name: str):
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{agent_name}.md"
    return prompt_path.read_text()