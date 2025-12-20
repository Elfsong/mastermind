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
        # Try loading from the package using importlib.resources (Python 3.9+)
        # This assumes 'mastermind' is the package name and prompts are included in it
        # However, since prompts/ is outside src/ in the current repo structure, 
        # we need to ensure they are packaged correctly.
        # If prompts are moved inside src/mastermind/prompts, this would be:
        # return importlib.resources.files("mastermind.prompts").joinpath(f"{agent_name}.md").read_text()
        
        # Fallback to current relative path approach if running locally/editable
        prompt_path = Path(__file__).parent.parent / "prompts" / f"{agent_name}.md"
        if prompt_path.exists():
            return prompt_path.read_text()
            
        # If installed as a package, 'prompts' might not be a direct sibling of the site-packages/mastermind.py 
        # depending on how it was installed.
        # Let's try to find it relative to the module location if packaged with MANIFEST.in or package_data
        
        # Alternative: check if prompts are adjacent to the module file (if we move them there)
        prompt_path = Path(__file__).parent / "prompts" / f"{agent_name}.md"
        if prompt_path.exists():
            return prompt_path.read_text()

        raise FileNotFoundError(f"Could not find prompt file for {agent_name}")

    except Exception as e:
        raise FileNotFoundError(f"Error loading prompt: {e}")