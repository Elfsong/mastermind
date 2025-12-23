# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-12-18

import os
import uuid
import argparse
import readline
import subprocess
import importlib.resources

from pathlib import Path
from rich.theme import Theme
from dotenv import load_dotenv
from rich.console import Console


HISTORY_FILE = os.path.expanduser("~/.agent_history")
if os.path.exists(HISTORY_FILE):
    try:
        readline.read_history_file(HISTORY_FILE)
    except OSError:
        pass

DEEP_AGENTS_ASCII = r"""[bold green]
    __  __           _            __  __ _           _ 
    |  \/  |         | |          |  \/  (_)         | |
    | \  / | __ _ ___| |_ ___ _ __| \  / |_ _ __   __| |
    | |\/| |/ _` / __| __/ _ \ '__| |\/| | | '_ \ / _` |
    | |  | | (_| \__ \ ||  __/ |  | |  | | | | | | (_| |
    |_|  |_|\__,_|___/\__\___|_|  |_|  |_|_|_| |_|\__,_|
[/bold green]"""

# Custom theme: define colors for different roles
CUSTOM_THEME = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "tool": "bold green",
    "ai": "bold blue",
})

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

def load_environment_variables(model):
    load_dotenv()
    if model == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            console = Console(theme=CUSTOM_THEME)
            console.print("[bold yellow]GOOGLE_API_KEY is not set.[/bold yellow]")
            api_key = console.input("Please enter your Google API key: ")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key.strip()
            else:
                console.print("[bold red]Google API Key is required for this model.[/bold red]")
                exit(1)
    
    elif model == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            console = Console(theme=CUSTOM_THEME)
            console.print("[bold yellow]OPENAI_API_KEY is not set.[/bold yellow]")
            api_key = console.input("Please enter your OpenAI API key: ")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key.strip()
            else:
                console.print("[bold red]OpenAI API Key is required for this model.[/bold red]")
                exit(1)
                
    elif model == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            console = Console(theme=CUSTOM_THEME)
            console.print("[bold yellow]ANTHROPIC_API_KEY is not set.[/bold yellow]")
            api_key = console.input("Please enter your Anthropic API key: ")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key.strip()
            else:
                console.print("[bold red]Anthropic API Key is required for this model.[/bold red]")
                exit(1)
                
    # Tavily is optional or used by tools    
    if not os.getenv("TAVILY_API_KEY"):
        console = Console(theme=CUSTOM_THEME)
        console.print("[bold yellow]TAVILY_API_KEY is not set.[/bold yellow]")
        api_key = console.input("Please enter your Tavily API key: ")
        if api_key:
            os.environ["TAVILY_API_KEY"] = api_key.strip()
        else:
            console.print("[bold red]Tavily API Key is required for this model.[/bold red]")
            exit(1)