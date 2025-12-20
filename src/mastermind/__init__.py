# coding: utf-8
import os
import uuid
import argparse
import readline
import subprocess
from pathlib import Path

from rich.live import Live
from rich.theme import Theme
from rich.panel import Panel
from rich.console import Console
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from dotenv import load_dotenv
from .utils import get_system_prompt
from .tools import web_search, shell_command

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, ToolMessage, AIMessage, AIMessageChunk


HISTORY_FILE = os.path.expanduser("~/.agent_history")
if os.path.exists(HISTORY_FILE):
    readline.read_history_file(HISTORY_FILE)

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

def run_interactive_agent(agent, config):
    console = Console(theme=CUSTOM_THEME)
    console.clear()
    system_info = "[dim][+] Offensive Attack Mode Enabled[/dim]\n[dim][+] Unattended Mode Enabled[/dim]\n[bold yellow][+] Command+C to Exit[/bold yellow]"
    console.print(
        Panel(
            DEEP_AGENTS_ASCII + "\n" + system_info, 
            title="[bold cyan]Mastermind[/bold cyan]", 
            border_style="cyan", 
            expand=False
        )
    )

    while True:
        # Get user input
        try:
            console.print()
            safe_prompt = "\001\033[1;32m\002➜\001\033[0m\002 "
            user_input = input(safe_prompt)
            readline.write_history_file(HISTORY_FILE)
            user_input = user_input.strip()
        except EOFError:
            console.print("\n[bold red]EOF Error[/bold red]")
            break
        except KeyboardInterrupt:
            console.print("\n[bold green]Goodbye![/bold green]")
            break
        
        # Exit commands
        if user_input.lower() in ["exit", "quit"]: break

        # Shell commands
        if user_input.startswith("!"):
            cmd = user_input[1:].strip()
            if not cmd:
                continue
                
            # Special handling for cd command
            if cmd.startswith("cd "):
                try:
                    target_dir = cmd[3:].strip()
                    # Handle ~ and other path expansion
                    target_dir = os.path.expanduser(target_dir)
                    os.chdir(target_dir)
                    console.print(f"[dim cyan]Changed directory to: {os.getcwd()}[/dim cyan]")
                except Exception as e:
                    console.print(f"[bold red]Error changing directory:[/bold red] {e}")
            else:
                try:
                    # Execute other Shell commands directly to terminal, support pipes and wildcards
                    subprocess.run(cmd, shell=True)
                except Exception as e:
                    console.print(f"[bold red]Execution Error:[/bold red] {e}")
                except KeyboardInterrupt:
                    console.print("\n[bold yellow]Shell Command Interrupted[/bold yellow]")
                    continue
            
            # Skip this loop, don't send to AI
            continue

        input_data = {"messages": [HumanMessage(content=user_input)]}
        full_msg_content = ""
        last_ai_message = None

        # Initial display
        initial_display = Spinner("dots", text="Thinking...\n", style="bold blue")

        # Live display
        try:
            with Live(initial_display, console=console, vertical_overflow="visible", refresh_per_second=10) as live:
                for mode, data in agent.stream(input_data, config, stream_mode=["messages", "updates"]):
                    
                    if mode == "messages":
                        msg, _ = data
                        if isinstance(msg, AIMessageChunk) and msg.content:
                            chunk_content = msg.content
                            if isinstance(chunk_content, list):
                                for part in chunk_content:
                                    if isinstance(part, str):
                                        full_msg_content += part
                                    elif isinstance(part, dict) and "text" in part:
                                        full_msg_content += part["text"]
                            elif isinstance(chunk_content, str):
                                full_msg_content += chunk_content

                            # Update Live display to Markdown
                            live.update(Panel(Markdown(full_msg_content), title="[ai]Mastermind[/ai]", border_style="blue", expand=False))

                    elif mode == "updates":
                        # Tool output processing logic remains the same
                        # Note: Direct console.print here will print above the Live component, which is allowed by Rich
                        for node_name, output in data.items():
                            if not output or "messages" not in output: continue
                            last_node_msg = output["messages"][-1]
                            
                            if isinstance(last_node_msg, AIMessage):
                                last_ai_message = last_node_msg
                            
                            if isinstance(last_node_msg, ToolMessage):
                                args = "N/A"
                                if last_ai_message and hasattr(last_ai_message, "tool_calls"):
                                    for tool_call in last_ai_message.tool_calls:
                                        if tool_call["id"] == last_node_msg.tool_call_id:
                                            args = tool_call["args"]
                                            break
                                
                                live.console.print(Panel(
                                    f"[bold yellow]Tool:[/bold yellow] {last_node_msg.name}\n"
                                    f"[bold yellow]Args:[/bold yellow] {args}\n"
                                    f"[bold yellow]Result:[/bold yellow] [dim]{str(last_node_msg.content)[:300]}...[/dim]", 
                                    title="Action", 
                                    border_style="yellow"
                                ))
                                full_msg_content = ""
                                live.update(Spinner("dots", text="Next step...\n", style="bold blue"))
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Task Cancelled[/bold yellow]")
            continue

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

def main():
    # args: --model gpt-4o-mini or --model gemini-3-pro-preview
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google", choices=["openai", "google", "anthropic"])
    args = parser.parse_args()

    load_environment_variables(args.model)

    if args.model == "openai":
        backend = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    elif args.model == "google":
        backend = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", streaming=True)
    elif args.model == "anthropic":
        backend = ChatAnthropic(model="claude-sonnet-4-5-20250929", streaming=True)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    mastermind = create_agent(
        model=backend, 
        name="mastermind",
        system_prompt=get_system_prompt("mastermind"), 
        tools=[web_search, shell_command],
        middleware=[SummarizationMiddleware(model=backend, trigger=("fraction", 0.85), keep=("messages", 6))],
    )

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    run_interactive_agent(mastermind, config)

if __name__ == "__main__":
    main()