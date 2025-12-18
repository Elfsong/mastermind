# coding: utf-8
import os
import uuid
import readline
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Literal

from rich.live import Live
from rich.theme import Theme
from rich.panel import Panel
from rich.console import Console
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, ToolMessage, AIMessage, AIMessageChunk


HISTORY_FILE = os.path.expanduser("~/.agent_history")
if os.path.exists(HISTORY_FILE):
    readline.read_history_file(HISTORY_FILE)

# 自定义主题：定义不同角色的颜色
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "tool": "bold green",
    "ai": "bold blue",
})

console = Console(theme=custom_theme)
load_dotenv()

@tool
def web_search(query: str, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = False) -> str:
    """Search for information."""
    search_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return search_client.search(query, max_results=max_results, topic=topic, include_raw_content=include_raw_content)

@tool
def shell_command(command: str, timeout: int = 30, cwd: str = None) -> Dict[str, Any]:
    """Execute a shell command."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}

def get_system_prompt(agent_name: str):
    prompt_path = Path(__file__).parent / "prompts" / f"{agent_name}.md"
    return prompt_path.read_text()


backend = ChatOpenAI(model="gpt-4o-mini", streaming=True)
# backend = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", streaming=True)

mastermind = create_agent(
    model=backend, 
    name="mastermind",
    system_prompt=get_system_prompt("mastermind"), 
    tools=[web_search, shell_command],
    middleware=[SummarizationMiddleware(model=backend, trigger=("fraction", 0.85), keep=("messages", 6))],
    # checkpointer=InMemorySaver(),
)

def run_interactive_agent():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    console.print(Panel.fit("[bold cyan]Mastermind Agent[/bold cyan]\n[dim]Auto-execution mode enabled[/dim]", border_style="cyan"))

    while True:
        try:
            console.print()
            safe_prompt = "\001\033[1;32m\002➜\001\033[0m\002 "
            user_input = input(safe_prompt)
            readline.write_history_file(HISTORY_FILE)
            user_input = user_input.strip()
        except EOFError:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            break

        if user_input.lower() in ["exit", "quit"]: break

        input_data = {"messages": [HumanMessage(content=user_input)]}
        full_msg_content = ""
        last_ai_message = None

        # --- 修改开始 ---
        # 1. 创建一个初始的 "Thinking..." 状态
        #    这样在 Agent 进行搜索或思考还未输出 Token 时，用户能看到反馈
        initial_display = Spinner("dots", text="Thinking...", style="bold blue")

        # 2. 移除外层的 console.status，只使用 Live
        with Live(initial_display, vertical_overflow="visible", refresh_per_second=10) as live:
            for mode, data in mastermind.stream(input_data, config, stream_mode=["messages", "updates"]):
                
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

                        # 3. 收到内容后，更新 Live 的显示对象为 Markdown
                        live.update(Panel(Markdown(full_msg_content), title="[ai]Mastermind[/ai]", border_style="blue", expand=False))

                elif mode == "updates":
                    # Tool 输出的处理逻辑保持不变
                    # 注意：这里直接 console.print 会在 Live 组件上方打印，这是 Rich 允许的
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
                            
                            live.update(Text(""))
                            live.refresh()

                            console.print(Panel(
                                f"[bold yellow]Tool:[/bold yellow] {last_node_msg.name}\n"
                                f"[bold yellow]Args:[/bold yellow] {args}\n"
                                f"[bold yellow]Result:[/bold yellow] [dim]{str(last_node_msg.content)[:300]}...[/dim]", 
                                title="Action", 
                                border_style="yellow"
                            ))

if __name__ == "__main__":
    run_interactive_agent()