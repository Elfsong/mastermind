# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-12-18

import os
import subprocess
from tavily import TavilyClient
from langchain.tools import tool
from typing import Dict, Any, Literal


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