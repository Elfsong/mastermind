#!/usr/bin/env python3
"""Claude-named entrypoint for the shared CyberGym parallel orchestrator."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/run_codex_cybergym_tasks_parallel.py"), run_name="__main__")
