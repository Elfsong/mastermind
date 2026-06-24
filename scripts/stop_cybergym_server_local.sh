#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
PID_FILE="${ROOT}/runs/cybergym_server/server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "no CyberGym server pid file"
    exit 0
fi

PID="$(cat "$PID_FILE")"
if sudo kill -0 "$PID" 2>/dev/null; then
    sudo kill "$PID"
    echo "stopped CyberGym server: pid=$PID"
else
    echo "CyberGym server is not running: pid=$PID"
    rm -f "$PID_FILE"
fi
