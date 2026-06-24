#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"
PID_FILE="${MASTERMIND_DOCKER_PID_FILE:-${ROOT}/runs/docker/dockerd.pid}"

# Also kill any stale dockerd processes attached to our socket, in case PID file is stale
pkill -f "dockerd --host=unix:///tmp/mastermind-docker.sock" 2>/dev/null || true

if [ ! -f "$PID_FILE" ]; then
    echo "no dockerd pid file"
    exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
    sudo kill "$PID"
    echo "stopped dockerd: pid=$PID"
else
    echo "dockerd is not running: pid=$PID"
fi
