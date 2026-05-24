#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/mlx_devbox/users/mz.du/repo/mastermind}"
PID_FILE="${MASTERMIND_DOCKER_PID_FILE:-${ROOT}/runs/docker/dockerd.pid}"

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
