#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/mlx_devbox/users/mz.du/repo/mastermind}"
source "${ROOT}/scripts/mastermind_env.sh"

DOCKER_HOST_VALUE="${MASTERMIND_DOCKER_HOST:-unix:///tmp/mastermind-docker.sock}"
RUN_DIR="${ROOT}/runs/cybergym_server"
PID_FILE="${RUN_DIR}/server.pid"
LOG_FILE="${RUN_DIR}/server.log"

mkdir -p "$RUN_DIR"
export DOCKER_HOST="$DOCKER_HOST_VALUE"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "CyberGym server already running: pid=$(cat "$PID_FILE")"
    exit 0
fi

cd "${ROOT}/cybergym"

setsid -f sudo --preserve-env=DOCKER_HOST,CYBERGYM_API_KEY \
    "$CYBERGYM_VENV/bin/python" -m cybergym.server \
    --host 0.0.0.0 \
    --port 8666 \
    --log_dir "${RUN_DIR}/logs" \
    --db_path "${RUN_DIR}/poc.db" \
    --binary_dir "$CYBERGYM_SERVER_DATA_DIR" \
    > "$LOG_FILE" 2>&1

sleep 1
ps -eo pid,args \
    | awk '$0 ~ /\.venv\/bin\/python -m cybergym\.server/ && $0 !~ /sudo/ && $0 !~ /awk/ {print $1; exit}' \
    > "$PID_FILE"
echo "CyberGym server started: pid=$(cat "$PID_FILE") url=http://127.0.0.1:8666"
