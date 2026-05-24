#!/usr/bin/env bash
set -euo pipefail

ROOT="${MASTERMIND_REPO_ROOT:-/mlx_devbox/users/mz.du/repo/mastermind}"
SOCKET="${MASTERMIND_DOCKER_HOST:-unix:///tmp/mastermind-docker.sock}"
SOCKET_PATH="${SOCKET#unix://}"
DATA_ROOT="${MASTERMIND_DOCKER_DATA_ROOT:-${ROOT}/runs/docker-data}"
EXEC_ROOT="${MASTERMIND_DOCKER_EXEC_ROOT:-${ROOT}/runs/docker-exec}"
PID_FILE="${MASTERMIND_DOCKER_PID_FILE:-${ROOT}/runs/docker/dockerd.pid}"
LOG_FILE="${MASTERMIND_DOCKER_LOG_FILE:-${ROOT}/runs/docker/dockerd.log}"

mkdir -p "$(dirname "$PID_FILE")" "$DATA_ROOT" "$EXEC_ROOT"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "dockerd already running: pid=$(cat "$PID_FILE") socket=$SOCKET"
    exit 0
fi

setsid -f sudo dockerd \
    --host="$SOCKET" \
    --data-root "$DATA_ROOT" \
    --exec-root "$EXEC_ROOT" \
    --pidfile "$PID_FILE" \
    --iptables=false \
    --ip-forward=false \
    --ip-masq=false \
    --bridge=none \
    --storage-driver=vfs \
    > "$LOG_FILE" 2>&1

for _ in $(seq 1 20); do
    if sudo env DOCKER_HOST="$SOCKET" docker info >/dev/null 2>&1; then
        echo "dockerd started: pid=$(cat "$PID_FILE") socket=$SOCKET"
        echo "Use: sudo env DOCKER_HOST=$SOCKET docker info"
        exit 0
    fi
    sleep 0.5
done

echo "dockerd did not become ready; see $LOG_FILE" >&2
exit 1
