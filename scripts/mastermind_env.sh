#!/usr/bin/env bash
# Local Mastermind/CyberGym environment for the TikTok MM5 workspace.
#
# Source this file before running dual_loops tools:
#   source scripts/mastermind_env.sh

MASTERMIND_REPO_ROOT="${MASTERMIND_REPO_ROOT:-/data00/home/mz.du/Projects/mastermind}"

if [ -f "${MASTERMIND_REPO_ROOT}/.env" ]; then
    set -a
    source "${MASTERMIND_REPO_ROOT}/.env"
    set +a
fi

export MASTERMIND_MANIFEST="${MASTERMIND_MANIFEST:-${MASTERMIND_REPO_ROOT}/mastermind.manifest.json}"

export CYBERGYM_BENCHMARK_ROOT="${CYBERGYM_BENCHMARK_ROOT:-${MASTERMIND_REPO_ROOT}/runs/cybergym_assets/cybergym_data}"
export CYBERGYM_DATA_DIR="${CYBERGYM_DATA_DIR:-${CYBERGYM_BENCHMARK_ROOT}/data}"
export CYBERGYM_TASKS_JSON="${CYBERGYM_TASKS_JSON:-${CYBERGYM_BENCHMARK_ROOT}/tasks.json}"

export CYBERGYM_SERVER_DATA_DIR="${CYBERGYM_SERVER_DATA_DIR:-${MASTERMIND_REPO_ROOT}/runs/cybergym_assets/cybergym-server-data}"
export CYBERGYM_TASKS_FILE="${CYBERGYM_TASKS_FILE:-${MASTERMIND_REPO_ROOT}/cybergym/TASKS_TRAIN}"
export CYBERGYM_TRAIN_ROOT="${CYBERGYM_TRAIN_ROOT:-${MASTERMIND_REPO_ROOT}/runs/dual_loops}"

export CYBERGYM_SERVER="${CYBERGYM_SERVER:-http://127.0.0.1:8666}"
export EXECUTOR_BASE_URL="${EXECUTOR_BASE_URL:-http://localhost:8001/v1}"
export EXECUTOR_MODEL="${EXECUTOR_MODEL:-openai/Qwen/Qwen3.5-27B}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MASTERMIND_REPO_ROOT}/runs/cache}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${MASTERMIND_REPO_ROOT}/runs/local/share}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${XDG_DATA_HOME}/uv/python}"
export UV_PYTHON="${UV_PYTHON:-3.12}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${MASTERMIND_REPO_ROOT}/.venv}"
export CYBERGYM_VENV="${CYBERGYM_VENV:-${MASTERMIND_REPO_ROOT}/.venv}"
export MASTERMIND_DOCKER_HOST="${MASTERMIND_DOCKER_HOST:-unix:///tmp/mastermind-docker.sock}"
