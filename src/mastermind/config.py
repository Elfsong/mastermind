"""Repo-level Mastermind manifest loading and validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "mastermind.manifest.json"
CYBERGYM_API_KEY_ENV = "CYBERGYM_API_KEY"


@dataclass(frozen=True)
class BenchmarkPaths:
    root: Path
    data_dir: Path
    tasks_json: Path


@dataclass(frozen=True)
class ServerDataPaths:
    root: Path
    arvo_dir: Path
    oss_fuzz_dir: Path


@dataclass(frozen=True)
class ServiceConfig:
    url: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    local_vllm_allows_empty_key: bool = False


@dataclass(frozen=True)
class OutputPaths:
    run_root: Path
    rollout_store: Path


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    name: str
    benchmark: BenchmarkPaths
    server_data: ServerDataPaths
    task_splits: dict[str, Path]
    services: dict[str, ServiceConfig]
    outputs: OutputPaths
    path: Path

    @property
    def cybergym_server_url(self) -> str:
        server = self.services.get("cybergym_server")
        return server.url if server and server.url else ""

    @property
    def executor_base_url(self) -> str:
        executor = self.services.get("executor")
        return executor.base_url if executor and executor.base_url else ""


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return data


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_file(path: Path, *, override: bool = False) -> list[str]:
    """Load a simple KEY=VALUE .env file without printing secret values."""
    loaded: list[str] = []
    if not path.exists():
        return loaded
    with path.open() as f:
        for line in f:
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def load_manifest(path: str | Path | None = None) -> Manifest:
    """Load the repo-level manifest.

    `MASTERMIND_MANIFEST` can override the default manifest path for local
    experiments without changing the checked-in config.
    """
    manifest_path = Path(
        path or os.getenv("MASTERMIND_MANIFEST") or DEFAULT_MANIFEST_PATH
    ).resolve()
    load_env_file(manifest_path.parent / ".env")
    data = _load_json(manifest_path)
    base_dir = manifest_path.parent

    benchmark = data.get("benchmark", {})
    server_data = data.get("server_data", {})
    outputs = data.get("outputs", {})

    services = {
        name: ServiceConfig(**value)
        for name, value in data.get("services", {}).items()
    }

    return Manifest(
        schema_version=int(data.get("schema_version", 1)),
        name=str(data.get("name", manifest_path.stem)),
        benchmark=BenchmarkPaths(
            root=_resolve_path(benchmark["root"], base_dir),
            data_dir=_resolve_path(benchmark["data_dir"], base_dir),
            tasks_json=_resolve_path(benchmark["tasks_json"], base_dir),
        ),
        server_data=ServerDataPaths(
            root=_resolve_path(server_data["root"], base_dir),
            arvo_dir=_resolve_path(server_data["arvo_dir"], base_dir),
            oss_fuzz_dir=_resolve_path(server_data["oss_fuzz_dir"], base_dir),
        ),
        task_splits={
            name: _resolve_path(path_value, base_dir)
            for name, path_value in data.get("task_splits", {}).items()
        },
        services=services,
        outputs=OutputPaths(
            run_root=_resolve_path(outputs.get("run_root", "runs"), base_dir),
            rollout_store=_resolve_path(
                outputs.get("rollout_store", "runs/rollouts.jsonl"), base_dir
            ),
        ),
        path=manifest_path,
    )


def check_manifest(manifest: Manifest) -> list[CheckResult]:
    """Return path/env checks without performing network calls."""
    checks: list[CheckResult] = []
    paths = {
        "benchmark.root": manifest.benchmark.root,
        "benchmark.data_dir": manifest.benchmark.data_dir,
        "benchmark.tasks_json": manifest.benchmark.tasks_json,
        "server_data.root": manifest.server_data.root,
        "server_data.arvo_dir": manifest.server_data.arvo_dir,
        "server_data.oss_fuzz_dir": manifest.server_data.oss_fuzz_dir,
        **{f"task_splits.{name}": path for name, path in manifest.task_splits.items()},
    }
    for name, path in paths.items():
        checks.append(CheckResult(name=name, ok=path.exists(), detail=str(path)))

    for service_name, service in manifest.services.items():
        env_name = service.api_key_env
        if not env_name:
            continue
        optional_empty_local_key = (
            service_name == "executor" and service.local_vllm_allows_empty_key
        )
        value_set = bool(os.getenv(env_name))
        code_default_available = (
            service_name == "cybergym_server"
            and env_name == CYBERGYM_API_KEY_ENV
        )
        checks.append(
            CheckResult(
                name=f"env.{env_name}",
                ok=optional_empty_local_key or value_set or code_default_available,
                detail=(
                    "set"
                    if value_set
                    else "optional for local vLLM"
                    if optional_empty_local_key
                    else "not set; code fallback present"
                    if code_default_available
                    else "not set"
                ),
            )
        )
    return checks
