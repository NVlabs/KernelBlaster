#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def _default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT_DIR / "out" / "portfolio" / "environment" / timestamp / "environment.json"


def _run(command: list[str], timeout: float = 15) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "status": "unavailable",
            "error_type": type(error).__name__,
        }
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[:20_000],
        "stderr": completed.stderr.strip()[:20_000],
    }


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.lower()] = value.strip().strip('"')
    return values


def _git() -> dict[str, Any]:
    commit = _run(["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(ROOT_DIR), "status", "--porcelain"])
    return {
        "commit": commit.get("stdout") if commit["status"] == "ok" else None,
        "dirty": bool(status.get("stdout")) if status["status"] == "ok" else None,
    }


def collect_environment(
    *,
    include_gpu: bool,
    include_docker: bool,
    container_image: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "wsl": "microsoft" in platform.release().lower(),
            "os_release": _os_release(),
        },
        "git": _git(),
        "tools": {
            "python": _run([sys.executable, "--version"]),
            "git": _run(["git", "--version"]),
        },
        "validation": {
            "gpu": "NOT RUN",
            "cuda_compiler": "NOT RUN",
            "ncu": "NOT RUN",
            "docker": "NOT RUN",
        },
    }
    windows_cmd = Path("/mnt/c/Windows/System32/cmd.exe")
    if windows_cmd.is_file():
        payload["platform"]["windows_version"] = _run(
            [str(windows_cmd), "/c", "ver"]
        )

    if include_gpu:
        payload["tools"]["nvidia_smi"] = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        )
        payload["tools"]["nvcc"] = _run(["nvcc", "--version"])
        payload["tools"]["ncu"] = _run(["ncu", "--version"])
        payload["validation"]["gpu"] = payload["tools"]["nvidia_smi"]["status"]
        payload["validation"]["cuda_compiler"] = payload["tools"]["nvcc"][
            "status"
        ]
        payload["validation"]["ncu"] = payload["tools"]["ncu"]["status"]

    if include_docker:
        payload["tools"]["docker"] = _run(
            ["docker", "version", "--format", "{{json .}}"]
        )
        payload["validation"]["docker"] = payload["tools"]["docker"]["status"]
        if container_image:
            payload["container"] = {
                "image": container_image,
                "inspect": _run(
                    [
                        "docker",
                        "image",
                        "inspect",
                        container_image,
                        "--format",
                        "{{json .RepoDigests}} {{.Id}}",
                    ]
                ),
            }
    return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect a sanitized, machine-readable KernelBlaster environment manifest."
    )
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument("--include-docker", action="store_true")
    parser.add_argument("--container-image", default="kernelblaster:validation-25.01")
    args = parser.parse_args()
    payload = collect_environment(
        include_gpu=args.include_gpu,
        include_docker=args.include_docker,
        container_image=args.container_image if args.include_docker else None,
    )
    _atomic_write(args.output.resolve(), payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
