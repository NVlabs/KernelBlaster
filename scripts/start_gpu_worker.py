#!/usr/bin/env python3
"""Validate the GPU runtime, then replace this process with the worker server."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from src.kernelblaster.servers.auth import validate_worker_token
from src.kernelblaster.servers.security import validate_worker_environment


ROOT_DIR = Path(__file__).resolve().parent.parent


def runtime_check_command() -> list[str]:
    return [
        sys.executable,
        str(ROOT_DIR / "scripts" / "check_runtime_versions.py"),
        "--require-gpu",
    ]


def gpu_server_command(arguments: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.kernelblaster.servers.gpu",
        *arguments,
    ]


def main() -> int:
    validate_worker_environment()
    validate_worker_token()
    subprocess.run(runtime_check_command(), cwd=ROOT_DIR, check=True)
    command = gpu_server_command(sys.argv[1:])
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
