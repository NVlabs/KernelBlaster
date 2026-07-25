#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from typing import Any


def _command_version(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error_type": type(error).__name__}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "version_output": (completed.stdout + completed.stderr).strip()[:2000],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report and optionally enforce the container CUDA toolchain contract."
    )
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "nvcc": _command_version(["nvcc", "--version"]),
        "ncu": _command_version(["ncu", "--version"]),
        "nvidia_smi": _command_version(["nvidia-smi"]),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        report["torch"] = None
        report["torch_cuda"] = None
        report["cuda_available"] = False
        failures.append("PyTorch is not installed.")

    expectations = {
        "python": os.getenv("KERNELBLASTER_EXPECT_PYTHON_VERSION"),
        "torch": os.getenv("KERNELBLASTER_EXPECT_TORCH_VERSION"),
        "torch_cuda": os.getenv("KERNELBLASTER_EXPECT_CUDA_VERSION"),
    }
    for key, expected in expectations.items():
        if expected and str(report.get(key)) != expected:
            failures.append(
                f"{key} mismatch: expected {expected}, observed {report.get(key)}"
            )
    if args.require_gpu and not report.get("cuda_available"):
        failures.append("CUDA is not available to PyTorch.")
    if args.require_gpu and not report["nvcc"]["available"]:
        failures.append("nvcc is unavailable.")

    report["status"] = "failed" if failures else "ok"
    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
