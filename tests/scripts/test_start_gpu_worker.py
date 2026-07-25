from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "start_gpu_worker", ROOT / "scripts" / "start_gpu_worker.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_worker_startup_uses_fixed_argv_for_probe_and_server():
    assert MODULE.runtime_check_command() == [
        sys.executable,
        str(ROOT / "scripts" / "check_runtime_versions.py"),
        "--require-gpu",
    ]
    assert MODULE.gpu_server_command(["--host", "127.0.0.1"]) == [
        sys.executable,
        "-m",
        "src.kernelblaster.servers.gpu",
        "--host",
        "127.0.0.1",
    ]
