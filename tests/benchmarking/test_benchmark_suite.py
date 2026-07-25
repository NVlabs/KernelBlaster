from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_suite", ROOT / "scripts" / "benchmark_suite.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_command_is_serial_baseline_only(tmp_path):
    command = MODULE._command(
        task_dir=tmp_path / "task",
        task_id="004",
        kernel="MatVec",
        output_dir=tmp_path / "out",
        warmup=20,
        repetitions=100,
        sessions=3,
        cooldown_seconds=0,
    )
    assert command[1].endswith("benchmark_cuda.py")
    assert "--candidate" not in command
    assert command[command.index("--task-id") + 1] == "004"
    assert command[command.index("--sessions") + 1] == "3"
