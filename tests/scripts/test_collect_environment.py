from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "collect_environment", ROOT / "scripts" / "collect_environment.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cpu_collection_does_not_probe_gpu_or_docker(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, timeout=15):
        commands.append(command)
        return {"status": "ok", "returncode": 0, "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(MODULE, "_run", fake_run)
    payload = MODULE.collect_environment(
        include_gpu=False,
        include_docker=False,
        container_image=None,
    )
    flattened = " ".join(part for command in commands for part in command)
    assert "nvidia-smi" not in flattened
    assert "nvcc" not in flattened
    assert "ncu" not in flattened
    assert "docker" not in flattened
    assert payload["validation"]["gpu"] == "NOT RUN"
