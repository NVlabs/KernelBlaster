from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_writes_parseable_artifacts_without_external_calls(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("nvidia-smi", "docker"):
        executable = fake_bin / name
        executable.write_text("#!/bin/sh\nexit 97\n")
        executable.chmod(0o755)

    output_dir = tmp_path / "dry-run"
    environment = os.environ.copy()
    for name in (
        "OPENAI_API_KEY",
        "KERNELBLASTER_LLM_API_KEY",
        "KERNELBLASTER_LLM_BASE_URL",
    ):
        environment.pop(name, None)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_portfolio.py"),
            "--suite",
            "core10",
            "--model",
            "gpt-5.6-terra",
            "--gpu",
            "rtx3080",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    events = [
        json.loads(line)
        for line in (output_dir / "events.jsonl").read_text().splitlines()
    ]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert manifest["schema_version"] == "3.0"
    assert manifest["validation"]["cuda"] == "NOT RUN"
    assert manifest["validation"]["llm_smoke_test"] == "NOT RUN"
    assert events[0]["data"]["network_calls"] == 0
    assert events[0]["data"]["cuda_calls"] == 0
    assert summary["status"] == "dry_run"
