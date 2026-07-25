#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed entrypoint executed inside one untrusted GPU Job container.

This program deliberately accepts only Supervisor-created ``/input/request.json``.
Candidate source controls neither the command line nor the environment of the
compiler/test subprocess.  Output is captured under the quota-backed ``/work``
tmpfs and imported by the Supervisor only after strict validation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from typing import Any


INPUT = Path("/input")
WORK = Path("/work")
OUTPUT = WORK / "out"


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _capture(
    command: list[str], *, environment: dict[str, str], stdout_limit: int, stderr_limit: int,
    wall_seconds: int,
) -> tuple[int, bytes, bytes, str | None]:
    """Run a command in its own process group and enforce pipe byte limits."""
    process = subprocess.Popen(
        command,
        cwd=WORK,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    streams = selectors.DefaultSelector()
    streams.register(process.stdout, selectors.EVENT_READ, "stdout")
    streams.register(process.stderr, selectors.EVENT_READ, "stderr")
    stdout = bytearray()
    stderr = bytearray()
    violation: str | None = None
    deadline = time.monotonic() + wall_seconds
    while streams.get_map():
        if time.monotonic() >= deadline:
            violation = "stage_timeout"
            _kill_process_group(process)
            break
        for key, _ in streams.select(timeout=0.1):
            chunk = key.fileobj.read1(64 * 1024)
            if not chunk:
                streams.unregister(key.fileobj)
                continue
            target, limit = (
                (stdout, stdout_limit) if key.data == "stdout" else (stderr, stderr_limit)
            )
            remaining = limit - len(target)
            target.extend(chunk[: max(0, remaining)])
            if len(chunk) > remaining:
                violation = "sandbox_violation"
                _kill_process_group(process)
        if violation is not None:
            break
        if process.poll() is not None and not streams.get_map():
            break
    if violation is not None:
        _kill_process_group(process)
    returncode = process.wait()
    return returncode, bytes(stdout), bytes(stderr), violation


def _fixed_environment(request: dict[str, Any]) -> dict[str, str]:
    target_arch = str(request["target_arch"])
    digits = target_arch.removeprefix("sm_")
    compute = digits[:-1] + "." + digits[-1]
    return {
        "HOME": "/work",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "TMPDIR": "/work/tmp",
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCH_CUDA_ARCH_LIST": compute,
        "CMAKE_CUDA_ARCHITECTURES": digits,
        "CUDAARCHS": digits,
    }


def _command(request: dict[str, Any]) -> list[str]:
    stage = str(request["stage"])
    if stage == "compile":
        sources = sorted(str(path) for path in (INPUT / "candidate").rglob("*.cu"))
        if not sources:
            raise ValueError("candidate source bundle contains no CUDA source")
        driver = INPUT / str(request["driver_path"])
        if not driver.is_file():
            raise ValueError("private driver is unavailable")
        return [
            "nvcc",
            "-O3",
            "-std=c++17",
            f"-arch={request['target_arch']}",
            *sources,
            str(driver),
            "-o",
            str(OUTPUT / "candidate"),
        ]
    executable = INPUT / "candidate" / "candidate"
    if not executable.is_file():
        raise ValueError("candidate executable is unavailable")
    command = [str(executable), "--mode", stage]
    if stage == "events":
        command.extend(("--protocol", str(request["benchmark_protocol_id"])))
    return command


def _stage_reason(stage: str, stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace").lower()
    if "out of memory" in text or "cudaerrormemoryallocation" in text:
        return "gpu_oom"
    return {
        "compile": "compile_failed",
        "correctness": "correctness_failed",
        "events": "events_failed",
    }[stage]


def main() -> int:
    try:
        request = json.loads((INPUT / "request.json").read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        stage = str(request["stage"])
        if stage not in {"compile", "correctness", "events"}:
            raise ValueError("unsupported stage")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (WORK / "tmp").mkdir(parents=True, exist_ok=True)
        returncode, stdout, stderr, violation = _capture(
            _command(request),
            environment=_fixed_environment(request),
            stdout_limit=int(request["stdout_bytes"]),
            stderr_limit=int(request["stderr_bytes"]),
            wall_seconds=int(request["wall_seconds"]),
        )
        (OUTPUT / "stdout.log").write_bytes(stdout)
        (OUTPUT / "stderr.log").write_bytes(stderr)
        reason = violation or ("none" if returncode == 0 else _stage_reason(stage, stderr))
        if reason == "none" and stage == "events":
            try:
                measurement = json.loads(stdout.decode("utf-8"))
                if not isinstance(measurement, dict):
                    raise ValueError("Events output must be an object")
                (OUTPUT / "measurement.json").write_text(
                    json.dumps(measurement, sort_keys=True), encoding="utf-8"
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                reason = "events_failed"
                (OUTPUT / "stderr.log").write_bytes(stderr + f"\n{error}".encode())
        (OUTPUT / "result.json").write_text(
            json.dumps({"reason": reason, "returncode": returncode}, sort_keys=True),
            encoding="utf-8",
        )
        return 0 if reason == "none" else 1
    except Exception as error:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "stdout.log").write_bytes(b"")
        (OUTPUT / "stderr.log").write_text(f"sandbox runner: {error}", encoding="utf-8")
        (OUTPUT / "result.json").write_text(
            json.dumps({"reason": "sandbox_violation"}, sort_keys=True), encoding="utf-8"
        )
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised in Docker integration
    raise SystemExit(main())
