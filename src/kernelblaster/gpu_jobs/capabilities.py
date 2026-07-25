"""Detect the selected GPU without treating product-name enums as truth."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Callable

from .contracts import (
    GpuCapabilities,
    GpuDeviceCapability,
    GpuRuntimeCapability,
)


CommandRunner = Callable[[list[str]], bytes]


def _default_runner(command: list[str]) -> bytes:
    return subprocess.check_output(command, stderr=subprocess.STDOUT)


def _cuda_version(runner: CommandRunner) -> str:
    try:
        output = runner(["nvcc", "--version"]).decode("utf-8", errors="replace")
        match = re.search(r"release\s+([0-9]+\.[0-9]+)", output)
        return match.group(1) if match else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def detect_gpu_capabilities(
    *,
    runner: CommandRunner = _default_runner,
    environment: dict[str, str] | None = None,
) -> GpuCapabilities:
    environment = environment or os.environ
    logical_id = environment.get("KERNELBLASTER_GPU_DEVICE", "0").strip()
    maximum = int(environment.get("KERNELBLASTER_GPU_MAX_CONCURRENCY", "1"))
    if maximum != 1:
        raise ValueError("PR 04 supports exactly one concurrent GPU job")
    query = runner(
        [
            "nvidia-smi",
            f"--id={logical_id}",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    ).decode("utf-8", errors="replace").strip()
    fields = [field.strip() for field in query.split(",")]
    if len(fields) != 4:
        raise RuntimeError("nvidia-smi did not return name,memory,driver,compute capability")
    name, memory_mib, driver_version, compute_capability = fields
    if not re.fullmatch(r"[0-9]+\.[0-9]+", compute_capability):
        raise RuntimeError("GPU compute capability could not be detected")
    target_arch = "sm_" + compute_capability.replace(".", "")
    configured = {
        value.strip()
        for value in environment.get("KERNELBLASTER_ALLOWED_TARGET_ARCHES", "").split(",")
        if value.strip()
    }
    if configured and target_arch not in configured:
        raise RuntimeError("detected target arch is not in KERNELBLASTER_ALLOWED_TARGET_ARCHES")
    generated = environment.get("KERNELBLASTER_ENABLE_GENERATED_GPU_JOBS", "false").lower()
    return GpuCapabilities(
        supervisor_id=environment.get("KERNELBLASTER_SUPERVISOR_ID", f"local-gpu-{logical_id}"),
        device=GpuDeviceCapability(
            logical_id=logical_id,
            name=name,
            compute_capability=compute_capability,
            target_arch=target_arch,
            total_memory_bytes=int(memory_mib) * 1024 * 1024,
        ),
        runtime=GpuRuntimeCapability(
            cuda_version=_cuda_version(runner),
            driver_version=driver_version,
            image_digest=environment.get("KERNELBLASTER_SUPERVISOR_IMAGE_DIGEST") or None,
        ),
        max_concurrent_jobs=1,
        generated_jobs_enabled=generated in {"1", "true", "yes", "on"},
    )
