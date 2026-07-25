from __future__ import annotations

import pytest

from src.kernelblaster.gpu_jobs.capabilities import detect_gpu_capabilities


def _runner(command: list[str]) -> bytes:
    if command[0] == "nvidia-smi":
        return b"Unlisted Future GPU, 24576, 600.12, 8.9\n"
    if command[:2] == ["nvcc", "--version"]:
        return b"Cuda compilation tools, release 12.8, V12.8.61\n"
    raise AssertionError(command)


def test_capability_uses_detected_compute_capability_not_product_enum():
    capability = detect_gpu_capabilities(
        runner=_runner,
        environment={
            "KERNELBLASTER_GPU_DEVICE": "0",
            "KERNELBLASTER_GPU_MAX_CONCURRENCY": "1",
        },
    )
    assert capability.device.name == "Unlisted Future GPU"
    assert capability.device.target_arch == "sm_89"
    assert capability.runtime.cuda_version == "12.8"


@pytest.mark.parametrize(
    ("compute_capability", "target_arch"),
    (("8.0", "sm_80"), ("8.6", "sm_86"), ("8.9", "sm_89"), ("9.0", "sm_90")),
)
def test_protocol_is_stable_across_supported_hardware(compute_capability, target_arch):
    def runner(command: list[str]) -> bytes:
        if command[0] == "nvidia-smi":
            return f"Any GPU, 1024, test-driver, {compute_capability}\n".encode()
        return b"Cuda compilation tools, release 12.8, V12.8.0\n"

    capability = detect_gpu_capabilities(runner=runner, environment={})
    assert capability.schema_version == "gpu-capabilities/v1"
    assert capability.device.target_arch == target_arch


def test_capability_rejects_arch_allowlist_mismatch_and_multi_gpu_concurrency():
    with pytest.raises(RuntimeError, match="not in"):
        detect_gpu_capabilities(
            runner=_runner,
            environment={"KERNELBLASTER_ALLOWED_TARGET_ARCHES": "sm_86"},
        )
    with pytest.raises(ValueError, match="exactly one"):
        detect_gpu_capabilities(
            runner=_runner,
            environment={"KERNELBLASTER_GPU_MAX_CONCURRENCY": "2"},
        )
