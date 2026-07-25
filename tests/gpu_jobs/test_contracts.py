from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.kernelblaster.gpu_jobs.bundles import build_deterministic_bundle, validate_bundle
from src.kernelblaster.gpu_jobs.contracts import GpuJobManifest


def _manifest(**updates) -> dict:
    payload = {
        "job_id": "job-1",
        "run_id": "run-1",
        "idempotency_key": "candidate-1",
        "stage": "compile",
        "source_bundle_digest": "a" * 64,
        "driver_digest": "b" * 64,
        "target_arch": "sm_86",
        "benchmark_protocol_id": "trusted-smoke-v1",
        "deadline": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    payload.update(updates)
    return payload


def test_manifest_forbids_paths_shell_env_profiler_and_unknown_fields():
    for field, value in (
        ("host_path", "/tmp/source.cu"),
        ("shell", "nvcc source.cu"),
        ("env", {"CUDA_VISIBLE_DEVICES": "0"}),
        ("profiler_argv", ["ncu", "--set", "full"]),
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GpuJobManifest.model_validate(_manifest(**{field: value}))


def test_manifest_validates_digest_arch_deadline_and_stage_inputs():
    manifest = GpuJobManifest.model_validate(_manifest())
    assert len(manifest.canonical_sha256()) == 64
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        GpuJobManifest.model_validate(_manifest(source_bundle_digest="../source"))
    with pytest.raises(ValidationError, match="target_arch"):
        GpuJobManifest.model_validate(_manifest(target_arch="RTX3080"))
    with pytest.raises(ValidationError, match="deadline_exceeded"):
        GpuJobManifest.model_validate(
            _manifest(deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
    with pytest.raises(ValidationError, match="events requires executable_digest"):
        GpuJobManifest.model_validate(
            _manifest(stage="events", source_bundle_digest=None, driver_digest=None)
        )


def test_deterministic_bundle_rejects_path_traversal_and_links():
    first = build_deterministic_bundle({"src/init.cu": b"// source\n", "driver.cpp": b"// driver\n"})
    second = build_deterministic_bundle({"driver.cpp": b"// driver\n", "src/init.cu": b"// source\n"})
    assert first == second
    assert validate_bundle(first) == ("driver.cpp", "src/init.cu")
    with pytest.raises(ValueError, match="safe relative"):
        build_deterministic_bundle({"../escape": b"bad"})
