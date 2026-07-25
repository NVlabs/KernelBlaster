from __future__ import annotations

from pathlib import Path
import hashlib
import json

from src.kernelblaster.gpu_jobs import build_deterministic_bundle
from src.kernelblaster.gpu_jobs import supervisor


ROOT = Path(__file__).resolve().parents[2]


def test_default_supervisor_does_not_expose_legacy_binary_or_profiler_api():
    paths = {route.path for route in supervisor.APP.routes}
    assert paths >= {
        "/health",
        "/ready",
        "/v1/capabilities",
        "/v1/jobs",
        "/v1/jobs/{job_id}",
        "/v1/jobs/{job_id}/cancel",
    }
    assert "/gpu/binary" not in paths


def test_runtime_arch_is_not_hardcoded_in_compose_or_shared_docker_stage():
    deployment = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
    for fixed in (
        'TORCH_CUDA_ARCH_LIST: "8.6"',
        'CMAKE_CUDA_ARCHITECTURES: "86"',
        'CUDAARCHS: "86"',
        "TORCH_CUDA_ARCH_LIST=8.6",
        "CMAKE_CUDA_ARCHITECTURES=86",
        "CUDAARCHS=86",
    ):
        assert fixed not in deployment
        assert fixed not in dockerfile


def test_control_image_stage_contains_no_cuda_toolchain_installation():
    dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
    control_stage = dockerfile.split("FROM ${GPU_BASE_IMAGE}", 1)[0]
    assert "nvcc" not in control_stage
    assert "CUDA_HOME" not in control_stage


def test_reviewed_vector_add_bundle_matches_allowlist_digests():
    smoke = ROOT / "portfolio" / "trusted_gpu_smoke"
    bundle = build_deterministic_bundle(
        {"vector_add.cu": (smoke / "vector_add.cu").read_bytes()}
    )
    driver = (smoke / "driver.cpp").read_bytes()
    manifest = json.loads(
        (ROOT / "portfolio" / "trusted-gpu-bundles.json").read_text(encoding="utf-8")
    )
    entry = manifest["bundles"][0]
    assert hashlib.sha256(bundle).hexdigest() == entry["source_bundle_digest"]
    assert hashlib.sha256(driver).hexdigest() == entry["driver_digest"]
    assert entry["source_bundle_digest"] in manifest["source_bundle_digests"]
