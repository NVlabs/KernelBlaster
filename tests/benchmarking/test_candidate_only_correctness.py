# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CUDA = _load("benchmark_cuda_candidate_only", "benchmark_cuda.py")
CANDIDATES = _load("benchmark_candidates_candidate_only", "benchmark_candidates.py")


RUNTIME_CONTRACT = {
    "device": "cuda",
    "gpu_architectures": ["sm_86"],
    "input_dtype": "fp16",
    "accumulation_dtype": "fp32",
    "layout": "contiguous_row_major",
    "stream_mode": "legacy_default",
    "max_streams": 1,
    "graph_capture": False,
    "directions": ["inference_forward"],
    "backward": False,
    "fallback": "none",
    "production_ready": False,
}

OFFICIAL_DRIVER = """
#include <iostream>
void launch_gpu_implementation(void*, void*, void*, int64_t, int64_t, int64_t);
int main() {
    void* value = nullptr;
    launch_gpu_implementation(value, value, value, 1, 1, 1);
    std::cout << "passed" << std::endl;
    return 0;
}
"""

CUDA_SOURCE = """
void launch_gpu_implementation(
    void*, void*, void*, int64_t, int64_t, int64_t
) {
    cudaDeviceSynchronize();
}
"""

CANDIDATE_ONLY_DRIVER = """
#include "correctness_metrics.h"
void launch_gpu_implementation(void*, void*, void*, int64_t, int64_t, int64_t);
int main() {
    void* value = nullptr;
    launch_gpu_implementation(value, value, value, 1, 1, 1);
    return 0;
}
"""


def _case(seed: int) -> dict[str, object]:
    return {
        "case_id": "canonical",
        "seed": seed,
        "shape": {"M": 1, "N": 1, "K": 1},
        "deterministic": True,
        "count": 1,
        "quantile_sample_count": 1,
        "quantile_sampling": "deterministic_stride",
        "quantile_max_samples": 1048576,
        "mismatch_count": 0,
        "nonfinite_count": 0,
        "normalized_max": 0.0,
    }


def _distribution() -> dict[str, object]:
    metrics: dict[str, object] = {
        "max_abs_error": 0.0,
        "p99_abs_error": 0.0,
        "finite": True,
        "deterministic": True,
        "count": 3,
        "quantile_sample_count": 3,
        "quantile_sampling": "deterministic_stride",
        "quantile_max_samples": 1048576,
        "aggregate_quantile_semantics": "max_per_case_quantile_envelope",
        "mismatch_count": 0,
        "nonfinite_count": 0,
        "cases": [_case(seed) for seed in (0, 42, 20260721)],
    }
    for name in CUDA.CORRECTNESS_DISTRIBUTION_METRICS:
        metrics[name] = 0.0
    return metrics


def _write_inner_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "driver.cpp").write_text(OFFICIAL_DRIVER, encoding="utf-8")
    (task_dir / "init.cu").write_text(CUDA_SOURCE, encoding="utf-8")
    candidate = tmp_path / "candidate.cu"
    candidate.write_text(CUDA_SOURCE, encoding="utf-8")
    strict_driver = tmp_path / "canonical.cpp"
    strict_driver.write_text(CANDIDATE_ONLY_DRIVER, encoding="utf-8")
    return task_dir, candidate, strict_driver


def _inner_argv(
    task_dir: Path,
    candidate: Path,
    strict_driver: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "benchmark_cuda.py",
        "--task-dir",
        str(task_dir),
        "--task-id",
        "007",
        "--kernel",
        "Small-K matrix multiplication",
        "--candidate",
        str(candidate),
        "--candidate-name",
        "cublas",
        "--candidate-only-correctness-driver",
        str(strict_driver),
        "--correctness-only",
        "--sessions",
        "1",
        "--output-dir",
        str(output_dir),
    ]


def test_candidate_only_driver_requires_distribution_marker(tmp_path, monkeypatch):
    executable = tmp_path / "main"
    monkeypatch.setattr(
        CUDA,
        "_run_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [str(executable)], 0, stdout="passed\n", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="distribution marker"):
        CUDA._run_correctness(
            executable,
            output_dir=tmp_path,
            label="candidate",
            mode="original-candidate-only",
            timeout=1,
            require_distribution=True,
        )


def test_distribution_rejects_duplicate_case_seed_pair():
    metrics = _distribution()
    metrics["cases"][1] = _case(0)
    with pytest.raises(RuntimeError, match="duplicate case_id/seed pair"):
        CUDA._validate_correctness_distribution(metrics)


def test_candidate_only_driver_is_hashed_and_excluded_from_regression(tmp_path, monkeypatch):
    task_dir, candidate, strict_driver = _write_inner_inputs(tmp_path)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        CUDA.sys, "argv", _inner_argv(task_dir, candidate, strict_driver, output_dir)
    )
    monkeypatch.setattr(CUDA, "_version", lambda command: "test")
    monkeypatch.setattr(CUDA, "_git_commit", lambda: "test-commit")
    monkeypatch.setattr(
        CUDA,
        "_compile_program",
        lambda root, **kwargs: root / "fake" / kwargs["label"] / kwargs["mode"],
    )

    def fake_correctness(executable, **kwargs):
        result = {"mode": kwargs["mode"], "returncode": 0, "passed": True}
        if kwargs["require_distribution"]:
            result["metrics"] = _distribution()
        return result

    monkeypatch.setattr(CUDA, "_run_correctness", fake_correctness)
    assert CUDA.main() == 0

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(CANDIDATE_ONLY_DRIVER.encode("utf-8")).hexdigest()
    assert manifest["candidate_only_correctness_drivers"] == [
        {
            "name": strict_driver.name,
            "scope": "candidate_only",
            "sha256": expected_hash,
        }
    ]
    assert manifest["candidate_only_correctness"] == {
        "status": "passed",
        "drivers": ["candidate-only-0-canonical"],
        "executions": 2,
    }
    assert manifest["correctness_error_regression"] == {
        "status": "passed",
        "relative_allowance": CUDA.CORRECTNESS_ERROR_RELATIVE_ALLOWANCE,
        "absolute_allowance": CUDA.CORRECTNESS_ERROR_ABSOLUTE_ALLOWANCE,
        "comparisons": [],
    }
    baseline_scopes = {
        result["driver_scope"] for result in manifest["variants"]["baseline"]["correctness"]
    }
    candidate_only = [
        result
        for result in manifest["variants"]["cublas"]["correctness"]
        if result["driver_scope"] == "candidate_only"
    ]
    assert baseline_scopes == {"shared"}
    assert len(candidate_only) == 2
    assert all(result["metrics"]["cases"] for result in candidate_only)


def test_candidate_only_failure_precedes_benchmark_and_ncu(tmp_path, monkeypatch):
    task_dir, candidate, strict_driver = _write_inner_inputs(tmp_path)
    output_dir = tmp_path / "out"
    argv = _inner_argv(task_dir, candidate, strict_driver, output_dir)
    argv.insert(-2, "--ncu")
    monkeypatch.setattr(CUDA.sys, "argv", argv)
    monkeypatch.setattr(CUDA, "_version", lambda command: "test")
    monkeypatch.setattr(CUDA, "_git_commit", lambda: "test-commit")
    compiled_modes: list[str] = []

    def fake_compile(root, **kwargs):
        compiled_modes.append(kwargs["mode"])
        return root / "fake" / kwargs["label"] / kwargs["mode"]

    def fake_correctness(executable, **kwargs):
        if kwargs["require_distribution"]:
            raise RuntimeError("strict candidate-only failure")
        return {"mode": kwargs["mode"], "returncode": 0, "passed": True}

    monkeypatch.setattr(CUDA, "_compile_program", fake_compile)
    monkeypatch.setattr(CUDA, "_run_correctness", fake_correctness)
    monkeypatch.setattr(
        CUDA,
        "_run_ncu",
        lambda *args, **kwargs: pytest.fail("NCU ran before candidate-only correctness"),
    )
    monkeypatch.setattr(
        CUDA,
        "_run_command",
        lambda *args, **kwargs: pytest.fail("timing ran before candidate-only correctness"),
    )

    with pytest.raises(RuntimeError, match="strict candidate-only failure"):
        CUDA.main()
    assert "benchmark" not in compiled_modes
    assert "ncu-profile" not in compiled_modes


def test_benchmark_candidates_resolves_and_routes_candidate_only_driver(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.cu"
    shared_driver = tmp_path / "edge.cpp"
    strict_driver = tmp_path / "canonical.cpp"
    for path in (candidate, shared_driver, strict_driver):
        path.write_text("// test\n", encoding="utf-8")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "init.cu").write_text("// baseline\n", encoding="utf-8")
    (task_dir / "driver.cpp").write_text("// official\n", encoding="utf-8")
    manifest_path = tmp_path / "candidates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "runtime_contract": RUNTIME_CONTRACT,
                "tasks": [
                    {
                        "id": "007",
                        "name": "candidate",
                        "source": candidate.name,
                        "capability_status": "hardened",
                        "extra_correctness_drivers": [shared_driver.name],
                        "candidate_only_correctness_drivers": [strict_driver.name],
                        "supported_cases": [
                            {
                                "case_id": "canonical",
                                "class": "canonical",
                                "shape": {"M": 1, "N": 1, "K": 1},
                                "correctness": True,
                                "performance": True,
                            }
                        ],
                        "numerics_profile": "test",
                        "resource_policy": {
                            "kind": "none",
                            "ownership": "none",
                            "workspace_bytes": 0,
                        },
                        "reentrant_under_contract": True,
                        "requires_prewarm": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "test",
                "subset": "level1",
                "precision": "fp16",
                "defaults": {"rollouts": 1, "steps": 1},
                "tasks": [
                    {
                        "number": 7,
                        "id": "007",
                        "name": "Small-K matrix multiplication",
                        "category": "matmul",
                        "path": "task",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "suite-out"
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        captured.append(command)
        task_output = Path(command[command.index("--output-dir") + 1])
        task_output.mkdir(parents=True)
        (task_output / "summary.json").write_text(
            json.dumps(
                {
                    "comparison": None,
                    "stable": None,
                    "performance_claim_allowed": False,
                    "validation_gates": {"correctness": "passed"},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(CANDIDATES.subprocess, "run", fake_run)
    monkeypatch.setattr(CANDIDATES, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(
        CANDIDATES.sys,
        "argv",
        [
            "benchmark_candidates.py",
            "--suite",
            str(suite_path),
            "--manifest",
            str(manifest_path),
            "--task-id",
            "007",
            "--shape",
            "canonical",
            "--correctness-only",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert CANDIDATES.main() == 0
    assert len(captured) == 1
    command = captured[0]
    assert command[command.index("--extra-correctness-driver") + 1] == str(shared_driver.resolve())
    assert command[command.index("--candidate-only-correctness-driver") + 1] == str(
        strict_driver.resolve()
    )


def test_007_canonical_driver_declares_strict_protocol(tmp_path):
    relative_driver = "correctness_drivers/007_small_k_matmul_canonical_driver.cpp"
    manifest = json.loads(
        (ROOT / "portfolio" / "case_studies" / "core10" / "candidates.json").read_text(
            encoding="utf-8"
        )
    )
    task = next(item for item in manifest["tasks"] if item["id"] == "007")
    assert task["candidate_only_correctness_drivers"] == [relative_driver]
    assert task["supported_cases"][0]["correctness_protocol"] == (
        "official_plus_candidate_only_three_seed_driver"
    )
    path = ROOT / "portfolio" / "case_studies" / "core10" / relative_driver
    source = path.read_text(encoding="utf-8")
    assert "constexpr int64_t kM = 16384;" in source
    assert "constexpr int64_t kN = 16384;" in source
    assert "constexpr int64_t kK = 32;" in source
    assert "constexpr int kRepeats = 5;" in source
    assert "{0, 42, 20260721}" in source
    assert source.count("torch::kFloat32") >= 3
    assert "summarize_fp32_golden_in_row_chunks" in source
    assert "constexpr int64_t kChunkRows = 256;" in source
    assert "const int64_t element_count = kM * kN;" in source
    assert "quantile_stride" in source
    assert "chunked matmul metric count mismatch" in source
    assert "chunked matmul quantile sample mismatch" in source
    assert "setAllowTF32CuBLAS(false)" in source
    assert "constexpr double kAtol = 1e-2;" in source
    assert "constexpr double kRtol = 1e-2;" in source
    assert "torch::equal(" in source
    assert source.count("view(torch::kInt16)") == 2
    assert "KERNELBLASTER_CORRECTNESS_JSON" in source
    CUDA.write_compilation_units(
        tmp_path / "static-output",
        source,
        (ROOT / "portfolio" / "case_studies" / "core10" / "007_small_k_matmul_cublas.cu").read_text(
            encoding="utf-8"
        ),
    )
