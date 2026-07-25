# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.kernelblaster.portfolio.capabilities import (
    CAPABILITY_MARKER,
    describe_capabilities,
    load_capability_manifest,
    task_map,
    validate_candidate_request,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    ROOT / "portfolio" / "case_studies" / "core10" / "candidates.json"
)


def _request(task_id: str = "036") -> dict:
    manifest = load_capability_manifest(MANIFEST_PATH)
    task = task_map(manifest)[task_id]
    return {
        "arch": "sm_86",
        "compile_arch": "sm_86",
        "dtype": "fp16",
        "target_dtype": "int64",
        "layout": "contiguous_row_major",
        "stream_mode": "legacy_default",
        "stream_count": 1,
        "graph_capture": False,
        "backward": False,
        "shape": task["supported_cases"][0]["shape"],
        "portability_replay": False,
    }


def test_real_manifest_is_schema_v2_and_all_hardened_cases_are_explicit():
    manifest = load_capability_manifest(MANIFEST_PATH)
    tasks = task_map(manifest)
    assert manifest["schema_version"] == "2.0"
    assert manifest["runtime_contract"]["production_ready"] is False
    for task_id in ("004", "007", "036", "040", "095"):
        task = tasks[task_id]
        assert task["capability_status"] == "hardened"
        assert task["reentrant_under_contract"] is True
        assert task["supported_cases"][0]["case_id"] == "canonical"
        assert all(case["correctness"] for case in task["supported_cases"])
        assert all(
            not case["performance"] for case in task["supported_cases"][1:]
        )
    description = describe_capabilities(manifest)
    assert {task["id"] for task in description["tasks"]} == {
        "004",
        "007",
        "036",
        "040",
        "095",
    }
    assert not description["unknown_tasks"]
    assert tasks["007"]["candidate_only_correctness_drivers"] == [
        "correctness_drivers/007_small_k_matmul_canonical_driver.cpp"
    ]
    assert (
        tasks["007"]["supported_cases"][0]["correctness_protocol"]
        == "official_plus_candidate_only_three_seed_driver"
    )


@pytest.mark.parametrize(
    ("task_id", "change", "reason", "exit_code"),
    [
        ("036", {"stream_count": 0}, "invalid_request", 2),
        ("999", {}, "unknown_task", 2),
        ("036", {"arch": "sm_80"}, "unsupported_arch", 5),
        ("036", {"backward": True}, "unsupported_backward", 5),
        ("036", {"dtype": "fp32"}, "unsupported_dtype", 5),
        ("095", {"target_dtype": "int32"}, "unsupported_target_dtype", 5),
        ("036", {"layout": "noncontiguous"}, "unsupported_layout", 5),
        ("036", {"stream_mode": "non_default"}, "unsupported_stream", 5),
        ("036", {"stream_count": 2}, "unsupported_stream", 5),
        ("036", {"graph_capture": True}, "unsupported_graph_capture", 5),
        (
            "036",
            {"shape": {"B": 1, "C": 64, "D1": 1, "D2": 1}},
            "unsupported_shape",
            5,
        ),
    ],
)
def test_capability_reason_codes(task_id, change, reason, exit_code):
    manifest = load_capability_manifest(MANIFEST_PATH)
    request = _request("095" if task_id == "095" else "036")
    request.update(change)
    result = validate_candidate_request(manifest, task_id, request)
    assert result.supported is False
    assert result.reason_code == reason
    assert result.exit_code == exit_code


def test_capability_reason_priority_is_stable():
    manifest = load_capability_manifest(MANIFEST_PATH)
    request = _request()
    request.update(
        {
            "arch": "sm_80",
            "dtype": "bf16",
            "layout": "noncontiguous",
            "graph_capture": True,
        }
    )
    result = validate_candidate_request(manifest, "036", request)
    assert result.reason_code == "unsupported_arch"


def test_target_dtype_is_required_only_for_the_cross_entropy_contract():
    manifest = load_capability_manifest(MANIFEST_PATH)
    request = _request("095")
    request.pop("target_dtype")
    result = validate_candidate_request(manifest, "095", request)
    assert result.reason_code == "invalid_request"
    assert result.exit_code == 2

    rmsnorm_request = _request("036")
    rmsnorm_request.pop("target_dtype")
    assert validate_candidate_request(
        manifest, "036", rmsnorm_request
    ).supported


def test_legacy_research_candidates_are_not_schema_v2_capabilities():
    manifest = load_capability_manifest(MANIFEST_PATH)
    result = validate_candidate_request(manifest, "019", _request("036"))
    assert result.reason_code == "unknown_task"
    description = describe_capabilities(manifest, ["019"])
    assert description["tasks"] == []
    assert description["unknown_tasks"] == ["019"]


def test_runner_rejects_before_output_directory_or_subprocess(
    tmp_path, monkeypatch, capsys
):
    spec = importlib.util.spec_from_file_location(
        "benchmark_candidates_capability_test",
        ROOT / "scripts" / "benchmark_candidates.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_dir = tmp_path / "must-not-exist"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("subprocess must not run for an unsupported request")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "benchmark_candidates.py",
            "--task-id",
            "036",
            "--dtype",
            "bf16",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert module.main() == 5
    assert not output_dir.exists()
    line = capsys.readouterr().out.strip()
    assert line.startswith(CAPABILITY_MARKER)
    payload = json.loads(line[len(CAPABILITY_MARKER) :])
    assert payload["reason_code"] == "unsupported_dtype"


def test_runner_prioritizes_invalid_request_before_unknown_task(
    tmp_path, monkeypatch, capsys
):
    spec = importlib.util.spec_from_file_location(
        "benchmark_candidates_invalid_priority_test",
        ROOT / "scripts" / "benchmark_candidates.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_dir = tmp_path / "must-not-exist"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("subprocess must not run for an invalid request")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "benchmark_candidates.py",
            "--task-id",
            "999",
            "--stream-count",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert module.main() == 2
    assert not output_dir.exists()
    line = capsys.readouterr().out.strip()
    assert line.startswith(CAPABILITY_MARKER)
    payload = json.loads(line[len(CAPABILITY_MARKER) :])
    assert payload["reason_code"] == "invalid_request"


def test_runner_can_describe_without_output_directory(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "benchmark_candidates_describe_test",
        ROOT / "scripts" / "benchmark_candidates.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_dir = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "benchmark_candidates.py",
            "--task-id",
            "095",
            "--describe-capabilities",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert module.main() == 0
    assert not output_dir.exists()
    line = capsys.readouterr().out.strip()
    assert line.startswith(CAPABILITY_MARKER)
    payload = json.loads(line[len(CAPABILITY_MARKER) :])
    assert payload["supported"] is True
    assert payload["tasks"][0]["target_dtype"] == "int64"


def test_thread_device_raii_replaces_raw_static_resources():
    sources = {
        "007": ("007_small_k_matmul_cublas.cu", "cublasSetStream"),
        "040": ("040_layernorm_sm86.cu", "32896"),
        "095": ("095_cross_entropy_sm86.cu", "2048"),
    }
    root = ROOT / "portfolio" / "case_studies" / "core10"
    for filename, required in sources.values():
        source = (root / filename).read_text(encoding="utf-8")
        assert "static thread_local" in source
        assert "thread_device_context" in source
        assert required in source
        assert "static cublasHandle_t" not in source
        assert "static float* partials" not in source
        assert "static int capacity" not in source
        assert "assert(cudaMalloc" not in source
        assert "assert(cudaGetDevice" not in source
        assert "assert(cublasCreate" not in source
        assert "assert(cublasSetStream" not in source


def test_hardened_edge_drivers_emit_three_seed_error_distributions():
    drivers = [
        ROOT / "portfolio/case_studies/core10/edge_drivers/004_matvec_edge_driver.cpp",
        ROOT / "portfolio/case_studies/core10/edge_drivers/007_small_k_matmul_edge_driver.cpp",
        ROOT / "portfolio/case_studies/rmsnorm/edge_driver.cpp",
        ROOT / "portfolio/case_studies/core10/edge_drivers/040_layernorm_edge_driver.cpp",
        ROOT / "portfolio/case_studies/core10/edge_drivers/095_cross_entropy_edge_driver.cpp",
    ]
    for driver in drivers:
        source = driver.read_text(encoding="utf-8")
        assert '#include "correctness_metrics.h"' in source
        assert "{0, 42, 20260721}" in source
        assert "merge_envelope" in source
    assert '"canonical"' not in drivers[1].read_text(encoding="utf-8")
    assert all(
        '"canonical"' in driver.read_text(encoding="utf-8")
        for index, driver in enumerate(drivers)
        if index != 1
    )
