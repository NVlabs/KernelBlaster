# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from src.kernelblaster.benchmarking import (
    find_launch_declaration,
    find_launch_definition,
    split_compilation_units,
)


ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "portfolio" / "case_studies" / "core10"

DRIVERS = {
    "007": {
        "path": "edge_drivers/007_resource_lifecycle_driver.cpp",
        "shape": {"M": 17, "N": 19, "K": 7},
        "atol": "1e-2",
        "golden": "torch::matmul(",
        "measured_waves": 8,
    },
    "040": {
        "path": "edge_drivers/040_resource_lifecycle_driver.cpp",
        "shape": {"B": 1, "F": 3, "D1": 5, "D2": 7},
        "atol": "5e-3",
        "golden": "torch::layer_norm(",
        "measured_waves": 8,
    },
    "095": {
        "path": "edge_drivers/095_resource_lifecycle_driver.cpp",
        "shape": {"B": 17, "classes": 7},
        "atol": "1e-2",
        "golden": "torch::nn::functional::cross_entropy(",
        "measured_waves": 32,
    },
}

CASE_IDS = {
    "reuse-5-calls",
    "parallel-host-thread-a",
    "parallel-host-thread-b",
    "thread-exit-waves-a",
    "thread-exit-waves-b",
}


def _manifest_tasks() -> dict[str, dict]:
    payload = json.loads((CASE_ROOT / "candidates.json").read_text(encoding="utf-8"))
    return {task["id"]: task for task in payload["tasks"]}


def test_manifest_declares_resource_lifecycle_protocols():
    tasks = _manifest_tasks()
    for task_id, expected in DRIVERS.items():
        task = tasks[task_id]
        assert expected["path"] in task["extra_correctness_drivers"]
        assert (CASE_ROOT / expected["path"]).is_file()
        assert any(
            case["shape"] == expected["shape"]
            and case["correctness"] is True
            and case["performance"] is False
            for case in task["supported_cases"]
        )

        policy = task["resource_policy"]
        assert policy["ownership"] == "thread_device_raii"
        assert policy["initialization_phase"] == "correctness_or_warmup"
        assert policy["timed_region_allocations"] is False
        assert policy["release_phase"] == "host_thread_exit"
        assert policy["stream_binding"] == "cudaStreamLegacy"

        protocol = policy["validation_protocol"]
        assert protocol == {
            "kind": "same_thread_reuse_dual_host_thread_release_v1",
            "same_thread_calls": 5,
            "host_threads": 2,
            "calls_per_host_thread": 5,
            "warmup_thread_waves": 4,
            "measured_thread_waves_per_group": expected["measured_waves"],
            "measured_groups": 2,
            "leak_allowance_bytes": 65536,
            "seeds": [0, 42, 20260721],
            "shape": expected["shape"],
            "driver": expected["path"],
        }
        if policy["workspace_bytes"]:
            leaked_group_bytes = (
                protocol["host_threads"]
                * protocol["measured_thread_waves_per_group"]
                * policy["workspace_bytes"]
            )
            assert leaked_group_bytes > protocol["leak_allowance_bytes"]


def test_resource_drivers_are_bounded_host_only_abi_consumers():
    tasks = _manifest_tasks()
    for task_id, expected in DRIVERS.items():
        driver = (CASE_ROOT / expected["path"]).read_text(encoding="utf-8")
        source = (CASE_ROOT / tasks[task_id]["source"]).read_text(encoding="utf-8")

        declaration, _ = find_launch_declaration(driver)
        definition, _ = find_launch_definition(source)
        main, header, cuda = split_compilation_units(driver, source)
        assert declaration in header
        assert definition in cuda
        assert "int main()" in main
        assert driver.count("KERNELBLASTER_CORRECTNESS_JSON") == 1
        assert '#include "correctness_metrics.h"' in driver
        assert "<<<" not in driver
        assert 'extern "C"' not in driver
        for forbidden in (
            "cudaMalloc(",
            "cudaFree(",
            "cublasCreate(",
            "cublasDestroy(",
            "cudaStreamCreate(",
        ):
            assert forbidden not in driver


def test_resource_drivers_cover_reuse_parallelism_release_and_numerics():
    for expected in DRIVERS.values():
        source = (CASE_ROOT / expected["path"]).read_text(encoding="utf-8")
        assert "constexpr int kReuseCalls = 5;" in source
        assert "StartGate gate(2);" in source
        assert "RepeatGate repeat_gate(2);" in source
        assert source.count("std::thread ") == 2
        assert source.count("cudaSetDevice(0)") >= 3
        assert "cudaStreamSynchronize(cudaStreamLegacy)" in source
        assert "cudaMemGetInfo" in source
        assert "kWarmupThreadWaves = 4" in source
        assert f"kMeasuredThreadWaves = {expected['measured_waves']}" in source
        assert "kLeakAllowanceBytes = 64 * 1024" in source
        assert "const std::vector<int64_t> seeds = {0, 42, 20260721};" in source
        assert ".device(torch::kCUDA, 0)" in source
        assert "Workload parallel_first" in source
        assert "Workload parallel_second" in source
        assert "Workload lifecycle_first" in source
        assert "Workload lifecycle_second" in source
        assert "buffers_are_independent(" in source
        assert "parallel_buffers_independent" in source
        assert "lifecycle_buffers_independent" in source
        assert expected["golden"] in source
        assert "to(torch::kFloat32)" in source
        assert f"constexpr double kAtol = {expected['atol']};" in source
        assert "constexpr double kRtol = 1e-2;" in source
        assert "metrics.nonfinite_count == 0" in source
        assert "metrics.mismatch_count == 0" in source
        assert "metrics.normalized_max <= 1.0" in source
        for case_id in CASE_IDS:
            assert f'"{case_id}"' in source
