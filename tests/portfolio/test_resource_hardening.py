# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

from src.kernelblaster.benchmarking import (
    find_launch_definition,
    normalize_cuda_source,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "portfolio" / "case_studies" / "core10"


def _read_source(filename: str) -> str:
    return (SOURCE_ROOT / filename).read_text(encoding="utf-8")


def test_release_build_resource_failures_are_unconditional_and_fail_loud():
    contexts = {
        "007_small_k_matmul_cublas.cu": "CublasContext",
        "040_layernorm_sm86.cu": "LayerNormContext",
        "095_cross_entropy_sm86.cu": "CrossEntropyContext",
    }
    sources = {filename: _read_source(filename) for filename in contexts}

    for filename, context in contexts.items():
        source = sources[filename]
        assert "#include <cassert>" not in source
        assert "KERNELBLASTER_CUDA_ERROR" in source
        assert "KERNELBLASTER_RESOURCE_BLOCKED" in source
        assert "[[noreturn]] void fail_cuda" in source
        assert "std::abort();" in source
        assert re.search(r'require_cuda_resource\(\s*"cudaGetDevice"', source)
        assert re.search(r'require_cuda\(\s*"cudaDeviceSynchronize"', source)
        assert f"~{context}() noexcept" in source
        assert "KERNELBLASTER_RESOURCE_CLEANUP_BLOCKED" in source
        assert "report_cuda_cleanup" in source
        normalized, transformations = normalize_cuda_source(source)
        normalized_launcher, _ = find_launch_definition(normalized)
        assert "cudaDeviceSynchronize" not in normalized_launcher
        assert "cudaDeviceSynchronize via require_cuda" in transformations

    cublas_source = sources["007_small_k_matmul_cublas.cu"]
    assert "KERNELBLASTER_CUBLAS_ERROR" in cublas_source
    assert "fail_cublas_resource" in cublas_source
    for operation in (
        "cublasCreate",
        "cublasSetMathMode",
        "cublasSetStream",
    ):
        assert re.search(
            rf'require_cublas_resource\(\s*"{operation}"', cublas_source
        )
    assert re.search(r'require_cublas\(\s*"cublasGemmEx"', cublas_source)
    assert re.search(
        r'report_cublas_cleanup\(\s*"cublasDestroy"', cublas_source
    )

    workspace_sources = {
        "040_layernorm_sm86.cu": "cudaMalloc(layernorm_workspace)",
        "095_cross_entropy_sm86.cu": "cudaMalloc(cross_entropy_workspace)",
    }
    for filename, operation in workspace_sources.items():
        source = sources[filename]
        assert re.search(
            rf'require_cuda_resource\(\s*"{re.escape(operation)}"', source
        )
        assert "KERNELBLASTER_CONTRACT_ERROR" in source
        assert "require_contract(" in source


def test_aggregate_quantiles_are_explicitly_labeled_as_an_envelope():
    header = (
        ROOT
        / "src/kernelblaster/servers/cuda_env/correctness_metrics.h"
    ).read_text(encoding="utf-8")
    assert "max_per_case_quantile_envelope" in header
    assert "merge_envelope" in header
