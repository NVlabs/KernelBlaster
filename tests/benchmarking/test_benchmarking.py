from __future__ import annotations

import pytest

from src.kernelblaster.benchmarking import (
    BENCHMARK_MARKER,
    comparison_validity,
    find_launch_call,
    instrument_profiler_driver,
    instrument_driver,
    latency_summary,
    ncu_metric_names,
    normalize_cuda_source,
    session_spread_percent,
    split_compilation_units,
)


DRIVER = r'''
#include <torch/torch.h>
void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t count
);

int main() {
    torch::Tensor input;
    torch::Tensor output;
    int64_t count = 4;
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        count
    );
    bool passed = true;
    if (passed) std::cout << "passed" << std::endl;
    return 0;
}
'''


CUDA = r'''
#include <cuda_runtime.h>
__global__ void kernel() {}
void launch_gpu_implementation(void*, void*, int64_t) {
    kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaStreamSynchronize(0);
}
'''


CUDA_WITH_NON_LAUNCHER_SYNC = r'''
void helper() { cudaDeviceSynchronize(); }
void launch_gpu_implementation(void*, void*, int64_t) {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
'''


CUDA_WITH_REQUIRE_CUDA_SYNC = r'''
void require_cuda(const char*, cudaError_t);
void launch_gpu_implementation(void*, void*, int64_t) {
    kernel<<<1, 1>>>();
    require_cuda("cudaDeviceSynchronize", cudaDeviceSynchronize());
}
'''


def test_normalization_only_removes_host_synchronization():
    normalized, transformations = normalize_cuda_source(CUDA)
    assert "cudaDeviceSynchronize" not in normalized
    assert "cudaStreamSynchronize" not in normalized
    assert normalized.count("cudaGetLastError") == 2
    assert transformations == [
        "cudaDeviceSynchronize via CUDA_CHECK",
        "cudaStreamSynchronize",
    ]


def test_normalization_does_not_rewrite_unrelated_host_functions():
    normalized, transformations = normalize_cuda_source(CUDA_WITH_NON_LAUNCHER_SYNC)
    assert "void helper() { cudaDeviceSynchronize(); }" in normalized
    assert normalized.count("cudaGetLastError") == 1
    assert transformations == ["cudaDeviceSynchronize"]


def test_normalization_removes_fail_loud_synchronize_wrapper():
    normalized, transformations = normalize_cuda_source(
        CUDA_WITH_REQUIRE_CUDA_SYNC
    )
    assert "cudaDeviceSynchronize" not in normalized
    assert 'require_cuda("cudaGetLastError", cudaGetLastError());' in normalized
    assert transformations == ["cudaDeviceSynchronize via require_cuda"]


def test_launch_call_parser_handles_multiline_arguments():
    call, _span = find_launch_call(DRIVER)
    assert call.startswith("launch_gpu_implementation(")
    assert call.rstrip().endswith(");")
    assert "output.data_ptr()" in call


def test_instrumented_driver_contains_events_seed_and_marker():
    instrumented = instrument_driver(
        DRIVER,
        seed=20260719,
        warmup=20,
        repetitions=100,
        inner_loops=0,
    )
    assert "torch::manual_seed(20260719)" in instrumented
    assert "cudaEventElapsedTime" in instrumented
    assert "kb_repetitions = 100" in instrumented
    assert BENCHMARK_MARKER in instrumented
    assert instrumented.count("launch_gpu_implementation(") >= 4


def test_profiler_driver_limits_collection_to_launcher_call():
    instrumented = instrument_profiler_driver(DRIVER)
    assert instrumented.startswith("#include <cuda_profiler_api.h>")
    assert "cudaProfilerStart();" in instrumented
    assert "cudaProfilerStop();" in instrumented
    main = instrumented[instrumented.index("int main") :]
    start = main.index("cudaProfilerStart")
    launch_positions = [
        index
        for index in range(len(main))
        if main.startswith("launch_gpu_implementation(", index)
    ]
    assert len(launch_positions) == 2
    assert launch_positions[0] < start < launch_positions[1]
    assert main.count("cudaDeviceSynchronize();") == 2


def test_split_compilation_units_moves_declaration_to_header():
    main, header, cuda = split_compilation_units(DRIVER, CUDA)
    assert '#include "cuda_model.cuh"' in main
    assert "void launch_gpu_implementation" not in main.split("int main", 1)[0]
    assert "void launch_gpu_implementation" in header
    assert cuda.startswith('#include "cuda_model.cuh"')


def test_latency_summary_reports_distribution_and_cv():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["count"] == 5
    assert summary["median_us"] == 3.0
    assert summary["p10_us"] == pytest.approx(1.4)
    assert summary["p90_us"] == pytest.approx(4.6)
    assert summary["cv_percent"] > 0


def test_session_spread_and_comparison_gates():
    assert session_spread_percent([10.0, 10.2, 9.9]) == pytest.approx(
        (10.2 / 9.9 - 1) * 100
    )
    gate = comparison_validity(
        baseline_source_sha256="same",
        candidate_source_sha256="same",
        baseline_session_medians=[10.0, 10.1, 9.9],
        candidate_session_medians=[10.1, 10.0, 10.0],
        speedup=1.01,
        max_session_spread_percent=5.0,
    )
    assert gate["comparison_kind"] == "self_check"
    assert gate["self_check_passed"] is True
    assert gate["formal_valid"] is True

    unstable = comparison_validity(
        baseline_source_sha256="baseline",
        candidate_source_sha256="candidate",
        baseline_session_medians=[10.0, 11.0, 9.0],
        candidate_session_medians=[9.0, 9.1, 9.2],
        speedup=1.1,
        max_session_spread_percent=5.0,
    )
    assert unstable["stable"] is False
    assert unstable["formal_valid"] is False


def test_ncu_metric_names_reads_metric_name_column():
    exported = (
        '==PROF== Connected\n'
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"1","kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","42"\n'
        '"1","kernel","dram__bytes_read.sum","byte","1024"\n'
    )
    assert ncu_metric_names(exported) == [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
    ]
