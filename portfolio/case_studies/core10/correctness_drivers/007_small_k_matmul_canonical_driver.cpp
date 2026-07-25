// SPDX-FileCopyrightText: Copyright (c) 2026 KernelBlaster contributors
// SPDX-License-Identifier: Apache-2.0

#include <ATen/Context.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "correctness_metrics.h"
#include "cuda_model.cuh"

void launch_gpu_implementation(
    void*, void*, void*, int64_t, int64_t, int64_t
);

namespace kc = kernelblaster::correctness;

namespace {

constexpr int64_t kM = 16384;
constexpr int64_t kN = 16384;
constexpr int64_t kK = 32;
constexpr int64_t kChunkRows = 256;
constexpr int64_t kQuantileMaxSamples = 1 << 20;
constexpr int kRepeats = 5;
constexpr double kAtol = 1e-2;
constexpr double kRtol = 1e-2;

kc::Metrics summarize_fp32_golden_in_row_chunks(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    const torch::Tensor& candidate
) {
    const int64_t element_count = kM * kN;
    const int64_t quantile_stride = (
        element_count + kQuantileMaxSamples - 1
    ) / kQuantileMaxSamples;
    const auto input_b_fp32 = input_b.to(torch::kFloat32);
    std::vector<torch::Tensor> absolute_samples;
    std::vector<torch::Tensor> normalized_samples;
    absolute_samples.reserve((kM + kChunkRows - 1) / kChunkRows);
    normalized_samples.reserve(absolute_samples.capacity());

    kc::Metrics metrics;
    double absolute_sum = 0.0;
    double absolute_square_sum = 0.0;
    for (int64_t row = 0; row < kM; row += kChunkRows) {
        const int64_t rows = std::min(kChunkRows, kM - row);
        const auto reference = torch::matmul(
            input_a.narrow(0, row, rows).to(torch::kFloat32),
            input_b_fp32
        );
        const auto candidate_fp32 = candidate
                                        .narrow(0, row, rows)
                                        .to(torch::kFloat32);
        const auto absolute = (candidate_fp32 - reference).abs();
        const auto normalized = absolute / (kAtol + kRtol * reference.abs());
        const int64_t chunk_count = absolute.numel();

        metrics.count += chunk_count;
        metrics.nonfinite_count += torch::logical_not(
            torch::isfinite(candidate_fp32)
        ).sum().item<int64_t>();
        metrics.mismatch_count += (normalized > 1.0).sum().item<int64_t>();
        absolute_sum += absolute.sum().item<double>();
        absolute_square_sum += absolute.square().sum().item<double>();
        metrics.abs_max = std::max(
            metrics.abs_max, absolute.max().item<double>()
        );
        metrics.normalized_max = std::max(
            metrics.normalized_max, normalized.max().item<double>()
        );

        // kN is divisible by quantile_stride, so every row chunk starts on
        // the same global deterministic-stride boundary as the full tensor.
        // Clone the sparse views so completed chunks can release their large
        // FP32 intermediates immediately.
        absolute_samples.push_back(
            absolute.flatten()
                .slice(0, 0, chunk_count, quantile_stride)
                .clone()
        );
        normalized_samples.push_back(
            normalized.flatten()
                .slice(0, 0, chunk_count, quantile_stride)
                .clone()
        );
    }

    if (metrics.count != element_count) {
        throw std::runtime_error("chunked matmul metric count mismatch");
    }
    auto absolute_sample = std::get<0>(
        torch::cat(absolute_samples).sort()
    );
    auto normalized_sample = std::get<0>(
        torch::cat(normalized_samples).sort()
    );
    if (
        absolute_sample.numel() != kQuantileMaxSamples
        || normalized_sample.numel() != kQuantileMaxSamples
    ) {
        throw std::runtime_error("chunked matmul quantile sample mismatch");
    }

    metrics.quantile_sample_count = absolute_sample.numel();
    metrics.abs_mean = absolute_sum / static_cast<double>(metrics.count);
    metrics.abs_rmse = std::sqrt(
        absolute_square_sum / static_cast<double>(metrics.count)
    );
    metrics.abs_p50 = kc::sample_quantile(absolute_sample, 0.50);
    metrics.abs_p90 = kc::sample_quantile(absolute_sample, 0.90);
    metrics.abs_p99 = kc::sample_quantile(absolute_sample, 0.99);
    metrics.abs_p999 = kc::sample_quantile(absolute_sample, 0.999);
    metrics.normalized_p50 = kc::sample_quantile(normalized_sample, 0.50);
    metrics.normalized_p90 = kc::sample_quantile(normalized_sample, 0.90);
    metrics.normalized_p99 = kc::sample_quantile(normalized_sample, 0.99);
    metrics.normalized_p999 = kc::sample_quantile(normalized_sample, 0.999);
    return metrics;
}

bool run_seed(
    int64_t seed,
    kc::Metrics& aggregate,
    std::vector<std::string>& case_results
) {
    torch::manual_seed(seed);
    const auto options = torch::TensorOptions()
                             .dtype(torch::kFloat16)
                             .device(torch::kCUDA);
    const auto input_a = torch::randn({kM, kK}, options);
    const auto input_b = torch::randn({kK, kN}, options);
    auto output = torch::empty({kM, kN}, options);
    torch::Tensor first;
    bool deterministic = true;

    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        output.fill_(std::numeric_limits<float>::quiet_NaN());
        launch_gpu_implementation(
            output.data_ptr(),
            input_a.data_ptr(),
            input_b.data_ptr(),
            kM,
            kN,
            kK
        );
        if (repeat == 0) {
            first = output.clone();
        } else {
            deterministic = deterministic && torch::equal(
                first.view(torch::kInt16), output.view(torch::kInt16)
            );
        }
    }
    first = torch::Tensor();

    const kc::Metrics metrics = summarize_fp32_golden_in_row_chunks(
        input_a, input_b, output
    );
    kc::merge_envelope(aggregate, metrics);
    case_results.push_back(kc::case_json(
        "canonical-16384x16384x32",
        seed,
        "{\"M\":16384,\"N\":16384,\"K\":32}",
        metrics,
        deterministic
    ));
    return deterministic && metrics.nonfinite_count == 0
        && metrics.mismatch_count == 0 && metrics.normalized_max <= 1.0;
}

}  // namespace

int main() {
    torch::NoGradGuard no_grad;
    at::globalContext().setAllowTF32CuBLAS(false);
    const std::vector<int64_t> seeds = {0, 42, 20260721};
    kc::Metrics aggregate;
    std::vector<std::string> case_results;
    bool passed = true;
    for (const int64_t seed : seeds) {
        passed = run_seed(seed, aggregate, case_results) && passed;
    }

    const bool finite = aggregate.nonfinite_count == 0;
    std::cout << "KERNELBLASTER_CORRECTNESS_JSON "
              << kc::result_json(aggregate, finite, passed, case_results)
              << std::endl;
    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
