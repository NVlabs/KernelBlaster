// SPDX-FileCopyrightText: Copyright (c) 2026 KernelBlaster contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/torch.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "correctness_metrics.h"
#include "cuda_model.cuh"

void launch_gpu_implementation(
    void*, void*, void*, void*, int64_t, int64_t, int64_t, int64_t
);

namespace kc = kernelblaster::correctness;

struct Shape {
    const char* case_id;
    int64_t batch;
    int64_t features;
    int64_t dim1;
    int64_t dim2;
};

static bool run_case(
    const Shape& shape,
    int64_t seed,
    kc::Metrics& aggregate,
    std::vector<std::string>& case_results
) {
    constexpr double atol = 5e-3;
    constexpr double rtol = 1e-2;
    torch::manual_seed(seed);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    const std::vector<int64_t> normalized = {
        shape.features, shape.dim1, shape.dim2
    };
    auto input = torch::randn(
        {shape.batch, shape.features, shape.dim1, shape.dim2}, options
    );
    auto weight = torch::randn(normalized, options);
    auto bias = torch::randn(normalized, options);
    auto reference = torch::layer_norm(
        input.to(torch::kFloat32), normalized,
        weight.to(torch::kFloat32), bias.to(torch::kFloat32), 1e-5, true
    );
    auto output = torch::empty_like(input);
    torch::Tensor first;
    bool deterministic = true;
    for (int repeat = 0; repeat < 5; ++repeat) {
        launch_gpu_implementation(
            output.data_ptr(), input.data_ptr(), weight.data_ptr(), bias.data_ptr(),
            shape.batch, shape.features, shape.dim1, shape.dim2
        );
        if (repeat == 0) first = output.clone();
        else deterministic = deterministic && torch::equal(first, output);
    }
    const kc::Metrics metrics = kc::summarize(reference, output, atol, rtol);
    kc::merge_envelope(aggregate, metrics);
    case_results.push_back(kc::case_json(
        shape.case_id,
        seed,
        "{\"B\":" + std::to_string(shape.batch)
            + ",\"F\":" + std::to_string(shape.features)
            + ",\"D1\":" + std::to_string(shape.dim1)
            + ",\"D2\":" + std::to_string(shape.dim2) + "}",
        metrics,
        deterministic
    ));
    return deterministic && metrics.nonfinite_count == 0
        && metrics.mismatch_count == 0 && metrics.normalized_max <= 1.0;
}

int main() {
    const std::vector<Shape> shapes = {
        {"canonical", 16, 64, 256, 256},
        {"boundary-1x3x5x7", 1, 3, 5, 7},
        {"odd-2x7x9x11", 2, 7, 9, 11},
        {"neighbor-3x16x17x18", 3, 16, 17, 18},
    };
    const std::vector<int64_t> seeds = {0, 42, 20260721};
    kc::Metrics aggregate;
    std::vector<std::string> case_results;
    bool passed = true;
    for (const Shape& shape : shapes) {
        for (const int64_t seed : seeds) {
            passed = run_case(shape, seed, aggregate, case_results) && passed;
        }
    }
    const bool finite = aggregate.nonfinite_count == 0;
    std::cout << "KERNELBLASTER_CORRECTNESS_JSON "
              << kc::result_json(aggregate, finite, passed, case_results)
              << std::endl;
    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
