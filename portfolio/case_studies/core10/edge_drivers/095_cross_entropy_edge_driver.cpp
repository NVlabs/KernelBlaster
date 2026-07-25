// SPDX-FileCopyrightText: Copyright (c) 2026 KernelBlaster contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/torch.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "correctness_metrics.h"
#include "cuda_model.cuh"

void launch_gpu_implementation(void*, void*, void*, int64_t, int64_t);

namespace kc = kernelblaster::correctness;

struct Shape {
    const char* case_id;
    int64_t batch;
    int64_t classes;
};

static bool run_case(
    const Shape& shape,
    int64_t seed,
    kc::Metrics& aggregate,
    std::vector<std::string>& case_results
) {
    constexpr double atol = 1e-2;
    constexpr double rtol = 1e-2;
    torch::manual_seed(seed);
    auto half_options = torch::TensorOptions()
        .dtype(torch::kFloat16).device(torch::kCUDA);
    auto target_options = torch::TensorOptions()
        .dtype(torch::kLong).device(torch::kCUDA);
    auto predictions = torch::randn(
        {shape.batch, shape.classes}, half_options
    );
    auto targets = torch::randint(
        0, shape.classes, {shape.batch}, target_options
    );
    auto reference = torch::nn::functional::cross_entropy(
        predictions.to(torch::kFloat32), targets
    );
    auto output = torch::empty({1}, half_options);
    torch::Tensor first;
    bool deterministic = true;
    for (int repeat = 0; repeat < 5; ++repeat) {
        launch_gpu_implementation(
            output.data_ptr(), predictions.data_ptr(), targets.data_ptr(),
            shape.batch, shape.classes
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
            + ",\"classes\":" + std::to_string(shape.classes) + "}",
        metrics,
        deterministic
    ));
    return deterministic && metrics.nonfinite_count == 0
        && metrics.mismatch_count == 0 && metrics.normalized_max <= 1.0;
}

int main() {
    const std::vector<Shape> shapes = {
        {"canonical", 4096, 10},
        {"boundary-1x2", 1, 2},
        {"odd-17x7", 17, 7},
        {"neighbor-65x9", 65, 9},
        {"neighbor-257x10", 257, 10},
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
