// SPDX-FileCopyrightText: Copyright (c) 2026 KernelBlaster contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>

#include "correctness_metrics.h"
#include "cuda_model.cuh"

void launch_gpu_implementation(void*, void*, void*, int64_t, int64_t, int64_t);

namespace kc = kernelblaster::correctness;

struct Shape {
    const char* case_id;
    int64_t m;
    int64_t n;
    int64_t k;
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
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto a = torch::randn({shape.m, shape.k}, options);
    auto b = torch::randn({shape.k, shape.n}, options);
    auto reference = torch::matmul(a.to(torch::kFloat32), b.to(torch::kFloat32));
    auto output = torch::empty({shape.m, shape.n}, options);
    torch::Tensor first;
    bool deterministic = true;
    for (int repeat = 0; repeat < 5; ++repeat) {
        launch_gpu_implementation(
            output.data_ptr(), a.data_ptr(), b.data_ptr(),
            shape.m, shape.n, shape.k
        );
        if (repeat == 0) first = output.clone();
        else deterministic = deterministic && torch::equal(first, output);
    }
    const kc::Metrics metrics = kc::summarize(reference, output, atol, rtol);
    kc::merge_envelope(aggregate, metrics);
    case_results.push_back(kc::case_json(
        shape.case_id,
        seed,
        "{\"M\":" + std::to_string(shape.m)
            + ",\"N\":" + std::to_string(shape.n)
            + ",\"K\":" + std::to_string(shape.k) + "}",
        metrics,
        deterministic
    ));
    return deterministic && metrics.nonfinite_count == 0
        && metrics.mismatch_count == 0 && metrics.normalized_max <= 1.0;
}

int main() {
    const std::vector<Shape> shapes = {
        {"boundary-1x1x1", 1, 1, 1},
        {"odd-17x19x7", 17, 19, 7},
        {"neighbor-65x33x31", 65, 33, 31},
    };
    const std::vector<int64_t> seeds = {0, 42, 20260721};
    kc::Metrics aggregate;
    std::vector<std::string> case_results;
    bool passed = true;
    for (const Shape& shape : shapes) {
        for (const int64_t seed : seeds) {
            passed = run_case(
                shape, seed, aggregate, case_results
            ) && passed;
        }
    }
    const bool finite = aggregate.nonfinite_count == 0;
    std::cout << "KERNELBLASTER_CORRECTNESS_JSON "
              << kc::result_json(aggregate, finite, passed, case_results)
              << std::endl;
    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
