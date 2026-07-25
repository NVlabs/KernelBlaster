/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/torch.h>
#include <iostream>
#include "cuda_model.cuh"

// Declaration for the GPU implementation of GELU.
// The inputs and outputs are raw pointers to GPU memory (float16).
void launch_gpu_implementation(
    void* output,         // Output tensor (float16), shape: [batch_size, dim]
    void* input,          // Input tensor (float16), shape: [batch_size, dim]
    int64_t batch_size,   // Batch size (16)
    int64_t dim           // Feature dimension (16384)
);

int main() {
    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Data type: float16
    torch::Dtype dtype = torch::kFloat16;
    float rtol = 1e-1f, atol = 1e-1f;

    int64_t batch_size = 16;
    int64_t dim = 16384;

    // Generate input tensor on CUDA with float16
    torch::Tensor input = torch::randn({batch_size, dim}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference output using libtorch GELU
    torch::Tensor ref_output = torch::nn::functional::gelu(input);

    // Allocate output tensor for GPU kernel (float16, CUDA)
    torch::Tensor output = torch::empty_like(ref_output);

    // Call GPU implementation (pass raw pointers)
    launch_gpu_implementation(
        output.data_ptr(),
        input.data_ptr(),
        batch_size,
        dim
    );

    // Compare outputs
    bool is_close = torch::allclose(output, ref_output, rtol, atol);

    if (is_close) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
