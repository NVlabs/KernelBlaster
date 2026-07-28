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

// Declaration for your CUDA implementation
void launch_gpu_implementation(
    void* output, 
    void* input_a, 
    void* input_b, 
    int64_t M, 
    int64_t K, 
    int64_t N
);

int main() {
    // Set device and dtype
    torch::Device device(torch::kCUDA);
    auto dtype = torch::kHalf; // fp16

    // Matrix sizes
    const int64_t M = 8205;
    const int64_t K = 2949;
    const int64_t N = 5921;

    // Generate random input tensors on GPU
    torch::Tensor A = torch::randn({M, K}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor B = torch::randn({K, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Reference libtorch output
    torch::Tensor ref_output = torch::matmul(A, B);

    // Allocate output tensor for CUDA kernel
    torch::Tensor output = torch::empty({M, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Call CUDA kernel (pass raw pointers)
    launch_gpu_implementation(
        output.data_ptr(), 
        A.data_ptr(), 
        B.data_ptr(), 
        M, 
        K, 
        N
    );

    // Compare outputs with relaxed tolerance for fp16
    bool result = torch::allclose(output, ref_output, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (result) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    return 0;
}
