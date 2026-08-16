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

// Declaration for the CUDA implementation
void launch_gpu_implementation(
    void* output,           // Output tensor pointer, shape (N, N), fp16, CUDA
    void* input_A,          // Input A tensor pointer, shape (N, N), fp16, CUDA
    void* input_B,          // Input B tensor pointer, shape (N, N), fp16, CUDA
    int64_t N               // Matrix size
);

int main() {
    torch::Dtype dtype = torch::kHalf;
    torch::Device device(torch::kCUDA);

    const int64_t N = 4096;

    // Generate upper triangular input matrices on GPU
    torch::Tensor A = torch::triu(torch::randn({N, N}, torch::TensorOptions().dtype(dtype).device(device)));
    torch::Tensor B = torch::triu(torch::randn({N, N}, torch::TensorOptions().dtype(dtype).device(device)));

    // Reference output using libtorch: C_ref = triu(matmul(A, B))
    torch::Tensor C_ref = torch::triu(torch::matmul(A, B));

    // Allocate output tensor for CUDA kernel and fill with a value that will never match the reference
    torch::Tensor C_cuda = torch::zeros({N, N}, torch::TensorOptions().dtype(dtype).device(device));

    // Synchronize before calling the kernel (just in case)
    torch::cuda::synchronize();

    // Launch the CUDA implementation
    launch_gpu_implementation(
        C_cuda.data_ptr(),    // output
        A.data_ptr(),         // input_A
        B.data_ptr(),         // input_B
        N                     // size
    );

    // Synchronize after kernel launch to ensure all operations are completed
    torch::cuda::synchronize();

    // Compare outputs using torch::allclose (rtol=1e-1, atol=1e-1 for fp16)
    bool passed = torch::allclose(C_cuda, C_ref, /*rtol=*/1e-1, /*atol=*/1e-1);

    if (passed) {
        std::cout << "passed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
        // Optionally print max absolute difference for debugging
        auto max_diff = (C_cuda - C_ref).abs().max().item<float>();
        std::cout << "Max absolute difference: " << max_diff << std::endl;
    }

    return 0;
}