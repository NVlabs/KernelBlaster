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
// File: cuda_model.cu

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

// CUDA kernel for upper-triangular matrix multiplication (C = triu(A @ B)), fp16 I/O, fp32 accumulation
__global__ void upper_triangular_matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int64_t N)
{
    constexpr int TILE_SIZE = 16;
    int tile_i = blockIdx.y * TILE_SIZE;
    int tile_j = blockIdx.x * TILE_SIZE;
    int local_i = threadIdx.y;
    int local_j = threadIdx.x;
    int i = tile_i + local_i;
    int j = tile_j + local_j;

    if (i < N && j < N && i <= j) {
        float acc = 0.0f;
        int k_start = i;
        int k_end = j;
        for (int k = k_start; k <= k_end; ++k) {
            float a = __half2float(A[i * N + k]);
            float b = __half2float(B[k * N + j]);
            acc += a * b;
        }
        C[i * N + j] = __float2half(acc);
    }
}

// Ensure C++ linkage so the function is visible to main.cpp
extern "C"
void launch_gpu_implementation(
    void* output,         // (fp16) pointer to output tensor, shape (N, N)
    void* input_A,        // (fp16) pointer to input A, shape (N, N)
    void* input_B,        // (fp16) pointer to input B, shape (N, N)
    int64_t N                 // matrix dimension
) {
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    int num_blocks_x = (N + TILE_SIZE - 1) / TILE_SIZE;
    int num_blocks_y = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(num_blocks_x, num_blocks_y);

    const half* A = static_cast<const half*>(input_A);
    const half* B = static_cast<const half*>(input_B);
    half* C = static_cast<half*>(output);

    upper_triangular_matmul_kernel<<<grid, block>>>(A, B, C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        assert(false);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution error: %s\n", cudaGetErrorString(err));
        assert(false);
    }
}
