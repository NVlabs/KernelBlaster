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
// cuda_model.cuh

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cassert>
#include <cstdio>

// --- Lower Triangular MatMul Kernel ---
// Computes C = tril(A * B), where A and B are lower triangular matrices.
// Input/output: fp16, accumulation: fp32 for numerical stability.

#define BLOCK_SIZE 32

// Utility: check CUDA error
inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

/**
 * Kernel for lower triangular matrix multiplication: C = tril(A * B)
 * Each thread computes one output element C[i, j] for i >= j.
 * Accumulation is done in FP32, result is stored as FP16.
 * 
 * Only computes the lower triangle (i >= j) of C; upper triangle is left zero.
 */
__global__ void tril_matmul_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute lower triangle
    if (row >= N || col >= N || row < col)
        return;

    float acc = 0.0f;
    // For lower triangular A and B, the valid range for k is:
    //   k in [col, row]
    //   because:
    //     A[row, k] nonzero for k <= row
    //     B[k, col] nonzero for col <= k
    //     So k >= col and k <= row (and k < N)
    for (int k = col; k <= row; ++k) {
        // A[row, k] is nonzero for k <= row (guaranteed)
        // B[k, col] is nonzero for col <= k (guaranteed)
        float a_fp32 = __half2float(A[row * N + k]);
        float b_fp32 = __half2float(B[k * N + col]);
        acc += a_fp32 * b_fp32;
    }
    // Store result as half
    C[row * N + col] = __float2half(acc);
}

/**
 * Host function to launch the lower triangular matrix multiplication kernel.
 * 
 * @param output   Pointer to output buffer (fp16), shape (N, N)
 * @param input_A  Pointer to input A (fp16), shape (N, N), lower-triangular
 * @param input_B  Pointer to input B (fp16), shape (N, N), lower-triangular
 * @param N        Matrix size (N x N)
 */
void launch_gpu_implementation(void* output, void* input_A, void* input_B, int64_t N) {
    // Cast to half pointers
    half* out = static_cast<half*>(output);
    const half* A = static_cast<const half*>(input_A);
    const half* B = static_cast<const half*>(input_B);

    // Use 2D grid/block for better occupancy and memory coalescing
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    tril_matmul_fp16_kernel<<<grid, block>>>(A, B, out, N);

    checkCuda(cudaGetLastError(), "Kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "Kernel sync failed");
}

