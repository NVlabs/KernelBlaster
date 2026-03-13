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
// 3D Tensor-Matrix Multiplication (fp16 I/O, fp32 accumulation)
// Performs: output[n, m, l] = sum_{k} A[n, m, k] * B[k, l]
// Input:  A (N, M, K) half, B (K, L) half
// Output: output (N, M, L) half

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>

// CUDA kernel: Each thread computes one output element (n, m, l)
__global__ void tensor3d_matmul_fp16_kernel(
    const half* __restrict__ A,  // (N, M, K)
    const half* __restrict__ B,  // (K, L)
    half* __restrict__ output,   // (N, M, L)
    int64_t N, int64_t M, int64_t K, int64_t L
) {
    // Flattened thread index for (n, m, l)
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * M * L;
    if (tid >= total) return;

    // Compute output indices
    int64_t l = tid % L;
    int64_t m = (tid / L) % M;
    int64_t n = tid / (M * L);

    // Compute dot product: output[n, m, l] = sum_{k} A[n, m, k] * B[k, l]
    float acc = 0.0f;
    int64_t a_base = n * M * K + m * K;
    int64_t b_base = l;
    for (int64_t k = 0; k < K; ++k) {
        float a_val = __half2float(A[a_base + k]);
        float b_val = __half2float(B[k * L + b_base]);
        acc += a_val * b_val;
    }
    // Store as fp16
    output[tid] = __float2half(acc);
}

// Host launcher
void launch_gpu_implementation(
    void* output,             // (N, M, L) - fp16, GPU
    void* input_A,            // (N, M, K) - fp16, GPU
    void* input_B,            // (K, L)    - fp16, GPU
    int64_t N,
    int64_t M,
    int64_t K,
    int64_t L
) {
    // Each thread computes one (n, m, l)
    int64_t total = N * M * L;
    int threads_per_block = 256;
    int num_blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);

    tensor3d_matmul_fp16_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const half*>(input_A),
        static_cast<const half*>(input_B),
        static_cast<half*>(output),
        N, M, K, L
    );
    cudaDeviceSynchronize();
}

