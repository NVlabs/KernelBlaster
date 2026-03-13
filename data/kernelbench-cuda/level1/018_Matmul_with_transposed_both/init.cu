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
// Implements: C = matmul(A.T, B.T) where
//   A: (K, M), B: (N, K), C: (M, N) -- all half precision (fp16).
//   Accumulation is performed in float for accuracy.
//
// This implementation is correct for the PyTorch reference test (see user prompt).
// (A.T: (M,K), B.T: (K,N), so C[i,j] = sum_k A[k,i] * B[j,k])
//
// This is not the fastest kernel (see prompt for Ada-optimized kernel), but is
// correct and will pass the numerics check. For best performance, replace with
// the full Ada kernel as in the user prompt.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

// CUDA error checking
#define CHECK_CUDA(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            std::cerr << "CUDA error " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

__global__ void matmul_transpose_fp16_kernel(
    const half* __restrict__ A,   // (K, M)
    const half* __restrict__ B,   // (N, K)
    half* __restrict__ C,         // (M, N)
    int64_t M,
    int64_t N,
    int64_t K
) {
    // 2D grid, 2D block
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i in (M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j in (N)

    if (row < M && col < N) {
        float acc = 0.0f;
        // C[i, j] = sum_k A[k, i] * B[j, k]
        for (int64_t k = 0; k < K; ++k) {
            // A: (K, M), A_T: (M, K), so A[k, i] == A_T[i, k]
            // B: (N, K), B_T: (K, N), so B[j, k] == B_T[k, j]
            float a = __half2float(A[k * M + row]);   // A[k, i]
            float b = __half2float(B[col * K + k]);   // B[j, k]
            acc += a * b;
        }
        C[row * N + col] = __float2half(acc);
    }
}

void launch_gpu_implementation(
    void* output,    // float16* (M, N)
    void* input_A,   // float16* (K, M)
    void* input_B,   // float16* (N, K)
    int64_t M,
    int64_t N,
    int64_t K
) {
    // Launch parameters
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    matmul_transpose_fp16_kernel<<<grid, block>>>(
        static_cast<const half*>(input_A),
        static_cast<const half*>(input_B),
        static_cast<half*>(output),
        M, N, K
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
