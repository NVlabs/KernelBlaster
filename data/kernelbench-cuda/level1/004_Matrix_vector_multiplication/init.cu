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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>

// Matrix-vector multiplication kernel: C = A * B
// A: (M, K) [half], B: (K, 1) [half], C: (M, 1) [half]
// Accumulation is done in fp32 for numerical stability.

#ifndef CHECK_CUDA
#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif

// Use 256 threads per block for good occupancy and memory coalescing
constexpr int THREADS_PER_BLOCK = 256;

// Kernel: Each thread computes one output element (row of A * B)
__global__ void matvec_fp16_acc_fp32(
    const half* __restrict__ A, // [M, K]
    const half* __restrict__ B, // [K, 1]
    half* __restrict__ C,       // [M, 1]
    int M, int K)
{
    // Each thread computes one output row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Accumulate in fp32 for numerical stability
    float acc = 0.0f;

    // Loop over K dimension with vectorized loads for B
    // Unroll by 8 for better performance
    int k = 0;
#if __CUDA_ARCH__ >= 530
    for (; k <= K - 8; k += 8) {
        // Load 8 elements of A and B
        half a_vals[8], b_vals[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_vals[i] = A[row * K + k + i];
            b_vals[i] = B[k + i];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc += __half2float(a_vals[i]) * __half2float(b_vals[i]);
        }
    }
#endif
    // Handle remainder
    for (; k < K; ++k) {
        acc += __half2float(A[row * K + k]) * __half2float(B[k]);
    }

    // Write result in fp16
    C[row] = __float2half(acc);
}

// Host launcher
void launch_gpu_implementation(void* output, void* input_A, void* input_B, int M, int K)
{
    // All pointers are device pointers, type: half (fp16)
    half* d_A = static_cast<half*>(input_A);
    half* d_B = static_cast<half*>(input_B);
    half* d_C = static_cast<half*>(output);

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    matvec_fp16_acc_fp32<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, M, K
    );

    // Synchronize to ensure completion
    CHECK_CUDA(cudaDeviceSynchronize());
}
