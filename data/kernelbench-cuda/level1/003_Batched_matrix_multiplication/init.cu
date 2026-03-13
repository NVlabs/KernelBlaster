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
#include <stdio.h>
#include <assert.h>

// Utility for checking CUDA errors
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        assert(0); \
    }

// Kernel: compute C[b, i, j] = sum_k A[b, i, k] * B[b, k, j], fp16 input/output, fp32 accum
__global__ void batched_gemm_fp16_ref_kernel(
    const half* __restrict__ A, // [batch, m, k]
    const half* __restrict__ B, // [batch, k, n]
    half* __restrict__ C,       // [batch, m, n]
    int batch_size, int m, int k, int n
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float acc = 0.f;
        // fp32 accumulation for best numerical stability
        for (int kk = 0; kk < k; ++kk) {
            half a = A[b * m * k + row * k + kk];
            half b_val = B[b * k * n + kk * n + col];
            acc += __half2float(a) * __half2float(b_val);
        }
        C[b * m * n + row * n + col] = __float2half(acc);
    }
}

// Host launcher
void launch_gpu_implementation(void* output, void* A, void* B,
                               int batch_size, int m, int k, int n) {
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (m + threads.y - 1) / threads.y,
                batch_size);

    batched_gemm_fp16_ref_kernel<<<blocks, threads>>>(
        static_cast<const half*>(A),
        static_cast<const half*>(B),
        static_cast<half*>(output),
        batch_size, m, k, n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
