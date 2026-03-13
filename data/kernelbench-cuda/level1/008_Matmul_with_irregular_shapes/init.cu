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
#include <algorithm>
#include <cstdio>

// Block tile sizes
#define TILE_M 128
#define TILE_N 128
#define TILE_K 8

// Utility for ceil-div
inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

// CUDA kernel: C = A x B, A[M,K], B[K,N], C[M,N], all half, accumulate in float
__global__ void matmul_fp16_accum_fp32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K, int N)
{
    // Block origin
    int row0 = blockIdx.y * TILE_M;
    int col0 = blockIdx.x * TILE_N;

    // Thread origin in block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int local_row = tid / TILE_N;
    int local_col = tid % TILE_N;

    // Each thread computes multiple C elements (thread striding)
    for (int row = row0 + threadIdx.y; row < min(row0 + TILE_M, M); row += blockDim.y) {
        for (int col = col0 + threadIdx.x; col < min(col0 + TILE_N, N); col += blockDim.x) {
            float acc = 0.0f;
            for (int kb = 0; kb < K; kb += TILE_K) {
                // Unroll K by 8 for fp16 vectorization
                #pragma unroll
                for (int k = 0; k < TILE_K; ++k) {
                    int kk = kb + k;
                    if (kk < K) {
                        half a = A[row * K + kk];
                        half b = B[kk * N + col];
                        acc += __half2float(a) * __half2float(b);
                    }
                }
            }
            // Write result as half
            C[row * N + col] = __float2half(acc);
        }
    }
}

// Host launcher, with safe grid/block sizing for large/irregular shapes
void launch_gpu_implementation(
    void* output,
    void* input_a,
    void* input_b,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // Use 16x16 threads per block for good occupancy and coalescing
    dim3 blockDim(16, 16);
    dim3 gridDim(iDivUp(N, TILE_N), iDivUp(M, TILE_M));
    matmul_fp16_accum_fp32_kernel<<<gridDim, blockDim>>>(
        static_cast<const half*>(input_a),
        static_cast<const half*>(input_b),
        static_cast<half*>(output),
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(N)
    );
    cudaDeviceSynchronize();
}
