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
#include <cstdio>
#include <cassert>

// CUDA error check macro
#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        assert(0); \
    } \
} while (0)

// Tile sizes for high occupancy and coalescing
constexpr int TM = 8; // Rows per thread tile
constexpr int TN = 8; // Cols per thread tile
constexpr int BK = 8; // K tile

// Kernel: each thread block computes a (blockDim.y*TM x blockDim.x*TN) tile of C
__global__ void matmul_tall_skinny_fp16_fp32acc_kernel(
    const half* __restrict__ A, // (M, N)
    const half* __restrict__ B, // (N, M)
    half* __restrict__ C,       // (M, M)
    int M, int N
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Each thread computes a TM x TN tile
    int row = by * blockDim.y * TM + ty * TM;
    int col = bx * blockDim.x * TN + tx * TN;

    float accum[TM][TN] = {0.0f};

    // Loop over K dimension in chunks of BK
    for (int k0 = 0; k0 < N; k0 += BK) {
        // Load A and B tiles into registers
        half A_tile[TM][BK];
        half B_tile[BK][TN];

        // Load A_tile: rows = row+i, cols = k0+k
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            int r = row + i;
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                int kk = k0 + k;
                if (r < M && kk < N) {
                    A_tile[i][k] = A[r * N + kk];
                } else {
                    A_tile[i][k] = __float2half(0.0f);
                }
            }
        }

        // Load B_tile: rows = k0+k, cols = col+j
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            int kk = k0 + k;
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = col + j;
                if (kk < N && c < M) {
                    B_tile[k][j] = B[kk * M + c];
                } else {
                    B_tile[k][j] = __float2half(0.0f);
                }
            }
        }

        // Compute local accumulators
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    accum[i][j] += __half2float(A_tile[i][k]) * __half2float(B_tile[k][j]);
                }
            }
        }
    }

    // Write results to output C
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = col + j;
                if (c < M) {
                    C[r * M + c] = __float2half(accum[i][j]);
                }
            }
        }
    }
}

// Host launcher (C linkage for compatibility with C++)
extern "C"
void launch_gpu_implementation(
    void* output,           // Output: (M, M), at::Half*
    void* input_A,          // Input: (M, N), at::Half*
    void* input_B,          // Input: (N, M), at::Half*
    int64_t M,              // Rows of A, cols of B, output rows/cols
    int64_t N               // Cols of A, rows of B
) {
    const half* d_A = static_cast<const half*>(input_A);
    const half* d_B = static_cast<const half*>(input_B);
    half* d_C = static_cast<half*>(output);

    // Tune block dimensions for high occupancy
    constexpr int block_x = 8; // threads in x
    constexpr int block_y = 8; // threads in y

    dim3 blockDim(block_x, block_y, 1);
    dim3 gridDim(
        (M + block_x * TN - 1) / (block_x * TN),
        (M + block_y * TM - 1) / (block_y * TM),
        1
    );

    matmul_tall_skinny_fp16_fp32acc_kernel<<<gridDim, blockDim>>>(
        d_A, d_B, d_C, static_cast<int>(M), static_cast<int>(N)
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
