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
#include <stdint.h>
#include <stdio.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 16
#define THREADS_PER_BLOCK 256

inline __device__ __host__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

/*
 * CUDA kernel for C = A^T * B
 *   A: (K, M), B: (K, N), C: (M, N)
 *   Accumulate in FP32, output in FP16.
 */
__global__ void matmul_ATB_fp16_kernel(
    const half *__restrict__ A, // (K, M)
    const half *__restrict__ B, // (K, N)
    half *__restrict__ C,       // (M, N)
    int M, int K, int N
) {
    // Each block computes a (BLOCK_M x BLOCK_N) output tile
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Shared memory for tiles
    __shared__ half As[BLOCK_K][BLOCK_M];
    __shared__ half Bs[BLOCK_K][BLOCK_N];

    // Each thread computes a (tile_m_per_thread x tile_n_per_thread) sub-tile
    // We'll use a 2D thread layout for simplicity
    const int tile_m_per_thread = 8;
    const int tile_n_per_thread = 8;
    const int THREADS_PER_ROW = BLOCK_M / tile_m_per_thread; // 8
    const int THREADS_PER_COL = BLOCK_N / tile_n_per_thread; // 8

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Output tile's top-left corner indices
    int row_start = block_row * BLOCK_M;
    int col_start = block_col * BLOCK_N;

    // Each thread computes a small register tile
    float acc[tile_m_per_thread][tile_n_per_thread] = {0};

    // Loop over K dimension in chunks of BLOCK_K
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Load A^T tile: A is (K, M), but we need A^T (M, K)
        // So for output row m in this block, and for k in k0...k0+BLOCK_K
        for (int i = 0; i < tile_m_per_thread; ++i) {
            int m = row_start + thread_row * tile_m_per_thread + i;
            for (int kk = 0; kk < BLOCK_K; ++kk) {
                int k = k0 + kk;
                if (m < M && k < K) {
                    // A^T[m, k] == A[k, m]
                    As[kk][m - row_start] = A[k * M + m];
                } else if (thread_col == 0) {
                    As[kk][m - row_start] = __float2half(0.0f);
                }
            }
        }
        // Load B tile: (K, N)
        for (int i = 0; i < tile_n_per_thread; ++i) {
            int n = col_start + thread_col * tile_n_per_thread + i;
            for (int kk = 0; kk < BLOCK_K; ++kk) {
                int k = k0 + kk;
                if (n < N && k < K) {
                    Bs[kk][n - col_start] = B[k * N + n];
                } else if (thread_row == 0) {
                    Bs[kk][n - col_start] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // Compute local output sub-tile
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            for (int i = 0; i < tile_m_per_thread; ++i) {
                int m = thread_row * tile_m_per_thread + i;
                for (int j = 0; j < tile_n_per_thread; ++j) {
                    int n = thread_col * tile_n_per_thread + j;
                    acc[i][j] += __half2float(As[kk][m]) * __half2float(Bs[kk][n]);
                }
            }
        }
        __syncthreads();
    }

    // Store results
    for (int i = 0; i < tile_m_per_thread; ++i) {
        int m = row_start + thread_row * tile_m_per_thread + i;
        if (m < M) {
            for (int j = 0; j < tile_n_per_thread; ++j) {
                int n = col_start + thread_col * tile_n_per_thread + j;
                if (n < N) {
                    C[m * N + n] = __float2half(acc[i][j]);
                }
            }
        }
    }
}

/*
 * Host launcher for the kernel.
 *   output: (M, N), float16
 *   input_A: (K, M), float16
 *   input_B: (K, N), float16
 *   M, K, N: sizes
 */
void launch_gpu_implementation(
    void* output,
    void* input_A,
    void* input_B,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // Thread block is (THREADS_PER_ROW, THREADS_PER_COL)
    constexpr int tile_m_per_thread = 8;
    constexpr int tile_n_per_thread = 8;
    constexpr int THREADS_PER_ROW = BLOCK_M / tile_m_per_thread; // 8
    constexpr int THREADS_PER_COL = BLOCK_N / tile_n_per_thread; // 8

    dim3 blockDim(THREADS_PER_ROW, THREADS_PER_COL); // 8x8 = 64 threads per block
    dim3 gridDim(div_up(N, BLOCK_N), div_up(M, BLOCK_M));

    size_t shared_mem_size = (BLOCK_K * BLOCK_M + BLOCK_K * BLOCK_N) * sizeof(half);

    matmul_ATB_fp16_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        static_cast<const half*>(input_A),
        static_cast<const half*>(input_B),
        static_cast<half*>(output),
        int(M), int(K), int(N)
    );
    cudaDeviceSynchronize();
}
