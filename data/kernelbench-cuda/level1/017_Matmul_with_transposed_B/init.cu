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
#include <assert.h>

// Tune these for best performance on large matrices
#define BM 64   // block rows
#define BN 64   // block cols
#define BK 32   // block depth

inline int ceil_div(int n, int d) { return (n + d - 1) / d; }

// Kernel for C = A * B^T, where:
//   - A: (M, K), row-major
//   - B: (N, K), row-major, but we want B^T (i.e., for C[i, j]: sum_k A[i,k] * B[j,k])
//   - C: (M, N), row-major
__global__ void matmul_fp16t_fp32acc(
    const half* __restrict__ A, // (M, K)
    const half* __restrict__ B, // (N, K)
    half* __restrict__ C,       // (M, N)
    int M, int K, int N
) {
    // Block indices
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Thread indices
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Each thread computes a tile of (TM, TN)
    const int TM = 8;
    const int TN = 8;

    int row = block_row + thread_row * TM;
    int col = block_col + thread_col * TN;

    float acc[TM][TN] = {0};

    // Loop over K in chunks of BK
    for (int bk = 0; bk < K; bk += BK) {
        // Shared memory for A and B tiles
        __shared__ half As[BM][BK];
        __shared__ half Bs[BN][BK];

        // Each thread loads TM x BK for A and TN x BK for B
        for (int i = 0; i < TM; ++i) {
            int ai = row + i;
            for (int kk = 0; kk < BK; ++kk) {
                int ak = bk + kk;
                if (ai < M && ak < K)
                    As[thread_row * TM + i][kk] = A[ai * K + ak];
                else
                    As[thread_row * TM + i][kk] = __float2half(0.f);
            }
        }
        for (int j = 0; j < TN; ++j) {
            int bj = col + j;
            for (int kk = 0; kk < BK; ++kk) {
                int bk_ = bk + kk;
                if (bj < N && bk_ < K)
                    Bs[thread_col * TN + j][kk] = B[bj * K + bk_];
                else
                    Bs[thread_col * TN + j][kk] = __float2half(0.f);
            }
        }
        __syncthreads();

        // Compute local tile
        for (int kk = 0; kk < BK; ++kk) {
            half a_frag[TM];
            half b_frag[TN];
            for (int i = 0; i < TM; ++i)
                a_frag[i] = As[thread_row * TM + i][kk];
            for (int j = 0; j < TN; ++j)
                b_frag[j] = Bs[thread_col * TN + j][kk];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += __half2float(a_frag[i]) * __half2float(b_frag[j]);
        }
        __syncthreads();
    }

    // Write results
    for (int i = 0; i < TM; ++i) {
        int ai = row + i;
        if (ai < M) {
            for (int j = 0; j < TN; ++j) {
                int bj = col + j;
                if (bj < N) {
                    C[ai * N + bj] = __float2half(acc[i][j]);
                }
            }
        }
    }
}

void launch_gpu_implementation(
    void* output,      // output: (M, N), fp16
    void* input_A,     // input: (M, K), fp16
    void* input_B,     // input: (N, K), fp16
    int64_t M,
    int64_t K,
    int64_t N
) {
    dim3 threads(BN / 8, BM / 8); // (8, 8)
    dim3 blocks(ceil_div(N, BN), ceil_div(M, BM));
    size_t smem_bytes = (BM * BK + BN * BK) * sizeof(half);

    matmul_fp16t_fp32acc<<<blocks, threads, smem_bytes>>>(
        static_cast<const half*>(input_A),
        static_cast<const half*>(input_B),
        static_cast<half*>(output),
        M, K, N
    );
    cudaDeviceSynchronize();
}
