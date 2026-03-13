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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>
#include <iostream>

// Tile sizes chosen for good balance on small-to-medium matrices
#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

// Kernel: C[M,N] = A[M,K] * B[N,K]^T + bias[N]; half inputs/outputs, fp32 accumulation.
// If ApplyRelu is true, apply ReLU after bias addition.
template <bool ApplyRelu>
__global__ void gemm_bias_act_fp16_fp32acc(
    const half* __restrict__ A,    // [M, K], row-major
    const half* __restrict__ B,    // [N, K], row-major
    const half* __restrict__ bias, // [N]
    half* __restrict__ C,          // [M, N], row-major
    int M, int N, int K
) {
    __shared__ half As[TILE_M][TILE_K];
    __shared__ half Bs[TILE_N][TILE_K];

    const int block_m0 = blockIdx.y * TILE_M;
    const int block_n0 = blockIdx.x * TILE_N;

    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15

    // Each thread computes a 4x4 output sub-block
    float acc[4][4] = {0.0f};

    // Loop over K dimension tiles
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load of A tile [TILE_M x TILE_K]
        for (int idx = ty * blockDim.x + tx; idx < TILE_M * TILE_K; idx += blockDim.x * blockDim.y) {
            int a_row = idx / TILE_K;         // 0..TILE_M-1
            int a_col = idx % TILE_K;         // 0..TILE_K-1
            int g_row = block_m0 + a_row;     // global row in A
            int g_col = k0 + a_col;           // global col in A
            if (g_row < M && g_col < K) {
                As[a_row][a_col] = A[g_row * K + g_col];
            } else {
                As[a_row][a_col] = __float2half(0.0f);
            }
        }

        // Cooperative load of B tile [TILE_N x TILE_K]
        for (int idx = ty * blockDim.x + tx; idx < TILE_N * TILE_K; idx += blockDim.x * blockDim.y) {
            int b_row = idx / TILE_K;         // 0..TILE_N-1  -> corresponds to N dimension
            int b_col = idx % TILE_K;         // 0..TILE_K-1  -> corresponds to K dimension
            int g_row = block_n0 + b_row;     // global row in B (i.e., n)
            int g_col = k0 + b_col;           // global col in B (i.e., k)
            if (g_row < N && g_col < K) {
                Bs[b_row][b_col] = B[g_row * K + g_col];
            } else {
                Bs[b_row][b_col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute on the loaded tiles
        // Thread computes 4 rows (stride 16) x 4 cols (stride 16)
        // Rows: ty + i*16, Cols: tx + j*16
        for (int kk = 0; kk < TILE_K; ++kk) {
            float aFrag[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int a_r = ty + i * 16;
                aFrag[i] = __half2float(As[a_r][kk]);
            }

            float bFrag[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int b_r = tx + j * 16;
                bFrag[j] = __half2float(Bs[b_r][kk]);
            }

#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += aFrag[i] * bFrag[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back results with bias and optional ReLU
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m = block_m0 + ty + i * 16;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int n = block_n0 + tx + j * 16;
            if (n >= N) continue;

            float val = acc[i][j] + __half2float(bias[n]);
            if (ApplyRelu) {
                val = val > 0.0f ? val : 0.0f;
            }
            C[m * N + n] = __float2half_rn(val);
        }
    }
}

// Helper: ceil div
static inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

void launch_one_gemm_with_bias_act(
    const half* A, const half* B, const half* bias, half* C,
    int M, int N, int K, bool relu
) {
    dim3 block(16, 16);
    dim3 grid(ceil_div_int(N, TILE_N), ceil_div_int(M, TILE_M));
    if (relu) {
        gemm_bias_act_fp16_fp32acc<true><<<grid, block>>>(A, B, bias, C, M, N, K);
    } else {
        gemm_bias_act_fp16_fp32acc<false><<<grid, block>>>(A, B, bias, C, M, N, K);
    }
}

// Public API: launches three GEMMs corresponding to the MLP: ReLU after first two, none after third.
void launch_gpu_implementation(
    void* output,
    void* input,
    void* w1,
    void* b1,
    void* w2,
    void* b2,
    void* w3,
    void* b3,
    int64_t batch_size,
    int64_t input_size,
    int64_t hidden_size1,
    int64_t hidden_size2,
    int64_t output_size
) {
    // Cast to half*
    const half* A0 = static_cast<const half*>(input);  // [B, I]
    const half* W1 = static_cast<const half*>(w1);     // [H1, I]
    const half* B1 = static_cast<const half*>(b1);     // [H1]

    const half* W2 = static_cast<const half*>(w2);     // [H2, H1]
    const half* B2 = static_cast<const half*>(b2);     // [H2]

    const half* W3 = static_cast<const half*>(w3);     // [O, H2]
    const half* B3 = static_cast<const half*>(b3);     // [O]

    half* Yout = static_cast<half*>(output);           // [B, O]

    // Dimensions (ensure they fit in int for kernels)
    int M0 = static_cast<int>(batch_size);
    int K0 = static_cast<int>(input_size);
    int N0 = static_cast<int>(hidden_size1);

    int M1 = static_cast<int>(batch_size);
    int K1 = static_cast<int>(hidden_size1);
    int N1 = static_cast<int>(hidden_size2);

    int M2 = static_cast<int>(batch_size);
    int K2 = static_cast<int>(hidden_size2);
    int N2 = static_cast<int>(output_size);

    assert(M0 >= 0 && K0 >= 0 && N0 >= 0);
    assert(M1 >= 0 && K1 >= 0 && N1 >= 0);
    assert(M2 >= 0 && K2 >= 0 && N2 >= 0);

    // Allocate intermediates
    half* Y1 = nullptr;
    half* Y2 = nullptr;
    size_t Y1_bytes = static_cast<size_t>(M0) * static_cast<size_t>(N0) * sizeof(half);
    size_t Y2_bytes = static_cast<size_t>(M1) * static_cast<size_t>(N1) * sizeof(half);

    cudaMalloc(&Y1, Y1_bytes);
    cudaMalloc(&Y2, Y2_bytes);

    // Layer 1: [B, I] x [H1, I]^T -> [B, H1], +b1, ReLU
    launch_one_gemm_with_bias_act(A0, W1, B1, Y1, M0, N0, K0, /*relu=*/true);

    // Layer 2: [B, H1] x [H2, H1]^T -> [B, H2], +b2, ReLU
    launch_one_gemm_with_bias_act(Y1, W2, B2, Y2, M1, N1, K1, /*relu=*/true);

    // Layer 3: [B, H2] x [O, H2]^T -> [B, O], +b3, no ReLU
    launch_one_gemm_with_bias_act(Y2, W3, B3, Yout, M2, N2, K2, /*relu=*/false);

    // Cleanup
    cudaFree(Y1);
    cudaFree(Y2);

    // Note: The caller performs cudaDeviceSynchronize() after this function.
}
