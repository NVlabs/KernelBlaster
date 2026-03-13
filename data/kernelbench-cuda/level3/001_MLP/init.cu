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
#include <cstdio>

// Tile configuration
constexpr int TN_REG = 4;            // Each thread computes 4 output columns
constexpr int THREADS_X = 32;        // Threads along N dimension
constexpr int THREADS_Y = 8;         // Threads along M dimension
constexpr int BLOCK_THREADS = THREADS_X * THREADS_Y;

constexpr int BM = THREADS_Y;                 // Tile size along M (rows)
constexpr int BN = THREADS_X * TN_REG;        // Tile size along N (cols)
constexpr int BK = 32;                        // Tile size along K

template <bool ApplyReLU>
__global__ void linear_bias_relu_kernel(
    const half* __restrict__ X,   // [M, K]
    const half* __restrict__ W,   // [N, K] (row-major) -> Y = X * W^T + b
    const half* __restrict__ b,   // [N]
    half* __restrict__ Y,         // [M, N]
    int64_t M, int64_t N, int64_t K
) {
    // Block tile origin
    const int64_t m0 = blockIdx.y * BM;
    const int64_t n0 = blockIdx.x * BN;

    // Thread indices
    const int tx = threadIdx.x; // [0, THREADS_X)
    const int ty = threadIdx.y; // [0, THREADS_Y)

    // Output coordinates computed by this thread
    const int64_t m = m0 + ty;
    const int64_t n_base = n0 + tx * TN_REG;

    // Shared memory tiles
    __shared__ half Asm[BM][BK];     // Tile from X: [BM, BK]
    __shared__ half Bsm[BK][BN];     // Tile from W: [BK, BN] (note: transposed for compute convenience)

    // Register accumulators in FP32 for numerical stability
    float acc[TN_REG] = {0.f, 0.f, 0.f, 0.f};

    // Iterate over K dimension in tiles
    for (int64_t k0 = 0; k0 < K; k0 += BK) {
        // Load A tile (X)
        int tid = ty * blockDim.x + tx;
        int A_elems = BM * BK;
        for (int idx = tid; idx < A_elems; idx += BLOCK_THREADS) {
            int row = idx / BK;
            int col = idx % BK;
            int64_t gm = m0 + row;
            int64_t gk = k0 + col;
            half val = __float2half(0.0f);
            if (gm < M && gk < K) {
                val = X[gm * K + gk];
            }
            Asm[row][col] = val;
        }

        // Load B tile (W)
        int B_elems = BK * BN;
        for (int idx = tid; idx < B_elems; idx += BLOCK_THREADS) {
            int row = idx / BN;       // k within tile
            int col = idx % BN;       // n within tile
            int64_t gk = k0 + row;
            int64_t gn = n0 + col;
            half val = __float2half(0.0f);
            if (gn < N && gk < K) {
                val = W[gn * K + gk]; // W is [N,K] row-major
            }
            Bsm[row][col] = val;
        }

        __syncthreads();

        // Compute partial products for this tile if m is in bounds
        if (m < M) {
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float a_val = __half2float(Asm[ty][kk]);
#pragma unroll
                for (int r = 0; r < TN_REG; ++r) {
                    int64_t n = n_base + r;
                    if (n < N) {
                        float b_val = __half2float(Bsm[kk][tx * TN_REG + r]);
                        acc[r] += a_val * b_val;
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write results with bias and activation
    if (m < M) {
#pragma unroll
        for (int r = 0; r < TN_REG; ++r) {
            int64_t n = n_base + r;
            if (n < N) {
                float v = acc[r] + __half2float(b[n]);
                if (ApplyReLU) v = v > 0.f ? v : 0.f;
                Y[m * N + n] = __float2half_rn(v);
            }
        }
    }
}

// Utility to launch a single linear + bias (+ ReLU) op
static void launch_linear_bias_relu(
    const half* X, const half* W, const half* b, half* Y,
    int64_t M, int64_t N, int64_t K, bool apply_relu, cudaStream_t stream = 0
) {
    dim3 block(THREADS_X, THREADS_Y, 1);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    if (apply_relu) {
        linear_bias_relu_kernel<true><<<grid, block, 0, stream>>>(X, W, b, Y, M, N, K);
    } else {
        linear_bias_relu_kernel<false><<<grid, block, 0, stream>>>(X, W, b, Y, M, N, K);
    }
}

// Public entry point called by the test harness
void launch_gpu_implementation(
    void* output,
    const void* input,
    const void* w1,
    const void* b1,
    const void* w2,
    const void* b2,
    const void* w3,
    const void* b3,
    int64_t batch_size,
    int64_t input_size,
    int64_t hidden1_size,
    int64_t hidden2_size,
    int64_t output_size
) {
    const half* x  = static_cast<const half*>(input); // [B, input_size]
    const half* W1 = static_cast<const half*>(w1);    // [hidden1_size, input_size]
    const half* B1 = static_cast<const half*>(b1);    // [hidden1_size]
    const half* W2 = static_cast<const half*>(w2);    // [hidden2_size, hidden1_size]
    const half* B2 = static_cast<const half*>(b2);    // [hidden2_size]
    const half* W3 = static_cast<const half*>(w3);    // [output_size, hidden2_size]
    const half* B3 = static_cast<const half*>(b3);    // [output_size]
    half* y_out    = static_cast<half*>(output);      // [B, output_size]

    // Allocate intermediate buffers on device: y1 [B, hidden1], y2 [B, hidden2]
    half* y1 = nullptr;
    half* y2 = nullptr;
    size_t bytes_y1 = static_cast<size_t>(batch_size) * static_cast<size_t>(hidden1_size) * sizeof(half);
    size_t bytes_y2 = static_cast<size_t>(batch_size) * static_cast<size_t>(hidden2_size) * sizeof(half);
    cudaMalloc(&y1, bytes_y1);
    cudaMalloc(&y2, bytes_y2);

    // Layer 1: y1 = relu(x @ W1^T + b1)
    launch_linear_bias_relu(x, W1, B1, y1, batch_size, hidden1_size, input_size, true);

    // Layer 2: y2 = relu(y1 @ W2^T + b2)
    launch_linear_bias_relu(y1, W2, B2, y2, batch_size, hidden2_size, hidden1_size, true);

    // Layer 3: y_out = y2 @ W3^T + b3
    launch_linear_bias_relu(y2, W3, B3, y_out, batch_size, output_size, hidden2_size, false);

    // Clean up
    cudaFree(y1);
    cudaFree(y2);
}
