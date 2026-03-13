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
// cuda_model.cuh
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <mma.h>

// ---- CUDA kernel for C = A * B, fp16 inputs/outputs, fp32 accumulation ----
// No symmetry optimization: matmul of symmetric matrices is not itself symmetric.

__global__ void matmul_fp16_fp32_accum(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N
) {
    // Each thread computes a tile of the output matrix C
    // Tile size: BLOCK_M x BLOCK_N
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int TILE_K = 32;

    // Shared memory for A and B tiles
    __shared__ half Asub[BLOCK_M][TILE_K];
    __shared__ half Bsub[TILE_K][BLOCK_N];

    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    // Accumulator in fp32
    float acc = 0.0f;

    // Loop over tiles of K dimension
    for (int tile_k = 0; tile_k < N; tile_k += TILE_K) {
        // Load tiles into shared memory
        int tiled_k = tile_k + threadIdx.x;
        if (row < N && tiled_k < N && threadIdx.x < TILE_K) {
            Asub[threadIdx.y][threadIdx.x] = A[row * N + tiled_k];
        } else if (threadIdx.x < TILE_K && threadIdx.y < BLOCK_M) {
            Asub[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        int tiled_k2 = tile_k + threadIdx.y;
        if (col < N && tiled_k2 < N && threadIdx.y < TILE_K) {
            Bsub[threadIdx.y][threadIdx.x] = B[tiled_k2 * N + col];
        } else if (threadIdx.y < TILE_K && threadIdx.x < BLOCK_N) {
            Bsub[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        __syncthreads();

        // Compute partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += __half2float(Asub[threadIdx.y][k]) * __half2float(Bsub[k][threadIdx.x]);
        }

        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

// Host launch function
void launch_gpu_implementation(
    void* output,
    void* input_A,
    void* input_B,
    int64_t N
) {
    // Each block computes a 32x32 tile
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (N + BLOCK_M - 1) / BLOCK_M);

    matmul_fp16_fp32_accum<<<grid, block>>>(
        static_cast<const half*>(input_A),
        static_cast<const half*>(input_B),
        static_cast<half*>(output),
        static_cast<int>(N)
    );
    cudaDeviceSynchronize();
}
