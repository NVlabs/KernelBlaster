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
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

// Utility: CUDA error checking
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,         \
            cudaGetErrorString(err));                                   \
        return;                                                         \
    }                                                                   \
} while (0)

// Utility: ceil division
static inline int div_up(int a, int b) { return (a + b - 1) / b; }

/*
    4D tensor-matrix multiplication:
    C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    - Input A: (b, i, j, l), half
    - Input B: (l, k), half
    - Output C: (b, i, j, k), half
    Accumulation must be done in float for numerical stability, then cast to half.
    Kernel is optimized for GPU, using block tiling and shared memory for B.
    Each thread computes one (b, i, j, k_out) output element.
*/

#define TILE_K 64
#define TILE_K_B 64
#define TILE_J 8
#define TILE_I 4
#define TILE_K_OUT 32

// Kernel: Each block computes a tile of (i, j, k_out) for a given batch b.
// Shared memory is used for B[l, k_out] tiles.
__global__ void tensor4d_matmul_kernel(
    const half* __restrict__ A,   // [b, i, j, l]
    const half* __restrict__ B,   // [l, k]
    half* __restrict__ C,         // [b, i, j, k]
    int bdim, int idim, int jdim, int ldim, int kdim
) {
    // Each block computes for a single batch b.
    int b = blockIdx.z;
    int i_tile = blockIdx.y * TILE_I;
    int j_tile = blockIdx.x * TILE_J;

    // Each thread computes a subset of k_out for one (i, j)
    int tid = threadIdx.x;
    int k_out_base = (tid / (TILE_J * TILE_I)) * TILE_K_OUT;
    int local_tid = tid % (TILE_J * TILE_I);
    int local_i = local_tid / TILE_J;
    int local_j = local_tid % TILE_J;

    int i = i_tile + local_i;
    int j = j_tile + local_j;

    // For output: compute (b, i, j, k_out) for k_out in [0, kdim)
    __shared__ half B_sh[TILE_K_B][TILE_K_OUT]; // [ldim_tile][k_out_tile]

    for (int k_out = k_out_base; k_out < kdim; k_out += blockDim.x / (TILE_J * TILE_I) * TILE_K_OUT) {
        // Accumulator for output
        float acc[TILE_K_OUT] = {0.0f};
        // Loop over l in tiles
        for (int l_tile = 0; l_tile < ldim; l_tile += TILE_K_B) {
            // Load B[l, k_out] into shared memory
            int l_sh = threadIdx.y;
            int k_sh = threadIdx.x % TILE_K_OUT;
            for (int l_off = l_sh; l_off < TILE_K_B && (l_tile + l_off) < ldim; l_off += blockDim.y) {
                for (int k_off = k_sh; k_off < TILE_K_OUT && (k_out + k_off) < kdim; k_off += blockDim.x / (TILE_J * TILE_I)) {
                    int l = l_tile + l_off;
                    int k = k_out + k_off;
                    if (l < ldim && k < kdim) {
                        B_sh[l_off][k_off] = B[l * kdim + k];
                    } else {
                        B_sh[l_off][k_off] = __float2half(0.0f);
                    }
                }
            }
            __syncthreads();

            // For each l in tile, accumulate
            for (int l = 0; l < TILE_K_B && (l_tile + l) < ldim; ++l) {
                int l_global = l_tile + l;
                // A[b, i, j, l]
                half a_val = __float2half(0.0f);
                if (i < idim && j < jdim && l_global < ldim && b < bdim) {
                    a_val = A[((b * idim + i) * jdim + j) * ldim + l_global];
                }
                float a_f = __half2float(a_val);
                // For all k_out in the tile
#pragma unroll
                for (int kk = 0; kk < TILE_K_OUT; ++kk) {
                    int k_global = k_out + kk;
                    if (k_global < kdim) {
                        float b_f = __half2float(B_sh[l][kk]);
                        acc[kk] += a_f * b_f;
                    }
                }
            }
            __syncthreads();
        }

        // Write back to C
        for (int kk = 0; kk < TILE_K_OUT; ++kk) {
            int k_global = k_out + kk;
            if (i < idim && j < jdim && k_global < kdim && b < bdim) {
                // Clamp to fp16 range
                C[((b * idim + i) * jdim + j) * kdim + k_global] = __float2half(acc[kk]);
            }
        }
    }
}

// Simpler, more generic kernel: Each thread computes one output element: (b, i, j, k)
__global__ void tensor4d_matmul_naive_kernel(
    const half* __restrict__ A,   // [b, i, j, l]
    const half* __restrict__ B,   // [l, k]
    half* __restrict__ C,         // [b, i, j, k]
    int bdim, int idim, int jdim, int ldim, int kdim
) {
    int b = blockIdx.z;
    int i = blockIdx.y;
    int j = blockIdx.x;
    int k = threadIdx.x;

    if (b >= bdim || i >= idim || j >= jdim || k >= kdim) return;

    float acc = 0.0f;
    for (int l = 0; l < ldim; ++l) {
        half a = A[((b * idim + i) * jdim + j) * ldim + l];
        half b_val = B[l * kdim + k];
        acc += __half2float(a) * __half2float(b_val);
    }
    C[((b * idim + i) * jdim + j) * kdim + k] = __float2half(acc);
}

// Host launcher
void launch_gpu_implementation(
    void* output,         // Output tensor (b, i, j, k), type: at::Half*
    void* input_A,        // Input tensor A (b, i, j, l), type: at::Half*
    void* input_B,        // Input matrix B (l, k), type: at::Half*
    uint64_t b, uint64_t i, uint64_t j, uint64_t l, uint64_t k // Tensor shapes
) {
    // Heuristic: Use a tiling kernel for large k, otherwise fallback to naive
    if (i % TILE_I == 0 && j % TILE_J == 0 && k % TILE_K_OUT == 0 && l % TILE_K_B == 0) {
        // Tuned tile/block sizes for L40S
        dim3 grid(div_up(j, TILE_J), div_up(i, TILE_I), b);
        dim3 block((k + TILE_K_OUT - 1) / TILE_K_OUT * (TILE_J * TILE_I), 1, 1);
        // Limit block size for occupancy
        int max_threads = 256;
        if (block.x > max_threads) block.x = max_threads;
        size_t shared_bytes = TILE_K_B * TILE_K_OUT * sizeof(half);
        tensor4d_matmul_kernel<<<grid, block, shared_bytes>>>(
            static_cast<const half*>(input_A),
            static_cast<const half*>(input_B),
            static_cast<half*>(output),
            b, i, j, l, k
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // Fallback to naive: each thread computes one (b, i, j, k)
        dim3 grid(j, i, b);
        int block = k;
        tensor4d_matmul_naive_kernel<<<grid, block>>>(
            static_cast<const half*>(input_A),
            static_cast<const half*>(input_B),
            static_cast<half*>(output),
            b, i, j, l, k
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
