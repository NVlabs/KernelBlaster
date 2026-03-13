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
#include <mma.h>

using namespace nvcuda;

const int WARP_SIZE = 32;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void fused_linear_scaled_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    float scaling_factor,
    int batch_size,
    int in_features,
    int out_features
) {
    // Tile coordinates
    const int tile_row = blockIdx.y * WMMA_M;
    const int tile_col = blockIdx.x * WMMA_N;

    // Warp and matrix fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Accumulate matrix product
    for (int k = 0; k < in_features; k += WMMA_K) {
        int k_offset = k;
        wmma::load_matrix_sync(a_frag, input + tile_row * in_features + k_offset, in_features);
        wmma::load_matrix_sync(b_frag, weight + tile_col * in_features + k_offset, in_features);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store to shared memory
    __shared__ float smem_acc[WMMA_M][WMMA_N];
    wmma::store_matrix_sync(smem_acc[0], acc_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    // Process 2 elements per thread with proper coverage
    const int row_in_tile = threadIdx.x / 8;  // 128 threads / 8 = 16 rows
    const int col_in_tile = (threadIdx.x % 8) * 2;
    const int global_row = tile_row + row_in_tile;
    const int global_col = tile_col + col_in_tile;

    if (global_row < batch_size && global_col + 1 < out_features) {
        // Load accumulated results
        float2 result = {
            smem_acc[row_in_tile][col_in_tile],
            smem_acc[row_in_tile][col_in_tile + 1]
        };

        // Load bias values
        half2 bias_val = *reinterpret_cast<const half2*>(&bias[global_col]);
        float2 fbias = __half22float2(bias_val);

        // Fused operations: (result + bias) * (scale + 1)
        result.x = (result.x + fbias.x) * (scaling_factor + 1.0f);
        result.y = (result.y + fbias.y) * (scaling_factor + 1.0f);

        // Store final result
        *reinterpret_cast<half2*>(&output[global_row * out_features + global_col]) = 
            __float22half2_rn(result);
    }
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, 
                              float scaling_factor, int batch_size, 
                              int in_features, int out_features) {
    dim3 grid((out_features + WMMA_N - 1) / WMMA_N,
              (batch_size + WMMA_M - 1) / WMMA_M);
    dim3 block(128);  // 4 warps to cover 16x16 tile

    fused_linear_scaled_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        scaling_factor,
        batch_size,
        in_features,
        out_features
    );
    cudaDeviceSynchronize();
}
