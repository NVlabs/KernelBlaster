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
#include <mma.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16
#define BLOCK_ROWS 128
#define BLOCK_COLS 128
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

__global__ void fused_gemm_bias_relu_div_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const half* __restrict__ bias,
    half* __restrict__ C,
    float divisor,
    int M, int N, int K) {

    using namespace nvcuda;

    // Warp-level matrix fragments
    wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    const int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int block_row = blockIdx.y * MMA_M;
    const unsigned int block_col = blockIdx.x * MMA_N;

    // Main GEMM loop
    for(int k = 0; k < K; k += MMA_K) {
        wmma::load_matrix_sync(a_frag, A + block_row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + block_col * K + k, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Convert float accumulator to half before storing
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, half> c_frag;
    
    #pragma unroll
    for(int i = 0; i < acc_frag.num_elements; ++i) {
        float val = acc_frag.x[i] + __half2float(bias[block_col + i % MMA_N]);
        val = fmaxf(val, 0.0f) / divisor;
        c_frag.x[i] = __float2half_rn(val);
    }

    wmma::store_matrix_sync(C + block_row * N + block_col, c_frag, N, wmma::mem_row_major);
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, float divisor, 
                              int batch_size, int in_features, int out_features) {
    dim3 grid(
        (out_features + MMA_N - 1) / MMA_N,
        (batch_size + MMA_M - 1) / MMA_M
    );
    dim3 block(THREADS_PER_BLOCK);

    fused_gemm_bias_relu_div_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        divisor,
        batch_size,
        out_features,
        in_features
    );
    cudaDeviceSynchronize();
}
