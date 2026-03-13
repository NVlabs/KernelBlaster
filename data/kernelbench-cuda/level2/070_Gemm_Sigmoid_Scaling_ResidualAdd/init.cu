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
#include <cmath>
#include <cassert>  // Add missing assert header

// Valid WMMA configuration for FP16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

__global__ void gemm_sigmoid_scaling_residual_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    float scaling_factor,
    int M, int N, int K
) {
    // Warp and lane IDs
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Declare WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // Calculate matrix offsets
    const int row_offset = blockIdx.y * WMMA_M * K;
    const int col_offset = blockIdx.x * WMMA_N;

    // Main GEMM loop
    for (int k = 0; k < K; k += WMMA_K) {
        nvcuda::wmma::load_matrix_sync(a_frag, input + row_offset + k, K);
        nvcuda::wmma::load_matrix_sync(b_frag, weight + col_offset * K + k, K);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Load bias through shared memory
    __shared__ half s_bias[WMMA_N];
    if (lane_id < WMMA_N) {
        s_bias[lane_id] = bias[col_offset + lane_id];
    }
    __syncthreads();

    // Element-wise operations
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> output_frag;
    
    #pragma unroll
    for (int i = 0; i < acc_frag.num_elements; ++i) {
        float val = acc_frag.x[i] + __half2float(s_bias[i % WMMA_N]);
        val = 1.0f / (1.0f + expf(-val));
        val = val * scaling_factor + acc_frag.x[i];
        output_frag.x[i] = __float2half_rn(val);
    }

    // Store result
    nvcuda::wmma::store_matrix_sync(
        output + blockIdx.y * WMMA_M * N + col_offset,
        output_frag,
        N,
        nvcuda::wmma::mem_row_major
    );
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, float scaling_factor) {
    const int batch_size = 128;
    const int input_size = 1024;
    const int hidden_size = 512;

    // Verify divisibility by WMMA tile sizes
    assert(batch_size % WMMA_M == 0);
    assert(hidden_size % WMMA_N == 0);
    assert(input_size % WMMA_K == 0);

    dim3 grid(hidden_size / WMMA_N, batch_size / WMMA_M);
    dim3 block(THREADS_PER_BLOCK);

    gemm_sigmoid_scaling_residual_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        scaling_factor,
        batch_size, hidden_size, input_size
    );

    cudaDeviceSynchronize();
}
