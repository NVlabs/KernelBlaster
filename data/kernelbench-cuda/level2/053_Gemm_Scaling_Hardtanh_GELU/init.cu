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
#include <cmath>

// Kernel declarations with proper __global__ qualifiers
__global__ void transpose_kernel(const half* input, half* output, int rows, int cols);
__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K);
__global__ void fused_ops_kernel(half* output, const half* bias, float scaling_factor,
                                float hardtanh_min, float hardtanh_max,
                                int batch_size, int out_features);

// Transpose kernel implementation
__global__ void transpose_kernel(const half* input, half* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

// GEMM kernel implementation with explicit launch syntax
__global__ void gemm_kernel(const half* __restrict__ A,
                           const half* __restrict__ B,
                           half* __restrict__ C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half_rn(sum);
    }
}

// Fused operations kernel
__global__ void fused_ops_kernel(half* output, const half* bias, float scaling_factor,
                                float hardtanh_min, float hardtanh_max,
                                int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx >= total) return;

    float val = __half2float(output[idx]) + __half2float(bias[idx % out_features]);
    val *= scaling_factor;
    val = fmaxf(hardtanh_min, fminf(val, hardtanh_max));
    val = 0.5f * val * (1.0f + erff(val / 1.41421356237f));
    output[idx] = __float2half_rn(val);
}

// Host launch function with proper kernel launches
void launch_gpu_implementation(
    void* output, void* input, void* weight, void* bias,
    float scaling_factor, float hardtanh_min, float hardtanh_max,
    int batch_size, int in_features, int out_features
) {
    // 1. Transpose weight matrix
    half* d_weight_t;
    cudaMalloc(&d_weight_t, in_features * out_features * sizeof(half));
    
    dim3 transpose_block(16, 16);
    dim3 transpose_grid(
        (in_features + transpose_block.x - 1) / transpose_block.x,
        (out_features + transpose_block.y - 1) / transpose_block.y
    );
    transpose_kernel<<<transpose_grid, transpose_block>>>(
        static_cast<const half*>(weight), d_weight_t, out_features, in_features
    );

    // 2. Matrix multiplication
    dim3 gemm_block(16, 16);
    dim3 gemm_grid(
        (out_features + gemm_block.x - 1) / gemm_block.x,
        (batch_size + gemm_block.y - 1) / gemm_block.y
    );
    gemm_kernel<<<gemm_grid, gemm_block>>>(
        static_cast<const half*>(input),
        d_weight_t,
        static_cast<half*>(output),
        batch_size, out_features, in_features
    );

    // 3. Fused operations
    int num_elements = batch_size * out_features;
    dim3 ops_block(256);
    dim3 ops_grid((num_elements + ops_block.x - 1) / ops_block.x);
    fused_ops_kernel<<<ops_grid, ops_block>>>(
        static_cast<half*>(output),
        static_cast<const half*>(bias),
        scaling_factor,
        hardtanh_min,
        hardtanh_max,
        batch_size,
        out_features
    );

    cudaFree(d_weight_t);
    cudaDeviceSynchronize();
}
