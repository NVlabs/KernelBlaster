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

__global__ void gemm_scale_sum_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ scale,
    half* __restrict__ scaled_values,
    float* sum,
    float* sum_sq,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= batch_size || col >= out_features) return;

    // FP32 accumulation for numerical stability
    float val = 0.0f;
    for(int k = 0; k < in_features; ++k) {
        val += __half2float(input[row * in_features + k]) * 
               __half2float(weight[col * in_features + k]);
    }
    val += __half2float(bias[col]);
    val *= __half2float(scale[col]);

    // Store intermediate scaled value
    scaled_values[row * out_features + col] = __float2half_rn(val);

    // Atomic accumulation of sum and sum of squares
    atomicAdd(&sum[col], val);
    atomicAdd(&sum_sq[col], val * val);
}

__global__ void bn_apply_kernel(
    const half* __restrict__ scaled_values,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    const float* sum,
    const float* sum_sq,
    float eps,
    half* __restrict__ output,
    int batch_size,
    int out_features
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= out_features) return;

    // Compute batch statistics
    const float mean = sum[col] / batch_size;
    const float variance = (sum_sq[col] / batch_size) - (mean * mean);
    const float inv_std = rsqrtf(variance + eps);

    // Apply batch normalization to all elements in this column
    for(int row = 0; row < batch_size; ++row) {
        const float x = __half2float(scaled_values[row * out_features + col]);
        const float bn_val = (x - mean) * __half2float(gamma[col]) * inv_std + __half2float(beta[col]);
        output[row * out_features + col] = __float2half_rn(bn_val);
    }
}

void launch_gpu_implementation(
    void* output,
    void* input,
    void* gemm_weight,
    void* gemm_bias,
    void* scale,
    void* bn_weight,
    void* bn_bias,
    void* bn_running_mean,
    void* bn_running_var,
    float bn_eps,
    int batch_size,
    int in_features,
    int out_features
) {
    // Allocate temporary buffers
    half* d_scaled_values;
    float *d_sum, *d_sum_sq;
    cudaMalloc(&d_scaled_values, batch_size * out_features * sizeof(half));
    cudaMalloc(&d_sum, out_features * sizeof(float));
    cudaMalloc(&d_sum_sq, out_features * sizeof(float));
    cudaMemset(d_sum, 0, out_features * sizeof(float));
    cudaMemset(d_sum_sq, 0, out_features * sizeof(float));

    // Launch GEMM + scaling + statistics kernel
    dim3 gemm_block(16, 16);
    dim3 gemm_grid(
        (batch_size + gemm_block.x - 1) / gemm_block.x,
        (out_features + gemm_block.y - 1) / gemm_block.y
    );
    gemm_scale_sum_kernel<<<gemm_grid, gemm_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(gemm_weight),
        static_cast<const half*>(gemm_bias),
        static_cast<const half*>(scale),
        d_scaled_values,
        d_sum,
        d_sum_sq,
        batch_size,
        in_features,
        out_features
    );

    // Launch batch normalization kernel
    const int bn_block_size = 256;
    dim3 bn_grid((out_features + bn_block_size - 1) / bn_block_size);
    bn_apply_kernel<<<bn_grid, bn_block_size>>>(
        d_scaled_values,
        static_cast<const half*>(bn_weight),
        static_cast<const half*>(bn_bias),
        d_sum,
        d_sum_sq,
        bn_eps,
        static_cast<half*>(output),
        batch_size,
        out_features
    );

    // Cleanup
    cudaFree(d_scaled_values);
    cudaFree(d_sum);
    cudaFree(d_sum_sq);
    cudaDeviceSynchronize();
}
