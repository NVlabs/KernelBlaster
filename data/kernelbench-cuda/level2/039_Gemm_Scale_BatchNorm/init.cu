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
#include <iostream>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define BLOCK_ROWS 256
#define BLOCK_COLS 128
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK 256
#define K_STAGE 4

__global__ void fused_gemm_scale_bn_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ scale,
    const half* __restrict__ bn_weight,
    const half* __restrict__ bn_bias,
    const half* __restrict__ running_mean,
    const half* __restrict__ running_var,
    half* __restrict__ output,
    float epsilon,
    int batch_size,
    int in_features,
    int out_features
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid / out_features;
    const int col = tid % out_features;

    if (row >= batch_size || col >= out_features) return;

    // GEMM with FP32 accumulation
    float sum = 0.0f;
    for(int k = 0; k < in_features; ++k) {
        sum += __half2float(input[row * in_features + k]) * 
               __half2float(weight[col * in_features + k]);
    }

    // Add bias
    sum += __half2float(bias[col]);

    // Apply scale
    sum *= __half2float(scale[col]);

    // Batch normalization
    const float mean = __half2float(running_mean[col]);    // Fixed extra bracket
    const float var = __half2float(running_var[col]);      // Fixed here too
    const float gamma = __half2float(bn_weight[col]);
    const float beta = __half2float(bn_bias[col]);

    sum = (sum - mean) / sqrtf(var + epsilon);
    sum = sum * gamma + beta;

    output[row * out_features + col] = __float2half_rn(sum);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* gemm_weight, void* gemm_bias,
    void* scale,
    void* bn_weight, void* bn_bias,
    void* bn_running_mean, void* bn_running_var,
    float bn_eps,
    int64_t batch_size, int64_t in_features, int64_t out_features
) {
    const int num_elements = batch_size * out_features;
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    fused_gemm_scale_bn_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(gemm_weight),
        static_cast<const half*>(gemm_bias),
        static_cast<const half*>(scale),
        static_cast<const half*>(bn_weight),
        static_cast<const half*>(bn_bias),
        static_cast<const half*>(bn_running_mean),
        static_cast<const half*>(bn_running_var),
        static_cast<half*>(output),
        bn_eps,
        batch_size,
        in_features,
        out_features
    );

    cudaDeviceSynchronize();
}
