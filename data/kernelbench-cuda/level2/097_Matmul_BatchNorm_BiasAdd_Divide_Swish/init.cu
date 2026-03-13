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
#include <vector>

__global__ void fused_matmul_bn_swish_kernel(
    const half* __restrict__ input,
    const half* __restrict__ matmul_weight,
    const half* __restrict__ matmul_bias,
    const half* __restrict__ bn_weight,
    const half* __restrict__ bn_bias,
    const half* __restrict__ bn_running_mean,
    const half* __restrict__ bn_running_var,
    const half* __restrict__ custom_bias,
    half* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float bn_eps,
    float divide_value
) {
    // 2D grid: x=out_features, y=batch_size
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx >= out_features || batch_idx >= batch_size) return;

    // Matmul with fused accumulation in float32
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        float input_val = __half2float(input[batch_idx * in_features + k]);
        float weight_val = __half2float(matmul_weight[out_idx * in_features + k]);
        sum += input_val * weight_val;
    }

    // Add matmul bias
    sum += __half2float(matmul_bias[out_idx]);

    // Batch normalization
    float mean = __half2float(bn_running_mean[out_idx]);
    float var = __half2float(bn_running_var[out_idx]);
    float gamma = __half2float(bn_weight[out_idx]);
    float beta = __half2float(bn_bias[out_idx]);
    
    sum = gamma * (sum - mean) / sqrtf(var + bn_eps) + beta;

    // Add custom bias (broadcast scalar)
    sum += __half2float(*custom_bias);

    // Divide by value
    sum /= divide_value;

    // Swish activation
    float sigmoid = 1.0f / (1.0f + expf(-sum));
    sum *= sigmoid;

    // Store final fp16 result
    output[batch_idx * out_features + out_idx] = __float2half_rn(sum);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* matmul_weight, void* matmul_bias,
    void* bn_weight, void* bn_bias, void* bn_running_mean, void* bn_running_var,
    void* custom_bias,
    int in_features, int out_features,
    float bn_eps, float bn_momentum,
    const int64_t* bias_shape, int64_t bias_shape_len,
    float divide_value
) {
    const int batch_size = 128;
    dim3 block(32, 4);
    dim3 grid(
        (out_features + block.x - 1) / block.x,
        (batch_size + block.y - 1) / block.y
    );

    fused_matmul_bn_swish_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(matmul_weight),
        static_cast<const half*>(matmul_bias),
        static_cast<const half*>(bn_weight),
        static_cast<const half*>(bn_bias),
        static_cast<const half*>(bn_running_mean),
        static_cast<const half*>(bn_running_var),
        static_cast<const half*>(custom_bias),
        static_cast<half*>(output),
        batch_size,
        in_features,
        out_features,
        bn_eps,
        divide_value
    );
    
    cudaDeviceSynchronize();
}
