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

__global__ void model_forward_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    float scale_factor
) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const half* row_in = input + batch_idx * in_features;
    float acc_features[5]; // Maximum expected out_features=5

    // Matrix multiplication with fp32 accumulation
    #pragma unroll
    for (int out_idx = 0; out_idx < out_features; ++out_idx) {
        float acc = 0.0f;
        #pragma unroll
        for (int in_idx = 0; in_idx < in_features; ++in_idx) {
            acc += __half2float(row_in[in_idx]) * 
                   __half2float(weight[out_idx * in_features + in_idx]);
        }
        acc += __half2float(bias[out_idx]);
        acc_features[out_idx] = acc;
    }

    // Max pooling with kernel_size=2 and stride=2
    const int pooled_size = (out_features - kernel_size) / 2 + 1;
    float pooled_sum = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < pooled_size; ++i) {
        const int start = i * 2;
        const float a = acc_features[start];
        const float b = (start + 1 < out_features) ? acc_features[start + 1] : -INFINITY;
        pooled_sum += fmaxf(a, b);
    }

    // Final scaling and store
    output[batch_idx] = __float2half_rn(pooled_sum * scale_factor);
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, 
                              int kernel_size, float scale_factor) {
    const int batch_size = 128;
    const int in_features = 10;
    const int out_features = 5;
    
    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;

    model_forward_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size,
        in_features,
        out_features,
        kernel_size,
        scale_factor
    );
    
    cudaDeviceSynchronize();
}
