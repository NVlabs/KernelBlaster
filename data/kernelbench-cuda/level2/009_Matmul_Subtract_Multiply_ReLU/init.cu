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

__global__ void fused_linear_ops_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ subtract_value,
    const half* __restrict__ multiply_value,
    half* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * out_features) return;

    const int batch_idx = tid / out_features;
    const int out_idx = tid % out_features;

    const half* input_row = input + batch_idx * in_features;
    const half* weight_col = weight + out_idx * in_features;

    float sum = 0.0f;
    int i = 0;

    // Vectorized load for better memory throughput
    for (; i < in_features - 1; i += 2) {
        const half2 input_val = *reinterpret_cast<const half2*>(input_row + i);
        const half2 weight_val = *reinterpret_cast<const half2*>(weight_col + i);
        sum += __half2float(input_val.x) * __half2float(weight_val.x);
        sum += __half2float(input_val.y) * __half2float(weight_val.y);
    }

    // Handle odd feature dimension
    if (i < in_features) {
        sum += __half2float(input_row[i]) * __half2float(weight_col[i]);
    }

    // Add bias
    sum += __half2float(bias[out_idx]);

    // Load scalar parameters once per thread
    const float sub_val = __half2float(*subtract_value);
    const float mul_val = __half2float(*multiply_value);

    // Fused pointwise operations
    sum = fmaxf((sum - sub_val) * mul_val, 0.0f);

    // Store with fp16 conversion
    output[tid] = __float2half_rn(sum);
}

void launch_gpu_implementation(
    void* output,
    void* input,
    void* linear_weight,
    void* linear_bias,
    void* subtract_value,
    void* multiply_value,
    int batch_size,
    int in_features,
    int out_features
) {
    const int num_elements = batch_size * out_features;
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    fused_linear_ops_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(linear_weight),
        static_cast<const half*>(linear_bias),
        static_cast<const half*>(subtract_value),
        static_cast<const half*>(multiply_value),
        static_cast<half*>(output),
        batch_size,
        in_features,
        out_features
    );
    
    cudaDeviceSynchronize();
}
