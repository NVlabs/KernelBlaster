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
#include <math.h>

__global__ void conv_relu_add_bias_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ model_bias,
    half* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int output_h,
    int output_w
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * output_h * output_w;
    if (tid >= total_elements) return;

    // Decompose linear index into tensor dimensions
    const int n = tid / (out_channels * output_h * output_w);
    const int c_out = (tid / (output_h * output_w)) % out_channels;
    const int h_out = (tid / output_w) % output_h;
    const int w_out = tid % output_w;

    float acc = 0.0f;

    // Optimized loop structure with reversed kernel order for better memory access
    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int h_in = h_out + kh;
        if (h_in >= height) continue;
        
        #pragma unroll
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int w_in = w_out + kw;
            if (w_in >= width) continue;

            #pragma unroll
            for (int c_in = 0; c_in < in_channels; ++c_in) {
                const int input_idx = ((n * in_channels + c_in) * height + h_in) * width + w_in;
                const int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                
                acc += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
            }
        }
    }

    // Fused operations with register caching
    const float conv_bias_val = __half2float(conv_bias[c_out]);
    const float model_bias_val = __half2float(model_bias[c_out]);
    
    acc = fmaxf(acc + conv_bias_val, 0.0f) + model_bias_val;
    output[tid] = __float2half_rn(acc);
}

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, void* model_bias,
                               int batch_size, int in_channels, int out_channels, int height, int width,
                               int kernel_size, int output_h, int output_w) {
    const int total_elements = batch_size * out_channels * output_h * output_w;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    conv_relu_add_bias_kernel<<<blocks_per_grid, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(model_bias),
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        output_h,
        output_w
    );
    
    cudaDeviceSynchronize();
}
