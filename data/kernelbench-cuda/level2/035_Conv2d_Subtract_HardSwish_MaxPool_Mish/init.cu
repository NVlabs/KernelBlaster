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

__global__ void conv_sub_hardswish_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ conv_output,
    float subtract_value,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size) {
    
    const int H_conv = height - kernel_size + 1;
    const int W_conv = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * H_conv * W_conv;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    // Unravel 4D indices
    const int n = tid / (out_channels * H_conv * W_conv);
    const int c_out = (tid / (H_conv * W_conv)) % out_channels;
    const int h_out = (tid / W_conv) % H_conv;
    const int w_out = tid % W_conv;

    float acc = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int h_in = h_out + kh;
            const int w_in = w_out + kw;
            if (h_in < height && w_in < width) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    const int input_idx = ((n * in_channels + c_in) * height + h_in) * width + w_in;
                    const int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    // Add bias and subtract value
    acc += __half2float(bias[c_out]);
    acc -= subtract_value;

    // HardSwish activation
    const float hardswish = acc * fminf(fmaxf(acc + 3.0f, 0.0f), 6.0f) / 6.0f;
    conv_output[tid] = __float2half_rn(hardswish);
}

__global__ void maxpool_mish_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size, int channels,
    int height, int width, int pool_size) {
    
    const int H_pool = height / pool_size;
    const int W_pool = width / pool_size;
    const int output_size = batch_size * channels * H_pool * W_pool;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    // Unravel 4D indices
    const int n = tid / (channels * H_pool * W_pool);
    const int c = (tid / (H_pool * W_pool)) % channels;
    const int h_p = (tid / W_pool) % H_pool;
    const int w_p = tid % W_pool;

    // Max pooling window
    float max_val = -INFINITY;
    const int h_start = h_p * pool_size;
    const int w_start = w_p * pool_size;
    
    for (int h = h_start; h < h_start + pool_size; ++h) {
        for (int w = w_start; w < w_start + pool_size; ++w) {
            if (h < height && w < width) {
                const int input_idx = ((n * channels + c) * height + h) * width + w;
                const float val = __half2float(input[input_idx]);
                max_val = fmaxf(max_val, val);
            }
        }
    }

    // Mish activation
    const float softplus = logf(1.0f + expf(max_val));
    const float mish = max_val * tanhf(softplus);
    output[tid] = __float2half_rn(mish);
}

void launch_gpu_implementation(void* output, void* input,
                              void* conv_weight, void* conv_bias,
                              int64_t in_channels, int64_t out_channels,
                              int64_t kernel_size, float subtract_value,
                              int64_t pool_kernel_size,
                              int64_t batch_size, int64_t height, int64_t width) {
    // Calculate intermediate dimensions
    const int H_conv = height - kernel_size + 1;
    const int W_conv = width - kernel_size + 1;
    const size_t conv_output_size = batch_size * out_channels * H_conv * W_conv;

    // Allocate intermediate buffer
    half* d_conv_output;
    cudaMalloc(&d_conv_output, conv_output_size * sizeof(half));

    // Launch convolution kernel
    const int block_size = 256;
    int grid_size = (conv_output_size + block_size - 1) / block_size;
    conv_sub_hardswish_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv_output,
        subtract_value,
        batch_size, in_channels, out_channels,
        height, width, kernel_size
    );

    // Launch maxpool+mish kernel
    const size_t pool_output_size = batch_size * out_channels * 
                                  (H_conv/pool_kernel_size) * 
                                  (W_conv/pool_kernel_size);
    grid_size = (pool_output_size + block_size - 1) / block_size;
    maxpool_mish_kernel<<<grid_size, block_size>>>(
        d_conv_output,
        static_cast<half*>(output),
        batch_size, out_channels,
        H_conv, W_conv, pool_kernel_size
    );

    cudaFree(d_conv_output);
    cudaDeviceSynchronize();
}
