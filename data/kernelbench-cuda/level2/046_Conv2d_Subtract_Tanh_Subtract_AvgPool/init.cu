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

__global__ void conv_sub_tanh_sub_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ intermediate,
    float subtract1, float subtract2,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size
) {
    const int oh = height - kernel_size + 1;
    const int ow = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * oh * ow;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    // Unpack 4D tensor coordinates
    int n = tid / (out_channels * oh * ow);
    int c = (tid % (out_channels * oh * ow)) / (oh * ow);
    int h = (tid % (oh * ow)) / ow;
    int w = tid % ow;

    float acc = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h + kh;
                int w_in = w + kw;
                if (h_in < height && w_in < width) {
                    int input_idx = ((n * in_channels + ic) * height + h_in) * width + w_in;
                    int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    // Add bias and apply pointwise operations
    acc += __half2float(bias[c]);
    acc -= subtract1;
    acc = tanhf(acc);
    acc -= subtract2;

    intermediate[tid] = __float2half_rn(acc);
}

__global__ void avg_pool_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size, int channels,
    int height, int width, int pool_size
) {
    const int pooled_h = height / pool_size;
    const int pooled_w = width / pool_size;
    const int output_size = batch_size * channels * pooled_h * pooled_w;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    // Unpack 4D tensor coordinates
    int n = tid / (channels * pooled_h * pooled_w);
    int c = (tid % (channels * pooled_h * pooled_w)) / (pooled_h * pooled_w);
    int ph = (tid % (pooled_h * pooled_w)) / pooled_w;
    int pw = tid % pooled_w;

    float sum = 0.0f;
    int h_start = ph * pool_size;
    int w_start = pw * pool_size;
    int count = 0;

    for (int h = h_start; h < h_start + pool_size; ++h) {
        for (int w = w_start; w < w_start + pool_size; ++w) {
            if (h < height && w < width) {
                int input_idx = ((n * channels + c) * height + h) * width + w;
                sum += __half2float(input[input_idx]);
                count++;
            }
        }
    }

    output[tid] = count > 0 ? __float2half_rn(sum / count) : __float2half(0.0f);
}

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, 
                              float subtract1_value, float subtract2_value, int kernel_size_pool) {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int height = 32, width = 32;
    const int kernel_size = 3;
    
    // Calculate convolution output dimensions
    const int conv_h = height - kernel_size + 1;
    const int conv_w = width - kernel_size + 1;
    const int intermediate_size = batch_size * out_channels * conv_h * conv_w;

    // Allocate intermediate tensor
    half* d_intermediate;
    cudaMalloc(&d_intermediate, intermediate_size * sizeof(half));

    // Launch fused convolution + pointwise operations kernel
    int block_size = 256;
    dim3 grid_conv((intermediate_size + block_size - 1) / block_size);
    conv_sub_tanh_sub_kernel<<<grid_conv, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_intermediate,
        subtract1_value, subtract2_value,
        batch_size, in_channels, out_channels,
        height, width, kernel_size
    );

    // Launch average pooling kernel
    const int pooled_h = conv_h / kernel_size_pool;
    const int pooled_w = conv_w / kernel_size_pool;
    const int output_size = batch_size * out_channels * pooled_h * pooled_w;
    
    dim3 grid_pool((output_size + block_size - 1) / block_size);
    avg_pool_kernel<<<grid_pool, block_size>>>(
        d_intermediate,
        static_cast<half*>(output),
        batch_size, out_channels,
        conv_h, conv_w, kernel_size_pool
    );

    cudaFree(d_intermediate);
    cudaDeviceSynchronize();
}
