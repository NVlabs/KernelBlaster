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

// 3D Convolution Kernel
__global__ void conv3d_kernel(
    const half* input, const half* weight, const half* bias, half* output,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width, int kernel_size,
    int stride, int padding, int out_depth, int out_height, int out_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx >= total) return;

    int b = idx / (out_channels * out_depth * out_height * out_width);
    int rem = idx % (out_channels * out_depth * out_height * out_width);
    int oc = rem / (out_depth * out_height * out_width);
    rem = rem % (out_depth * out_height * out_width);
    int d = rem / (out_height * out_width);
    rem = rem % (out_height * out_width);
    int h = rem / out_width;
    int w = rem % out_width;

    float acc = 0.0f;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    int in_d = d * stride - padding + kd;
                    int in_h = h * stride - padding + kh;
                    int in_w = w * stride - padding + kw;
                    
                    if (in_d >= 0 && in_d < depth && in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        int input_idx = b * in_channels * depth * height * width +
                                      ic * depth * height * width +
                                      in_d * height * width +
                                      in_h * width +
                                      in_w;
                        int weight_idx = oc * in_channels * kernel_size * kernel_size * kernel_size +
                                      ic * kernel_size * kernel_size * kernel_size +
                                      kd * kernel_size * kernel_size +
                                      kh * kernel_size +
                                      kw;
                        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }

    if (bias) acc += __half2float(bias[oc]);
    output[idx] = __float2half_rn(acc);
}

// 3D Max Pooling Kernel
__global__ void max_pool3d_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int depth, int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_depth = depth / 2;
    int out_height = height / 2;
    int out_width = width / 2;
    int total = batch_size * channels * out_depth * out_height * out_width;
    if (idx >= total) return;

    int b = idx / (channels * out_depth * out_height * out_width);
    int rem = idx % (channels * out_depth * out_height * out_width);
    int c = rem / (out_depth * out_height * out_width);
    rem = rem % (out_depth * out_height * out_width);
    int d = rem / (out_height * out_width);
    rem = rem % (out_height * out_width);
    int h = rem / out_width;
    int w = rem % out_width;

    float max_val = -INFINITY;
    for (int kd = 0; kd < 2; ++kd) {
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                int in_d = d * 2 + kd;
                int in_h = h * 2 + kh;
                int in_w = w * 2 + kw;
                if (in_d < depth && in_h < height && in_w < width) {
                    int input_idx = b * channels * depth * height * width +
                                  c * depth * height * width +
                                  in_d * height * width +
                                  in_h * width +
                                  in_w;
                    max_val = fmaxf(max_val, __half2float(input[input_idx]));
                }
            }
        }
    }
    output[idx] = __float2half_rn(max_val);
}

// LogSumExp Kernel
__global__ void logsumexp_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int depth, int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = depth * height * width;
    int total = batch_size * spatial;
    if (idx >= total) return;

    int b = idx / spatial;
    int s = idx % spatial;
    int d = s / (height * width);
    s = s % (height * width);
    int h = s / width;
    int w = s % width;

    float max_val = -INFINITY;
    for (int c = 0; c < channels; ++c) {
        int input_idx = b * channels * depth * height * width +
                      c * depth * height * width +
                      d * height * width +
                      h * width +
                      w;
        max_val = fmaxf(max_val, __half2float(input[input_idx]));
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < channels; ++c) {
        int input_idx = b * channels * depth * height * width +
                      c * depth * height * width +
                      d * height * width +
                      h * width +
                      w;
        sum_exp += expf(__half2float(input[input_idx]) - max_val);
    }

    output[idx] = __float2half_rn(max_val + logf(sum_exp));
}

// ReLU Kernel
__global__ void relu_kernel(const half* input, half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __hgt(input[idx], __float2half(0.0f)) ? input[idx] : __float2half(0.0f);
}

// Host launch function
void launch_gpu_implementation(void* output, void* input,
    int in_channels, int out_channels, int kernel_size,
    int stride, int padding,
    const void* conv_weight, const void* conv_bias) {

    // Fixed dimensions from test case
    const int batch_size = 128;
    const int depth = 16, height = 32, width = 32;

    // Calculate convolution output dimensions
    const int out_depth = (depth + 2*padding - kernel_size)/stride + 1;
    const int out_height = (height + 2*padding - kernel_size)/stride + 1;
    const int out_width = (width + 2*padding - kernel_size)/stride + 1;

    // Allocate intermediate buffers
    half *d_conv, *d_pool, *d_lse;
    size_t conv_size = batch_size * out_channels * out_depth * out_height * out_width;
    cudaMalloc(&d_conv, conv_size * sizeof(half));

    // Launch convolution
    const int block_size = 256;
    int grid_size = (conv_size + block_size - 1) / block_size;
    conv3d_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input), 
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv,
        batch_size, in_channels, out_channels,
        depth, height, width, kernel_size,
        stride, padding, out_depth, out_height, out_width
    );

    // Max Pool 3D
    const int pool_depth = out_depth / 2;
    const int pool_height = out_height / 2;
    const int pool_width = out_width / 2;
    size_t pool_size = batch_size * out_channels * pool_depth * pool_height * pool_width;
    cudaMalloc(&d_pool, pool_size * sizeof(half));
    
    grid_size = (pool_size + block_size - 1) / block_size;
    max_pool3d_kernel<<<grid_size, block_size>>>(
        d_conv, d_pool,
        batch_size, out_channels,
        out_depth, out_height, out_width
    );
    cudaFree(d_conv);

    // LogSumExp
    const int lse_size = batch_size * pool_depth * pool_height * pool_width;
    cudaMalloc(&d_lse, lse_size * sizeof(half));
    
    grid_size = (lse_size + block_size - 1) / block_size;
    logsumexp_kernel<<<grid_size, block_size>>>(
        d_pool, d_lse,
        batch_size, out_channels,
        pool_depth, pool_height, pool_width
    );
    cudaFree(d_pool);

    // ReLU
    grid_size = (lse_size + block_size - 1) / block_size;
    relu_kernel<<<grid_size, block_size>>>(
        d_lse, static_cast<half*>(output), lse_size
    );
    cudaFree(d_lse);
}
