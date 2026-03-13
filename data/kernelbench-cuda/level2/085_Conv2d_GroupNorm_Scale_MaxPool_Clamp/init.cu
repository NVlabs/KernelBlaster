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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

// Convolution kernel using direct computation
__global__ void conv2d_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_size) {
    
    const int output_h = input_h - kernel_size + 1;
    const int output_w = input_w - kernel_size + 1;
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * out_channels * output_h * output_w) return;

    const int n = output_idx / (out_channels * output_h * output_w);
    const int c_out = (output_idx / (output_h * output_w)) % out_channels;
    const int oh = (output_idx / output_w) % output_h;
    const int ow = output_idx % output_w;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            for (int c_in = 0; c_in < in_channels; ++c_in) {
                const int h = oh + kh;
                const int w = ow + kw;
                if (h < input_h && w < input_w) {
                    const int input_idx = n * in_channels * input_h * input_w + c_in * input_h * input_w + h * input_w + w;
                    const int weight_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }
    sum += __half2float(bias[c_out]);
    output[output_idx] = __float2half_rn(sum);
}

// GroupNorm reduction kernel
__global__ void group_norm_reduce_kernel(
    const half* __restrict__ input,
    float* __restrict__ sum,
    float* __restrict__ sum_sq,
    int batch_size, int channels, int height, int width,
    int num_groups) {
    
    const int c_per_group = channels / num_groups;
    const int group_idx = blockIdx.x;
    const int n = group_idx / num_groups;
    const int g = group_idx % num_groups;
    const int c_start = g * c_per_group;
    const int c_end = c_start + c_per_group;

    float thread_sum = 0.0f, thread_sum_sq = 0.0f;
    for (int c = c_start; c < c_end; ++c) {
        for (int h = threadIdx.x; h < height; h += blockDim.x) {
            for (int w = threadIdx.y; w < width; w += blockDim.y) {
                const int idx = n * channels * height * width + c * height * width + h * width + w;
                const float val = __half2float(input[idx]);
                thread_sum += val;
                thread_sum_sq += val * val;
            }
        }
    }

    __shared__ float s_sum[256], s_sum_sq[256];
    const int tid = threadIdx.x * blockDim.y + threadIdx.y;
    s_sum[tid] = thread_sum;
    s_sum_sq[tid] = thread_sum_sq;
    __syncthreads();

    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[group_idx] = s_sum[0];
        sum_sq[group_idx] = s_sum_sq[0];
    }
}

// GroupNorm normalization + scaling kernel
__global__ void group_norm_scale_kernel(
    const half* __restrict__ input,
    const float* __restrict__ sum,
    const float* __restrict__ sum_sq,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    const half* __restrict__ scale,
    half* __restrict__ output,
    int batch_size, int channels, int height, int width,
    int num_groups, float eps) {
    
    const int c_per_group = channels / num_groups;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    const int n = idx / (channels * height * width);
    const int c = (idx / (height * width)) % channels;
    const int g = c / c_per_group;
    const int group_idx = n * num_groups + g;

    const float mean = sum[group_idx] / (c_per_group * height * width);
    const float var = sum_sq[group_idx] / (c_per_group * height * width) - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    const float val = __half2float(input[idx]);
    float normalized = (val - mean) * inv_std;
    normalized = normalized * __half2float(gamma[c]) + __half2float(beta[c]);
    normalized *= __half2float(scale[c]);

    output[idx] = __float2half_rn(normalized);
}

// Max pooling + clamp kernel
__global__ void max_pool_clamp_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size, int channels, int input_h, int input_w,
    int pool_size, float clamp_min, float clamp_max) {
    
    const int output_h = input_h / pool_size;
    const int output_w = input_w / pool_size;
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_h * output_w) return;

    const int n = output_idx / (channels * output_h * output_w);
    const int c = (output_idx / (output_h * output_w)) % channels;
    const int oh = (output_idx / output_w) % output_h;
    const int ow = output_idx % output_w;

    float max_val = -INFINITY;
    for (int kh = 0; kh < pool_size; ++kh) {
        for (int kw = 0; kw < pool_size; ++kw) {
            const int h = oh * pool_size + kh;
            const int w = ow * pool_size + kw;
            if (h < input_h && w < input_w) {
                const int idx = n * channels * input_h * input_w + c * input_h * input_w + h * input_w + w;
                max_val = fmaxf(max_val, __half2float(input[idx]));
            }
        }
    }
    output[output_idx] = __float2half_rn(fmaxf(fminf(max_val, clamp_max), clamp_min));
}

void launch_gpu_implementation(
    void* output, void* input,
    int in_channels, int out_channels, int kernel_size,
    int num_groups, const std::vector<int64_t>& scale_shape,
    int maxpool_kernel_size, float clamp_min, float clamp_max,
    void* conv_weight, void* conv_bias,
    void* group_norm_weight, void* group_norm_bias,
    void* scale) {
    
    const int batch_size = 128;
    const int input_h = 32, input_w = 32;
    const int conv_output_h = input_h - kernel_size + 1;
    const int conv_output_w = input_w - kernel_size + 1;
    const int conv_output_size = batch_size * out_channels * conv_output_h * conv_output_w;

    // Allocate intermediate buffers
    half *d_conv, *d_pool;
    float *d_sum, *d_sum_sq;
    cudaMalloc(&d_conv, conv_output_size * sizeof(half));
    
    // Launch convolution
    const int conv_block = 256;
    const int conv_grid = (conv_output_size + conv_block - 1) / conv_block;
    conv2d_kernel<<<conv_grid, conv_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv,
        batch_size, in_channels, out_channels,
        input_h, input_w, kernel_size
    );

    // GroupNorm reduction
    const int num_group_instances = batch_size * num_groups;
    cudaMalloc(&d_sum, num_group_instances * sizeof(float));
    cudaMalloc(&d_sum_sq, num_group_instances * sizeof(float));
    dim3 reduce_block(16, 16);
    group_norm_reduce_kernel<<<num_group_instances, reduce_block>>>(
        d_conv, d_sum, d_sum_sq,
        batch_size, out_channels, conv_output_h, conv_output_w, num_groups
    );

    // GroupNorm + scaling
    const int norm_size = batch_size * out_channels * conv_output_h * conv_output_w;
    const int norm_block = 256;
    group_norm_scale_kernel<<<(norm_size + norm_block - 1)/norm_block, norm_block>>>(
        d_conv, d_sum, d_sum_sq,
        static_cast<const half*>(group_norm_weight),
        static_cast<const half*>(group_norm_bias),
        static_cast<const half*>(scale),
        d_conv,
        batch_size, out_channels, conv_output_h, conv_output_w,
        num_groups, 1e-5f
    );

    // MaxPool + Clamp
    const int pool_output_h = conv_output_h / maxpool_kernel_size;
    const int pool_output_w = conv_output_w / maxpool_kernel_size;
    const int pool_output_size = batch_size * out_channels * pool_output_h * pool_output_w;
    cudaMalloc(&d_pool, pool_output_size * sizeof(half));
    const int pool_block = 256;
    max_pool_clamp_kernel<<<(pool_output_size + pool_block - 1)/pool_block, pool_block>>>(
        d_conv, d_pool,
        batch_size, out_channels, conv_output_h, conv_output_w,
        maxpool_kernel_size, clamp_min, clamp_max
    );

    // Copy result and cleanup
    cudaMemcpy(output, d_pool, pool_output_size * sizeof(half), cudaMemcpyDeviceToDevice);
    cudaFree(d_conv); cudaFree(d_sum); cudaFree(d_sum_sq); cudaFree(d_pool);
}
