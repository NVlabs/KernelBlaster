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

__global__ void conv_transpose_kernel(
    const half* input, const half* weight, const half* bias,
    half* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int stride, int padding) {
    int output_height = (input_height - 1) * stride + kernel_size - 2 * padding;
    int output_width = (input_width - 1) * stride + kernel_size - 2 * padding;
    int output_elements = batch_size * out_channels * output_height * output_width;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_elements) return;

    int n = tid / (out_channels * output_height * output_width);
    int remainder = tid % (out_channels * output_height * output_width);
    int c_out = remainder / (output_height * output_width);
    remainder %= output_height * output_width;
    int h_out = remainder / output_width;
    int w_out = remainder % output_width;

    float acc = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int dh = 0; dh < kernel_size; ++dh) {
            for (int dw = 0; dw < kernel_size; ++dw) {
                int h_in = (h_out - dh + padding) / stride;
                int w_in = (w_out - dw + padding) / stride;
                if ((h_out - dh + padding) % stride != 0 || (w_out - dw + padding) % stride != 0)
                    continue;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int input_idx = n * in_channels * input_height * input_width +
                                    c_in * input_height * input_width + h_in * input_width + w_in;
                    int weight_idx = c_in * out_channels * kernel_size * kernel_size +
                                     c_out * kernel_size * kernel_size + dh * kernel_size + dw;
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }
    acc += __half2float(bias[c_out]);
    output[tid] = __float2half_rn(acc);
}

__global__ void max_pool_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int input_height, int input_width,
    int kernel_size, int stride) {
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;
    int output_elements = batch_size * channels * output_height * output_width;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_elements) return;

    int n = tid / (channels * output_height * output_width);
    int remainder = tid % (channels * output_height * output_width);
    int c = remainder / (output_height * output_width);
    remainder %= output_height * output_width;
    int h_pool = remainder / output_width;
    int w_pool = remainder % output_width;

    int h_start = h_pool * stride;
    int w_start = w_pool * stride;
    float max_val = -INFINITY;
    for (int h = h_start; h < h_start + kernel_size; ++h) {
        for (int w = w_start; w < w_start + kernel_size; ++w) {
            if (h < input_height && w < input_width) {
                int idx = n * channels * input_height * input_width +
                          c * input_height * input_width + h * input_width + w;
                max_val = fmaxf(max_val, __half2float(input[idx]));
            }
        }
    }
    output[tid] = __float2half_rn(max_val);
}

__global__ void hardtanh_kernel(
    half* data, int num_elements, float min_val, float max_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    float val = __half2float(data[tid]);
    data[tid] = __float2half_rn(fmaxf(fminf(val, max_val), min_val));
}

__global__ void mean_reduction_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int height, int width) {
    extern __shared__ float sdata[];
    int n = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < height * width; i += blockDim.x) {
        int h = i / width;
        int w = i % width;
        int idx = n * channels * height * width + c * height * width + h * width + w;
        sum += __half2float(input[idx]);
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        output[n * channels + c] = __float2half_rn(sdata[0] / (height * width));
}

__global__ void tanh_kernel(half* data, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    float val = __half2float(data[tid]);
    data[tid] = __float2half_rn(tanhf(val));
}

void launch_gpu_implementation(
    void* output, void* input, 
    void* conv_weight, void* conv_bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int maxpool_kernel_size, int maxpool_stride,
    float hardtanh_min, float hardtanh_max,
    int batch_size, int input_height, int input_width) {

    int conv_out_h = (input_height - 1)*stride + kernel_size - 2*padding;
    int conv_out_w = (input_width -1)*stride + kernel_size - 2*padding;
    int pool_out_h = (conv_out_h - maxpool_kernel_size)/maxpool_stride + 1;
    int pool_out_w = (conv_out_w - maxpool_kernel_size)/maxpool_stride + 1;

    half *d_conv, *d_pool, *d_mean;
    size_t conv_size = batch_size * out_channels * conv_out_h * conv_out_w * sizeof(half);
    size_t pool_size = batch_size * out_channels * pool_out_h * pool_out_w * sizeof(half);
    size_t mean_size = batch_size * out_channels * sizeof(half);

    cudaMalloc(&d_conv, conv_size);
    cudaMalloc(&d_pool, pool_size);
    cudaMalloc(&d_mean, mean_size);

    int block = 256;
    int grid = (batch_size * out_channels * conv_out_h * conv_out_w + block -1)/block;
    conv_transpose_kernel<<<grid, block>>>(
        (const half*)input, (const half*)conv_weight, (const half*)conv_bias, d_conv,
        batch_size, in_channels, out_channels, input_height, input_width,
        kernel_size, stride, padding
    );

    grid = (batch_size * out_channels * pool_out_h * pool_out_w + block -1)/block;
    max_pool_kernel<<<grid, block>>>(
        d_conv, d_pool, batch_size, out_channels, conv_out_h, conv_out_w,
        maxpool_kernel_size, maxpool_stride
    );
    cudaFree(d_conv);

    hardtanh_kernel<<<grid, block>>>(d_pool, batch_size * out_channels * pool_out_h * pool_out_w, hardtanh_min, hardtanh_max);

    dim3 grid_mean(batch_size, out_channels);
    int shmem = 256 * sizeof(float);
    mean_reduction_kernel<<<grid_mean, 256, shmem>>>(
        d_pool, d_mean, batch_size, out_channels, pool_out_h, pool_out_w
    );
    cudaFree(d_pool);

    grid = (batch_size * out_channels + block -1)/block;
    tanh_kernel<<<grid, block>>>(d_mean, batch_size * out_channels);

    cudaMemcpy(output, d_mean, mean_size, cudaMemcpyDeviceToDevice);
    cudaFree(d_mean);
}
