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

__global__ void conv2d_bias_kernel(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int H_out,
    int W_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * H_out * W_out;
    if (idx >= total_elements) return;

    int n = idx / (out_channels * H_out * W_out);
    int remainder = idx % (out_channels * H_out * W_out);
    int c_out = remainder / (H_out * W_out);
    remainder = remainder % (H_out * W_out);
    int h_out = remainder / W_out;
    int w_out = remainder % W_out;

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out + kh;
            int w_in = w_out + kw;
            if (h_in < height && w_in < width) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int input_idx = n * in_channels * height * width + c_in * height * width + h_in * width + w_in;
                    int weight_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    sum += __half2float(bias[c_out]);
    output[idx] = __float2half_rn(sum);
}

__global__ void min_reduce_kernel(
    const half* conv_output,
    half* min_output,
    int batch_size,
    int out_channels,
    int H_out,
    int W_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * H_out * W_out;
    if (idx >= total_elements) return;

    int n = idx / (H_out * W_out);
    int remainder = idx % (H_out * W_out);
    int h = remainder / W_out;
    int w = remainder % W_out;

    float min_val = INFINITY;
    for (int c = 0; c < out_channels; ++c) {
        int conv_idx = n * out_channels * H_out * W_out + c * H_out * W_out + h * W_out + w;
        float val = __half2float(conv_output[conv_idx]);
        min_val = fminf(min_val, val);
    }

    min_output[idx] = __float2half_rn(min_val);
}

__global__ void tanh_tanh_kernel(
    const half* input,
    half* output,
    int num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = __half2float(input[idx]);
    val = tanhf(val);
    val = tanhf(val);
    output[idx] = __float2half_rn(val);
}

void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size) {

    int H_out = height - kernel_size + 1;
    int W_out = width - kernel_size + 1;

    half* d_conv_output;
    size_t conv_output_size = batch_size * out_channels * H_out * W_out * sizeof(half);
    cudaMalloc(&d_conv_output, conv_output_size);

    half* d_min_output;
    size_t min_output_size = batch_size * H_out * W_out * sizeof(half);
    cudaMalloc(&d_min_output, min_output_size);

    int block_size = 256;
    int grid_size = (batch_size * out_channels * H_out * W_out + block_size - 1) / block_size;
    conv2d_bias_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        d_conv_output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        H_out,
        W_out
    );

    grid_size = (batch_size * H_out * W_out + block_size - 1) / block_size;
    min_reduce_kernel<<<grid_size, block_size>>>(
        d_conv_output,
        d_min_output,
        batch_size,
        out_channels,
        H_out,
        W_out
    );

    grid_size = (batch_size * H_out * W_out + block_size - 1) / block_size;
    tanh_tanh_kernel<<<grid_size, block_size>>>(
        d_min_output,
        static_cast<half*>(output),
        batch_size * H_out * W_out
    );

    cudaFree(d_conv_output);
    cudaFree(d_min_output);
    cudaDeviceSynchronize();
}
