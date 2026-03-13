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
#include <iostream>

// ConvTranspose3D Kernel
__global__ void conv_transpose_3d_kernel(
    const half* input, const half* weight, const half* bias,
    half* output,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int batch_size, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx >= total_output) return;

    // Unravel output coordinates
    int n = idx / (out_channels * out_depth * out_height * out_width);
    int rem = idx % (out_channels * out_depth * out_height * out_width);
    int oc = rem / (out_depth * out_height * out_width);
    rem = rem % (out_depth * out_height * out_width);
    int d = rem / (out_height * out_width);
    rem = rem % (out_height * out_width);
    int h = rem / out_width;
    int w = rem % out_width;

    float acc = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int d_in = (d - kd + padding) / stride;
                    int h_in = (h - kh + padding) / stride;
                    int w_in = (w - kw + padding) / stride;

                    if ((d - kd + padding) % stride == 0 &&
                        (h - kh + padding) % stride == 0 &&
                        (w - kw + padding) % stride == 0 &&
                        d_in >= 0 && d_in < in_depth &&
                        h_in >= 0 && h_in < in_height &&
                        w_in >= 0 && w_in < in_width) {

                        int input_idx = n * in_channels * in_depth * in_height * in_width +
                                        ic * in_depth * in_height * in_width +
                                        d_in * in_height * in_width +
                                        h_in * in_width +
                                        w_in;

                        int weight_idx = ic * out_channels * kernel_size * kernel_size * kernel_size +
                                         oc * kernel_size * kernel_size * kernel_size +
                                         kd * kernel_size * kernel_size +
                                         kh * kernel_size +
                                         kw;

                        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }

    if (bias) {
        acc += __half2float(bias[oc]);
    }

    output[idx] = __float2half_rn(acc);
}

// MaxPool3D Kernel
__global__ void max_pool_3d_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int in_depth, int in_height, int in_width,
    int kernel_size, int stride) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_depth = (in_depth - kernel_size) / stride + 1;
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    int total_output = batch_size * channels * out_depth * out_height * out_width;
    if (idx >= total_output) return;

    // Unravel output coordinates
    int n = idx / (channels * out_depth * out_height * out_width);
    int rem = idx % (channels * out_depth * out_height * out_width);
    int c = rem / (out_depth * out_height * out_width);
    rem = rem % (out_depth * out_height * out_width);
    int d = rem / (out_height * out_width);
    rem = rem % (out_height * out_width);
    int h = rem / out_width;
    int w = rem % out_width;

    int d_start = d * stride;
    int d_end = d_start + kernel_size;
    int h_start = h * stride;
    int h_end = h_start + kernel_size;
    int w_start = w * stride;
    int w_end = w_start + kernel_size;

    float max_val = -__int_as_float(0x7f800000); // -INF

    for (int di = d_start; di < d_end; ++di) {
        for (int hi = h_start; hi < h_end; ++hi) {
            for (int wi = w_start; wi < w_end; ++wi) {
                if (di < in_depth && hi < in_height && wi < in_width) {
                    int input_idx = n * channels * in_depth * in_height * in_width +
                                    c * in_depth * in_height * in_width +
                                    di * in_height * in_width +
                                    hi * in_width +
                                    wi;
                    float val = __half2float(input[input_idx]);
                    max_val = fmaxf(max_val, val);
                }
            }
        }
    }

    output[idx] = __float2half_rn(max_val);
}

// Sum Reduction Kernel
__global__ void sum_reduce_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int depth, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * depth * height * width;
    if (idx >= total_output) return;

    // Unravel output coordinates
    int n = idx / (depth * height * width);
    int rem = idx % (depth * height * width);
    int d = rem / (height * width);
    rem = rem % (height * width);
    int h = rem / width;
    int w = rem % width;

    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        int input_idx = n * channels * depth * height * width +
                        c * depth * height * width +
                        d * height * width +
                        h * width +
                        w;
        sum += __half2float(input[input_idx]);
    }

    output[idx] = __float2half_rn(sum);
}

void launch_gpu_implementation(void* output, void* input, 
                              int in_channels, int out_channels,
                              int kernel_size, int stride, int padding,
                              void* weight, void* bias) {
    // Fixed input dimensions from test case
    const int batch_size = 16;
    const int in_depth = 16, in_height = 32, in_width = 32;

    // Calculate ConvTranspose output dimensions
    const int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size;
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;

    // Allocate intermediate buffers
    half *d_conv, *d_pool1, *d_pool2;
    size_t conv_size = batch_size * out_channels * out_depth * out_height * out_width;
    cudaMalloc(&d_conv, conv_size * sizeof(half));

    // Launch ConvTranspose kernel
    const int block_size = 256;
    int grid_size = (conv_size + block_size - 1) / block_size;
    conv_transpose_3d_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input), 
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        d_conv,
        in_channels, out_channels,
        kernel_size, stride, padding,
        batch_size, in_depth, in_height, in_width,
        out_depth, out_height, out_width
    );

    // First MaxPool (kernel=2, stride=2)
    const int pool1_depth = (out_depth - 2) / 2 + 1;
    const int pool1_height = (out_height - 2) / 2 + 1;
    const int pool1_width = (out_width - 2) / 2 + 1;
    size_t pool1_size = batch_size * out_channels * pool1_depth * pool1_height * pool1_width;
    cudaMalloc(&d_pool1, pool1_size * sizeof(half));
    max_pool_3d_kernel<<<(pool1_size + block_size - 1) / block_size, block_size>>>(
        d_conv, d_pool1,
        batch_size, out_channels,
        out_depth, out_height, out_width,
        2, 2
    );

    // Second MaxPool (kernel=3, stride=3)
    const int pool2_depth = (pool1_depth - 3) / 3 + 1;
    const int pool2_height = (pool1_height - 3) / 3 + 1;
    const int pool2_width = (pool1_width - 3) / 3 + 1;
    size_t pool2_size = batch_size * out_channels * pool2_depth * pool2_height * pool2_width;
    cudaMalloc(&d_pool2, pool2_size * sizeof(half));
    max_pool_3d_kernel<<<(pool2_size + block_size - 1) / block_size, block_size>>>(
        d_pool1, d_pool2,
        batch_size, out_channels,
        pool1_depth, pool1_height, pool1_width,
        3, 3
    );

    // Sum reduction
    const int sum_size = batch_size * pool2_depth * pool2_height * pool2_width;
    sum_reduce_kernel<<<(sum_size + block_size - 1) / block_size, block_size>>>(
        d_pool2, static_cast<half*>(output),
        batch_size, out_channels,
        pool2_depth, pool2_height, pool2_width
    );

    // Cleanup
    cudaFree(d_conv);
    cudaFree(d_pool1);
    cudaFree(d_pool2);
    cudaDeviceSynchronize();
}
