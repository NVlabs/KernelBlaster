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

__global__ void conv_scale_kernel(
    const half* input, const half* weight, const half* bias,
    half* conv_output, float scale_factor,
    int in_channels, int out_channels, int kernel_size,
    int batch_size, int height, int width
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z % out_channels;
    int n = blockIdx.z / out_channels;

    if (n >= batch_size || c_out >= out_channels || h >= height || w >= width) return;

    int kernel_radius = kernel_size / 2;
    float sum = 0.0f;

    for (int kh = -kernel_radius; kh <= kernel_radius; ++kh) {
        for (int kw = -kernel_radius; kw <= kernel_radius; ++kw) {
            int h_in = h + kh;
            int w_in = w + kw;

            if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int input_idx = n * in_channels * height * width + c_in * height * width + h_in * width + w_in;
                    int weight_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + (kh + kernel_radius) * kernel_size + (kw + kernel_radius);

                    sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    sum += __half2float(bias[c_out]);
    sum *= scale_factor;

    int output_idx = n * out_channels * height * width + c_out * height * width + h * width + w;
    conv_output[output_idx] = __float2half_rn(sum);
}

__global__ void min_reduce_kernel(
    const half* conv_output, half* output,
    int batch_size, int out_channels, int height, int width
) {
    int h = blockIdx.y;
    int w = blockIdx.x;
    int n = blockIdx.z;

    if (n >= batch_size || h >= height || w >= width) return;

    float min_val = INFINITY;
    for (int c = 0; c < out_channels; ++c) {
        int idx = n * out_channels * height * width + c * height * width + h * width + w;
        float val = __half2float(conv_output[idx]);
        if (val < min_val) {
            min_val = val;
        }
    }

    int output_idx = n * height * width + h * width + w;
    output[output_idx] = __float2half_rn(min_val);
}

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, float scale_factor,
                               int in_channels, int out_channels, int kernel_size,
                               int batch_size, int height, int width) {
    half* d_conv_output;
    size_t conv_output_size = batch_size * out_channels * height * width;
    cudaMalloc(&d_conv_output, conv_output_size * sizeof(half));

    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y, batch_size * out_channels);

    conv_scale_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv_output,
        scale_factor,
        in_channels,
        out_channels,
        kernel_size,
        batch_size,
        height,
        width
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Conv kernel error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 min_grid(width, height, batch_size);
    min_reduce_kernel<<<min_grid, dim3(1)>>>(
        d_conv_output,
        static_cast<half*>(output),
        batch_size,
        out_channels,
        height,
        width
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Min reduce kernel error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaFree(d_conv_output);
}
