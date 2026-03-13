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

__global__ void conv_transpose_3d_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor,
    int batch_size,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_output = batch_size * out_channels * output_depth * output_height * output_width;
    if (idx >= total_output) return;

    // Calculate output indices
    const int n = idx / (out_channels * output_depth * output_height * output_width);
    const int c_out = (idx % (out_channels * output_depth * output_height * output_width)) / (output_depth * output_height * output_width);
    const int d_out = (idx % (output_depth * output_height * output_width)) / (output_height * output_width);
    const int h_out = (idx % (output_height * output_width)) / output_width;
    const int w_out = idx % output_width;

    float acc = 0.0f;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int d_in = (d_out - kd + padding) / stride;
                const int h_in = (h_out - kh + padding) / stride;
                const int w_in = (w_out - kw + padding) / stride;

                if ((d_out - kd + padding) % stride != 0) continue;
                if ((h_out - kh + padding) % stride != 0) continue;
                if ((w_out - kw + padding) % stride != 0) continue;
                if (d_in < 0 || d_in >= input_depth) continue;
                if (h_in < 0 || h_in >= input_height) continue;
                if (w_in < 0 || w_in >= input_width) continue;

                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    const int input_idx = ((n * in_channels + c_in) * input_depth + d_in) * input_height * input_width + h_in * input_width + w_in;
                    const int weight_idx = ((c_in * out_channels + c_out) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;

                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    if (bias) {
        acc += __half2float(bias[c_out]);
    }

    acc = fmaxf(acc, min_value);
    acc /= divisor;

    output[idx] = __float2half_rn(acc);
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias,
                               int in_channels, int out_channels, int kernel_size,
                               int stride, int padding, float min_value, float divisor) {
    const int input_depth = 16, input_height = 32, input_width = 32;
    const int batch_size = 16;
    
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    
    const int total_output = batch_size * out_channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int grid_size = (total_output + block_size - 1) / block_size;

    conv_transpose_3d_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        min_value,
        divisor,
        batch_size,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width
    );
    
    cudaDeviceSynchronize();
}
