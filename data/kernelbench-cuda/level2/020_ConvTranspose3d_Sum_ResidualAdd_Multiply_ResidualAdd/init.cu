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

__global__ void fused_conv_transpose_ops_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ model_bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int output_padding,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    if (idx >= total_elements) return;

    // Calculate 5D output indices
    const int n = idx / (out_channels * output_depth * output_height * output_width);
    int remainder = idx % (out_channels * output_depth * output_height * output_width);
    const int c_out = remainder / (output_depth * output_height * output_width);
    remainder %= (output_depth * output_height * output_width);
    const int d_out = remainder / (output_height * output_width);
    remainder %= (output_height * output_width);
    const int h_out = remainder / output_width;
    const int w_out = remainder % output_width;

    float acc = 0.0f;

    // Iterate over kernel and input channels (corrected weight indexing)
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int d_in = (d_out - kd + padding) / stride;
                    const int h_in = (h_out - kh + padding) / stride;
                    const int w_in = (w_out - kw + padding) / stride;

                    // Check valid input position and divisibility
                    if (d_in < 0 || d_in >= input_depth || (d_out - kd + padding) % stride != 0) continue;
                    if (h_in < 0 || h_in >= input_height || (h_out - kh + padding) % stride != 0) continue;
                    if (w_in < 0 || w_in >= input_width || (w_out - kw + padding) % stride != 0) continue;

                    const int input_idx = ((n * in_channels + c_in) * input_depth + d_in) * input_height * input_width + h_in * input_width + w_in;
                    // CORRECTED WEIGHT INDEXING: [in_channels, out_channels, kd, kh, kw]
                    const int weight_idx = ((c_in * out_channels + c_out) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;

                    acc += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                }
            }
        }
    }

    // Add convolution bias
    acc += __half2float(conv_bias[c_out]);

    // Element-wise operations
    const float original_x = acc;
    const float model_bias_val = __half2float(model_bias[c_out]);

    float x = original_x + model_bias_val;
    x += original_x;          // x = 2*original_x + model_bias_val
    x *= original_x;          // x = (2*original_x + model_bias_val) * original_x
    x += original_x;          // x = (2*original_x + model_bias_val) * original_x + original_x

    output[idx] = __float2half_rn(x);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* model_bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int output_padding,
    int64_t bias_dim0, int64_t bias_dim1, int64_t bias_dim2, int64_t bias_dim3) {

    const int batch_size = 16;
    const int input_depth = 16, input_height = 32, input_width = 32;

    // Calculate output dimensions
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_conv_transpose_ops_kernel<<<blocks_per_grid, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(model_bias),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        kernel_size, stride, padding, output_padding,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width
    );

    cudaDeviceSynchronize();
}
