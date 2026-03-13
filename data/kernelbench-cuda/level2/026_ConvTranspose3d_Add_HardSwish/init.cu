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
#include <cmath>

__device__ __forceinline__ half hardswish(half x) {
    float in = __half2float(x);
    float out = in * fminf(fmaxf(in + 3.0f, 0.0f), 6.0f) / 6.0f;
    return __float2half_rn(out);
}

__global__ void conv_transpose_add_activate_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ add_input,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_d, int input_h, int input_w,
    int kernel_size, int stride, int padding, int output_padding
) {
    const int output_d = (input_d - 1) * stride + kernel_size - 2 * padding + output_padding;
    const int output_h = (input_h - 1) * stride + kernel_size - 2 * padding + output_padding;
    const int output_w = (input_w - 1) * stride + kernel_size - 2 * padding + output_padding;
    
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * out_channels * output_d * output_h * output_w) return;

    // Unravel output index
    const int n = output_idx / (out_channels * output_d * output_h * output_w);
    const int oc = (output_idx / (output_d * output_h * output_w)) % out_channels;
    const int d = (output_idx / (output_h * output_w)) % output_d;
    const int h = (output_idx / output_w) % output_h;
    const int w = output_idx % output_w;

    float acc = 0.0f;

    // Iterate over kernel dimensions
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int d_in = (d - kd + padding) / stride;
                const int h_in = (h - kh + padding) / stride;
                const int w_in = (w - kw + padding) / stride;

                if ((d - kd + padding) % stride != 0 ||
                    (h - kh + padding) % stride != 0 ||
                    (w - kw + padding) % stride != 0) continue;

                if (d_in < 0 || d_in >= input_d ||
                    h_in < 0 || h_in >= input_h ||
                    w_in < 0 || w_in >= input_w) continue;

                // Iterate over input channels
                for (int ic = 0; ic < in_channels; ++ic) {
                    const int input_idx = n * in_channels * input_d * input_h * input_w +
                                        ic * input_d * input_h * input_w +
                                        d_in * input_h * input_w +
                                        h_in * input_w +
                                        w_in;

                    const int weight_idx = ic * kernel_size * kernel_size * kernel_size * out_channels +
                                         oc * kernel_size * kernel_size * kernel_size +
                                         kd * kernel_size * kernel_size +
                                         kh * kernel_size +
                                         kw;

                    acc += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                }
            }
        }
    }

    // Add bias and input
    acc += __half2float(conv_bias[oc]) + __half2float(add_input[output_idx]);

    // Apply HardSwish
    output[output_idx] = __float2half_rn(acc * __half2float(hardswish(__float2half_rn(acc))));
}

void launch_gpu_implementation(
    void* output,
    void* input,
    void* add_input,
    const void* conv_weight,
    const void* conv_bias,
    const void* /*model_bias*/,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    const std::vector<int64_t>& bias_shape
) {
    const int batch_size = 128;
    const int input_d = 16, input_h = 16, input_w = 16;
    const int output_size = (input_d - 1) * stride + kernel_size - 2 * padding + output_padding;
    const int total_output = batch_size * out_channels * output_size * output_size * output_size;

    const int threads = 256;
    const int blocks = (total_output + threads - 1) / threads;

    conv_transpose_add_activate_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(add_input),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        input_d, input_h, input_w,
        kernel_size, stride, padding, output_padding
    );

    cudaDeviceSynchronize();
}
