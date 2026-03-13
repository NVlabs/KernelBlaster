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
#include <mma.h>
#include <iostream>

__global__ void fused_transposed_conv_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ model_bias,
    half* __restrict__ output,
    float scaling_factor,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_size,
    int stride, int padding, int output_padding
) {
    // Calculate output dimensions
    const int output_h = (input_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_w = (input_w - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_size = batch_size * out_channels * output_h * output_w;

    // Calculate global thread ID using 3D grid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    // Decompose output index
    int n = tid / (out_channels * output_h * output_w);
    int c_out = (tid / (output_h * output_w)) % out_channels;
    int h_out = (tid / output_w) % output_h;
    int w_out = tid % output_w;

    // Initialize accumulator with conv_bias
    float acc = __half2float(conv_bias[c_out]);

    // Iterate over kernel elements
    for (int k_h = 0; k_h < kernel_size; ++k_h) {
        for (int k_w = 0; k_w < kernel_size; ++k_w) {
            // Calculate input position based on transposed conv arithmetic
            int h_in = (h_out - k_h + padding) / stride;
            int w_in = (w_out - k_w + padding) / stride;

            // Check if input position is valid and integer division matches
            if (h_in >= 0 && h_in < input_h &&
                w_in >= 0 && w_in < input_w &&
                (h_out - k_h + padding) % stride == 0 &&
                (w_out - k_w + padding) % stride == 0) {

                // Iterate over input channels
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int input_idx = n * in_channels * input_h * input_w +
                                  c_in * input_h * input_w +
                                  h_in * input_w + w_in;
                                  
                    int weight_idx = c_in * out_channels * kernel_size * kernel_size +
                                   c_out * kernel_size * kernel_size +
                                   k_h * kernel_size + k_w;

                    acc += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                }
            }
        }
    }

    // Add model bias (broadcasted)
    acc += __half2float(model_bias[c_out]);

    // Apply activation and scaling operations
    acc = fmaxf(fminf(acc, 1.0f), 0.0f);
    acc *= scaling_factor;
    acc = fmaxf(fminf(acc, 1.0f), 0.0f);
    acc /= scaling_factor;

    // Write final result
    output[tid] = __float2half_rn(acc);
}

void launch_gpu_implementation(void* output, void* input, void* conv_weight, 
                              void* conv_bias, void* model_bias, float scaling_factor) {
    // Model parameters
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int input_h = 32, input_w = 32;
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const int output_padding = 1;
    
    const int output_h = (input_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_w = (input_w - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_size = batch_size * out_channels * output_h * output_w;

    // Configure kernel launch
    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    fused_transposed_conv_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(model_bias),
        static_cast<half*>(output),
        scaling_factor,
        batch_size, in_channels, out_channels,
        input_h, input_w, kernel_size,
        stride, padding, output_padding
    );
    
    cudaDeviceSynchronize();
}
