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

__global__ void fused_conv3d_ops_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ scaling_factor,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_size
) {
    // Calculate output dimensions
    const int output_depth = input_depth - kernel_size + 1;
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    const int output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    // Global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_elements) return;

    // Unravel output indices
    const int n = tid / (out_channels * output_depth * output_height * output_width);
    int remainder = tid % (out_channels * output_depth * output_height * output_width);
    const int c_out = remainder / (output_depth * output_height * output_width);
    remainder %= output_depth * output_height * output_width;
    const int d_out = remainder / (output_height * output_width);
    remainder %= output_height * output_width;
    const int h_out = remainder / output_width;
    const int w_out = remainder % output_width;

    // Convolution window bounds
    const int d_start = d_out;
    const int d_end = d_start + kernel_size;
    const int h_start = h_out;
    const int h_end = h_start + kernel_size;
    const int w_start = w_out;
    const int w_end = w_start + kernel_size;

    // Accumulate in FP32 for numerical stability
    float acc = 0.0f;

    // Perform 3D convolution
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int d_in = d_start + kd;
                    const int h_in = h_start + kh;
                    const int w_in = w_start + kw;
                    
                    if (d_in < input_depth && h_in < input_height && w_in < input_width) {
                        const int input_idx = ((n * in_channels + c_in) * input_depth + d_in) * input_height * input_width +
                                            h_in * input_width + w_in;
                        const int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kd) * kernel_size * kernel_size +
                                            kh * kernel_size + kw;
                        
                        acc += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                    }
                }
            }
        }
    }

    // Add conv bias
    acc += __half2float(conv_bias[c_out]);

    // Apply scaling factor (per-channel)
    acc *= __half2float(scaling_factor[c_out]);

    // Apply tanh activation
    acc = tanhf(acc);

    // Multiply by bias (per-channel)
    acc *= __half2float(bias[c_out]);

    // Apply sigmoid
    acc = 1.0f / (1.0f + expf(-acc));

    // Store final result
    output[tid] = __float2half_rn(acc);
}

void launch_gpu_implementation(void* output, void* input, 
                              void* conv_weight, void* conv_bias,
                              void* scaling_factor, void* bias) {
    // Problem dimensions
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int input_depth = 16, input_height = 32, input_width = 32;
    const int kernel_size = 3;
    
    // Calculate output elements
    const int output_depth = input_depth - kernel_size + 1;
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    const int output_elements = batch_size * out_channels * output_depth * output_height * output_width;

    // Kernel configuration
    const int block_size = 256;
    const int grid_size = (output_elements + block_size - 1) / block_size;

    fused_conv3d_ops_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(scaling_factor),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_size
    );
    
    cudaDeviceSynchronize();
}
