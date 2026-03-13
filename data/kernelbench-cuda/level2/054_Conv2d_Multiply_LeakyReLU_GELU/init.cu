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

__global__ void fused_conv_ops_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ multiplier,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size
) {
    const int H_out = height - kernel_size + 1;
    const int W_out = width - kernel_size + 1;
    const int output_elements = batch_size * out_channels * H_out * W_out;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_elements) return;

    // Unravel output indices
    const int n = tid / (out_channels * H_out * W_out);
    const int c_out = (tid % (out_channels * H_out * W_out)) / (H_out * W_out);
    const int hw_idx = tid % (H_out * W_out);
    const int h_out = hw_idx / W_out;
    const int w_out = hw_idx % W_out;

    float acc = 0.0f;

    // Convolution computation with FP32 accumulation
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int h_in = h_out + kh;
                const int w_in = w_out + kw;
                
                if (h_in < height && w_in < width) {
                    const int input_idx = n * in_channels * height * width +
                                        c_in * height * width +
                                        h_in * width +
                                        w_in;
                    const int weight_idx = c_out * in_channels * kernel_size * kernel_size +
                                         c_in * kernel_size * kernel_size +
                                         kh * kernel_size +
                                         kw;

                    acc += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                }
            }
        }
    }

    // Add bias and apply pointwise operations
    acc += __half2float(conv_bias[c_out]);
    acc *= __half2float(multiplier[c_out]);
    acc = acc > 0 ? acc : 0.01f * acc;  // LeakyReLU
    acc = 0.5f * acc * (1.0f + erff(acc / 1.41421356237f));  // GELU

    output[tid] = __float2half_rn(acc);
}

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, void* multiplier, 
                              int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    const int H_out = height - kernel_size + 1;
    const int W_out = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * H_out * W_out;

    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    fused_conv_ops_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(multiplier),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        height, width, kernel_size
    );

    cudaDeviceSynchronize();
}
