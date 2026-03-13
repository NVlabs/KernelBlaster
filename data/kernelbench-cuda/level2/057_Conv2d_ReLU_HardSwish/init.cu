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

__global__ void fused_conv_relu_hardswish_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size) {
    
    const int out_h = height - kernel_size + 1;
    const int out_w = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * out_h * out_w;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    // Calculate output tensor indices
    const int n = tid / (out_channels * out_h * out_w);
    const int c_out = (tid % (out_channels * out_h * out_w)) / (out_h * out_w);
    const int h_out = (tid % (out_h * out_w)) / out_w;
    const int w_out = tid % out_w;

    // Convolution accumulator in FP32
    float acc = 0.0f;

    // Perform convolution
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int h_in = h_out + kh;
                const int w_in = w_out + kw;
                
                if (h_in < height && w_in < width) {
                    const int input_idx = n * in_channels * height * width + 
                                        c_in * height * width + h_in * width + w_in;
                    const int weight_idx = c_out * in_channels * kernel_size * kernel_size + 
                                         c_in * kernel_size * kernel_size + kh * kernel_size + kw;
                    
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    // Add bias and apply activations
    acc += __half2float(bias[c_out]);
    acc = fmaxf(acc, 0.0f);  // ReLU
    acc = acc * fminf(fmaxf((acc + 3.0f) / 6.0f, 0.0f), 1.0f);  // HardSwish

    // Store final result
    output[tid] = __float2half_rn(acc);
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias,
                               int batch_size, int in_channels, int out_channels,
                               int height, int width, int kernel_size) {
    const int out_h = height - kernel_size + 1;
    const int out_w = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * out_h * out_w;

    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    fused_conv_relu_hardswish_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        height, width, kernel_size
    );

    cudaDeviceSynchronize();
}
