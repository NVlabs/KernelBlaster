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

__global__ void conv_div_relu_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    float divisor,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_size) {
    
    const int output_h = input_h - kernel_size + 1;
    const int output_w = input_w - kernel_size + 1;
    const int total_elements = batch_size * out_channels * output_h * output_w;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Unflatten output index
    const int n = tid / (out_channels * output_h * output_w);
    const int c_out = (tid / (output_h * output_w)) % out_channels;
    const int h_out = (tid / output_w) % output_h;
    const int w_out = tid % output_w;

    float acc = 0.0f;
    
    // Convolution computation
    for(int c_in = 0; c_in < in_channels; ++c_in) {
        for(int kh = 0; kh < kernel_size; ++kh) {
            for(int kw = 0; kw < kernel_size; ++kw) {
                const int h_in = h_out + kh;
                const int w_in = w_out + kw;
                
                if(h_in < input_h && w_in < input_w) {
                    const int input_idx = ((n * in_channels + c_in) * input_h + h_in) * input_w + w_in;
                    const int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    // Add bias and apply operations
    acc += __half2float(bias[c_out]);
    acc /= divisor;
    acc = fmaxf(acc, 0.01f * acc);

    output[tid] = __float2half_rn(acc);
}

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, 
                              int in_channels, int out_channels, int kernel_size, void* divisor,
                              int batch_size, int height, int width) {
    // Copy divisor value from device to host
    half h_divisor;
    cudaMemcpy(&h_divisor, divisor, sizeof(half), cudaMemcpyDeviceToHost);
    const float divisor_val = __half2float(h_divisor);

    // Calculate output dimensions
    const int output_h = height - kernel_size + 1;
    const int output_w = width - kernel_size + 1;
    const int total_elements = batch_size * out_channels * output_h * output_w;

    // Launch kernel
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    conv_div_relu_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        divisor_val,
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        height, width, kernel_size
    );
    
    cudaDeviceSynchronize();
}
