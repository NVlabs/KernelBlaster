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

__global__ void model_forward_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ subtract_1,
    const half* __restrict__ subtract_2,
    half* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int padding
) {
    const int output_h = height;
    const int output_w = width;
    const int kernel_radius = kernel_size / 2;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * output_h * output_w;
    if (idx >= total_elements) return;

    // NCHW output index calculation
    const int n = idx / (out_channels * output_h * output_w);
    const int c_out = (idx / (output_h * output_w)) % out_channels;  // Fixed variable name
    const int h_out = (idx / output_w) % output_h;
    const int w_out = idx % output_w;

    float acc = 0.0f;
    #pragma unroll
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int h_in = h_out + kh - kernel_radius;
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int w_in = w_out + kw - kernel_radius;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    // NCHW input indexing
                    const int input_idx = n * in_channels * height * width 
                                       + c_in * height * width 
                                       + h_in * width 
                                       + w_in;
                    // OIHW weight indexing (fixed c_out -> c_out)
                    const int weight_idx = c_out * in_channels * kernel_size * kernel_size 
                                        + c_in * kernel_size * kernel_size 
                                        + kh * kernel_size 
                                        + kw;
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    // Add bias and subtract values (fixed c_out_ -> c_out)
    acc += __half2float(bias[c_out]);
    acc -= __half2float(*subtract_1);
    acc -= __half2float(*subtract_2);

    // Mish activation
    const float softplus = logf(1.0f + expf(acc));
    const float mish = acc * tanhf(softplus);
    output[idx] = __float2half_rn(mish);
}

void launch_gpu_implementation(
    void* output,
    void* input,
    void* conv_weight,
    void* conv_bias,
    void* subtract_1,
    void* subtract_2,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size
) {
    const int output_size = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    model_forward_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(subtract_1),
        static_cast<const half*>(subtract_2),
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        kernel_size / 2
    );
    
    cudaDeviceSynchronize();
}
