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

__global__ void model_kernel(const half* input, const half* conv_weight, const half* conv_bias, const half* model_bias,
                             half* output, int batch_size, int in_channels, int out_channels, int kernel_size,
                             int input_h, int input_w, int output_h, int output_w) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= batch_size) return;

    // Calculate output dimensions after conv_transpose
    const int conv_output_h = input_h + kernel_size - 1;
    const int conv_output_w = input_w + kernel_size - 1;
    const float spatial_size = conv_output_h * conv_output_w;

    // Shared memory for intermediate sums
    __shared__ float smem_sum_input[3]; // in_channels=3
    __shared__ float smem_sum_kernel[16][3]; // out_channels=16, in_channels=3

    // Compute sum_input for each channel
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        float sum = 0.0f;
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                int idx = n * in_channels * input_h * input_w + c_in * input_h * input_w + h * input_w + w;
                sum += __half2float(input[idx]);
            }
        }
        smem_sum_input[c_in] = sum;
    }

    // Compute sum_kernel for each (c_out, c_in)
    for (int c_out = 0; c_out < out_channels; ++c_out) {
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            float sum = 0.0f;
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    int idx = c_in * out_channels * kernel_size * kernel_size + c_out * kernel_size * kernel_size + i * kernel_size + j;
                    sum += __half2float(conv_weight[idx]);
                }
            }
            smem_sum_kernel[c_out][c_in] = sum;
        }
    }

    // Compute GEMM result and process
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float channel_vals[16]; // out_channels=16

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        float val = 0.0f;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            val += smem_sum_input[c_in] * smem_sum_kernel[c_out][c_in];
        }
        val = val / spatial_size + __half2float(conv_bias[c_out]) + __half2float(model_bias[c_out]);
        
        channel_vals[c_out] = val;
        if (val > max_val) max_val = val;
    }

    // Compute logsumexp
    for (int c_out = 0; c_out < out_channels; ++c_out) {
        sum_exp += expf(channel_vals[c_out] - max_val);
    }
    float result = logf(sum_exp) + max_val;

    // Multiply by 10 and store
    output[n] = __float2half_rn(result * 10.0f);
}

void launch_gpu_implementation(void* output, void* input,
                               void* conv_weight, void* conv_bias, void* model_bias,
                               int in_channels, int out_channels, int kernel_size,
                               int64_t bias_c, int64_t bias_h, int64_t bias_w) {
    const int batch_size = 128;
    const int input_h = 32, input_w = 32;
    const int output_h = input_h + kernel_size - 1;
    const int output_w = input_w + kernel_size - 1;

    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    model_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(model_bias),
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_h,
        input_w,
        output_h,
        output_w
    );
    
    cudaDeviceSynchronize();
}
