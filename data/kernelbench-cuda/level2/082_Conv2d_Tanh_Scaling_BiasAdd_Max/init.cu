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
    half* __restrict__ output,
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ model_bias,
    float scaling_factor,
    int pool_kernel_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int batch_size,
    int height,
    int width
) {
    const int H_conv = height - kernel_size + 1;
    const int W_conv = width - kernel_size + 1;
    const int H_pool = H_conv / pool_kernel_size;
    const int W_pool = W_conv / pool_kernel_size;
    const int total_output_elements = batch_size * out_channels * H_pool * W_pool;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_output_elements) return;

    const int n = idx / (out_channels * H_pool * W_pool);
    const int residual = idx % (out_channels * H_pool * W_pool);
    const int c_out = residual / (H_pool * W_pool);
    const int pool_idx = residual % (H_pool * W_pool);
    const int pool_h = pool_idx / W_pool;
    const int pool_w = pool_idx % W_pool;

    const int h_conv_start = pool_h * pool_kernel_size;
    const int w_conv_start = pool_w * pool_kernel_size;

    float max_val = -INFINITY;

    for (int dh = 0; dh < pool_kernel_size; ++dh) {
        for (int dw = 0; dw < pool_kernel_size; ++dw) {
            const int h_conv = h_conv_start + dh;
            const int w_conv = w_conv_start + dw;

            if (h_conv >= H_conv || w_conv >= W_conv) continue;

            float sum = __half2float(conv_bias[c_out]);

            for (int c_in = 0; c_in < in_channels; ++c_in) {
                for (int dy = 0; dy < kernel_size; ++dy) {
                    for (int dx = 0; dx < kernel_size; ++dx) {
                        const int input_h = h_conv + dy;
                        const int input_w = w_conv + dx;

                        if (input_h >= height || input_w >= width) continue;

                        const int input_idx = n * in_channels * height * width 
                                           + c_in * height * width 
                                           + input_h * width 
                                           + input_w;
                        const int weight_idx = c_out * in_channels * kernel_size * kernel_size 
                                            + c_in * kernel_size * kernel_size 
                                            + dy * kernel_size 
                                            + dx;

                        sum += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                    }
                }
            }

            float activated = tanhf(sum) * scaling_factor + __half2float(model_bias[c_out]);
            max_val = fmaxf(max_val, activated);
        }
    }

    const int output_idx = n * out_channels * H_pool * W_pool 
                         + c_out * H_pool * W_pool 
                         + pool_h * W_pool 
                         + pool_w;
    output[output_idx] = __float2half_rn(max_val);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* model_bias,
    float scaling_factor, int pool_kernel_size,
    int in_channels, int out_channels, int kernel_size,
    int batch_size, int height, int width
) {
    const int H_conv = height - kernel_size + 1;
    const int W_conv = width - kernel_size + 1;
    const int H_pool = H_conv / pool_kernel_size;
    const int W_pool = W_conv / pool_kernel_size;
    const int total_output_elements = batch_size * out_channels * H_pool * W_pool;

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (total_output_elements + threadsPerBlock - 1) / threadsPerBlock;

    model_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(model_bias),
        scaling_factor,
        pool_kernel_size,
        in_channels,
        out_channels,
        kernel_size,
        batch_size,
        height,
        width
    );

    cudaDeviceSynchronize();
}
