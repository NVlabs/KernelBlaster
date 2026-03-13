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
#include <iostream>

__global__ void model_forward_kernel(
    const half* input,
    const half* conv_weight,
    const half* conv_bias,
    const half scale1,
    const half* model_bias,
    const half scale2,
    half* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding
) {
    // Calculate output dimensions after transposed convolution
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;

    // Calculate pooled dimensions after average pooling
    const int pooled_depth = output_depth / 2;
    const int pooled_height = output_height / 2;
    const int pooled_width = output_width / 2;

    // Global thread index calculation
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_final_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    if (tid >= num_final_elements) return;

    // Unravel indices for final output
    const int n = tid / (out_channels * pooled_depth * pooled_height * pooled_width);
    const int c_out = (tid % (out_channels * pooled_depth * pooled_height * pooled_width)) 
                      / (pooled_depth * pooled_height * pooled_width);
    const int d_pool = (tid % (pooled_depth * pooled_height * pooled_width)) 
                       / (pooled_height * pooled_width);
    const int h_pool = (tid % (pooled_height * pooled_width)) / pooled_width;
    const int w_pool = tid % pooled_width;

    // Calculate window in transposed output
    const int d_out_start = d_pool * 2;
    const int h_out_start = h_pool * 2;
    const int w_out_start = w_pool * 2;

    float sum = 0.0f;

    // Process 2x2x2 window
    for (int d = 0; d < 2; ++d) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                const int d_out = d_out_start + d;
                const int h_out = h_out_start + h;
                const int w_out = w_out_start + w;

                if (d_out >= output_depth || h_out >= output_height || w_out >= output_width) continue;

                float conv_value = 0.0f;

                // Transposed convolution calculation
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    for (int kd = 0; kd < kernel_size; ++kd) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                const int d_in = (d_out - kd + padding) / stride;
                                const int h_in = (h_out - kh + padding) / stride;
                                const int w_in = (w_out - kw + padding) / stride;

                                if (d_in < 0 || h_in < 0 || w_in < 0 ||
                                    d_in >= input_depth || h_in >= input_height || w_in >= input_width) continue;

                                if ((d_out - kd + padding) % stride != 0 ||
                                    (h_out - kh + padding) % stride != 0 ||
                                    (w_out - kw + padding) % stride != 0) continue;

                                const int input_idx = n * in_channels * input_depth * input_height * input_width +
                                                    c_in * input_depth * input_height * input_width +
                                                    d_in * input_height * input_width +
                                                    h_in * input_width +
                                                    w_in;

                                // CORRECTED WEIGHT INDEXING
                                const int weight_idx = c_in * out_channels * kernel_size * kernel_size * kernel_size +
                                                     c_out * kernel_size * kernel_size * kernel_size +
                                                     kd * kernel_size * kernel_size +
                                                     kh * kernel_size +
                                                     kw;

                                conv_value += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                            }
                        }
                    }
                }

                // Add conv bias and apply scale1
                conv_value += __half2float(conv_bias[c_out]);
                conv_value *= __half2float(scale1);
                sum += conv_value;
            }
        }
    }

    // Average pool and final operations
    float result = sum / 8.0f;
    result += __half2float(model_bias[c_out]);
    result *= __half2float(scale2);
    output[tid] = __float2half_rn(result);
}

void launch_gpu_implementation(
    void* output, void* input,
    const void* conv_weight, const void* conv_bias,
    const void* scale1, const void* model_bias, const void* scale2,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int batch_size, int input_depth, int input_height, int input_width
) {
    // Convert scalar parameters to half
    half h_scale1, h_scale2;
    cudaMemcpy(&h_scale1, scale1, sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_scale2, scale2, sizeof(half), cudaMemcpyDeviceToHost);

    // Calculate final output dimensions
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    const int pooled_depth = output_depth / 2;
    const int pooled_height = ((input_height - 1) * stride - 2 * padding + kernel_size) / 2;
    const int pooled_width = ((input_width - 1) * stride - 2 * padding + kernel_size) / 2;

    const int num_final_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_final_elements + threadsPerBlock - 1) / threadsPerBlock;

    model_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        h_scale1,
        static_cast<const half*>(model_bias),
        h_scale2,
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding
    );

    cudaDeviceSynchronize();
}
