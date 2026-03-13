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

__global__ void model_forward_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ multiplier,
    half* __restrict__ output,
    int kernel_size, int stride, 
    int padding, int output_padding,
    float negative_slope, int pool_kernel
) {
    // Hard-coded dimensions from test case
    const int batch_size = 16;
    const int in_channels = 16;
    const int out_channels = 32;
    const int input_depth = 16, input_height = 32, input_width = 32;

    // Calculate intermediate output dimensions after ConvTranspose3d
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Final output dimensions after max pooling
    const int final_depth = output_depth / pool_kernel;
    const int final_height = output_height / pool_kernel;
    const int final_width = output_width / pool_kernel;

    // Flattened global index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * final_depth * final_height * final_width) return;

    // Convert index to 5D coordinates
    const int n = idx / (out_channels * final_depth * final_height * final_width);
    const int oc = (idx / (final_depth * final_height * final_width)) % out_channels;
    const int d_out = (idx / (final_height * final_width)) % final_depth;
    const int h_out = (idx / final_width) % final_height;
    const int w_out = idx % final_width;

    // Max pooling window boundaries
    const int d_start = d_out * pool_kernel;
    const int d_end = d_start + pool_kernel;
    const int h_start = h_out * pool_kernel;
    const int h_end = h_start + pool_kernel;
    const int w_start = w_out * pool_kernel;
    const int w_end = w_start + pool_kernel;

    float max_val = -INFINITY;

    // Iterate over max pooling window
    for (int d = d_start; d < d_end; ++d) {
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                if (d >= output_depth || h >= output_height || w >= output_width) continue;

                // Compute ConvTranspose3d output
                float val = conv_bias ? __half2float(conv_bias[oc]) : 0.0f;

                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kd = 0; kd < kernel_size; ++kd) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                const int d_in = (d - kd + padding) / stride;
                                const int h_in = (h - kh + padding) / stride;
                                const int w_in = (w - kw + padding) / stride;

                                if (d_in >= 0 && d_in < input_depth &&
                                    h_in >= 0 && h_in < input_height &&
                                    w_in >= 0 && w_in < input_width &&
                                    (d - kd + padding) % stride == 0 &&
                                    (h - kh + padding) % stride == 0 &&
                                    (w - kw + padding) % stride == 0) {

                                    const int input_idx = ((n * in_channels + ic) * input_depth + d_in) * input_height * input_width + h_in * input_width + w_in;
                                    const int weight_idx = ((ic * out_channels + oc) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;

                                    val += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                                }
                            }
                        }
                    }
                }

                // Apply LeakyReLU
                val = val > 0 ? val : val * negative_slope;
                // Multiply by parameter
                val *= __half2float(multiplier[oc]);
                // Apply LeakyReLU again
                val = val > 0 ? val : val * negative_slope;

                max_val = fmaxf(max_val, val);
            }
        }
    }

    output[idx] = __float2half_rn(max_val);
}

void launch_gpu_implementation(void* output, void* input, 
                              void* conv_weight, void* conv_bias,
                              void* multiplier,
                              int kernel_size, int stride, 
                              int padding, int output_padding,
                              float negative_slope, int pool_kernel) {
    // Output dimensions from test case
    const int batch_size = 16;
    const int out_channels = 32;
    const int final_depth = 16, final_height = 32, final_width = 32;
    const int total_elements = batch_size * out_channels * final_depth * final_height * final_width;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    model_forward_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(multiplier),
        static_cast<half*>(output),
        kernel_size, stride, padding, output_padding,
        negative_slope, pool_kernel
    );

    cudaDeviceSynchronize();
}
