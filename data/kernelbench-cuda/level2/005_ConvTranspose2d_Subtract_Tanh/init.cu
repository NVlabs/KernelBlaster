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
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ model_bias,
    half* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int output_height,
    int output_width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * output_height * output_width;
    if (idx >= total_elements) return;

    // Unravel the index into output tensor coordinates (NCHW layout)
    const int n = idx / (out_channels * output_height * output_width);
    const int oc = (idx / (output_height * output_width)) % out_channels;
    const int oh = (idx / output_width) % output_height;
    const int ow = idx % output_width;

    float acc = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Calculate input positions with transposed conv arithmetic
                const int h_in_num = oh - kh + padding;
                if (h_in_num % stride != 0) continue;
                const int ih = h_in_num / stride;
                if (ih < 0 || ih >= input_height) continue;

                const int w_in_num = ow - kw + padding;
                if (w_in_num % stride != 0) continue;
                const int iw = w_in_num / stride;
                if (iw < 0 || iw >= input_width) continue;

                // Input index (NCHW layout)
                const int input_idx = n * in_channels * input_height * input_width +
                                    ic * input_height * input_width +
                                    ih * input_width + iw;

                // Weight index (ICOC layout: in_channel, out_channel, kh, kw)
                const int weight_idx = ic * out_channels * kernel_size * kernel_size +
                                    oc * kernel_size * kernel_size +
                                    kh * kernel_size + kw;

                acc += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
            }
        }
    }

    // Add convolution bias if present
    if (conv_bias) {
        acc += __half2float(conv_bias[oc]);
    }

    // Subtract model bias and apply tanh
    acc -= __half2float(model_bias[oc]);
    acc = tanhf(acc);

    output[idx] = __float2half_rn(acc);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* model_bias,
    int64_t batch_size, int64_t in_channels, int64_t out_channels,
    int64_t input_height, int64_t input_width,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding,
    int64_t output_height, int64_t output_width
) {
    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    model_forward_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(model_bias),
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        output_height,
        output_width
    );

    cudaDeviceSynchronize();
}
