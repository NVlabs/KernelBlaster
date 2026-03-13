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
#include <cstdio>
#include <cassert>

// Depthwise 2D convolution kernel for fp16 input/output, fp32 accumulation
// Supports arbitrary batch size, channels, kernel size, stride, dilation, padding, and bias
// Depthwise: each input channel is convolved with its own kernel (groups == in_channels == out_channels)

__global__ void depthwise_conv2d_fp16_kernel(
    const half* __restrict__ input,         // [B, C, H, W]
    const half* __restrict__ weight,        // [C, 1, KH, KW]
    const half* __restrict__ bias,          // [C] or nullptr
    half* __restrict__ output,              // [B, C, OH, OW]
    int batch_size,
    int in_channels,
    int out_channels,   // == in_channels for depthwise
    int input_height,
    int input_width,
    int kernel_size_h,
    int kernel_size_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    // Output shape: [B, C, OH, OW]
    const int OH = (input_height + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1;
    const int OW = (input_width + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = batch_size * in_channels * OH * OW;
    if (tid >= nthreads) return;

    // Compute output indices
    int ow = tid % OW;
    int oh = (tid / OW) % OH;
    int c  = (tid / (OW * OH)) % in_channels;
    int b  = tid / (OW * OH * in_channels);

    float acc = 0.0f;

    // Loop over kernel
#pragma unroll
    for (int kh = 0; kh < kernel_size_h; ++kh) {
        int ih = oh * stride_h - padding_h + kh * dilation_h;
        if (ih < 0 || ih >= input_height) continue;
#pragma unroll
        for (int kw = 0; kw < kernel_size_w; ++kw) {
            int iw = ow * stride_w - padding_w + kw * dilation_w;
            if (iw < 0 || iw >= input_width) continue;

            // Input: [B, C, H, W]
            int input_idx = ((b * in_channels + c) * input_height + ih) * input_width + iw;
            // Weight: [C, 1, KH, KW] (PyTorch depthwise)
            int weight_idx = ((c * kernel_size_h) + kh) * kernel_size_w + kw;

            acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        acc += __half2float(bias[c]);
    }

    // Store result as half
    int output_idx = ((b * out_channels + c) * OH + oh) * OW + ow;
    output[output_idx] = __float2half(acc);
}

void launch_gpu_implementation(
    void* output,                         // Output tensor pointer [B, C, OH, OW]
    void* input,                          // Input tensor pointer [B, C, H, W]
    void* weight,                         // Weight tensor pointer [C, 1, KH, KW]
    void* bias,                           // Bias tensor pointer (can be nullptr)
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size_h,
    int kernel_size_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    // Only depthwise convolution is supported (groups == in_channels == out_channels)
    assert(groups == in_channels && in_channels == out_channels);

    // Calculate output dimensions
    const int OH = (input_height + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1;
    const int OW = (input_width  + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1;
    const int nthreads = batch_size * in_channels * OH * OW;

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (nthreads + threadsPerBlock - 1) / threadsPerBlock;

    depthwise_conv2d_fp16_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w
    );

    cudaDeviceSynchronize();
}
