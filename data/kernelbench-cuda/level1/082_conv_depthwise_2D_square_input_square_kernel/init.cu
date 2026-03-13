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
// Depthwise 2D Convolution (fp16, PyTorch-style NHWC and NCHW equivalent) CUDA kernel
// Supports: batch, in_channels, H, W, kernel_size (square), stride, padding, bias (optional)
// Accumulation is always done in fp32 for numerical stability, output in fp16

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

// Utility: CUDA kernel to perform depthwise 2D convolution (NCHW layout)
__global__ void depthwise_conv2d_fp16_kernel(
    const half* __restrict__ input,      // [N, C, H, W]
    const half* __restrict__ weight,     // [C, kH, kW] (PyTorch depthwise: [in_channels, k, k])
    const half* __restrict__ bias,       // [C] or nullptr
    half* __restrict__ output,           // [N, C, H_out, W_out]
    int N, int C, int H, int W,
    int k, int stride, int padding,
    int H_out, int W_out
) {
    // Each thread computes one output pixel per channel per batch.
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || c >= C || hw >= H_out*W_out) return;
    int h_out = hw / W_out;
    int w_out = hw % W_out;

    float acc = 0.0f; // Accumulate in fp32

    // For each element in the kernel
    for (int kh = 0; kh < k; ++kh) {
        for (int kw = 0; kw < k; ++kw) {
            int h_in = h_out * stride + kh - padding;
            int w_in = w_out * stride + kw - padding;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int input_idx = ((n * C + c) * H + h_in) * W + w_in; // NCHW
                int weight_idx = (c * k + kh) * k + kw;
                acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
            }
        }
    }
    // Add bias if present
    if (bias != nullptr) {
        acc += __half2float(bias[c]);
    }
    // Store result as fp16
    int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = __float2half(acc);
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(
    void* output,                        // Output tensor (GPU memory)
    void* input,                         // Input tensor (GPU memory)
    void* weight,                        // Conv2d weight (GPU memory)
    void* bias,                          // Conv2d bias (GPU memory), nullptr if bias is not used
    int batch_size,
    int in_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
) {
    // Output dimensions
    int H_out = (height + 2 * padding - kernel_size) / stride + 1;
    int W_out = (width  + 2 * padding - kernel_size) / stride + 1;

    // Launch configuration
    int threads_per_block = 128;
    int blocks_x = (H_out * W_out + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_x, in_channels, batch_size);

    depthwise_conv2d_fp16_kernel<<<grid, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size, in_channels, height, width, kernel_size, stride, padding, H_out, W_out
    );
    cudaDeviceSynchronize();
}
