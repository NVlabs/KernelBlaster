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
// cuda_model.cuh

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include <cstdio>

// Pointwise 1x1 2D convolution kernel (fp16 I/O, fp32 accumulation).
// Layout: NCHW for input/output, OIHW for weight.
//   input:  [N, C_in, H, W]   (half*)
//   weight: [C_out, C_in, 1, 1] (half*)
//   bias:   [C_out] (half*) or nullptr if no bias
//   output: [N, C_out, H, W] (half*)

__global__ void pointwise_conv2d_nchw_fp16_kernel(
    const half* __restrict__ input,   // [N, C_in, H, W]
    const half* __restrict__ weight,  // [C_out, C_in, 1, 1]
    const half* __restrict__ bias,    // [C_out] or nullptr
    half* __restrict__ output,        // [N, C_out, H, W]
    int N,
    int C_in,
    int C_out,
    int H,
    int W,
    bool has_bias
) {
    // Flattened output index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;
    if (idx >= total) return;

    // Compute output tensor indices
    int w = idx % W;
    int h = (idx / W) % H;
    int c_out = (idx / (W * H)) % C_out;
    int n = idx / (W * H * C_out);

    // Compute output[n, c_out, h, w] = sum_{c_in} input[n, c_in, h, w] * weight[c_out, c_in, 0, 0] + bias[c_out]
    float acc = 0.0f;
    int input_base = ((n * C_in * H + 0) * W + 0); // For pointer arithmetic
    int weight_base = c_out * C_in;

    // Loop over input channel (vectorize if possible)
    int c_in = 0;
#if __CUDA_ARCH__ >= 530
    // Vectorize with half2 if C_in even and aligned
    for (; c_in + 1 < C_in; c_in += 2) {
        int input_idx0 = ((n * C_in + c_in) * H + h) * W + w;
        int input_idx1 = ((n * C_in + c_in + 1) * H + h) * W + w;
        int weight_idx0 = weight_base + c_in;
        int weight_idx1 = weight_base + c_in + 1;
        half2 in2 = __halves2half2(input[input_idx0], input[input_idx1]);
        half2 w2  = __halves2half2(weight[weight_idx0], weight[weight_idx1]);
        float2 prod = __half22float2(__hmul2(in2, w2));
        acc += prod.x + prod.y;
    }
#endif
    for (; c_in < C_in; ++c_in) {
        int input_idx = ((n * C_in + c_in) * H + h) * W + w;
        int weight_idx = weight_base + c_in;
        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
    }

    if (has_bias && bias != nullptr) {
        acc += __half2float(bias[c_out]);
    }

    // Convert to half for output
    output[idx] = __float2half(acc);
}

// Host launcher
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    bool has_bias
) {
    // Alias to half*
    half* d_output = static_cast<half*>(output);
    const half* d_input = static_cast<const half*>(input);
    const half* d_weight = static_cast<const half*>(weight);
    const half* d_bias = static_cast<const half*>(bias);

    int N = batch_size, C_in = in_channels, C_out = out_channels, H = height, W = width;
    int total = N * C_out * H * W;

    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    pointwise_conv2d_nchw_fp16_kernel<<<blocks, threads_per_block>>>(
        d_input, d_weight, d_bias, d_output,
        N, C_in, C_out, H, W, has_bias
    );
    cudaDeviceSynchronize();
}

