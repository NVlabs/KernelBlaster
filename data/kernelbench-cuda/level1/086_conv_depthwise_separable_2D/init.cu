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
// Depthwise Separable 2D Convolution (fp16 I/O, fp32 accumulate) for L40S+ (Ada) GPUs
// Host launch code for: 
// void launch_gpu_implementation(
//     void* output, void* input, void* depthwise_weight, void* depthwise_bias, 
//     void* pointwise_weight, void* pointwise_bias, 
//     int batch_size, int in_channels, int out_channels, 
//     int input_height, int input_width, 
//     int kernel_size, int stride, int padding, int dilation
// );

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cassert>
#include <cstdio>

// ---- Depthwise Conv2d Kernel ----
// Each thread computes one output element (N, C, H_out, W_out)
__global__ void depthwise_conv2d_fp16_kernel(
    const half* __restrict__ input,         // [N, C, H, W]
    const half* __restrict__ weight,        // [C, kH, kW]
    const half* __restrict__ bias,          // [C] or nullptr
    half* __restrict__ output,              // [N, C, H_out, W_out]
    int N, int C, int H, int W,
    int kH, int kW,
    int stride, int padding, int dilation,
    int H_out, int W_out
) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    if (hw >= H_out * W_out) return;
    int h_out = hw / W_out;
    int w_out = hw % W_out;

    float acc = 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
        int h_in = h_out * stride - padding + kh * dilation;
        if (h_in < 0 || h_in >= H) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int w_in = w_out * stride - padding + kw * dilation;
            if (w_in < 0 || w_in >= W) continue;
            int input_idx = ((n * C + c) * H + h_in) * W + w_in;
            int weight_idx = (c * kH + kh) * kW + kw;
            acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
        }
    }
    // Add bias if present
    if (bias != nullptr) {
        acc += __half2float(bias[c]);
    }
    // Store as half
    int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = __float2half(acc);
}

// ---- Pointwise Conv2d (1x1) Kernel ----
// Each thread computes one output element (N, Cout, H, W)
__global__ void pointwise_conv2d_fp16_kernel(
    const half* __restrict__ input,         // [N, Cin, H, W]
    const half* __restrict__ weight,        // [Cout, Cin, 1, 1] (PyTorch format)
    const half* __restrict__ bias,          // [Cout] or nullptr
    half* __restrict__ output,              // [N, Cout, H, W]
    int N, int Cin, int Cout, int H, int W
) {
    int n = blockIdx.x;
    int cout = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    if (hw >= H * W) return;
    int h = hw / W;
    int w = hw % W;

    float acc = 0.0f;
    for (int cin = 0; cin < Cin; ++cin) {
        int in_idx = ((n * Cin + cin) * H + h) * W + w;
        int w_idx = ((cout * Cin) + cin);
        acc += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
    }
    // Add bias if present
    if (bias != nullptr) {
        acc += __half2float(bias[cout]);
    }
    int out_idx = ((n * Cout + cout) * H + h) * W + w;
    output[out_idx] = __float2half(acc);
}

// ---- Host Launcher ----
void launch_gpu_implementation(
    void* output,                        // Output tensor (float16, GPU memory)
    void* input,                         // Input tensor (float16, GPU memory)
    void* depthwise_weight,              // Depthwise Conv2d weight (float16, GPU memory)
    void* depthwise_bias,                // Depthwise Conv2d bias (float16, GPU memory) or nullptr if no bias
    void* pointwise_weight,              // Pointwise Conv2d weight (float16, GPU memory)
    void* pointwise_bias,                // Pointwise Conv2d bias (float16, GPU memory) or nullptr if no bias
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    using namespace std;

    // 1. Compute output shapes
    int H_out = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int W_out = (input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // 2. Allocate intermediate buffer for depthwise output: [N, C, H_out, W_out]
    half* depthwise_out;
    size_t dw_bytes = batch_size * in_channels * H_out * W_out * sizeof(half);
    cudaMalloc(&depthwise_out, dw_bytes);

    // 3. Launch depthwise kernel
    // Grid: (N, C, ceil(H_out*W_out/threads))
    int threads = 256;
    int hw_out = H_out * W_out;
    int blocks_z = (hw_out + threads - 1) / threads;
    dim3 grid_dw(batch_size, in_channels, blocks_z);
    dim3 block_dw(threads);

    depthwise_conv2d_fp16_kernel<<<grid_dw, block_dw>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(depthwise_weight),
        static_cast<const half*>(depthwise_bias),
        depthwise_out,
        batch_size, in_channels, input_height, input_width,
        kernel_size, kernel_size, stride, padding, dilation,
        H_out, W_out
    );

    // 4. Launch pointwise kernel (1x1 conv, PyTorch format: [out_channels, in_channels, 1, 1])
    // Grid: (N, out_channels, ceil(H_out*W_out/threads))
    int blocks_z_pw = (hw_out + threads - 1) / threads;
    dim3 grid_pw(batch_size, out_channels, blocks_z_pw);
    dim3 block_pw(threads);

    // PyTorch's pointwise weight is [out_channels, in_channels, 1, 1], flatten to [out_channels, in_channels]
    pointwise_conv2d_fp16_kernel<<<grid_pw, block_pw>>>(
        depthwise_out,
        static_cast<const half*>(pointwise_weight),
        static_cast<const half*>(pointwise_bias),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels, H_out, W_out
    );

    cudaFree(depthwise_out);

    // Synchronize for correctness
    cudaDeviceSynchronize();
}

