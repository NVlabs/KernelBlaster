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
#include <mma.h>
#include <cassert>
#include <cstdio>

// Helper to check for CUDA errors
#define CUDA_CHECK(call)  \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; } } while (0)

// Helper to compute output shape
inline int out_hw(int in_hw, int kernel, int stride, int pad, int dilation) {
    // Follows PyTorch's Conv2d output shape formula
    return (in_hw + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

// Simple, high-performance, numerically stable convolution kernel for fp16 input/output,
// with fp32 accumulation, supporting stride, padding, dilation, groups, bias.
// Layout: NCHW for all tensors.
__global__ void conv2d_fp16_nchw_kernel(
    const half* __restrict__ input,     // [N, C_in, H, W]
    const half* __restrict__ weight,    // [C_out, C_in/groups, K, K]
    const half* __restrict__ bias,      // [C_out] or nullptr if no bias
    half* __restrict__ output,          // [N, C_out, H_out, W_out]
    int N, int C_in, int C_out,
    int H, int W,
    int H_out, int W_out,
    int K, int stride, int pad, int dilation, int groups, bool has_bias
) {
    // Compute global thread index (over output tensor)
    int n = blockIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int hw_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_out >= C_out || hw_out >= H_out*W_out) return;

    int h_out = hw_out / W_out;
    int w_out = hw_out % W_out;

    // Calculate which group this output channel belongs to
    int group_idx = c_out / (C_out / groups);
    int c_in_start = group_idx * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);

    // Accumulator in fp32 for numerical stability
    float acc = 0.0f;

    // For each input channel in this group
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        int c_in_g = c_in - c_in_start;
        // For each kernel row
        for (int k_h = 0; k_h < K; ++k_h) {
            int in_h = h_out * stride - pad + k_h * dilation;
            if (in_h < 0 || in_h >= H) continue;
            // For each kernel col
            for (int k_w = 0; k_w < K; ++k_w) {
                int in_w = w_out * stride - pad + k_w * dilation;
                if (in_w < 0 || in_w >= W) continue;
                // Indexes
                int in_idx = ((n * C_in + c_in) * H + in_h) * W + in_w;
                int w_idx = (((c_out * (C_in / groups) + c_in_g) * K + k_h) * K + k_w);
                acc += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
            }
        }
    }
    // Add bias if present
    if (has_bias && bias) {
        acc += __half2float(bias[c_out]);
    }
    // Store output as half
    int out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[out_idx] = __float2half(acc);
}

// Host launch function for the above kernel
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
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool has_bias
) {
    // Compute output spatial dimensions (NCHW layout)
    int H_out = out_hw(height, kernel_size, stride, padding, dilation);
    int W_out = out_hw(width, kernel_size, stride, padding, dilation);

    dim3 threads(16, 8, 1); // 16 spatial, 8 channels (tuned for occupancy)
    dim3 blocks(
        (H_out * W_out + threads.x - 1) / threads.x,
        (out_channels + threads.y - 1) / threads.y,
        batch_size
    );

    conv2d_fp16_nchw_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        height, width,
        H_out, W_out,
        kernel_size, stride, padding, dilation, groups, has_bias
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
