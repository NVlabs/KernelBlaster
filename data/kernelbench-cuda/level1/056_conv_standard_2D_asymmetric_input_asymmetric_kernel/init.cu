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
// cuda_model.cuh (implementation)
// Efficient FP16 2D convolution kernel for asymmetric input and kernel sizes.
// I/O tensors are in fp16; accumulation is in fp32 for accuracy.
//
// Signature:
// void launch_gpu_implementation(
//     void* output,
//     void* input,
//     void* weight,
//     void* bias,
//     int batch_size,
//     int in_channels,
//     int out_channels,
//     int input_height,
//     int input_width,
//     int kernel_h,
//     int kernel_w,
//     int stride_h,
//     int stride_w,
//     int pad_h,
//     int pad_w,
//     int dilation_h,
//     int dilation_w,
//     int groups,
//     bool has_bias
// );

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

// Utility for bounds-safe __half2float (for fp16 IO)
__device__ __forceinline__ float h2f(half v) { return __half2float(v); }
__device__ __forceinline__ half f2h(float v) { return __float2half_rn(v); }

// Efficient NHWC-like indexing for this convolution
// Input:  (N, C, H, W)
// Weight: (O, C/g, kH, kW)
// Output: (N, O, outH, outW)

// CUDA Kernel for 2D convolution, supports asymmetric kernel/input/stride/pad/dilation/groups, fp16 I/O, fp32 accum
__global__ void conv2d_nchw_fp16_kernel(
    const half* __restrict__ input,   // [N, C, H, W]
    const half* __restrict__ weight,  // [O, Cg, kH, kW]
    const half* __restrict__ bias,    // [O] or nullptr
    half* __restrict__ output,        // [N, O, outH, outW]
    int N, int C, int O,
    int H, int W,
    int kH, int kW,
    int outH, int outW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    bool has_bias
) {
    // Each thread computes one output pixel (n, o, h, w)
    int n = blockIdx.z;
    int o = blockIdx.y * blockDim.y + threadIdx.y;
    int out_hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) return;
    int out_h = out_hw / outW;
    int out_w = out_hw % outW;
    if (out_h >= outH || out_w >= outW) return;

    int group = o / (O / groups);
    int c_per_group = C / groups;

    float acc = 0.0f;
    for (int c = 0; c < c_per_group; ++c) {
        int c_in = group * c_per_group + c;
        for (int kh = 0; kh < kH; ++kh) {
            int in_h = out_h * stride_h - pad_h + kh * dilation_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int in_w = out_w * stride_w - pad_w + kw * dilation_w;
                if (in_w < 0 || in_w >= W) continue;
                // input: [N, C, H, W]
                int in_idx = ((n * C + c_in) * H + in_h) * W + in_w;
                // weight: [O, Cg, kH, kW]
                int w_idx = (((o * c_per_group + c) * kH) + kh) * kW + kw;
                acc += h2f(input[in_idx]) * h2f(weight[w_idx]);
            }
        }
    }
    if (has_bias && bias != nullptr) acc += h2f(bias[o]);
    output[(((n * O + o) * outH) + out_h) * outW + out_w] = f2h(acc);
}

// Host-side launcher for the above kernel
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    bool has_bias
) {
    // Compute output shape
    int c_per_group = in_channels / groups;
    int outH = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int outW = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Kernel launch parameters: Each thread computes one output pixel (n, o, h, w)
    // Grid: (ceil(outH*outW/tx), ceil(out_channels/ty), batch_size)
    // Block: (tx, ty)
    constexpr int tx = 16, ty = 8;
    dim3 block(tx, ty);
    dim3 grid(
        (outH * outW + tx - 1) / tx,
        (out_channels + ty - 1) / ty,
        batch_size
    );

    conv2d_nchw_fp16_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        has_bias ? static_cast<const half*>(bias) : nullptr,
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_h, kernel_w,
        outH, outW,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        has_bias
    );

    cudaDeviceSynchronize();
}

