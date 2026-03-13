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

// This kernel implements a fast fp16 ConvTranspose2d (transposed convolution, "deconvolution").
// It supports arbitrary batch size, groups, stride, padding, and output padding.
// Input, weight, output are in fp16; accumulation is performed in fp32 for accuracy.
// Kernel layout matches PyTorch: weight shape (in_channels, out_channels // groups, kernel_h, kernel_w)
// Input: (N, in_channels, H, W), Output: (N, out_channels, H_out, W_out)
// No bias for this version (bias support is easy to add).

// Helper: ceil division
__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// Out-of-bounds-safe load for input, returns zero if out-of-bounds
__device__ inline float input_load(
    const half* input, int n, int c, int h, int w,
    int N, int C, int H, int W)
{
    if (n < 0 || n >= N || c < 0 || c >= C || h < 0 || h >= H || w < 0 || w >= W) return 0.f;
    int idx = ((n * C + c) * H + h) * W + w;
    return __half2float(input[idx]);
}

// Main kernel: each thread computes one (n, out_c, h_out, w_out) output element
__global__ void conv_transpose2d_fp16_kernel(
    const half* __restrict__ input,      // [N, in_channels, H, W]
    const half* __restrict__ weight,     // [in_channels, out_channels_per_group, kH, kW]
    half* __restrict__ output,           // [N, out_channels, H_out, W_out]
    int N, int in_channels, int out_channels, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding, int output_padding, int groups,
    const half* __restrict__ bias, bool has_bias)
{
    int out_c = blockIdx.x * blockDim.x + threadIdx.x; // out channel
    int w_out = blockIdx.y * blockDim.y + threadIdx.y; // output width
    int h_out = blockIdx.z % H_out;                    // output height
    int n = blockIdx.z / H_out;                        // batch

    if (n >= N || out_c >= out_channels || h_out >= H_out || w_out >= W_out) return;

    // Determine group and local indices
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group_idx = out_c / out_channels_per_group;
    int out_c_in_group = out_c % out_channels_per_group;

    // Accumulate in fp32 for accuracy
    float acc = 0.f;

    // For transposed conv, input spatial index (h_in, w_in) that contributes to (h_out, w_out) is:
    // h_in = (h_out + padding - kH * dilation + output_padding) / stride + kH
    // But for stride > 1, not every (h_out, w_out) is covered.
    // For each possible input position, compute which kernel pos (kh, kw) can contribute.

    for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // The input location that contributes to this output pixel
                int h_in = (h_out + padding - kh) / stride;
                int w_in = (w_out + padding - kw) / stride;
                // Check if this (h_out, w_out) is reached by some (h_in, kh)
                if (((h_out + padding - kh) % stride != 0) || ((w_out + padding - kw) % stride != 0)) continue;
                if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;

                int c_in_abs = group_idx * in_channels_per_group + c_in;
                int weight_idx =
                    ((c_in_abs * out_channels_per_group + out_c_in_group) * kernel_size + kh) * kernel_size + kw;
                int input_idx =
                    ((n * in_channels + c_in_abs) * H + h_in) * W + w_in;

                float ix = __half2float(input[input_idx]);
                float wx = __half2float(weight[weight_idx]);
                acc += ix * wx;
            }
        }
    }

    // Add bias if present
    if (has_bias && bias != nullptr)
        acc += __half2float(bias[out_c]);

    // Write result
    int out_idx = ((n * out_channels + out_c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = __float2half(acc);
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
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    bool has_bias)
{
    // Output shape calculation (PyTorch ConvTranspose2d formula):
    // H_out = (H-1)*stride - 2*padding + kernel_size + output_padding
    int H_out = (height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (width - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Grid/block config: tile out_channels, W_out, (N*H_out)
    // Each thread computes one output pixel (n, out_c, h_out, w_out)
    const int block_out_c = 32;
    const int block_w_out = 8;
    dim3 block(block_out_c, block_w_out, 1);
    dim3 grid(
        div_up(out_channels, block_out_c),
        div_up(W_out, block_w_out),
        batch_size * H_out
    );

    conv_transpose2d_fp16_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels, height, width,
        H_out, W_out,
        kernel_size, stride, padding, output_padding, groups,
        static_cast<const half*>(bias), has_bias
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
        assert(false);
    }
}
