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
#include <cstdint>
#include <cstdio>

// Use CUDA built-in half for fp16 tensors
// Accumulation is always in float (fp32) for numerical stability

// Utility for fast division and ceil
inline __host__ __device__ int div_up(int a, int b) { return (a + b - 1) / b; }

// Convolution kernel for fp16 input/output with fp32 accumulation
// NCHW layout for input/output, OIHW for weights
// Supports asymmetric kernels, stride, padding, dilation, groups, optional bias
// No fused activation
__global__ void conv2d_fp16_nchw_kernel(
    const half* __restrict__ input,         // [N, C_in, H_in, W_in]
    const half* __restrict__ weight,        // [C_out, C_in/groups, K_h, K_w]
    const half* __restrict__ bias,          // [C_out] or nullptr
    half* __restrict__ output,              // [N, C_out, H_out, W_out]
    int N, int C_in, int C_out, int H_in, int W_in,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    bool has_bias,
    int H_out, int W_out
) {
    // Each thread computes one output pixel: (n, oc, h_out, w_out)
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C_out * H_out * W_out;
    if (out_idx >= total_outputs) return;

    int w_out = out_idx % W_out;
    int h_out = (out_idx / W_out) % H_out;
    int oc = (out_idx / (W_out * H_out)) % C_out;
    int n = out_idx / (C_out * H_out * W_out);

    // Find group and channel within group
    int group_id = oc / (C_out / groups);
    int c_out_per_group = C_out / groups;
    int c_in_per_group = C_in / groups;
    int oc_in_group = oc % c_out_per_group;

    float acc = 0.0f;

    // Loop over input channels of this group
    for (int ic_in_group = 0; ic_in_group < c_in_per_group; ++ic_in_group) {
        int ic = group_id * c_in_per_group + ic_in_group;
        // Loop over kernel
        for (int kh = 0; kh < K_h; ++kh) {
            int h_in = h_out * stride_h - pad_h + kh * dilation_h;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < K_w; ++kw) {
                int w_in = w_out * stride_w - pad_w + kw * dilation_w;
                if (w_in < 0 || w_in >= W_in) continue;
                int inp_idx = ((n * C_in + ic) * H_in + h_in) * W_in + w_in;
                int wgt_idx = (((oc) * c_in_per_group + ic_in_group) * K_h + kh) * K_w + kw;
                float inp = __half2float(input[inp_idx]);
                float wgt = __half2float(weight[wgt_idx]);
                acc += inp * wgt;
            }
        }
    }

    // Add bias if present
    if (has_bias && bias != nullptr) {
        acc += __half2float(bias[oc]);
    }

    // Store as half
    int out_offset = ((n * C_out + oc) * H_out + h_out) * W_out + w_out;
    output[out_offset] = __float2half(acc);
}

// Host launcher for the above kernel
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    bool has_bias
) {
    // Compute output size
    int64_t H_out = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    int64_t W_out = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
    int64_t total_outputs = batch_size * out_channels * H_out * W_out;

    // Launch configuration
    int threads_per_block = 256;
    int blocks = div_up(total_outputs, threads_per_block);

    conv2d_fp16_nchw_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        has_bias ? static_cast<const half*>(bias) : nullptr,
        static_cast<half*>(output),
        static_cast<int>(batch_size),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(input_height),
        static_cast<int>(input_width),
        static_cast<int>(kernel_height),
        static_cast<int>(kernel_width),
        static_cast<int>(stride_h),
        static_cast<int>(stride_w),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        static_cast<int>(dilation_h),
        static_cast<int>(dilation_w),
        static_cast<int>(groups),
        has_bias,
        static_cast<int>(H_out),
        static_cast<int>(W_out)
    );
    cudaDeviceSynchronize();
}
