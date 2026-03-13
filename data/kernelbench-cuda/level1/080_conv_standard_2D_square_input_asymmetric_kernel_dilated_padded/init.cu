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

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>

// Utility: for fp16 accumulation, always accumulate in fp32 for numerical stability
__device__ __forceinline__ float half2float(half h) { return __half2float(h); }

// CUDA kernel for 2D convolution with asymmetric kernel, dilation, padding, stride
// - All pointer arguments are expected to be device pointers
// - All tensors are fp16
// - Accumulation is in fp32
// - No bias if has_bias == false
// - Input layout: (N, C_in, H, W), Weight: (C_out, C_in, KH, KW), Output: (N, C_out, OH, OW)
__global__ void conv2d_asym_pad_dil_fp16_kernel(
    half* __restrict__ output,          // (N, C_out, OH, OW)
    const half* __restrict__ input,     // (N, C_in, H, W)
    const half* __restrict__ weight,    // (C_out, C_in, KH, KW)
    const half* __restrict__ bias,      // (C_out) or nullptr if !has_bias
    int N, int C_in, int C_out,
    int H, int W,           // input H, W
    int KH, int KW,         // kernel H, W
    int stride, int pad_h, int pad_w,
    int dil_h, int dil_w,
    int OH, int OW,         // output H, W (precomputed)
    bool has_bias
) {
    // Each thread computes (n, c_out, oh, ow)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * OH * OW;
    if (tid >= total) return;

    // Decompose index
    int ow = tid % OW;
    int oh = (tid / OW) % OH;
    int c_out = (tid / (OH * OW)) % C_out;
    int n = tid / (C_out * OH * OW);

    float acc = 0.f;

    // For each input channel
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // For each kernel position
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh * stride - pad_h + kh * dil_h;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow * stride - pad_w + kw * dil_w;
                if (iw < 0 || iw >= W) continue;
                // Input index: n, c_in, ih, iw
                int in_idx = ((n * C_in + c_in) * H + ih) * W + iw;
                // Weight index: c_out, c_in, kh, kw
                int w_idx = ((c_out * C_in + c_in) * KH + kh) * KW + kw;
                acc += half2float(input[in_idx]) * half2float(weight[w_idx]);
            }
        }
    }
    // Add bias if requested
    if (has_bias && bias != nullptr) {
        acc += half2float(bias[c_out]);
    }
    // Convert to fp16 for output
    output[tid] = __float2half_rn(acc);
}

// Host launcher
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

void launch_gpu_implementation(
    void* output,            // Output tensor, shape: (batch_size, out_channels, out_h, out_w)
    void* input,             // Input tensor, shape: (batch_size, in_channels, height, width)
    void* weight,            // Conv2d weights, shape: (out_channels, in_channels, kernel_h, kernel_w)
    void* bias,              // Conv2d bias, shape: (out_channels,) or nullptr if bias == false
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    bool has_bias
) {
    // Compute output spatial size
    int OH = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int OW = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;
    int total = batch_size * out_channels * OH * OW;

    // Kernel launch configuration
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_asym_pad_dil_fp16_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(output),
        reinterpret_cast<const half*>(input),
        reinterpret_cast<const half*>(weight),
        reinterpret_cast<const half*>(bias),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_h, kernel_w,
        stride, pad_h, pad_w,
        dilation_h, dilation_w,
        OH, OW,
        has_bias
    );
    cudaDeviceSynchronize();
}
