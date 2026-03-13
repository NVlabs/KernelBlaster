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

// Utility for division rounding up
inline int div_up(int a, int b) { return (a + b - 1) / b; }

// __global__ kernel: Each thread computes one output pixel (n, oc, oh, ow)
__global__ void grouped_conv2d_transpose_fp16_kernel(
    half* __restrict__ output,        // [N, out_channels, outH, outW]
    const half* __restrict__ input,   // [N, in_channels, H, W]
    const half* __restrict__ weight,  // [in_channels, out_channels/groups, kH, kW]
    const half* __restrict__ bias,    // [out_channels] or nullptr
    int N, int in_channels, int out_channels,
    int H, int W, int outH, int outW,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_channels * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW; idx /= outW;
    int oh = idx % outH; idx /= outH;
    int oc = idx % out_channels; idx /= out_channels;
    int n  = idx;

    int out_cpg = out_channels / groups;
    int c_per_group = in_channels / groups;
    int group = oc / out_cpg;
    int ocg = oc % out_cpg;

    float acc = 0.0f;

    // For each input channel in the group
    for (int icg = 0; icg < c_per_group; ++icg) {
        int ic = group * c_per_group + icg;
        // For each kernel position
        for (int kh = 0; kh < kH; ++kh) {
            int ih_unstrided = oh + pad_h - kh * dil_h;
            if (ih_unstrided % stride_h != 0) continue;
            int ih = ih_unstrided / stride_h;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int iw_unstrided = ow + pad_w - kw * dil_w;
                if (iw_unstrided % stride_w != 0) continue;
                int iw = iw_unstrided / stride_w;
                if (iw < 0 || iw >= W) continue;

                int input_idx = ((n * in_channels + ic) * H + ih) * W + iw;
                int weight_idx = ((ic * out_cpg + ocg) * kH + kh) * kW + kw;

                float inp = __half2float(input[input_idx]);
                float wgt = __half2float(weight[weight_idx]);
                acc += inp * wgt;
            }
        }
    }
    if (bias != nullptr) {
        acc += __half2float(bias[oc]);
    }
    output[
        ((n * out_channels + oc) * outH + oh) * outW + ow
    ] = __float2half(acc);
}

// Host launcher
void launch_gpu_implementation(
    void* output, // float16* output, shape: [batch_size, out_channels, output_h, output_w]
    void* input,  // float16* input, shape: [batch_size, in_channels, height, width]
    void* weight, // float16* weight, shape: [in_channels, out_channels/groups, kernel_h, kernel_w]
    void* bias,   // float16* bias, shape: [out_channels] or nullptr if not used
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int total = batch_size * out_channels * output_h * output_w;
    int threads = 256;
    int blocks = div_up(total, threads);

    grouped_conv2d_transpose_fp16_kernel<<<blocks, threads>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        batch_size, in_channels, out_channels,
        input_h, input_w, output_h, output_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
