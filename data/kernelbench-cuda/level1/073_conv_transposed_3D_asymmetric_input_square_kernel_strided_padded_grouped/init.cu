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
// Implementation of 3D transposed convolution (ConvTranspose3d) in half precision (fp16)
// using CUDA, with accumulation in fp32 for numerical stability.
// This kernel supports groups and arbitrary padding/stride/output_padding.
//
// I/O tensor layout matches PyTorch: NCDHW (channels first)
// Weight layout is (in_channels, out_channels/groups, kD, kH, kW)
// Bias is nullptr since bias=False in this case.
//
// This code is tuned for L40S-class GPUs, but is portable and correct for all CUDA devices.
//
// Author: CUDA Programming Assistant

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdint.h>
#include <cstdio>

// Utility macro for CUDA calls
#define CUDA_CHECK(call)                                            \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                   cudaGetErrorString(err));                       \
            assert(0);                                             \
        }                                                          \
    } while (0)

// CUDA kernel for 3D transposed convolution (ConvTranspose3d)
__global__ void conv_transpose3d_fp16_ncdhw_kernel(
    half* __restrict__ output,      // [N, out_channels, D_out, H_out, W_out]
    const half* __restrict__ input, // [N, in_channels, D, H, W]
    const half* __restrict__ weight,// [in_channels, out_channels/groups, kD, kH, kW]
    const half* __restrict__ bias,  // [out_channels] or nullptr
    int N,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int ksize,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    // Compute output spatial sizes
    const int kD = ksize, kH = ksize, kW = ksize;
    const int stride_d = stride, stride_h = stride, stride_w = stride;
    const int pad_d = padding, pad_h = padding, pad_w = padding;
    const int out_pad_d = output_padding, out_pad_h = output_padding, out_pad_w = output_padding;

    // See PyTorch ConvTranspose3d output shape formula:
    // out_D = (D - 1) * stride - 2*padding + kD + output_padding
    const int D_out = (D - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    const int H_out = (H - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    const int W_out = (W - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    // Each thread computes 1 output voxel (n, oc, od, oh, ow)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * out_channels * D_out * H_out * W_out;
    if (tid >= total) return;

    // Decompose flat index
    int ow = tid % W_out;
    int oh = (tid / W_out) % H_out;
    int od = (tid / (W_out * H_out)) % D_out;
    int oc = (tid / (W_out * H_out * D_out)) % out_channels;
    int n  = tid / (W_out * H_out * D_out * out_channels);

    // Compute group and indices in group
    int oc_group = oc / (out_channels / groups);
    int oc_in_group = oc % (out_channels / groups);
    int in_ch_start = oc_group * (in_channels / groups);
    int in_ch_end   = in_ch_start + (in_channels / groups);

    // Accumulator in fp32 for numerical stability
    float acc = 0.0f;

    // For all in_channels in this group
    for (int ic = in_ch_start; ic < in_ch_end; ++ic) {
        // For each filter position (kd, kh, kw)
        for (int kd = 0; kd < kD; ++kd) {
            int id_unstrided = od + pad_d - kd;
            if (id_unstrided % stride_d != 0) continue;
            int id = id_unstrided / stride_d;
            if (id < 0 || id >= D) continue;
            for (int kh = 0; kh < kH; ++kh) {
                int ih_unstrided = oh + pad_h - kh;
                if (ih_unstrided % stride_h != 0) continue;
                int ih = ih_unstrided / stride_h;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; ++kw) {
                    int iw_unstrided = ow + pad_w - kw;
                    if (iw_unstrided % stride_w != 0) continue;
                    int iw = iw_unstrided / stride_w;
                    if (iw < 0 || iw >= W) continue;

                    // Input: [N, in_channels, D, H, W]
                    int input_idx = ((n * in_channels + ic) * D + id) * H * W + ih * W + iw;
                    // Weight: [in_channels, out_channels/groups, kD, kH, kW]
                    // For group g: ic in [g * (in_channels/groups), ...], oc_in_group in [0, out_channels/groups)
                    int weight_idx = (((ic) * (out_channels / groups) + oc_in_group) * kD + kd) * kH * kW + kh * kW + kw;

                    float inp = __half2float(input[input_idx]);
                    float wgt = __half2float(weight[weight_idx]);
                    acc += inp * wgt;
                }
            }
        }
    }
    // Add bias if present
    if (bias != nullptr) {
        acc += __half2float(bias[oc]);
    }
    // Store as fp16
    int out_idx = ((n * out_channels + oc) * D_out + od) * H_out * W_out + oh * W_out + ow;
    output[out_idx] = __float2half(acc);
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(
    void* output,                  // Output tensor (float16)
    void* input,                   // Input tensor (float16)
    void* weight,                  // Weight tensor (float16)
    void* bias,                    // Bias tensor (float16) or nullptr
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t depth,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
) {
    // Compute output shape
    int kD = kernel_size, kH = kernel_size, kW = kernel_size;
    int stride_d = stride, stride_h = stride, stride_w = stride;
    int pad_d = padding, pad_h = padding, pad_w = padding;
    int out_pad_d = output_padding, out_pad_h = output_padding, out_pad_w = output_padding;

    int D_out = (depth - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    int H_out = (height - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    int W_out = (width - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    int64_t total = batch_size * out_channels * D_out * H_out * W_out;

    // Set up launch
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    conv_transpose3d_fp16_ncdhw_kernel<<<blocks, threads>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        (int)batch_size,
        (int)in_channels,
        (int)out_channels,
        (int)depth, (int)height, (int)width,
        (int)kernel_size,
        (int)stride,
        (int)padding,
        (int)output_padding,
        (int)groups
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
