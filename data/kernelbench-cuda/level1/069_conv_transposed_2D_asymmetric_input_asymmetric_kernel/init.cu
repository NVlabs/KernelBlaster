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
#include <cassert>
#include <cstdio>

// -------------------------
// CUDA Transposed Conv2d Kernel
// -------------------------
// This kernel implements a 2D transposed convolution (ConvTranspose2d) for fp16 input/output.
// Accumulation is performed in fp32 for numerical stability.
// The kernel supports asymmetric input/kernel, stride, padding, dilation, output_padding, and grouping.
// Bias is optionally supported.

__global__ void conv2d_transpose_fp16_kernel(
    half* __restrict__ output,           // [N, OC, OH, OW]
    const half* __restrict__ input,      // [N, IC, IH, IW]
    const half* __restrict__ weight,     // [ICG, OC_per_group, KH, KW] (PyTorch: [IC, OC/groups, KH, KW])
    const half* __restrict__ bias,       // [OC] or nullptr
    int N,
    int IC,
    int OC,
    int IH,
    int IW,
    int KH,
    int KW,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    bool bias_enabled,
    int OH,
    int OW
) {
    // Indexing for each output element (n, oc, oh, ow)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int image_size = OC * OH * OW;
    int total = N * image_size;
    if (tid >= total) return;

    int n = tid / image_size;
    int rem = tid % image_size;
    int oc = rem / (OH * OW);
    int tmp = rem % (OH * OW);
    int oh = tmp / OW;
    int ow = tmp % OW;

    // Grouping
    int OC_per_group = OC / groups;
    int IC_per_group = IC / groups;
    int group_idx = oc / OC_per_group;

    float acc = 0.0f;

    // For each input channel in the same group
    for (int icg = 0; icg < IC_per_group; ++icg) {
        int ic = group_idx * IC_per_group + icg;
        // For each kernel position
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                // Compute input spatial position corresponding to this output
                // Formula derived from PyTorch's ConvTranspose2d doc:
                //   h_in = (oh + pad_h - kh * dilation_h) / stride_h
                //   w_in = (ow + pad_w - kw * dilation_w) / stride_w
                int h_in_nom = oh + pad_h - kh * dilation_h;
                int w_in_nom = ow + pad_w - kw * dilation_w;
                if (h_in_nom % stride_h != 0 || w_in_nom % stride_w != 0) continue;
                int ih = h_in_nom / stride_h;
                int iw = w_in_nom / stride_w;
                // Account for output_padding
                if ((oh + output_pad_h) < 0 || (ow + output_pad_w) < 0) continue;
                // Input bounds
                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;

                // PyTorch weight layout: [in_channels, out_channels/groups, kH, kW]
                // Each group contains IC_per_group input and OC_per_group output channels.
                // Our oc is relative to entire OC, need ocg = oc % OC_per_group
                int ocg = oc % OC_per_group;
                int w_idx = ((ic * OC_per_group + ocg) * KH + kh) * KW + kw;
                int i_idx = ((n * IC + ic) * IH + ih) * IW + iw;

                float inp = __half2float(input[i_idx]);
                float wgt = __half2float(weight[w_idx]);
                acc += inp * wgt;
            }
        }
    }

    if (bias_enabled && bias != nullptr)
        acc += __half2float(bias[oc]);

    // Store result in fp16
    output[tid] = __float2half(acc);
}

// Host launcher for the above kernel
void launch_gpu_implementation(
    void* output, 
    void* input, 
    void* weight, 
    void* bias, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int height_in, 
    int width_in, 
    int kernel_h, 
    int kernel_w, 
    int stride_h, 
    int stride_w, 
    int pad_h, 
    int pad_w, 
    int output_pad_h, 
    int output_pad_w, 
    int dilation_h, 
    int dilation_w, 
    int groups, 
    bool bias_enabled
) {
    // Compute output spatial shape
    int height_out = (height_in - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1;
    int width_out  = (width_in - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1;

    int N = batch_size;
    int IC = in_channels;
    int OC = out_channels;
    int IH = height_in;
    int IW = width_in;
    int KH = kernel_h;
    int KW = kernel_w;
    int OH = height_out;
    int OW = width_out;

    int total = N * OC * OH * OW;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    conv2d_transpose_fp16_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        N, IC, OC, IH, IW, KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        output_pad_h, output_pad_w,
        dilation_h, dilation_w,
        groups,
        bias_enabled,
        OH, OW
    );
    cudaDeviceSynchronize();
}
