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
#include <stdint.h>
#include <cassert>

// The kernel below implements a performant fp16 3D transposed convolution (deconvolution) in NCDHW format.
// Accumulation is done in fp32 for numerical stability. 
// This kernel supports arbitrary kernel size, stride, padding, output padding, groups, and optional bias.
// Weight tensor layout: [in_channels, out_channels/groups, kernel_d, kernel_w, kernel_h] (matching PyTorch's ConvTranspose3d).
// Input: [N, in_channels, D, W, H]
// Output: [N, out_channels, D_out, W_out, H_out]
// This kernel is designed for L40S and newer architectures (compute capability >= 8.0) for optimal fp16 performance.

#define CUDA_CHECK(err) assert((err) == cudaSuccess)

__device__ __forceinline__ int out_idx(
    int n, int oc, int od, int ow, int oh,
    int N, int OC, int OD, int OW, int OH
) {
    // Output is [N, OC, OD, OW, OH]
    return (((n * OC + oc) * OD + od) * OW + ow) * OH + oh;
}

__device__ __forceinline__ int in_idx(
    int n, int ic, int id, int iw, int ih,
    int N, int IC, int ID, int IW, int IH
) {
    // Input is [N, IC, ID, IW, IH]
    return (((n * IC + ic) * ID + id) * IW + iw) * IH + ih;
}

__device__ __forceinline__ int weight_idx(
    int ic, int oc_g, int kd, int kw, int kh,
    int OCg, int KD, int KW, int KH
) {
    // Weight is [IC, OCg, KD, KW, KH]
    return (((ic * OCg + oc_g) * KD + kd) * KW + kw) * KH + kh;
}

__global__ void conv3d_transpose_fp16_kernel(
    half* __restrict__ output,        // [N, OC, OD, OW, OH]
    const half* __restrict__ input,   // [N, IC, ID, IW, IH]
    const half* __restrict__ weight,  // [IC, OCg, KD, KW, KH]
    const half* __restrict__ bias,    // [OC] or nullptr
    int N, int IC, int OC, int ID, int IW, int IH,
    int KD, int KW, int KH,
    int SD, int SW, int SH,
    int PD, int PW, int PH,
    int OPD, int OPW, int OPH,
    int groups,
    bool has_bias,
    int OD, int OW, int OH,           // output shape
    int OCg                         // out_channels per group
) {
    // Each thread computes one output element: (n, oc, od, ow, oh)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OC * OD * OW * OH;
    if (tid >= total) return;

    // Decompose tid to (n, oc, od, ow, oh)
    int t = tid;
    int oh = t % OH;       t /= OH;
    int ow = t % OW;       t /= OW;
    int od = t % OD;       t /= OD;
    int oc = t % OC;       t /= OC;
    int n  = t;

    // Find group and local channel index
    int g = oc / OCg;
    int oc_g = oc % OCg;
    int ic_start = g * (IC / groups);
    int ic_end = ic_start + (IC / groups);

    // Compute corresponding region in input
    // For each kernel position, find if there's a contributing input.
    float acc = 0.0f;

    // Loop over kernel
    for (int kd = 0; kd < KD; ++kd) {
        int id = (od + PD - kd) / SD;
        if (id < 0 || id >= ID) continue;
        if ((od + PD - kd) % SD != 0) continue;
        for (int kw = 0; kw < KW; ++kw) {
            int iw = (ow + PW - kw) / SW;
            if (iw < 0 || iw >= IW) continue;
            if ((ow + PW - kw) % SW != 0) continue;
            for (int kh = 0; kh < KH; ++kh) {
                int ih = (oh + PH - kh) / SH;
                if (ih < 0 || ih >= IH) continue;
                if ((oh + PH - kh) % SH != 0) continue;
                // For all input channels in group
                for (int ic = ic_start; ic < ic_end; ++ic) {
                    int i_idx = in_idx(n, ic, id, iw, ih, N, IC, ID, IW, IH);
                    int w_idx = weight_idx(ic - ic_start, oc_g, kd, kw, kh, OCg, KD, KW, KH);
                    float inp = __half2float(input[i_idx]);
                    float w = __half2float(weight[w_idx]);
                    acc += inp * w;
                }
            }
        }
    }
    // Add bias if present
    if (has_bias) {
        acc += __half2float(bias[oc]);
    }
    // Cast to fp16 and store
    output[out_idx(n, oc, od, ow, oh, N, OC, OD, OW, OH)] = __float2half(acc);
}

// Host-side launcher
void launch_gpu_implementation(
    void* output_,
    void* input_,
    void* weight_,
    void* bias_,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int width,
    int height,
    int kernel_depth,
    int kernel_width,
    int kernel_height,
    int stride_d,
    int stride_w,
    int stride_h,
    int pad_d,
    int pad_w,
    int pad_h,
    int out_pad_d,
    int out_pad_w,
    int out_pad_h,
    int groups,
    bool has_bias
) {
    // Input: [N, IC, ID, IW, IH]
    // Weight: [IC, OC/groups, KD, KW, KH]
    // Output: [N, OC, OD, OW, OH]

    int N = batch_size;
    int IC = in_channels;
    int OC = out_channels;
    int ID = depth;
    int IW = width;
    int IH = height;
    int KD = kernel_depth;
    int KW = kernel_width;
    int KH = kernel_height;
    int SD = stride_d;
    int SW = stride_w;
    int SH = stride_h;
    int PD = pad_d;
    int PW = pad_w;
    int PH = pad_h;
    int OPD = out_pad_d;
    int OPW = out_pad_w;
    int OPH = out_pad_h;

    int OCg = OC / groups;
    int ICg = IC / groups;

    // Compute output spatial dims (formula from PyTorch docs)
    int OD = (ID - 1) * SD - 2 * PD + KD + OPD;
    int OW = (IW - 1) * SW - 2 * PW + KW + OPW;
    int OH = (IH - 1) * SH - 2 * PH + KH + OPH;

    half* output = static_cast<half*>(output_);
    const half* input = static_cast<const half*>(input_);
    const half* weight = static_cast<const half*>(weight_);
    const half* bias = static_cast<const half*>(bias_);

    // Launch one thread per output element
    int total = N * OC * OD * OW * OH;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv3d_transpose_fp16_kernel<<<blocks, threads>>>(
        output, input, weight, bias,
        N, IC, OC, ID, IW, IH,
        KD, KW, KH,
        SD, SW, SH,
        PD, PW, PH,
        OPD, OPW, OPH,
        groups,
        has_bias,
        OD, OW, OH,
        OCg
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
