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
/*
CUDA implementation of 1D transposed convolution (ConvTranspose1d) in fp16.

PyTorch reference:
    y = F.conv_transpose1d(
        x, weight, bias, stride, padding, output_padding=0, groups=1, dilation
    )

Input:
    x: (batch_size, in_channels, input_length), dtype=half
    weight: (in_channels, out_channels, kernel_size), dtype=half
        (PyTorch: [in_channels, out_channels, kernel_size])
    bias: (out_channels,) or nullptr, dtype=half
    output: (batch_size, out_channels, output_length), dtype=half

Kernel supports stride, padding, and dilation, with fp16 I/O and fp32 accumulation.
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cmath>

// Helper: compute output length for ConvTranspose1d
inline int conv_transpose1d_output_length(
    int input_length, int kernel_size, int stride, int padding, int dilation
) {
    // PyTorch formula (output_padding=0, groups=1):
    // output_length = (input_length - 1) * stride - 2*padding + dilation * (kernel_size - 1) + 1
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// CUDA kernel for fp16 ConvTranspose1d (NCHW format, out: NCO)
__global__ void conv1d_transpose_fp16_kernel(
    const half* __restrict__ input,      // [N, IC, IL]
    const half* __restrict__ weight,     // [IC, OC, K]
    const half* __restrict__ bias,       // [OC] or nullptr
    half* __restrict__ output,           // [N, OC, OL]
    int N, int IC, int OC,
    int IL, int OL, int K,
    int stride, int padding, int dilation,
    bool has_bias
) {
    // Each thread computes one output element: (n, oc, ol)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OC * OL;
    if (idx >= total) return;

    int ol = idx % OL;
    int oc = (idx / OL) % OC;
    int n = idx / (OC * OL);

    float acc = 0.0f;

    // For ConvTranspose1d, for each output position ol, find all (ic, il, k) that contribute:
    // For all il in [0, IL)
    //   For all ic in [0, IC)
    //     For all k in [0, K)
    //       If ol + padding - k*dilation is divisible by stride, and
    //          il = (ol + padding - k*dilation) / stride in [0, IL)
    //         acc += input[n, ic, il] * weight[ic, oc, k]
    for (int ic = 0; ic < IC; ++ic) {
        for (int k = 0; k < K; ++k) {
            int il_numer = ol + padding - k * dilation;
            if (il_numer % stride != 0) continue;
            int il = il_numer / stride;
            if (il < 0 || il >= IL) continue;
            // input: [N, IC, IL]
            // weight: [IC, OC, K]
            float inp = __half2float(input[n * (IC * IL) + ic * IL + il]);
            float w = __half2float(weight[ic * (OC * K) + oc * K + k]);
            acc += inp * w;
        }
    }
    if (has_bias && bias != nullptr) {
        acc += __half2float(bias[oc]);
    }
    // Store as fp16 (clamp to fp16 range)
    output[n * (OC * OL) + oc * OL + ol] = __float2half(acc);
}

// Host launch function for kernel
void launch_gpu_implementation(
    void* output, void* input,
    void* weight, void* bias,
    int batch_size, int in_channels, int out_channels,
    int input_length, int kernel_size, int stride, int padding, int dilation, bool has_bias
) {
    // Compute output length using PyTorch formula (output_padding=0)
    int output_length = conv_transpose1d_output_length(
        input_length, kernel_size, stride, padding, dilation
    );

    int N = batch_size;
    int IC = in_channels;
    int OC = out_channels;
    int IL = input_length;
    int OL = output_length;
    int K = kernel_size;

    // Kernel launch config
    int total = N * OC * OL;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv1d_transpose_fp16_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        N, IC, OC, IL, OL, K,
        stride, padding, dilation,
        has_bias
    );

    cudaDeviceSynchronize();
}

