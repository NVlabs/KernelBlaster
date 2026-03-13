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
Implements a fast CUDA kernel for 2D transposed convolution (ConvTranspose2d) with support for:
- Arbitrary batch size
- Asymmetric input
- Square kernel
- Stride, padding, dilation
- Optional bias
- Half precision (fp16) for I/O, fp32 accumulator for numeric stability

Tensor layouts:
- Input:  (N, IC, H_IN, W_IN)   (NCHW, contiguous)
- Weight: (IC, OC, KH, KW)      (PyTorch ConvTranspose2d layout)
- Output: (N, OC, H_OUT, W_OUT) (NCHW, contiguous)

This kernel is optimized for modern NVIDIA GPUs (e.g., L40S), but is portable to all CUDA GPUs.
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cassert>
#include <cstdio>

// Utility: Compute output shape for ConvTranspose2d
inline void calc_convtranspose2d_output_shape(
    int height_in, int width_in,
    int kernel_size, int stride, int padding, int dilation,
    int& height_out, int& width_out)
{
    // PyTorch formula:
    // H_out = (H_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + 1
    height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    width_out  = (width_in  - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// CUDA kernel for fp16 ConvTranspose2d with fp32 accumulation
__global__ void conv_transpose2d_fp16_kernel(
    const half* __restrict__ input,    // [N, IC, H_IN, W_IN]
    const half* __restrict__ weight,   // [IC, OC, KH, KW]
    const half* __restrict__ bias,     // [OC] or nullptr
    half* output,                      // [N, OC, H_OUT, W_OUT]
    int N, int IC, int OC,
    int H_IN, int W_IN,
    int H_OUT, int W_OUT,
    int K, int stride, int padding, int dilation,
    bool has_bias)
{
    // Each thread computes one output pixel (n, oc, h_out, w_out)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OC * H_OUT * W_OUT;
    if (tid >= total) return;

    // Map tid to n, oc, h_out, w_out
    int w_out = tid % W_OUT;
    int h_out = (tid / W_OUT) % H_OUT;
    int oc    = (tid / (W_OUT * H_OUT)) % OC;
    int n     = tid / (W_OUT * H_OUT * OC);

    // FP32 accumulator for output
    float acc = 0.0f;

    // For each input channel
    for (int ic = 0; ic < IC; ++ic) {
        // For each kernel position (kh, kw)
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // Compute corresponding input pixel (h_in, w_in) for this output position
                // PyTorch formula:
                // h_in = (h_out + padding - kh*dilation) / stride
                // w_in = (w_out + padding - kw*dilation) / stride
                int h_in_nom = h_out + padding - kh * dilation;
                int w_in_nom = w_out + padding - kw * dilation;

                // For ConvTranspose2d, input pixel is used if divisible by stride
                if (h_in_nom % stride != 0 || w_in_nom % stride != 0)
                    continue;

                int h_in = h_in_nom / stride;
                int w_in = w_in_nom / stride;

                // Check input bounds
                if (h_in < 0 || h_in >= H_IN || w_in < 0 || w_in >= W_IN)
                    continue;

                // PyTorch weight layout for ConvTranspose2d: [IC, OC, KH, KW]
                int w_idx = ic * (OC * K * K) + oc * (K * K) + kh * K + kw;
                int x_idx = n * (IC * H_IN * W_IN) + ic * (H_IN * W_IN) + h_in * W_IN + w_in;

                // Accumulate
                float x = __half2float(input[x_idx]);
                float w = __half2float(weight[w_idx]);
                acc += x * w;
            }
        }
    }
    // Optional bias
    if (has_bias && bias != nullptr)
        acc += __half2float(bias[oc]);

    // Cast back to half
    output[tid] = __float2half(acc);
}

void launch_gpu_implementation(
    void* output,                     // output tensor (float16, GPU)
    void* input,                      // input tensor (float16, GPU)
    void* weight,                     // weight tensor (float16, GPU)
    void* bias,                       // bias tensor (nullptr if bias==false, float16, GPU)
    int batch_size,                   // N
    int in_channels,                  // IC
    int out_channels,                 // OC
    int height_in,                    // H_IN
    int width_in,                     // W_IN
    int kernel_size,                  // K
    int stride,                       // stride
    int padding,                      // padding
    int dilation,                     // dilation
    bool has_bias                     // whether bias is present
)
{
    int height_out, width_out;
    calc_convtranspose2d_output_shape(
        height_in, width_in, kernel_size, stride, padding, dilation,
        height_out, width_out);

    int N = batch_size;
    int IC = in_channels;
    int OC = out_channels;
    int H_IN = height_in;
    int W_IN = width_in;
    int H_OUT = height_out;
    int W_OUT = width_out;
    int K = kernel_size;

    int total = N * OC * H_OUT * W_OUT;
    int block = 256;
    int grid = (total + block - 1) / block;

    conv_transpose2d_fp16_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        has_bias ? static_cast<const half*>(bias) : nullptr,
        static_cast<half*>(output),
        N, IC, OC, H_IN, W_IN, H_OUT, W_OUT, K, stride, padding, dilation, has_bias
    );
    cudaDeviceSynchronize();
}

