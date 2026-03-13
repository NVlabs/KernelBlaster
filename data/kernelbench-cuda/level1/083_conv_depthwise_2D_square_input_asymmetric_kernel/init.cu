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
  CUDA implementation of a depthwise 2D convolution with asymmetric kernel (kernel_size, 1).

  - All tensors are half-precision (fp16).
  - Follows PyTorch's NCHW layout: (batch, channel, height, width).
  - Each input channel has its own filter (depthwise).
  - Handles stride, padding, dilation.
  - Optional bias add.
  - Accumulation is done in FP32 for accuracy, final output is cast to FP16.

  Launch function:
    void launch_gpu_implementation(
        void* output, void* input,
        void* weight, void* bias,
        int batch_size, int in_channels,
        int height, int width,
        int kernel_size, int stride,
        int padding, int dilation, bool has_bias
    );
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>

// CUDA: Fast, coalesced, and parallel depthwise conv2d (NCHW, asymmetric kernel (K,1)), fp16 I/O, fp32 accum
__global__ void depthwise_conv2d_asym_fp16_kernel(
    const half* __restrict__ input,    // [N, C, H, W]
    const half* __restrict__ weight,   // [C, K, 1]
    const half* __restrict__ bias,     // [C] or nullptr
    half* __restrict__ output,         // [N, C, H_out, W_out]
    int N, int C, int H, int W,
    int K, int stride, int padding, int dilation,
    int H_out, int W_out,
    bool has_bias
) {
    // Each thread computes one output element: (n, c, h_out, w_out)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // Compute output indices
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int n     = idx / (C * H_out * W_out);

    // Compute input origin for this output position
    int h_in_origin = h_out * stride - padding;
    int w_in = w_out;  // kernel width is 1, stride always 1 for W in this model

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        int h_in = h_in_origin + k * dilation;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            int input_offset = ((n * C + c) * H + h_in) * W + w_in;
            int weight_offset = c * K + k; // weight: [C, K, 1]
            acc += __half2float(input[input_offset]) * __half2float(weight[weight_offset]);
        }
    }
    if (has_bias && bias != nullptr) {
        acc += __half2float(bias[c]);
    }

    // Cast to fp16
    int output_offset = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[output_offset] = __float2half_rn(acc);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* weight, void* bias,
    int batch_size, int in_channels,
    int height, int width,
    int kernel_size, int stride,
    int padding, int dilation, bool has_bias
) {
    // Output shape calculation (PyTorch's Conv2d formula)
    int H_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int W_out = (width + 2 * padding - dilation * (1 - 1) - 1) / stride + 1; // kernel_width=1

    const int N = batch_size;
    const int C = in_channels;
    const int H = height, W = width;
    const int K = kernel_size;

    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    depthwise_conv2d_asym_fp16_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        N, C, H, W,
        K, stride, padding, dilation,
        H_out, W_out,
        has_bias
    );
    cudaDeviceSynchronize();
}
