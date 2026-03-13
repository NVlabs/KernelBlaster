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
    CUDA fast convolution kernel for fp16 input/output, using fp32 accumulation.
    Implements the same logic as PyTorch's nn.Conv2d (NCHW layout):
        - Input: (N, C_in, H_in, W_in), fp16
        - Weight: (C_out, C_in/groups, K, K), fp16
        - Bias: (C_out) or nullptr (can be nullptr if has_bias==false), fp16
        - Output: (N, C_out, H_out, W_out), fp16
    Handles stride, padding, dilation, and groups, as per PyTorch semantics.
    Accumulation is always done in fp32 for numerical stability.

    Host function:
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
        int dilation,
        int groups,
        bool has_bias
    );
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>

// Utility: CUDA error checking macro for debugging
#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
#ifndef NDEBUG
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
#endif
}

// CUDA kernel for NCHW 2D convolution with groups, stride, padding, dilation, fp16 I/O, fp32 accumulation
__global__ void conv2d_nchw_fp16_kernel(
    const half* __restrict__ input,     // [N, C_in, H_in, W_in]
    const half* __restrict__ weight,    // [C_out, C_in_per_group, K, K]
    const half* __restrict__ bias,      // [C_out] or nullptr
    half* __restrict__ output,          // [N, C_out, H_out, W_out]
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int K, int stride, int padding, int dilation,
    int groups, bool has_bias
) {
    // Output dims
    const int C_in_per_group = C_in / groups;
    const int C_out_per_group = C_out / groups;
    const int H_out = (H_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    const int W_out = (W_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    // Flattened thread index for output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // Calculate N, C_out, H_out, W_out indices
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    // Find group and local c_in/c_out
    int group = c_out / C_out_per_group;
    int c_out_local = c_out % C_out_per_group;

    // Pointer offsets
    const int input_group_offset = n * C_in * H_in * W_in + group * C_in_per_group * H_in * W_in;
    const int weight_group_offset = (group * C_out_per_group + c_out_local) * C_in_per_group * K * K;
    const int out_offset = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;

    // Accumulator (fp32 for stability)
    float accum = has_bias ? __half2float(bias[c_out]) : 0.0f;

    // Loop over kernel window and input channels for this group
#pragma unroll
    for (int c_in_local = 0; c_in_local < C_in_per_group; ++c_in_local) {
#pragma unroll
        for (int k_h = 0; k_h < K; ++k_h) {
#pragma unroll
            for (int k_w = 0; k_w < K; ++k_w) {
                // Compute input coordinates (with stride, padding, dilation)
                int in_h = h_out * stride - padding + k_h * dilation;
                int in_w = w_out * stride - padding + k_w * dilation;
                if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                    // Index: [n, c_in, in_h, in_w]
                    int input_idx = input_group_offset
                        + c_in_local * H_in * W_in
                        + in_h * W_in + in_w;
                    // Weight: [c_out, c_in, k_h, k_w] (contiguous)
                    int weight_idx = weight_group_offset
                        + c_in_local * K * K
                        + k_h * K + k_w;
                    accum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }
    // Store the result as fp16 (round to nearest)
    output[out_offset] = __float2half_rn(accum);
}

// Host function for launching the CUDA kernel
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
    int dilation,
    int groups,
    bool has_bias
) {
    // Calculate output dims
    int H_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int W_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int total_out = batch_size * out_channels * H_out * W_out;

    // Kernel launch parameters
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

    // Cast pointers to CUDA half type
    const half* input_ptr = static_cast<const half*>(input);
    const half* weight_ptr = static_cast<const half*>(weight);
    const half* bias_ptr = has_bias ? static_cast<const half*>(bias) : nullptr;
    half* output_ptr = static_cast<half*>(output);

    // Launch kernel
    conv2d_nchw_fp16_kernel<<<blocks, threads>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_channels, out_channels,
        height, width, kernel_size, stride, padding, dilation,
        groups, has_bias
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
