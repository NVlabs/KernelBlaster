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
// Efficient CUDA implementation of 1D transposed convolution (ConvTranspose1d) for fp16 tensors

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <assert.h>

// Utility: Calculate output length for ConvTranspose1d
__host__ __device__ inline int calc_out_length(
    int input_length, int kernel_size, int stride, int padding, int output_padding, int dilation = 1
) {
    // PyTorch's ConvTranspose1d output length formula:
    // out_len = (input_len - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
}

// CUDA kernel for 1D transposed convolution (fp16 I/O, fp32 accumulator)
__global__ void conv1d_transpose_fp16_kernel(
    half* __restrict__ output,         // [B, out_channels, L_out]
    const half* __restrict__ input,    // [B, in_channels, L_in]
    const half* __restrict__ weight,   // [in_channels, out_channels/groups, kernel_size] (PyTorch layout)
    const half* __restrict__ bias,     // [out_channels] or nullptr
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    bool has_bias
) {
    // Each thread computes (b, c_out, l_out)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * output_length;
    if (tid >= total) return;

    int l_out = tid % output_length;
    int c_out = (tid / output_length) % out_channels;
    int b = tid / (output_length * out_channels);

    // Find group for this output channel
    int group_id = c_out / (out_channels / groups);
    int c_out_per_group = out_channels / groups;
    int c_in_per_group = in_channels / groups;

    float acc = 0.0f;

    // For each c_in in this group
    for (int c_in = group_id * c_in_per_group; c_in < (group_id + 1) * c_in_per_group; ++c_in) {
        // For each k in kernel
        for (int k = 0; k < kernel_size; ++k) {
            // Compute the corresponding input index for this output position
            // l_in = (l_out + padding - k) / stride
            int l_in_nom = l_out + padding - k;
            if (l_in_nom % stride != 0) continue; // Must be integer
            int l_in = l_in_nom / stride;
            if (l_in < 0 || l_in >= input_length) continue;

            // Weight indexing: [in_channels, out_channels/groups, kernel_size]
            int w_out = c_out - group_id * c_out_per_group;
            int w_idx = c_in * c_out_per_group * kernel_size + w_out * kernel_size + k;

            // Input: [B, in_channels, L_in]
            int inp_idx = b * in_channels * input_length + c_in * input_length + l_in;

            float inp = __half2float(input[inp_idx]);
            float w = __half2float(weight[w_idx]);
            acc += inp * w;
        }
    }

    // Add bias if present
    if (has_bias && bias != nullptr) {
        acc += __half2float(bias[c_out]);
    }

    // Write result in fp16
    int out_idx = b * out_channels * output_length + c_out * output_length + l_out;
    output[out_idx] = __float2half_rn(acc);
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(
    void* output,                  // Output tensor pointer (fp16, GPU)
    void* input,                   // Input tensor pointer (fp16, GPU)
    void* weight,                  // Weight tensor pointer (fp16, GPU)
    void* bias,                    // Bias tensor pointer (fp16, GPU) (nullptr if no bias)
    int batch_size,                // Batch size
    int in_channels,               // Number of input channels
    int out_channels,              // Number of output channels
    int input_length,              // Input sequence length
    int kernel_size,               // Convolution kernel size
    int stride,                    // Stride
    int padding,                   // Padding
    int output_padding,            // Output padding
    int groups,                    // Number of groups
    bool has_bias                  // Bias flag
) {
    // Calculate output length
    int output_length = calc_out_length(input_length, kernel_size, stride, padding, output_padding);

    // Launch kernel: one thread per (b, c_out, l_out)
    int total = batch_size * out_channels * output_length;
    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    conv1d_transpose_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        has_bias
    );

    cudaDeviceSynchronize();
}

