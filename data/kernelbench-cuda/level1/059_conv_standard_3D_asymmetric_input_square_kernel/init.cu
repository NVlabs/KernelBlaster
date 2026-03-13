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
    Efficient CUDA kernel for 3D convolution (with square 2D kernel) in fp16, using fp32 accumulation.

    Corresponds to PyTorch's nn.Conv3d with (kernel_size, kernel_size, 1) kernel, arbitrary stride/padding/dilation/groups.
    All input/output/weight/bias tensors are fp16.

    Kernel layout: 
        - Input:  (N, C_in, H_in, W_in, D_in)
        - Weight: (C_out, C_in // groups, K, K, 1)
        - Bias:   (C_out) or nullptr if not used
        - Output: (N, C_out, H_out, W_out, D_out)

    Accumulation is always done in fp32 for numerical stability, result is written as fp16.
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

// Utility: Divides a by b and rounds up
inline int div_up(int a, int b) { return (a + b - 1) / b; }

// CUDA kernel for 3D convolution, NCDHW layout, square kernel, fp16 I/O with fp32 accumulation, supports stride/padding/dilation/groups/bias
__global__ void conv3d_fp16_ncdhw_kernel(
    const half* __restrict__ input,    // [N, C_in, H_in, W_in, D_in]
    const half* __restrict__ weight,   // [C_out, C_in/groups, K, K, 1]
    const half* __restrict__ bias,     // [C_out] or nullptr
    half* __restrict__ output,         // [N, C_out, H_out, W_out, D_out]
    int N,
    int C_in,
    int C_out,
    int H_in,
    int W_in,
    int D_in,
    int K,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool use_bias,
    int H_out,
    int W_out,
    int D_out
) {
    // Each thread computes a single output element (n, c_out, h_out, w_out, d_out)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out * D_out;
    if (tid >= total) return;

    // Decompose linear index
    int d_out = tid % D_out;
    int w_out = (tid / D_out) % W_out;
    int h_out = (tid / (D_out * W_out)) % H_out;
    int c_out = (tid / (D_out * W_out * H_out)) % C_out;
    int n = tid / (D_out * W_out * H_out * C_out);

    // Figure out group info
    int c_out_per_group = C_out / groups;
    int c_in_per_group = C_in / groups;
    int group_idx = c_out / c_out_per_group;
    int c_out_local = c_out % c_out_per_group;

    // Output pointer
    int out_offset = (((n * C_out + c_out) * H_out + h_out) * W_out + w_out) * D_out + d_out;

    // Accumulate result in fp32
    float acc = 0.0f;

    // For 3D convolution, kernel size is (K, K, 1)
    // So only loop over 2D kernel in (kh, kw), single slice in depth
    for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
            // For depth, kernel size is always 1 (per the model), so only one index
            int kd = 0;

            // Compute input coordinates
            int h_in = h_out * stride - padding + kh * dilation;
            int w_in = w_out * stride - padding + kw * dilation;
            int d_in = d_out * stride - padding + kd * dilation;

            // Bounds check
            if (h_in < 0 || h_in >= H_in) continue;
            if (w_in < 0 || w_in >= W_in) continue;
            if (d_in < 0 || d_in >= D_in) continue;

            // Loop over input channels for this group
            for (int c_in_local = 0; c_in_local < c_in_per_group; ++c_in_local) {
                int c_in_idx = group_idx * c_in_per_group + c_in_local;

                // Input index: [n, c_in, h_in, w_in, d_in]
                int in_offset = (((n * C_in + c_in_idx) * H_in + h_in) * W_in + w_in) * D_in + d_in;
                half val_in = input[in_offset];

                // Weight index: [c_out, c_in_per_group, kh, kw, 0]
                int w_offset = ((((c_out) * c_in_per_group + c_in_local) * K + kh) * K + kw) * 1 + kd;
                half val_w = weight[w_offset];

                // Accumulate in fp32
                acc += __half2float(val_in) * __half2float(val_w);
            }
        }
    }

    // Add bias if present
    if (use_bias && bias != nullptr) {
        acc += __half2float(bias[c_out]);
    }

    // Convert to fp16 and store
    output[out_offset] = __float2half(acc);
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
    int input_height,
    int input_width,
    int input_depth,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool use_bias
) {
    // Compute output dimensions
    int H_out = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int W_out = (input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int D_out = (input_depth  + 2 * padding - dilation * (1 - 1) - 1) / stride + 1; // kernel_depth = 1

    int total = batch_size * out_channels * H_out * W_out * D_out;

    // Launch parameters
    int threads_per_block = 256;
    int num_blocks = div_up(total, threads_per_block);

    conv3d_fp16_ncdhw_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        use_bias ? static_cast<const half*>(bias) : nullptr,
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        input_depth,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        use_bias,
        H_out,
        W_out,
        D_out
    );

    cudaDeviceSynchronize();
}
