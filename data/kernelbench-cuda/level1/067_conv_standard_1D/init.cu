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
// Fast CUDA kernel for PyTorch Conv1d (N, C_in, L) with fp16 I/O and fp32 accumulation.
// Handles stride, padding, dilation, and groups. Output shape: (N, C_out, L_out).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cassert>

// Utility: Compute output length for 1D convolution as in PyTorch
__host__ __device__ inline
int64_t conv1d_out_length(int64_t L, int64_t pad, int64_t dilation, int64_t k, int64_t stride) {
    return (L + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
}

// CUDA Kernel: 1D Conv1d, fp16 I/O, fp32 accumulation, NCHW layout
__global__ void conv1d_fp16_nchw_kernel(
    const half* __restrict__ input,    // [N, C_in, L]
    const half* __restrict__ weight,   // [C_out, C_in/groups, K]
    const half* __restrict__ bias,     // [C_out] or nullptr
    half* __restrict__ output,         // [N, C_out, L_out]
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t L,
    int64_t K,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    int64_t L_out
) {
    // Each thread computes one output element: (n, oc, out_pos)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (tid >= total) return;

    int out_pos = tid % L_out;
    int oc = (tid / L_out) % C_out;
    int n = tid / (C_out * L_out);

    int group_size_out = C_out / groups;
    int group_size_in  = C_in / groups;
    int group_id = oc / group_size_out;
    int in_c_start = group_id * group_size_in;
    int in_c_end = in_c_start + group_size_in;
    int oc_in_group = oc % group_size_out;

    float acc = 0.0f;
    // For each input channel in group
    for (int ic = in_c_start; ic < in_c_end; ++ic) {
        int w_ic = ic - in_c_start;
        // For each kernel position
        for (int k = 0; k < K; ++k) {
            // Compute input position
            int in_pos = out_pos * stride - padding + k * dilation;
            if (in_pos < 0 || in_pos >= L) continue;

            // Input: [n, ic, in_pos]
            int input_idx = n * (C_in * L) + ic * L + in_pos;
            // Weight: [oc, group_size_in, K]
            int weight_idx = oc * (group_size_in * K) + w_ic * K + k;

            acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
        }
    }
    // Add bias if present
    if (bias) {
        acc += __half2float(bias[oc]);
    }
    output[tid] = __float2half(acc);
}

void launch_gpu_implementation(
    void* output,                   // Output tensor (fp16, CUDA)
    void* input,                    // Input tensor (fp16, CUDA)
    void* weight,                   // Conv1d weight parameter (fp16, CUDA)
    void* bias,                     // Conv1d bias parameter (nullptr if not used, fp16, CUDA)
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    int64_t length_out = conv1d_out_length(length, padding, dilation, kernel_size, stride);
    int64_t total = batch_size * out_channels * length_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv1d_fp16_nchw_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        length,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        length_out
    );
    cudaDeviceSynchronize();
}
