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
#include <cstdio>
#include <cstdint>
#include <algorithm>

// CUDA kernel for 1D AvgPool (fp16 I/O, fp32 accumulation, count_include_pad=True)
__global__ void avgpool1d_fp16_kernel(
    const __half* __restrict__ input,  // [N, C, L_in]
    __half* __restrict__ output,       // [N, C, L_out]
    int N,
    int C,
    int L_in,
    int L_out,
    int kernel_size,
    int stride,
    int padding
) {
    int out_linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_linear >= N * C * L_out) return;

    int l_out = out_linear % L_out;
    int c = (out_linear / L_out) % C;
    int n = out_linear / (C * L_out);

    // Compute window for this output position
    int window_start = l_out * stride - padding;
    int window_end = window_start + kernel_size;

    float acc = 0.0f;

    // Accumulate only in-bounds elements
    for (int l = window_start; l < window_end; ++l) {
        if (l >= 0 && l < L_in) {
            int idx = n * C * L_in + c * L_in + l;
            acc += __half2float(input[idx]);
        }
        // If out-of-bounds (padding), treat as zero (i.e., skip, which is equivalent)
    }

    // ALWAYS divide by kernel_size (count_include_pad=True)
    float avg = acc / (float)kernel_size;
    int out_idx = n * C * L_out + c * L_out + l_out;
    output[out_idx] = __float2half(avg);
}

void launch_gpu_implementation(
    void* output,                    // Output tensor (GPU memory, fp16)
    void* input,                     // Input tensor (GPU memory, fp16)
    int64_t batch_size,
    int64_t in_channels,
    int64_t input_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding
) {
    // PyTorch's formula for output length:
    // L_out = floor((L_in + 2*padding - kernel_size) / stride) + 1
    int64_t L_in = input_length;
    int64_t temp = L_in + 2 * padding - kernel_size;
    int64_t L_out = (temp / stride) + 1;
    if (L_out < 0) L_out = 0;

    int total = batch_size * in_channels * L_out;
    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    avgpool1d_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<const __half*>(input),
        static_cast<__half*>(output),
        static_cast<int>(batch_size),
        static_cast<int>(in_channels),
        static_cast<int>(input_length),
        static_cast<int>(L_out),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding)
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
