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
#include <algorithm>
#include <stdint.h>

// Efficient CUDA kernel for 2D Average Pooling (NCHW, fp16 I/O, fp32 accumulate)
// Each thread computes one output element in (N,C,H_out,W_out)
__global__ void avgpool2d_nchw_fp16(
    half* __restrict__ output,                // (N, C, H_out, W_out)
    const half* __restrict__ input,           // (N, C, H, W)
    int N,
    int C,
    int H,
    int W,
    int kernel_size,
    int stride,
    int padding,
    int H_out,
    int W_out
) {
    // Compute the linear thread index in the output tensor
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;

    if (tid >= total) return;

    // Compute n, c, h_out, w_out indices from tid
    int n = tid / (C * H_out * W_out);
    int rem = tid % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;

    // Compute pooling window in input coordinates
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    int h_end = min(h_start + kernel_size, H);
    int w_end = min(w_start + kernel_size, W);
    h_start = max(h_start, 0);
    w_start = max(w_start, 0);

    // Compute number of elements (for correct division at borders)
    int pool_h = h_end - h_start;
    int pool_w = w_end - w_start;
    int pool_size = pool_h * pool_w;

    // Accumulate in fp32 for numerical stability
    float acc = 0.0f;

    // Pool over the window
#pragma unroll
    for (int h = h_start; h < h_end; ++h) {
#pragma unroll
        for (int w = w_start; w < w_end; ++w) {
            // Input is NCHW
            int input_idx = n * (C * H * W) + c * (H * W) + h * W + w;
            acc += __half2float(input[input_idx]);
        }
    }

    // Compute average (if pool_size > 0, always true here)
    float avg = (pool_size > 0) ? acc / pool_size : 0.0f;

    // Write result (convert back to half)
    int output_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
    output[output_idx] = __float2half_rn(avg);
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(
    void* output,                // pointer to output tensor memory (GPU)
    void* input,                 // pointer to input tensor memory (GPU)
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
) {
    // Compute output dimensions (as in PyTorch nn.AvgPool2d)
    // Formula: H_out = floor((H + 2*padding - kernel_size)/stride + 1)
    int H_out = (height + 2 * padding - kernel_size) / stride + 1;
    int W_out = (width + 2 * padding - kernel_size) / stride + 1;

    int total = batch_size * channels * H_out * W_out;
    int threads_per_block = 256;
    int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;

    avgpool2d_nchw_fp16<<<blocks_per_grid, threads_per_block>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        batch_size,
        channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        H_out,
        W_out
    );
    cudaDeviceSynchronize();
}
