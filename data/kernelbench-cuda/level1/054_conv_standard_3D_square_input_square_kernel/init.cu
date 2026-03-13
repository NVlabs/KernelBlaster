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
#include <cassert>
#include <cstdio>

// Utility macro for CUDA error checking
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            assert(false); \
        } \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////
// 3D Convolution CUDA Kernel (fp16 I/O, fp32 accumulation)
//
// Input:  input   [B, C_in, D, W, H]   (NCDHW, contiguous)
// Weight: weight  [C_out, C_in/groups, Kd, Kw, Kh]
// Output: output  [B, C_out, D_out, W_out, H_out]
//
// Supports: stride, padding, dilation, groups, optional bias
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void conv3d_fp16_ncdhw_kernel(
    const half* __restrict__ input,   // [B, C_in, D, W, H]
    const half* __restrict__ weight,  // [C_out, C_in/groups, Kd, Kw, Kh]
    const half* __restrict__ bias,    // [C_out] or nullptr
    half* __restrict__ output,        // [B, C_out, D_out, W_out, H_out]
    int B, int C_in, int C_out,
    int D, int W, int H,
    int Kd, int Kw, int Kh,
    int stride, int padding, int dilation, int groups,
    int D_out, int W_out, int H_out
) {
    // Output index calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_out * D_out * W_out * H_out;
    if (tid >= total) return;

    // Compute output tensor indices
    int n = tid / (C_out * D_out * W_out * H_out);
    int rem = tid % (C_out * D_out * W_out * H_out);
    int c_out = rem / (D_out * W_out * H_out);
    rem = rem % (D_out * W_out * H_out);
    int d_out = rem / (W_out * H_out);
    rem = rem % (W_out * H_out);
    int w_out = rem / H_out;
    int h_out = rem % H_out;

    // Group index and input channel range for this out channel
    int c_out_g = c_out / (C_out / groups);
    int c_in_per_group = C_in / groups;
    int c_out_per_group = C_out / groups;
    int c_out_local = c_out % c_out_per_group;

    float acc = 0.0f;

    // For each input channel in this group
    for (int c_in_g = 0; c_in_g < c_in_per_group; ++c_in_g) {
        int c_in = c_out_g * c_in_per_group + c_in_g;

        // For each kernel depth/width/height
        for (int kd = 0; kd < Kd; ++kd) {
            int d_in = d_out * stride - padding + kd * dilation;
            if (d_in < 0 || d_in >= D) continue;
            for (int kw = 0; kw < Kw; ++kw) {
                int w_in = w_out * stride - padding + kw * dilation;
                if (w_in < 0 || w_in >= W) continue;
                for (int kh = 0; kh < Kh; ++kh) {
                    int h_in = h_out * stride - padding + kh * dilation;
                    if (h_in < 0 || h_in >= H) continue;

                    // Input index: [n, c_in, d_in, w_in, h_in] (NCDHW)
                    int input_idx = (((n * C_in + c_in) * D + d_in) * W + w_in) * H + h_in;
                    // Weight index: [c_out, c_in_g, kd, kw, kh]
                    int weight_idx = ((((c_out) * c_in_per_group + c_in_g) * Kd + kd) * Kw + kw) * Kh + kh;

                    float inp = __half2float(input[input_idx]);
                    float wgt = __half2float(weight[weight_idx]);
                    acc += inp * wgt;
                }
            }
        }
    }

    // Add bias if present
    if (bias) {
        acc += __half2float(bias[c_out]);
    }

    // Convert to fp16 and store
    int out_idx = (((n * C_out + c_out) * D_out + d_out) * W_out + w_out) * H_out + h_out;
    output[out_idx] = __float2half(acc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host code for launching the CUDA kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

void launch_gpu_implementation(
    void* output,    // [B, C_out, D_out, W_out, H_out], half
    void* input,     // [B, C_in, D, W, H], half
    void* weight,    // [C_out, C_in/groups, Kd, Kw, Kh], half
    void* bias,      // [C_out], half or nullptr
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int width,
    int height,
    int kernel_size, // Kd = Kw = Kh
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Compute output dimensions
    int Kd = kernel_size, Kw = kernel_size, Kh = kernel_size;
    int D_out = (depth + 2 * padding - dilation * (Kd - 1) - 1) / stride + 1;
    int W_out = (width + 2 * padding - dilation * (Kw - 1) - 1) / stride + 1;
    int H_out = (height + 2 * padding - dilation * (Kh - 1) - 1) / stride + 1;

    int total = batch_size * out_channels * D_out * W_out * H_out;

    // Launch configuration
    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    conv3d_fp16_ncdhw_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        depth, width, height,
        Kd, Kw, Kh,
        stride, padding, dilation, groups,
        D_out, W_out, H_out
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Notes:
// - This kernel is memory-bound and parallelizes over output elements.
// - Accumulation is done in fp32 for stability, output is written in fp16.
// - For large kernels, shared-memory blocking and tensor core acceleration can further optimize performance.
// - For the given test size (16, 3, 64, 64, 64), this kernel is efficient and correct.
// - The kernel supports bias and groups.
// - All arguments must be on GPU and contiguous.
// - Data layout: input/output in NCDHW, weight in [C_out, C_in/groups, Kd, Kw, Kh].
////////////////////////////////////////////////////////////////////////////////////////////////////
