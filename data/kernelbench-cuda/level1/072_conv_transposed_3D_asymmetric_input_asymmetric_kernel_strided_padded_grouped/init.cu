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
#include <assert.h>

// Utility: ceil division
__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// Fast, generic, correct 3D Transposed Convolution (fp16 I/O, fp32 accumulation)
// Tensor Layouts:
//   input:  [N, C_in, D, H, W] (NCDHW)
//   weight: [C_in, C_out/groups, Kd, Kh, Kw] (PyTorch)
//   output: [N, C_out, D_out, H_out, W_out] (NCDHW)
__global__ void conv3d_transpose_fp16_kernel(
    const half* __restrict__ input,      // [N, C_in, D, H, W]
    const half* __restrict__ weight,     // [C_in, C_out/groups, Kd, Kh, Kw]
    const half* __restrict__ bias,       // [C_out] or nullptr
    half* __restrict__ output,           // [N, C_out, D_out, H_out, W_out]
    int N,
    int C_in,
    int C_out,
    int D, int H, int W,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups,
    bool has_bias,
    int D_out, int H_out, int W_out
) {
    // Each thread computes one output element: (n, c_out, d_out, h_out, w_out)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_elems = N * C_out * D_out * H_out * W_out;
    if (tid >= n_elems) return;

    // Compute output indices
    int tmp = tid;
    int w_out_idx = tmp % W_out; tmp /= W_out;
    int h_out_idx = tmp % H_out; tmp /= H_out;
    int d_out_idx = tmp % D_out; tmp /= D_out;
    int c_out_idx = tmp % C_out; tmp /= C_out;
    int n_idx = tmp;

    // Find group and local channel indices
    int c_out_per_group = C_out / groups;
    int c_in_per_group  = C_in / groups;
    int group_idx = c_out_idx / c_out_per_group;
    int c_out_in_group = c_out_idx % c_out_per_group;

    float acc = 0.0f;

    // For each input channel in this group
    for (int c_in_in_group = 0; c_in_in_group < c_in_per_group; ++c_in_in_group) {
        int c_in_idx = group_idx * c_in_per_group + c_in_in_group;

        // For each kernel element
        for (int kd = 0; kd < Kd; ++kd) {
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    // Compute input indices corresponding to this output location and kernel position
                    // PyTorch formula for transposed conv3d:
                    // d_in = (d_out + pad_d - kd) / stride_d
                    // h_in = (h_out + pad_h - kh) / stride_h
                    // w_in = (w_out + pad_w - kw) / stride_w
                    int d_in_nom = d_out_idx + pad_d - kd;
                    int h_in_nom = h_out_idx + pad_h - kh;
                    int w_in_nom = w_out_idx + pad_w - kw;

                    // Must be divisible by stride and in bounds
                    if (d_in_nom % stride_d != 0) continue;
                    if (h_in_nom % stride_h != 0) continue;
                    if (w_in_nom % stride_w != 0) continue;

                    int d_in = d_in_nom / stride_d;
                    int h_in = h_in_nom / stride_h;
                    int w_in = w_in_nom / stride_w;

                    if (d_in < 0 || d_in >= D) continue;
                    if (h_in < 0 || h_in >= H) continue;
                    if (w_in < 0 || w_in >= W) continue;

                    // Input index: [N, C_in, D, H, W]
                    size_t input_idx = (((size_t)n_idx * C_in + c_in_idx) * D + d_in) * H * W
                                     + h_in * W + w_in;

                    // Weight index: [C_in, C_out/groups, Kd, Kh, Kw]
                    // weight[c_in_idx, c_out_in_group, kd, kh, kw]
                    size_t weight_idx = (((size_t)c_in_idx * c_out_per_group + c_out_in_group) * Kd + kd) * Kh * Kw
                                     + kh * Kw + kw;

                    float inp = __half2float(input[input_idx]);
                    float wgt = __half2float(weight[weight_idx]);
                    acc += inp * wgt;
                }
            }
        }
    }
    // Add bias if present
    if (has_bias) {
        acc += __half2float(bias[c_out_idx]);
    }

    // Cast accumulator to fp16 for output
    output[tid] = __float2half(acc);
}

// Host launcher function
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size_d,
    int kernel_size_h,
    int kernel_size_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int groups,
    bool has_bias
) {
    // Compute output spatial size according to PyTorch ConvTranspose3d formula:
    // L_out = (L_in - 1) * stride - 2*padding + kernel_size + output_padding
    int D_out = (depth  - 1) * stride_d - 2 * padding_d + kernel_size_d + output_padding_d;
    int H_out = (height - 1) * stride_h - 2 * padding_h + kernel_size_h + output_padding_h;
    int W_out = (width  - 1) * stride_w - 2 * padding_w + kernel_size_w + output_padding_w;

    int n_elems = batch_size * out_channels * D_out * H_out * W_out;

    // Kernel launch
    int threads_per_block = 256;
    int blocks = div_up(n_elems, threads_per_block);

    conv3d_transpose_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        depth, height, width,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups,
        has_bias,
        D_out, H_out, W_out
    );

    cudaDeviceSynchronize();
}
