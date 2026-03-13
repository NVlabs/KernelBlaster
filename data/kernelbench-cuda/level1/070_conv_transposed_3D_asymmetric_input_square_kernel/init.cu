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
#include <cassert>
#include <cstdio>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            assert(0); \
        } \
    } while (0)

__host__ __device__
inline int convtranspose_out_dim(
    int input, int stride, int padding, int kernel, int output_padding, int dilation
) {
    // PyTorch formula: (input - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
    return (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1;
}

__global__ void convtranspose3d_fp16_kernel(
    half* __restrict__ output,    // [N, out_ch, out_D, out_H, out_W]
    const half* __restrict__ input, // [N, in_ch, D, H, W]
    const half* __restrict__ weight, // [in_ch, out_ch_per_group, kD, kH, kW]
    const half* __restrict__ bias,   // [out_ch] or nullptr
    int N, int in_ch, int out_ch, int ksize,
    int D, int H, int W,
    int stride, int padding, int output_padding, int dilation, int groups,
    int out_D, int out_H, int out_W,
    bool bias_flag
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_ch * out_D * out_H * out_W;
    if (tid >= total) return;

    int o_w = tid % out_W;
    int o_h = (tid / out_W) % out_H;
    int o_d = (tid / (out_W * out_H)) % out_D;
    int o_c = (tid / (out_W * out_H * out_D)) % out_ch;
    int n   = tid / (out_W * out_H * out_D * out_ch);

    int group = o_c / (out_ch / groups);
    int out_c_g = o_c % (out_ch / groups);

    float acc = 0.0f;
    int in_ch_per_group = in_ch / groups;
    int out_ch_per_group = out_ch / groups;
    int weight_group_offset = group * in_ch_per_group * out_ch_per_group * ksize * ksize * ksize;

    for (int in_c_g = 0; in_c_g < in_ch_per_group; ++in_c_g) {
        int in_c = group * in_ch_per_group + in_c_g;
        for (int k_d = 0; k_d < ksize; ++k_d) {
            for (int k_h = 0; k_h < ksize; ++k_h) {
                for (int k_w = 0; k_w < ksize; ++k_w) {
                    int i_d = (o_d + padding - k_d * dilation) / stride;
                    int i_h = (o_h + padding - k_h * dilation) / stride;
                    int i_w = (o_w + padding - k_w * dilation) / stride;

                    if ((o_d + padding - k_d * dilation) % stride != 0) continue;
                    if ((o_h + padding - k_h * dilation) % stride != 0) continue;
                    if ((o_w + padding - k_w * dilation) % stride != 0) continue;
                    if (i_d < 0 || i_d >= D) continue;
                    if (i_h < 0 || i_h >= H) continue;
                    if (i_w < 0 || i_w >= W) continue;

                    size_t in_idx = ((size_t)n * in_ch + in_c) * D * H * W
                        + i_d * H * W + i_h * W + i_w;

                    size_t w_idx = weight_group_offset
                        + in_c_g * out_ch_per_group * ksize * ksize * ksize
                        + out_c_g * ksize * ksize * ksize
                        + k_d * ksize * ksize
                        + k_h * ksize
                        + k_w;

                    acc += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
                }
            }
        }
    }

    if (bias_flag && bias != nullptr) {
        acc += __half2float(bias[o_c]);
    }

    output[tid] = __float2half_rn(acc);
}

// DO NOT use extern "C", do not put in a namespace, do not mark static.
// This must have external linkage and the *exact* signature!
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int depth,
    int height,
    int width,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups,
    bool bias_flag
) {
    int out_D = convtranspose_out_dim(depth, stride, padding, kernel_size, output_padding, dilation);
    int out_H = convtranspose_out_dim(height, stride, padding, kernel_size, output_padding, dilation);
    int out_W = convtranspose_out_dim(width, stride, padding, kernel_size, output_padding, dilation);

    size_t total = (size_t)batch_size * out_channels * out_D * out_H * out_W;
    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    convtranspose3d_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        batch_size, in_channels, out_channels, kernel_size,
        depth, height, width,
        stride, padding, output_padding, dilation, groups,
        out_D, out_H, out_W,
        bias_flag
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
