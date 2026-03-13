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
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            assert(false); \
        } \
    } while (0)

// Accumulate in float for numerical stability
__device__ __forceinline__ float accumulate_conv1d_fp16(
    const half* __restrict__ x,    // Input: [batch_size, in_channels, input_length]
    const half* __restrict__ w,    // Weight: [out_channels, in_channels, kernel_size]
    int in_channels,
    int kernel_size,
    int input_length,
    int b, int oc, int out_pos,    // batch, out_channel, output position
    int stride,
    int dilation
) {
    float acc = 0.0f;
    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int in_pos = out_pos * stride + k * dilation;
            if (in_pos < input_length) {
                int x_idx = b * in_channels * input_length + ic * input_length + in_pos;
                int w_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                acc += __half2float(x[x_idx]) * __half2float(w[w_idx]);
            }
        }
    }
    return acc;
}

/*
 * CUDA kernel for 1D convolution in N, C, L layout, with fp16 I/O and fp32 accumulation.
 *   input:  [batch_size, in_channels, input_length]
 *   weight: [out_channels, in_channels, kernel_size]
 *   bias:   [out_channels] or nullptr
 *   output: [batch_size, out_channels, output_length]
 */
__global__ void conv1d_ncl_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int dilation,
    bool has_bias
) {
    // 3D grid: [batch, out_channel, output_pos]
    int b = blockIdx.z;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int out_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || oc >= out_channels || out_pos >= output_length) return;

    float acc = accumulate_conv1d_fp16(
        input, weight,
        in_channels, kernel_size, input_length,
        b, oc, out_pos,
        stride, dilation
    );

    // Add bias if present
    if (has_bias && bias != nullptr) {
        acc += __half2float(bias[oc]);
    }

    // Write result as fp16
    int out_idx = b * out_channels * output_length + oc * output_length + out_pos;
    output[out_idx] = __float2half(acc);
}

// Host launcher for the above kernel
void launch_gpu_implementation(
    void* output,                   // Output tensor pointer (float16)
    void* input,                    // Input tensor pointer (float16)
    void* weight,                   // Weight tensor pointer (float16)
    void* bias,                     // Bias tensor pointer (float16 or nullptr)
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t input_length,
    int64_t stride,
    int64_t dilation,
    bool has_bias                  // Indicates if bias is present
) {
    // Calculate output length as per PyTorch's formula
    int64_t output_length = (input_length - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Use a 3D grid: (output_length, out_channels, batch_size)
    // Tune block sizes for best occupancy and memory access
    const int out_pos_block = 32;
    const int oc_block = 8;
    dim3 block(out_pos_block, oc_block, 1);

    int grid_x = (output_length + out_pos_block - 1) / out_pos_block;
    int grid_y = (out_channels + oc_block - 1) / oc_block;
    int grid_z = batch_size;

    dim3 grid(grid_x, grid_y, grid_z);

    conv1d_ncl_fp16_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        has_bias ? static_cast<const half*>(bias) : nullptr,
        static_cast<half*>(output),
        static_cast<int>(batch_size),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(kernel_size),
        static_cast<int>(input_length),
        static_cast<int>(output_length),
        static_cast<int>(stride),
        static_cast<int>(dilation),
        has_bias
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
