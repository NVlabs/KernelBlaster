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
 * Fast CUDA kernel for 3D MaxPool (fp16)
 * Input:  (batch_size, channels, dim1, dim2, dim3)    (fp16)
 * Output: (batch_size, channels, out1, out2, out3)    (fp16)
 * Kernel parameters: kernel_size, stride, padding, dilation (all int)
 *
 * The kernel is optimized for memory coalescing and warp-level parallelism.
 * The accumulator is always in fp16, as required by the I/O spec and MaxPool's nature.
 *
 * Ceil_mode is always false.
 * No indices are returned.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

// Utility: Compute output shape for 3D MaxPool (ceil_mode == false)
__host__ __device__ inline int pool_out_dim(int in, int kernel, int stride, int pad, int dilation) {
    return (in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

// CUDA kernel for 3D MaxPool (NHWDC format)
__global__ void maxpool3d_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size, int channels,
    int in1, int in2, int in3,
    int out1, int out2, int out3,
    int kernel_size, int stride, int padding, int dilation
) {
    // Flat output index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out1 * out2 * out3;
    if (tid >= total) return;

    // Compute output indices: n, c, od1, od2, od3
    int o3 = tid % out3;
    int o2 = (tid / out3) % out2;
    int o1 = (tid / (out3 * out2)) % out1;
    int c  = (tid / (out3 * out2 * out1)) % channels;
    int n  = tid / (out3 * out2 * out1 * channels);

    // Compute input window start
    int id1_start = o1 * stride - padding;
    int id2_start = o2 * stride - padding;
    int id3_start = o3 * stride - padding;

    // Max accumulator (init to lowest possible fp16 value)
    half max_val = __float2half(-65504.0f);

    // For each window element
#pragma unroll
    for (int k1 = 0; k1 < kernel_size; ++k1) {
        int id1 = id1_start + k1 * dilation;
        if (id1 < 0 || id1 >= in1) continue;
#pragma unroll
        for (int k2 = 0; k2 < kernel_size; ++k2) {
            int id2 = id2_start + k2 * dilation;
            if (id2 < 0 || id2 >= in2) continue;
#pragma unroll
            for (int k3 = 0; k3 < kernel_size; ++k3) {
                int id3 = id3_start + k3 * dilation;
                if (id3 < 0 || id3 >= in3) continue;

                // Compute input index
                size_t inp_idx =
                    ((size_t)n * channels * in1 * in2 * in3) +
                    ((size_t)c * in1 * in2 * in3) +
                    ((size_t)id1 * in2 * in3) +
                    ((size_t)id2 * in3) +
                    id3;

                half v = input[inp_idx];
                max_val = __hgt(v, max_val) ? v : max_val;
            }
        }
    }

    // Write output
    size_t out_idx =
        ((size_t)n * channels * out1 * out2 * out3) +
        ((size_t)c * out1 * out2 * out3) +
        ((size_t)o1 * out2 * out3) +
        ((size_t)o2 * out3) +
        o3;
    output[out_idx] = max_val;
}

// Host function to launch CUDA MaxPool3d kernel
void launch_gpu_implementation(
    void* output,                   // Output tensor (GPU memory, fp16)
    void* input,                    // Input tensor (GPU memory, fp16)
    int batch_size,
    int channels,
    int dim1,
    int dim2,
    int dim3,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Compute output dimensions (ceil_mode == false)
    int out1 = pool_out_dim(dim1, kernel_size, stride, padding, dilation);
    int out2 = pool_out_dim(dim2, kernel_size, stride, padding, dilation);
    int out3 = pool_out_dim(dim3, kernel_size, stride, padding, dilation);

    size_t total = (size_t)batch_size * channels * out1 * out2 * out3;

    // Configure kernel launch
    int threads_per_block = 256;
    int num_blocks = (total + threads_per_block - 1) / threads_per_block;

    maxpool3d_fp16_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        batch_size, channels,
        dim1, dim2, dim3,
        out1, out2, out3,
        kernel_size, stride, padding, dilation
    );

    cudaDeviceSynchronize();
}
