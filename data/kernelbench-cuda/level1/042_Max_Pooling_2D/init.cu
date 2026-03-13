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
#include <cstdint>

// Fast MaxPool2D kernel for NHWC and NCHW, here NCHW for PyTorch default layout.
// All I/O tensors are in half (fp16).
// Accumulation is in half, as max-pooling is not subject to catastrophic cancellation as in sum reductions.

// Utility: Ceiling division
inline __host__ __device__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// CUDA kernel: MaxPool2D with dilation+padding+stride support. NCHW layout.
__global__ void maxpool2d_nchw_kernel(
    const half* __restrict__ input,  // [N, C, Hin, Win]
    half* __restrict__ output,       // [N, C, Hout, Wout]
    int N, int C,
    int Hin, int Win,
    int Hout, int Wout,
    int kernel_size, int stride, int padding, int dilation
) {
    // Output indices
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= Hout * Wout) return;
    int h_out = hw / Wout;
    int w_out = hw % Wout;

    // Compute pooling window
    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;

    half maxval = __half(-65504.0f); // Smallest fp16

    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_in_start + kh * dilation;
        if (h_in < 0 || h_in >= Hin) continue;
        #pragma unroll
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = w_in_start + kw * dilation;
            if (w_in < 0 || w_in >= Win) continue;
            int in_idx = ((n * C + c) * Hin + h_in) * Win + w_in;
            half v = input[in_idx];
            maxval = __hgt(v, maxval) ? v : maxval;
        }
    }
    int out_idx = ((n * C + c) * Hout + h_out) * Wout + w_out;
    output[out_idx] = maxval;
}

// Host launch code
void launch_gpu_implementation(
    void* output,            // Pointer to output tensor memory (GPU)
    void* input,             // Pointer to input tensor memory (GPU)
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Calculate output shape as per PyTorch's nn.MaxPool2d
    int Hout = div_up(height + 2 * padding - dilation * (kernel_size - 1) - 1, stride) + 1;
    int Wout = div_up(width  + 2 * padding - dilation * (kernel_size - 1) - 1, stride) + 1;

    // NCHW layout
    int N = batch_size;
    int C = channels;
    int Hin = height;
    int Win = width;

    // Kernel config
    int HWout = Hout * Wout;
    int threads = 256;
    int blocks = div_up(HWout, threads);

    dim3 grid(blocks, C, N);
    dim3 block(threads);

    maxpool2d_nchw_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        N, C, Hin, Win, Hout, Wout,
        kernel_size, stride, padding, dilation
    );
    cudaDeviceSynchronize();
}
