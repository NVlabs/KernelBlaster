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
// 3D Average Pooling CUDA kernel (fp16 input/output, fp32 accumulation for accuracy)
// Correctly matches PyTorch nn.AvgPool3d with count_include_pad=True (default for torch.nn.AvgPool3d!)
// Output shape and divisor exactly as PyTorch, including padded regions in the divisor.
//
// Input:  [N, C, D, H, W] (NCDHW, contiguous)
// Output: [N, C, OD, OH, OW] (NCDHW)
// All pointers are device pointers to half (fp16).
//
// Test case: batch_size=16, channels=32, depth=64, height=64, width=64, kernel_size=3, stride=2, padding=1
// kernel_size, stride, padding are all INT (not tuples)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

inline __host__ __device__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// CUDA kernel for 3D average pooling (fp16 I/O, fp32 accumulation, count_include_pad=True)
__global__ void avgpool3d_fp16_kernel(
    const half* __restrict__ input,     // [N, C, D, H, W]
    half* __restrict__ output,          // [N, C, OD, OH, OW]
    int N, int C, int D, int H, int W,
    int K, int S, int P,
    int OD, int OH, int OW
) {
    // Linear output index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OD * OH * OW;
    if (tid >= total) return;

    // Decompose tid into indices: n, c, od, oh, ow
    int ow = tid % OW;
    int oh = (tid / OW) % OH;
    int od = (tid / (OW * OH)) % OD;
    int c  = (tid / (OW * OH * OD)) % C;
    int n  = tid / (OW * OH * OD * C);

    // Compute pooling window (start/end in input space)
    int dstart = od * S - P;
    int hstart = oh * S - P;
    int wstart = ow * S - P;
    int dend = dstart + K;
    int hend = hstart + K;
    int wend = wstart + K;

    // Pooling window may overlap padding, so for count_include_pad=True the divisor is always K*K*K
    float acc = 0.0f;

    for (int d = dstart; d < dend; ++d) {
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                // Only accumulate if in-bounds
                if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                    int input_idx = (((n * C + c) * D + d) * H + h) * W + w;
                    acc += __half2float(input[input_idx]);
                }
            }
        }
    }

    // PyTorch's AvgPool3d (count_include_pad=True): divisor = K^3
    float avg = acc / float(K * K * K);

    int output_idx = (((n * C + c) * OD + od) * OH + oh) * OW + ow;
    output[output_idx] = __float2half(avg);
}

// Host function to launch the CUDA avgpool3d kernel
void launch_gpu_implementation(
    void* output,           // Output tensor pointer (GPU memory, half)
    void* input,            // Input tensor pointer (GPU memory, half)
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
) {
    // Calculate output dimensions using PyTorch's formula for 3D pooling
    int OD = (depth  + 2 * padding - kernel_size) / stride + 1;
    int OH = (height + 2 * padding - kernel_size) / stride + 1;
    int OW = (width  + 2 * padding - kernel_size) / stride + 1;
    int total = batch_size * channels * OD * OH * OW;

    int threads = 256;
    int blocks = div_up(total, threads);

    avgpool3d_fp16_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        batch_size, channels, depth, height, width,
        kernel_size, stride, padding,
        OD, OH, OW
    );

    cudaDeviceSynchronize();
}
