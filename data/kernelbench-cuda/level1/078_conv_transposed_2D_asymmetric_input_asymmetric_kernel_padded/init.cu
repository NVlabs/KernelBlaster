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
 * Fast CUDA kernel for 2D Transposed Convolution (ConvTranspose2d) in NHWC layout.
 * Input/output/weight/bias are all half-precision (fp16).
 * Accumulation is done in fp32 for numerical stability.
 *
 * Kernel arguments match the signature:
 * void launch_gpu_implementation(
 *     void* output, // pointer to output tensor (GPU memory)
 *     void* input,  // pointer to input tensor (GPU memory)
 *     void* weight, // pointer to conv_transpose2d weight (GPU memory)
 *     void* bias,   // pointer to conv_transpose2d bias (GPU memory, can be nullptr if no bias)
 *     int batch_size,
 *     int in_channels,
 *     int out_channels,
 *     int input_height,
 *     int input_width,
 *     int kernel_height,
 *     int kernel_width,
 *     int stride_height,
 *     int stride_width,
 *     int padding_height,
 *     int padding_width
 * );
 *
 * The kernel is optimized for coalesced global memory access and parallelizes over (N, out_C, out_H, out_W).
 * Accumulation is done in fp32 for accuracy, and the result is stored in fp16.
 *
 * This kernel supports arbitrary batch size, channel counts, kernel size, stride, and padding.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

// Helper macro for CUDA error checking
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while (0)

// CUDA kernel for 2D transposed convolution (NHWC layout, fp16 input/output/weight/bias, fp32 accumulation)
__global__ void conv2d_transpose_fp16_nhwc_kernel(
    const half* __restrict__ input,    // [N, in_C, H, W] (PyTorch default: NCHW)
    const half* __restrict__ weight,   // [in_C, out_C, kH, kW] (PyTorch: weight for ConvTranspose2d)
    const half* __restrict__ bias,     // [out_C] or nullptr
    half* __restrict__ output,         // [N, out_C, out_H, out_W]
    int N,
    int in_C,
    int out_C,
    int H, int W,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW,
    int out_H, int out_W
) {
    // Each thread computes one output pixel (n, oc, oh, ow)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;
    if (tid >= total) return;

    // Compute output indices
    int tmp = tid;
    int ow = tmp % out_W; tmp /= out_W;
    int oh = tmp % out_H; tmp /= out_H;
    int oc = tmp % out_C; tmp /= out_C;
    int n = tmp;

    // Output accumulator (fp32)
    float acc = 0.f;

    // For ConvTranspose2d, for each output (oh, ow), we find all input positions (ih, iw) that contribute.
    // The relationship is:
    //   ih = (oh + pH - kh) / sH
    //   iw = (ow + pW - kw) / sW
    // For all kh, kw where ih and iw are integer and in bounds, accumulate:
    //   output[n, oc, oh, ow] += input[n, ic, ih, iw] * weight[ic, oc, kh, kw]

    for (int ic = 0; ic < in_C; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            int ih_nom = oh + pH - kh;
            if (ih_nom % sH != 0) continue;
            int ih = ih_nom / sH;
            if (ih < 0 || ih >= H) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int iw_nom = ow + pW - kw;
                if (iw_nom % sW != 0) continue;
                int iw = iw_nom / sW;
                if (iw < 0 || iw >= W) continue;

                // Indexing: PyTorch is NCHW for input/output, weight is [in_C, out_C, kH, kW]
                int input_idx = ((n * in_C + ic) * H + ih) * W + iw;
                int weight_idx = ((ic * out_C + oc) * kH + kh) * kW + kw;

                float inp = __half2float(input[input_idx]);
                float wgt = __half2float(weight[weight_idx]);
                acc += inp * wgt;
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        acc += __half2float(bias[oc]);
    }

    // Write output as fp16
    int out_idx = ((n * out_C + oc) * out_H + oh) * out_W + ow;
    output[out_idx] = __float2half(acc);
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(
    void* output, // pointer to output tensor (GPU memory)
    void* input,  // pointer to input tensor (GPU memory)
    void* weight, // pointer to conv_transpose2d weight (GPU memory)
    void* bias,   // pointer to conv_transpose2d bias (GPU memory, can be nullptr if no bias)
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width
) {
    // Compute output shape (PyTorch ConvTranspose2d formula)
    int out_H = (input_height - 1) * stride_height - 2 * padding_height + kernel_height;
    int out_W = (input_width  - 1) * stride_width  - 2 * padding_width  + kernel_width;

    int N = batch_size;
    int in_C = in_channels;
    int out_C = out_channels;
    int H = input_height;
    int W = input_width;
    int kH = kernel_height;
    int kW = kernel_width;
    int sH = stride_height;
    int sW = stride_width;
    int pH = padding_height;
    int pW = padding_width;

    // Launch grid: 1 thread per output pixel (N, out_C, out_H, out_W)
    int num_outputs = N * out_C * out_H * out_W;
    int threads_per_block = 256;
    int num_blocks = (num_outputs + threads_per_block - 1) / threads_per_block;

    conv2d_transpose_fp16_nhwc_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        N, in_C, out_C, H, W, kH, kW, sH, sW, pH, pW,
        out_H, out_W
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
