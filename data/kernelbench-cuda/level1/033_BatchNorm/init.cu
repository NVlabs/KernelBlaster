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
// BatchNorm2d (inference) CUDA kernel for fp16 tensors
//
// This kernel applies BatchNorm2d (inference mode, using running_mean/running_var, not batch statistics)
// to a 4D tensor (N, C, H, W) in fp16, with fp32 accumulation for accuracy.
// The operation is, for each element:
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
//
// All tensors are fp16: input, output, weight (gamma), bias (beta), running_mean, running_var
// Accumulation for normalization is performed in fp32 for stability.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <cmath>

// Use the same epsilon as PyTorch BatchNorm2d default for fp16
#ifndef BN_EPSILON
#define BN_EPSILON 1e-5f
#endif

// Kernel for BatchNorm2d inference, fp16 I/O, fp32 math for normalization.
// Launch with enough threads to cover all elements in (N, C, H, W).
__global__ void batchnorm2d_inference_fp16_kernel(
    half* __restrict__ out,         // (N, C, H, W) output, fp16
    const half* __restrict__ inp,   // (N, C, H, W) input, fp16
    const half* __restrict__ weight, // (C,) gamma, fp16
    const half* __restrict__ bias,   // (C,) beta, fp16
    const half* __restrict__ running_mean, // (C,) running mean, fp16
    const half* __restrict__ running_var,  // (C,) running var, fp16
    int64_t N, int64_t C, int64_t H, int64_t W
) {
    // Flattened index for 4D tensor
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H * W;
    if (idx >= total) return;

    // Recover 4D indices: (n, c, h, w)
    int64_t w = idx % W;
    int64_t h = (idx / W) % H;
    int64_t c = (idx / (W * H)) % C;
    int64_t n = idx / (C * H * W);

    // Compute offset for (n, c, h, w)
    int64_t offset = ((n * C + c) * H + h) * W + w;

    // Load input and BN params
    float x = __half2float(inp[offset]);
    float mean = __half2float(running_mean[c]);
    float var = __half2float(running_var[c]);
    float gamma = __half2float(weight[c]);
    float beta = __half2float(bias[c]);

    // BatchNorm2d (inference): (x - mean) / sqrt(var + eps) * gamma + beta
    float y = (x - mean) / sqrtf(var + BN_EPSILON);
    y = y * gamma + beta;

    // Convert back to fp16
    out[offset] = __float2half_rn(y);
}

// Host function to launch the kernel
void launch_gpu_implementation(
    void* output,               // output tensor (fp16, GPU)
    void* input,                // input tensor (fp16, GPU)
    void* weight,               // BatchNorm weight (gamma) (fp16, GPU)
    void* bias,                 // BatchNorm bias (beta) (fp16, GPU)
    void* running_mean,         // running mean (fp16, GPU)
    void* running_var,          // running var (fp16, GPU)
    int64_t batch_size,
    int64_t num_features,
    int64_t dim1,
    int64_t dim2
) {
    // N, C, H, W
    int64_t N = batch_size;
    int64_t C = num_features;
    int64_t H = dim1;
    int64_t W = dim2;
    int64_t total = N * C * H * W;

    // Use 256 threads per block for good occupancy and memory bandwidth
    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    batchnorm2d_inference_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<const half*>(running_mean),
        static_cast<const half*>(running_var),
        N, C, H, W
    );

    cudaDeviceSynchronize();
}
