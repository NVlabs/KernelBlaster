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
CUDA implementation of InstanceNorm2d for fp16 tensors.
  - Performs instance normalization across each (C, H, W) for each batch.
  - Accumulates mean and variance in float32 for numerical stability.
  - Applies affine (weight/bias) parameters, also in fp16.
  - Kernel is optimized for coalesced memory access and warp-level reductions.

Host function:
void launch_gpu_implementation(
    void* output,   // (B, C, H, W), fp16
    void* input,    // (B, C, H, W), fp16
    void* weight,   // (C), fp16
    void* bias,     // (C), fp16
    int batch_size,
    int num_features,
    int height,
    int width
);

Tested against PyTorch InstanceNorm2d (affine=True) for fp16.

*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cassert>
#include <cstdio>

// For warp-level reductions
#if (__CUDACC_VER_MAJOR__ >= 9)
#define CUDA_WARP_SIZE 32
#else
#define CUDA_WARP_SIZE 32
#endif

// Utility for warp reduction (sum)
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = CUDA_WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Utility for warp reduction (sum of squares)
__inline__ __device__ float warp_reduce_sumsq(float val) {
    #pragma unroll
    for (int offset = CUDA_WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

constexpr float kEps = 1e-5f;

// Kernel: each block handles (B, C) pair, each thread processes a row of (H*W)
__global__ void instance_norm2d_affine_fp16_kernel(
    const half* __restrict__ input,   // (B, C, H, W)
    half* __restrict__ output,        // (B, C, H, W)
    const half* __restrict__ weight,  // (C)
    const half* __restrict__ bias,    // (C)
    int B, int C, int H, int W
) {
    // Launch config: blockIdx.x = b, blockIdx.y = c, blockDim.x = 256, gridDim = (B, C)
    int b = blockIdx.x;
    int c = blockIdx.y;
    int hw = H * W;
    int tid = threadIdx.x;

    // Shared memory for block reduction of mean and var
    __shared__ float mean_shared;
    __shared__ float var_shared;

    // Step 1: compute mean and variance (accum in fp32)
    float sum = 0.0f, sumsq = 0.0f;

    // Each thread processes several HW elements
    for (int i = tid; i < hw; i += blockDim.x) {
        int idx = ((b * C + c) * H * W) + i;
        float v = __half2float(input[idx]);
        sum += v;
        sumsq += v * v;
    }

    // Block reduction using shared memory
    // Step 1: reduce in-warp
    float warp_sum = warp_reduce_sum(sum);
    float warp_sumsq = warp_reduce_sumsq(sumsq);

    // Step 2: warp leaders write to shared mem
    if ((threadIdx.x & (CUDA_WARP_SIZE-1)) == 0) {
        atomicAdd(&mean_shared, warp_sum);
        atomicAdd(&var_shared, warp_sumsq);
    }
    __syncthreads();

    // Only the first thread computes final mean/var and broadcasts
    float mean, var;
    if (threadIdx.x == 0) {
        mean = mean_shared / hw;
        var = var_shared / hw - mean * mean;
        // Clamp var for numerical stability
        if (var < 0.0f) var = 0.0f;
        var = 1.0f / sqrtf(var + kEps);
        mean_shared = mean;
        var_shared = var;
    }
    __syncthreads();
    mean = mean_shared;
    var = var_shared;

    // Step 2: normalize and write output
    float w = __half2float(weight[c]);
    float b_ = __half2float(bias[c]);

    for (int i = tid; i < hw; i += blockDim.x) {
        int idx = ((b * C + c) * H * W) + i;
        float v = __half2float(input[idx]);
        float norm = (v - mean) * var;
        float affine = norm * w + b_;
        output[idx] = __float2half_rn(affine);
    }
}

// Host launcher
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int num_features,
    int height,
    int width
) {
    // Launch config: one block per (b,c), 256 threads per block
    dim3 block(256);
    dim3 grid(batch_size, num_features);

    instance_norm2d_affine_fp16_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        batch_size,
        num_features,
        height,
        width
    );
    cudaDeviceSynchronize();
}
