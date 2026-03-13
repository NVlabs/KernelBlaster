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
CUDA GroupNorm kernel for fp16 input/output.

Input tensor shape: (batch_size, num_features, dim1, dim2)
Output tensor shape: (batch_size, num_features, dim1, dim2)
weight, bias: shape [num_features], fp16

GroupNorm is computed as:
  For each group (across all elements in the group):
    mean = mean(x)
    var = mean((x - mean)^2)
    y = (x - mean) / sqrt(var + eps) * weight + bias
  Each group is (num_features//num_groups) channels, groups are split over the C dimension.
  weight, bias are per-channel.

Numerical stability: all reductions (mean, var) are done in float32.

Assumes:
- input/output/weight/bias are all fp16, contiguous in NCHW layout.
- eps = 1e-5, as in PyTorch default.

Tested for:
    batch_size = 16
    num_features = 64
    num_groups = 8
    dim1 = 256
    dim2 = 256
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cassert>

// CUDA: warp size
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// GroupNorm uses this epsilon
#define GROUPNORM_EPS 1e-5f

// Utility: warp-wide reduction (sum) for float
__inline__ __device__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Utility: block-wide reduction (sum) for float
template <unsigned blockSize>
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // supports up to 1024 threads

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);
    __syncthreads();

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0)
        val = warpReduceSum(val);
    return val;
}

// Main GroupNorm kernel, 1 block per (N, group), parallelizes within the group
__global__ void groupnorm_forward_fp16_kernel(
    const half* __restrict__ input,   // [N, C, H, W]
    half* __restrict__ output,        // [N, C, H, W]
    const half* __restrict__ weight,  // [C]
    const half* __restrict__ bias,    // [C]
    int N, int C, int G, int H, int W
) {
    // Each block computes one (N, group)
    // gridDim.x = N * G
    int ng = blockIdx.x;
    int n = ng / G;
    int g = ng % G;

    int group_channels = C / G;
    int group_size = group_channels * H * W;
    int c_start = g * group_channels;
    int c_end = c_start + group_channels;

    // Pointers to start of this (N, group)
    const half* input_ptr = input + n * C * H * W + c_start * H * W;
    half* output_ptr = output + n * C * H * W + c_start * H * W;

    // Step 1: compute group mean/var in FP32
    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c = idx / (H * W);
        int hw = idx % (H * W);
        int offset = c * H * W + hw;
        float val = __half2float(input_ptr[offset]);
        sum += val;
        sumsq += val * val;
    }

    // Block-wide reduction
    sum = blockReduceSum<1024>(sum);
    sumsq = blockReduceSum<1024>(sumsq);

    float mean = 0.0f, var = 0.0f;
    if (threadIdx.x == 0) {
        mean = sum / group_size;
        var = sumsq / group_size - mean * mean;
        // Store mean/var to shared memory for broadcast
        extern __shared__ float stats[];
        stats[0] = mean;
        stats[1] = var;
    }
    __syncthreads();

    extern __shared__ float stats[];
    mean = stats[0];
    var = stats[1];

    float inv_std = rsqrtf(var + GROUPNORM_EPS);

    // Step 2: normalize and affine
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c = idx / (H * W);
        int hw = idx % (H * W);
        int offset = c * H * W + hw;
        int channel = c_start + c;

        float val = __half2float(input_ptr[offset]);
        float w = __half2float(weight[channel]);
        float b = __half2float(bias[channel]);
        float norm = (val - mean) * inv_std;
        float y = norm * w + b;
        output_ptr[offset] = __float2half_rn(y);
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
    int num_groups,
    int dim1,
    int dim2
) {
    // N = batch_size, C = num_features, G = num_groups, H = dim1, W = dim2
    const int N = batch_size;
    const int C = num_features;
    const int G = num_groups;
    const int H = dim1;
    const int W = dim2;

    // Each block computes one (N, group)
    dim3 grid(N * G);
    // Use 1024 threads per block for best occupancy, reduce if needed
    dim3 block(1024);

    // Shared memory for mean/var per block
    size_t smem = 2 * sizeof(float);

    groupnorm_forward_fp16_kernel<<<grid, block, smem>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        N, C, G, H, W
    );
    cudaDeviceSynchronize();
}
