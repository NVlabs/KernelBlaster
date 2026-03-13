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
#include <cmath>
#include <stdint.h>
#include <stdio.h>

// Utility: CUDA warp-level reduction for float
__inline__ __device__ float warpReduceSum(float val) {
    // Reduce within a warp using shuffle instructions
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Utility: block-level reduction for float, returns the sum for the thread with threadIdx.x == 0 in each block
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // max 1024 threads/block, 32 warps max
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val); // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();

    // Read from shared memory only if that warp existed
    float sum = 0.0f;
    if (threadIdx.x < blockDim.x / 32)
        sum = shared[lane];
    if (wid == 0) {
        sum = warpReduceSum(sum);
    }
    return sum;
}

// Kernel: L2 normalization along dim=1 (features), input/output in fp16, accumulation in fp32
// Each block processes one row (sample) of the input
__global__ void l2norm_fp16_kernel(
    const half* __restrict__ input, // [batch_size, dim]
    half* __restrict__ output,      // [batch_size, dim]
    int64_t batch_size,
    int64_t dim
) {
    extern __shared__ float s_norm[]; // Shared memory for storing norm per block

    int row = blockIdx.x; // Each block processes one row
    int tid = threadIdx.x;

    if (row >= batch_size)
        return;

    // Step 1: Compute sum of squares in FP32 for this row
    float sum = 0.0f;
    for (int col = tid; col < dim; col += blockDim.x) {
        const int idx = row * dim + col;
        float val = __half2float(input[idx]);
        sum += val * val;
    }

    // Block reduction to get the total sum of squares for this row
    float l2 = blockReduceSum(sum);

    // Only one thread writes the norm to shared memory
    if (threadIdx.x == 0) {
        float norm = sqrtf(l2 + 1e-6f); // Add epsilon for numerical stability
        s_norm[0] = norm;
    }
    __syncthreads();

    float norm = s_norm[0];

    // Step 2: Normalize each element in the row
    for (int col = tid; col < dim; col += blockDim.x) {
        const int idx = row * dim + col;
        float val = __half2float(input[idx]);
        float normed = val / norm;
        output[idx] = __float2half_rn(normed);
    }
}

// Host launcher function
void launch_gpu_implementation(
    void* output,        // output: pointer to GPU memory, shape [batch_size, dim], type half
    void* input,         // input: pointer to GPU memory, shape [batch_size, dim], type half
    int64_t batch_size,  // batch size (16)
    int64_t dim          // feature dimension (16384)
) {
    // Each block processes one row, launch batch_size blocks
    // Use 256 threads per block (tunable, 256 is good for large dims)
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>(batch_size);

    // Shared memory for one float (the norm)
    size_t sharedMemBytes = sizeof(float);

    l2norm_fp16_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        batch_size,
        dim
    );

    cudaDeviceSynchronize();
}

