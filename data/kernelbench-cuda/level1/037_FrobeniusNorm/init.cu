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
#include <cstdint>
#include <cmath>
#include <cassert>

// Helper: warp-wide sum (FP32)
__inline__ __device__ float warpReduceSum(float val) {
    // Use warp shuffle for intra-warp reduction
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-wide reduction (FP32), returns sum in thread 0
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // max 1024 threads/32 = 32 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val); // Each warp reduces to one value

    __syncthreads();
    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f; // Only first warp loads

    if (wid == 0)
        val = warpReduceSum(val); // Final reduction within first warp

    return val; // Only thread 0 has the total sum
}

// ========================
// Kernel: Frobenius Norm (L2) reduction to FP32
// ========================
__global__ void frobenius_norm_kernel(
    const half* __restrict__ input,
    float* __restrict__ block_sums,
    int64_t total_elems
) {
    // Each thread accumulates a chunk in fp32
    float sum = 0.0f;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elems; i += gridDim.x * blockDim.x) {
        float v = __half2float(input[i]);
        sum += v * v;
    }
    // Reduce within block
    float block_sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        block_sums[blockIdx.x] = block_sum;
}

// ========================
// Kernel: Normalize with Frobenius norm
// ========================
__global__ void normalize_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    float inv_norm,
    int64_t total_elems
) {
    // Write normalized output: output[i] = input[i] * inv_norm (in fp32, write to fp16)
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elems; i += gridDim.x * blockDim.x) {
        float v = __half2float(input[i]);
        float o = v * inv_norm;
        output[i] = __float2half_rn(o);
    }
}

// ========================
// Host launcher
// ========================
void launch_gpu_implementation(
    void* output,                // Output tensor (GPU memory)
    void* input,                 // Input tensor (GPU memory)
    int64_t batch_size,
    int64_t features,
    int64_t dim1,
    int64_t dim2
) {
    using half = __half;
    int64_t total_elems = batch_size * features * dim1 * dim2;

    // Kernel config
    int threads = 256;
    int blocks = 128;
    if (total_elems < threads * blocks)
        blocks = (total_elems + threads - 1) / threads;

    // Allocate block_sums on device for reduction
    float* d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(float));

    // 1. Compute sum of squares (Frobenius norm^2)
    frobenius_norm_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input), d_block_sums, total_elems
    );
    cudaDeviceSynchronize();

    // 2. Copy block_sums to host, sum up, sqrt
    float h_block_sums[blocks];
    cudaMemcpy(h_block_sums, d_block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i)
        sum += h_block_sums[i];
    float norm = sqrtf(sum);
    // Avoid division by zero: if norm==0, set inv_norm=0 (output will be zero)
    float inv_norm = (norm > 0.f) ? 1.0f / norm : 0.0f;

    cudaFree(d_block_sums);

    // 3. Launch normalization kernel
    normalize_kernel<<<blocks, threads>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        inv_norm,
        total_elems
    );
    cudaDeviceSynchronize();
}
