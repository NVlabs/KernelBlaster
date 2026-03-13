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
#include <math_constants.h>
#include <cstdio>
#include <cassert>

// CUDA kernel for KL divergence (batchmean reduction)
//   output: (float16*) pointer to 1-element tensor
//   predictions: (float16*) [batch_size, feature_size] (softmax probabilities, sum to 1)
//   targets:     (float16*) [batch_size, feature_size] (softmax probabilities, sum to 1)
//   batch_size:  number of rows
//   feature_size: number of columns (4096)
//
// Computes (in fp32 for accumulation):
//   kl = sum_{i,j} targets[i,j] * (log(targets[i,j]) - log(predictions[i,j]))
//   output[0] = kl / batch_size
//
// Performance: 
//   - Vectorized fp16 loads (float4/half2 when possible)
//   - Each block reduces to shared memory, then final reduction in block 0
//   - All arithmetic (log, division, etc.) is done in fp32 for stability

constexpr int THREADS_PER_BLOCK = 256;

// Helper: numerically safe logf(x) for x > 0, returns -1e4 for x <= 0 (shouldn't happen with softmax, but guard anyway)
__device__ inline float safe_logf(float x) {
    return x > 0.f ? logf(x) : -1e4f;
}

__global__ void kl_div_batchmean_kernel(
    const half* __restrict__ predictions, // [batch_size, feature_size]
    const half* __restrict__ targets,     // [batch_size, feature_size]
    float* block_sums,                    // partial sums, one per block
    int64_t batch_size,
    int64_t feature_size
) {
    extern __shared__ float sdata[];
    float thread_sum = 0.0f;

    // Total number of elements
    int64_t total = batch_size * feature_size;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    // Vectorized loads: process 4 elements at a time when possible
    int64_t vec4_end = (total / 4) * 4;

    for (int64_t i = tid * 4; i < vec4_end; i += stride * 4) {
        int64_t idx = i;
        // Load 4 predictions and 4 targets at once
        float4 pred4, targ4;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            pred4.x = __half2float(predictions[idx + 0]);
            targ4.x = __half2float(targets[idx + 0]);
            pred4.y = __half2float(predictions[idx + 1]);
            targ4.y = __half2float(targets[idx + 1]);
            pred4.z = __half2float(predictions[idx + 2]);
            targ4.z = __half2float(targets[idx + 2]);
            pred4.w = __half2float(predictions[idx + 3]);
            targ4.w = __half2float(targets[idx + 3]);
        }
        thread_sum += targ4.x * (safe_logf(targ4.x) - safe_logf(pred4.x));
        thread_sum += targ4.y * (safe_logf(targ4.y) - safe_logf(pred4.y));
        thread_sum += targ4.z * (safe_logf(targ4.z) - safe_logf(pred4.z));
        thread_sum += targ4.w * (safe_logf(targ4.w) - safe_logf(pred4.w));
    }

    // Handle leftovers (if any)
    for (int64_t i = vec4_end + tid; i < total; i += stride) {
        float pred = __half2float(predictions[i]);
        float targ = __half2float(targets[i]);
        thread_sum += targ * (safe_logf(targ) - safe_logf(pred));
    }

    // Reduce within block
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction (in shared memory, fp32)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    // Write block result
    if (threadIdx.x == 0)
        block_sums[blockIdx.x] = sdata[0];
}

// Final reduction kernel: sum all block results to output[0]
__global__ void final_reduce_kernel(const float* block_sums, int num_blocks, int64_t batch_size, half* output) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += block_sums[i];
    }
    // Block reduction
    __shared__ float sdata[THREADS_PER_BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float batchmean = sdata[0] / (float)batch_size;
        output[0] = __float2half_rn(batchmean);
    }
}

void launch_gpu_implementation(
    void* output,      // float16*, shape [1]
    void* predictions, // float16*, [batch_size, feature_size]
    void* targets,     // float16*, [batch_size, feature_size]
    int64_t batch_size,
    int64_t feature_size
) {
    // Use a large number of blocks for full GPU utilization
    const int threads = THREADS_PER_BLOCK;
    int blocks = (batch_size * feature_size + threads * 8 - 1) / (threads * 8);
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096; // Reasonable upper bound

    // Allocate workspace for block sums
    float* d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(float));

    // Launch main reduction kernel
    size_t smem = threads * sizeof(float);
    kl_div_batchmean_kernel<<<blocks, threads, smem>>>(
        static_cast<const half*>(predictions),
        static_cast<const half*>(targets),
        d_block_sums,
        batch_size,
        feature_size
    );
    cudaDeviceSynchronize();

    // Final reduction to single value, write to output[0]
    final_reduce_kernel<<<1, threads>>>(d_block_sums, blocks, batch_size, static_cast<half*>(output));
    cudaDeviceSynchronize();

    cudaFree(d_block_sums);
}

