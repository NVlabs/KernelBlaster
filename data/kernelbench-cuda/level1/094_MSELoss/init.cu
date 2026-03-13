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
#include <cassert>
#include <cstdio>

// Efficient block size for reduction kernel
constexpr int BLOCK_SIZE = 256;

// CUDA kernel to compute squared error and block-level sum in FP32
__global__ void mse_square_sum_kernel(
    const half* __restrict__ predictions,
    const half* __restrict__ targets,
    float* __restrict__ block_sums,
    int64_t N,
    int64_t D
) {
    extern __shared__ float sdata[]; // Shared memory for block reduction

    // Compute global linear thread index
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int total = N * D;

    // Each thread computes partial sum
    float local_sum = 0.0f;

    // Loop with grid-stride for full coverage
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        float pred = __half2float(predictions[i]);
        float targ = __half2float(targets[i]);
        float diff = pred - targ;
        local_sum += diff * diff;
    }

    // Store in shared memory for reduction
    sdata[tid] = local_sum;
    __syncthreads();

    // Block-wide reduction (in shared memory)
#pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Final reduction kernel to sum block_sums into a scalar output, then divide by total for mean
__global__ void mse_final_reduce_kernel(
    const float* __restrict__ block_sums,
    int num_blocks,
    int64_t total_elements,
    half* __restrict__ output
) {
    float sum = 0.0f;
    // Use a single thread to finish reduction, as num_blocks is small
    for (int i = 0; i < num_blocks; ++i) {
        sum += block_sums[i];
    }
    float mean = sum / static_cast<float>(total_elements);
    // Store result as half
    output[0] = __float2half_rn(mean);
}

// Host launcher function
void launch_gpu_implementation(
    void* output,           // output: pointer to float16 (fp16) scalar [1 element]
    void* predictions,      // input: pointer to float16 tensor [batch_size, 4096]
    void* targets,          // input: pointer to float16 tensor [batch_size, 4096]
    int64_t batch_size,     // batch size (128)
    int64_t input_dim       // input dimension (4096)
) {
    int64_t total = batch_size * input_dim;

    // Configure kernel launch
    int threads = BLOCK_SIZE;
    // Use enough blocks to saturate GPU but not too many for reduction
    int blocks = (total + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024; // Clamp for safety (shared mem and launch bound)
    
    // Allocate temporary buffer for block results (on device)
    float* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, blocks * sizeof(float));

    // Launch first kernel: compute squared error sum per block
    mse_square_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        static_cast<const half*>(predictions),
        static_cast<const half*>(targets),
        d_block_sums,
        batch_size,
        input_dim
    );

    // Launch second kernel: final reduction and mean
    mse_final_reduce_kernel<<<1, 1>>>(
        d_block_sums,
        blocks,
        total,
        static_cast<half*>(output)
    );

    cudaFree(d_block_sums);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
}
