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
CUDA implementation of Hinge Loss (mean(clamp(1 - predictions * targets, min=0)))
All tensors are half-precision (fp16), but accumulation is done in fp32 for numerical stability.

Function signature:
void launch_gpu_implementation(
    void* output, 
    void* predictions, 
    void* targets, 
    int batch_size, 
    int input_dim
);

- output: pointer to device scalar half (output[0])
- predictions: [batch_size, input_dim] (half)
- targets: [batch_size, 1] (half)
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

// Warp-level reduction for float
__inline__ __device__ float warp_reduce_sum(float val) {
    // Reduce within a warp
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction for float
__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32]; // One float per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    // Read from shared memory only by first warp
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0)
        val = warp_reduce_sum(val);
    return val;
}

// Hinge loss kernel: each thread processes multiple elements
__global__ void hinge_loss_kernel(
    const half* __restrict__ predictions, // [batch_size, input_dim]
    const half* __restrict__ targets,     // [batch_size, 1]
    int batch_size,
    int input_dim,
    float* __restrict__ partial_sums,     // output: one float per block
    int* __restrict__ partial_counts      // output: one int per block
) {
    // Linearize the 2D array [batch_size, input_dim]
    int total = batch_size * input_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    int count = 0;

    for (int idx = tid; idx < total; idx += stride) {
        int batch = idx / input_dim;
        int dim = idx % input_dim;
        // predictions: [batch_size, input_dim], targets: [batch_size, 1]
        float pred = __half2float(predictions[idx]);
        float targ = __half2float(targets[batch]);
        float hinge = 1.0f - pred * targ;
        if (hinge < 0.0f) hinge = 0.0f;
        sum += hinge;
        ++count;
    }

    // Block-level reduction for sum and count
    float block_sum = block_reduce_sum(sum);
    int block_count = block_reduce_sum((float)count);

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = block_sum;
        partial_counts[blockIdx.x] = block_count;
    }
}

// Final reduction kernel: reduce all partial_sums and counts to scalar mean, write as half
__global__ void final_reduce_kernel(
    const float* __restrict__ partial_sums,
    const int* __restrict__ partial_counts,
    int num_blocks,
    half* __restrict__ output // scalar
) {
    float sum = 0.0f;
    int count = 0;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
        count += partial_counts[i];
    }
    // Reduce within block
    sum = block_reduce_sum(sum);
    count = (int)block_reduce_sum((float)count);

    if (threadIdx.x == 0) {
        float mean = (count > 0) ? (sum / count) : 0.0f;
        output[0] = __float2half_rn(mean);
    }
}

void launch_gpu_implementation(
    void* output,
    void* predictions,
    void* targets,
    int batch_size,
    int input_dim
) {
    // All pointers are device pointers
    const int threadsPerBlock = 256;
    const int total = batch_size * input_dim;
    // Use enough blocks to saturate the GPU, but not more than needed
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    // Allocate temporary space for reduction
    float* d_partial_sums = nullptr;
    int* d_partial_counts = nullptr;
    cudaMalloc(&d_partial_sums, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_partial_counts, blocksPerGrid * sizeof(int));

    // First kernel: compute partial sums/counts
    hinge_loss_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const half*>(predictions),
        static_cast<const half*>(targets),
        batch_size,
        input_dim,
        d_partial_sums,
        d_partial_counts
    );
    cudaDeviceSynchronize();

    // Second kernel: final reduction to scalar mean (output is half)
    // Use a single block, 256 threads
    final_reduce_kernel<<<1, 256>>>(
        d_partial_sums,
        d_partial_counts,
        blocksPerGrid,
        static_cast<half*>(output)
    );
    cudaDeviceSynchronize();

    cudaFree(d_partial_sums);
    cudaFree(d_partial_counts);
}
