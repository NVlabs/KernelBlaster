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
#include <cassert>

/*
 * CUDA implementation of Triplet Margin Loss for metric learning.
 *
 * This kernel computes the triplet margin loss for a batch of anchor, positive, and negative vectors.
 * All input/output tensors are half-precision (fp16), but accumulation is done in float32 for numerical stability.
 *
 * The loss for each sample is:
 *   loss = max(||anchor - positive||_2 - ||anchor - negative||_2 + margin, 0)
 *
 * Inputs:
 *   anchor   [batch_size, input_dim]  (half)
 *   positive [batch_size, input_dim]  (half)
 *   negative [batch_size, input_dim]  (half)
 *   margin   (float)
 *   batch_size, input_dim (int64_t)
 *
 * Output:
 *   output [1] (half) -- the mean triplet margin loss (as in PyTorch)
 */

#define WARP_SIZE 32

// Utility: warp-level reduction for float
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Utility: block-level reduction for float
__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE]; // One float per warp
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val); // Each warp reduces its values

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();

    // Only first warp loads shared memory and reduces
    float sum = 0.0f;
    if (wid == 0) {
        sum = (lane < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    return sum;
}

// CUDA kernel: one thread per sample (per batch)
__global__ void triplet_margin_loss_kernel(
    const half* __restrict__ anchor,
    const half* __restrict__ positive,
    const half* __restrict__ negative,
    float margin,
    int64_t batch_size,
    int64_t input_dim,
    float* loss_per_block // One float per block for block-level reduction
) {
    extern __shared__ float sdata[]; // Used for block reduction if needed

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;

    if (tid < batch_size) {
        // Compute squared L2 norm between anchor and positive
        float ap_sq = 0.0f, an_sq = 0.0f;
        for (int64_t d = 0; d < input_dim; ++d) {
            float a = __half2float(anchor[tid * input_dim + d]);
            float p = __half2float(positive[tid * input_dim + d]);
            float n = __half2float(negative[tid * input_dim + d]);
            float ap = a - p;
            float an = a - n;
            ap_sq += ap * ap;
            an_sq += an * an;
        }
        float ap_norm = sqrtf(ap_sq);
        float an_norm = sqrtf(an_sq);
        float loss = ap_norm - an_norm + margin;
        if (loss > 0.0f)
            local_loss = loss;
    }

    // Block-wide reduction
    float block_sum = block_reduce_sum(local_loss);

    // Write block result to global memory
    if (threadIdx.x == 0) {
        loss_per_block[blockIdx.x] = block_sum;
    }
}

// Host launcher
void launch_gpu_implementation(
    void* output,    // half* [1]
    void* anchor,    // half* [batch_size, input_dim]
    void* positive,  // half* [batch_size, input_dim]
    void* negative,  // half* [batch_size, input_dim]
    float margin,
    int64_t batch_size,
    int64_t input_dim
) {
    // Set up block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((batch_size + threadsPerBlock - 1) / threadsPerBlock);

    // Allocate device buffer for block-level loss reductions
    float* d_loss_per_block = nullptr;
    cudaMalloc(&d_loss_per_block, sizeof(float) * blocksPerGrid);

    // Launch kernel
    triplet_margin_loss_kernel<<<blocksPerGrid, threadsPerBlock, 0>>>(
        static_cast<const half*>(anchor),
        static_cast<const half*>(positive),
        static_cast<const half*>(negative),
        margin,
        batch_size,
        input_dim,
        d_loss_per_block
    );
    cudaDeviceSynchronize();

    // Reduce block loss sums on host
    float* h_loss_per_block = new float[blocksPerGrid];
    cudaMemcpy(h_loss_per_block, d_loss_per_block, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);

    float final_loss = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        final_loss += h_loss_per_block[i];
    }
    // Mean reduction as in PyTorch (divide by batch_size)
    final_loss = final_loss / (float)batch_size;

    // Write output as half
    half h_final_loss = __float2half_rn(final_loss);
    cudaMemcpy(static_cast<half*>(output), &h_final_loss, sizeof(half), cudaMemcpyHostToDevice);

    // Cleanup
    delete[] h_loss_per_block;
    cudaFree(d_loss_per_block);
}
