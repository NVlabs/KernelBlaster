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
#include <cmath>
#include <cassert>

// Fast CUDA kernel for Cosine Similarity Loss (see PyTorch code for reference).
// All computation is in FP32 for accumulators, I/O is FP16.

__global__ void cosine_similarity_loss_kernel(
    const half* __restrict__ predictions, // [batch_size, input_dim]
    const half* __restrict__ targets,     // [batch_size, input_dim]
    float* __restrict__ partial_sums,     // [num_blocks * 2], FP32
    int64_t batch_size,
    int64_t input_dim
) {
    extern __shared__ float smem[]; // [blockDim.x * 2]
    float* smem_sum = smem;                 // cosine sum
    float* smem_count = smem + blockDim.x;  // count sum (for mean)

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int block_id = blockIdx.x;

    // Each thread processes multiple samples in the batch
    float local_sum = 0.0f;
    float local_count = 0.0f;

    // Stride loop over the batch
    for (int i = block_id * block_size + tid; i < batch_size; i += gridDim.x * block_size) {
        // Compute dot(pred, target) and norms in FP32
        float dot = 0.0f;
        float pred_norm_sq = 0.0f;
        float targ_norm_sq = 0.0f;

        // Loop over input_dim in vectorized fashion
        int j = 0;
#ifdef __CUDA_ARCH__
        // Vectorize using half2 if possible
        for (; j + 1 < input_dim; j += 2) {
            half2 p = reinterpret_cast<const half2*>(predictions + i * input_dim)[j/2];
            half2 t = reinterpret_cast<const half2*>(targets     + i * input_dim)[j/2];
            float2 pf = __half22float2(p);
            float2 tf = __half22float2(t);

            dot         += pf.x * tf.x + pf.y * tf.y;
            pred_norm_sq += pf.x * pf.x + pf.y * pf.y;
            targ_norm_sq += tf.x * tf.x + tf.y * tf.y;
        }
#endif
        // Handle last odd element if needed
        for (; j < input_dim; ++j) {
            float pf = __half2float(predictions[i * input_dim + j]);
            float tf = __half2float(targets[i * input_dim + j]);
            dot         += pf * tf;
            pred_norm_sq += pf * pf;
            targ_norm_sq += tf * tf;
        }

        // Compute cosine similarity
        float denom = sqrtf(pred_norm_sq * targ_norm_sq + 1e-12f); // add epsilon for stability
        float cosine = (denom > 0.f) ? (dot / denom) : 0.f;
        // Clamp to [-1, 1] for numerical safety (matches PyTorch)
        cosine = fmaxf(fminf(cosine, 1.f), -1.f);

        local_sum += (1.f - cosine);
        local_count += 1.f;
    }

    // Write to shared memory for reduction
    smem_sum[tid] = local_sum;
    smem_count[tid] = local_count;
    __syncthreads();

    // Parallel reduction in shared memory (sum)
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_sum[tid] += smem_sum[tid + s];
            smem_count[tid] += smem_count[tid + s];
        }
        __syncthreads();
    }

    // Write per-block partial results to global memory
    if (tid == 0) {
        partial_sums[block_id] = smem_sum[0];
        partial_sums[gridDim.x + block_id] = smem_count[0];
    }
}

// Second reduction kernel: sum partial_sums[0..num_blocks-1], partial_sums[num_blocks..2*num_blocks-1]
__global__ void final_reduce_kernel(
    const float* __restrict__ partial_sums, // [num_blocks * 2]
    half* __restrict__ output,              // [1]
    int num_blocks
) {
    float sum = 0.0f;
    float count = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
        count += partial_sums[num_blocks + i];
    }

    // Reduce within block
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_count = sdata + blockDim.x;

    s_sum[threadIdx.x] = sum;
    s_count[threadIdx.x] = count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_count[threadIdx.x] += s_count[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = (s_count[0] > 0.f) ? (s_sum[0] / s_count[0]) : 0.f;
        output[0] = __float2half_rn(mean);
    }
}

// Host launch function
void launch_gpu_implementation(
    void* output,
    void* predictions,
    void* targets,
    int64_t batch_size,
    int64_t input_dim
) {
    // Use 256 threads per block for reduction efficiency
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;
    // Cap number of blocks to avoid excessive memory usage
    num_blocks = min(num_blocks, 1024);

    // Allocate partial_sums buffer on device (2*num_blocks floats)
    float* d_partial_sums = nullptr;
    cudaMalloc(&d_partial_sums, sizeof(float) * num_blocks * 2);

    // Launch first kernel
    size_t shmem_bytes = block_size * 2 * sizeof(float);
    cosine_similarity_loss_kernel<<<num_blocks, block_size, shmem_bytes>>>(
        static_cast<const half*>(predictions),
        static_cast<const half*>(targets),
        d_partial_sums,
        batch_size,
        input_dim
    );

    // Launch final reduction kernel
    // Only a single block is needed, use block_size threads
    final_reduce_kernel<<<1, block_size, block_size * 2 * sizeof(float)>>>(
        d_partial_sums,
        static_cast<half*>(output),
        num_blocks
    );

    cudaFree(d_partial_sums);
    cudaDeviceSynchronize();
}
