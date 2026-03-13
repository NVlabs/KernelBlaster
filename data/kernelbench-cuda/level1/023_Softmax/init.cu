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
// cuda_model.cuh

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <float.h>

// Fast warp-level reduction for the maximum (fp16->fp32), using shuffle
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Fast warp-level reduction for the sum (fp16->fp32), using shuffle
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-wide reduction for maximum (fp16->fp32)
__inline__ __device__ float block_reduce_max(float val) {
    static __shared__ float shared[32]; // Up to 1024 threads: 32 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final reduction within the first warp
    float max_val = -FLT_MAX;
    if (threadIdx.x < blockDim.x / 32) max_val = shared[lane];
    if (wid == 0) {
        max_val = warp_reduce_max(max_val);
    }
    return max_val;
}

// Block-wide reduction for sum (fp16->fp32)
__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32]; // Up to 1024 threads: 32 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final reduction within the first warp
    float sum_val = 0.0f;
    if (threadIdx.x < blockDim.x / 32) sum_val = shared[lane];
    if (wid == 0) {
        sum_val = warp_reduce_sum(sum_val);
    }
    return sum_val;
}

/*
    CUDA Kernel for batched softmax over axis-1 (row softmax):

    - Input:  input  (half*), shape (batch_size, dim)
    - Output: output (half*), shape (batch_size, dim)
    - For each row: output[i, :] = softmax(input[i, :])
    - All computation and IO in fp16, but accumulations (max, sum) in fp32 for accuracy.

    Efficient for large dim (e.g., 16,384).
    Uses block-per-row strategy, vectorized loads, and shared memory for reductions.
*/

__global__ void softmax_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size,
    int dim
) {
    // Each block handles one row (one batch element)
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Vectorized IO for speed
    using half2 = __half2;
    constexpr int VEC = 8; // 8 * 16 = 128b, good for coalescing
    int n_vec = dim / VEC;
    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    const half* row_in = input + row * dim;
    half* row_out = output + row * dim;

    // 1. Compute max for numerical stability (fp32 accumulator)
    float local_max = -FLT_MAX;
    for (int i = tid * VEC; i < n_vec * VEC; i += block_threads * VEC) {
        half vals[VEC];
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            vals[v] = row_in[i + v];
        }
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            float f = __half2float(vals[v]);
            if (f > local_max) local_max = f;
        }
    }
    // Handle leftovers (if dim not divisible by VEC)
    for (int i = n_vec * VEC + tid; i < dim; i += block_threads) {
        float f = __half2float(row_in[i]);
        if (f > local_max) local_max = f;
    }

    // Block-wide max reduction
    float max_val = block_reduce_max(local_max);
    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = max_val;
    __syncthreads();
    max_val = row_max;

    // 2. Compute sum of exp(x - max) (fp32 accumulator)
    float local_sum = 0.0f;
    for (int i = tid * VEC; i < n_vec * VEC; i += block_threads * VEC) {
        half vals[VEC];
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            vals[v] = row_in[i + v];
        }
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            float f = __half2float(vals[v]);
            local_sum += expf(f - max_val);
        }
    }
    for (int i = n_vec * VEC + tid; i < dim; i += block_threads) {
        float f = __half2float(row_in[i]);
        local_sum += expf(f - max_val);
    }

    // Block-wide sum reduction
    float sum_val = block_reduce_sum(local_sum);
    __shared__ float row_sum;
    if (threadIdx.x == 0) row_sum = sum_val;
    __syncthreads();
    sum_val = row_sum;

    // 3. Write output: y_i = exp(x_i - max) / sum
    for (int i = tid * VEC; i < n_vec * VEC; i += block_threads * VEC) {
        half vals[VEC], outs[VEC];
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            vals[v] = row_in[i + v];
        }
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            float f = __half2float(vals[v]);
            float ex = expf(f - max_val);
            float softmax = ex / sum_val;
            outs[v] = __float2half_rn(softmax);
        }
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            row_out[i + v] = outs[v];
        }
    }
    for (int i = n_vec * VEC + tid; i < dim; i += block_threads) {
        float f = __half2float(row_in[i]);
        float ex = expf(f - max_val);
        float softmax = ex / sum_val;
        row_out[i] = __float2half_rn(softmax);
    }
}

void launch_gpu_implementation(
    void* output,           // float16* output, shape: (batch_size, dim)
    void* input,            // float16* input, shape: (batch_size, dim)
    int64_t batch_size,
    int64_t dim
) {
    // Each row gets its own block, block size chosen for memory bandwidth and reduction efficiency
    constexpr int threads_per_block = 256;
    int grid = static_cast<int>(batch_size);

    softmax_fp16_kernel<<<grid, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    cudaDeviceSynchronize();
}
