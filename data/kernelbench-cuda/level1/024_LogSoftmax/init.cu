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
#include <cstdio>
#include <math_constants.h>

// Use 256 threads per block for good occupancy
#define THREADS_PER_BLOCK 256

// Utility: vectorized load of __half2. Handles out-of-bounds safely.
__device__ inline __half2 load_half2_vec(const __half* base, int idx, int n) {
    if (idx + 1 < n) {
        return *reinterpret_cast<const __half2*>(base + idx);
    } else {
        // Handle tail: load one, set second to -inf
        return __halves2half2(base[idx], __float2half(-CUDART_INF_F));
    }
}

// Kernel 1: Compute per-row max (in float) for each row of fp16 input
__global__ void logsoftmax_rowwise_max_fp16(
    const __half* __restrict__ x,
    float* __restrict__ max_per_row,
    int batch_size,
    int dim
) {
    extern __shared__ float sdata[]; // shared memory

    int row = blockIdx.x;
    if (row >= batch_size) return;

    int tid = threadIdx.x;
    float thread_max = -CUDART_INF_F;

    const __half* x_row = x + row * dim;
    int n_vec = dim / 2;  // number of __half2s
    int n_rem = dim % 2;

    // Vectorized max
    for (int i = tid; i < n_vec; i += blockDim.x) {
        __half2 h2 = load_half2_vec(x_row, 2*i, dim);
        float v0 = __half2float(h2.x);
        float v1 = __half2float(h2.y);
        thread_max = fmaxf(thread_max, fmaxf(v0, v1));
    }
    // Tail (if dim is odd)
    if (tid == 0 && n_rem) {
        float v = __half2float(x_row[dim-1]);
        thread_max = fmaxf(thread_max, v);
    }

    // Reduction in shared mem
    sdata[tid] = thread_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) max_per_row[row] = sdata[0];
}

// Kernel 2: Compute per-row sum(exp(x-rowmax)) (in float)
__global__ void logsoftmax_rowwise_expsum_fp16(
    const __half* __restrict__ x,
    const float* __restrict__ max_per_row,
    float* __restrict__ expsum_per_row,
    int batch_size,
    int dim
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= batch_size) return;

    int tid = threadIdx.x;
    float row_max = max_per_row[row];
    float thread_sum = 0.0f;

    const __half* x_row = x + row * dim;
    int n_vec = dim / 2;
    int n_rem = dim % 2;

    // Vectorized sum
    for (int i = tid; i < n_vec; i += blockDim.x) {
        __half2 h2 = load_half2_vec(x_row, 2*i, dim);
        float v0 = __half2float(h2.x) - row_max;
        float v1 = __half2float(h2.y) - row_max;
        thread_sum += expf(v0) + expf(v1);
    }
    // Tail
    if (tid == 0 && n_rem) {
        float v = __half2float(x_row[dim-1]) - row_max;
        thread_sum += expf(v);
    }

    // Block reduction
    sdata[tid] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) expsum_per_row[row] = sdata[0];
}

// Kernel 3: Write out logsoftmax
__global__ void logsoftmax_rowwise_write_fp16(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    const float* __restrict__ max_per_row,
    const float* __restrict__ expsum_per_row,
    int batch_size,
    int dim
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    float row_max = max_per_row[row];
    float row_logsum = logf(expsum_per_row[row]);

    const __half* x_row = x + row * dim;
    __half* y_row = y + row * dim;
    int tid = threadIdx.x;

    int n_vec = dim / 2;
    int n_rem = dim % 2;

    // Vectorized
    for (int i = tid; i < n_vec; i += blockDim.x) {
        __half2 h2 = load_half2_vec(x_row, 2*i, dim);
        float v0 = __half2float(h2.x) - row_max - row_logsum;
        float v1 = __half2float(h2.y) - row_max - row_logsum;
        __half2 out2 = __halves2half2(__float2half_rn(v0), __float2half_rn(v1));
        *reinterpret_cast<__half2*>(y_row + 2*i) = out2;
    }
    // Tail
    if (tid == 0 && n_rem) {
        float v = __half2float(x_row[dim-1]) - row_max - row_logsum;
        y_row[dim-1] = __float2half_rn(v);
    }
}

// Host launcher
void launch_gpu_implementation(
    void* output,    // (batch_size, dim), fp16
    void* input,     // (batch_size, dim), fp16
    int64_t batch_size,
    int64_t dim,
    int64_t logsoftmax_dim // Must be 1
) {
    if (logsoftmax_dim != 1) {
        printf("Only dim=1 (rowwise) logsoftmax is supported.\n");
        return;
    }

    float* d_max_per_row = nullptr;
    float* d_expsum_per_row = nullptr;
    cudaMalloc(&d_max_per_row, batch_size * sizeof(float));
    cudaMalloc(&d_expsum_per_row, batch_size * sizeof(float));

    int threads = THREADS_PER_BLOCK;
    int blocks = static_cast<int>(batch_size);
    size_t smem_bytes = threads * sizeof(float);

    logsoftmax_rowwise_max_fp16<<<blocks, threads, smem_bytes>>>(
        static_cast<const __half*>(input), d_max_per_row, batch_size, dim);

    logsoftmax_rowwise_expsum_fp16<<<blocks, threads, smem_bytes>>>(
        static_cast<const __half*>(input), d_max_per_row, d_expsum_per_row, batch_size, dim);

    logsoftmax_rowwise_write_fp16<<<blocks, threads>>>(
        static_cast<const __half*>(input),
        static_cast<__half*>(output),
        d_max_per_row, d_expsum_per_row,
        batch_size, dim);

    cudaFree(d_max_per_row);
    cudaFree(d_expsum_per_row);

    cudaDeviceSynchronize();
}
