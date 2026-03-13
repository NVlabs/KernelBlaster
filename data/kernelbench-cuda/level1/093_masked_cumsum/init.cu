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
#include <stdint.h>
#include <assert.h>

// For large rows, use 256 threads per block for occupancy and memory coalescing
constexpr int THREADS_PER_BLOCK = 256;

// Masked cumsum kernel: one block per row (dim == 1)
__global__ void masked_cumsum_fp16_kernel(
    half* __restrict__ output,
    const half* __restrict__ x,
    const bool* __restrict__ mask,
    int64_t batch_size,
    int64_t input_size
) {
    extern __shared__ float shared[]; // for block-wide scan

    int row = blockIdx.x;
    if (row >= batch_size) return;
    int tid = threadIdx.x;

    // Each thread processes a strided segment of the row.
    int n_threads = blockDim.x;
    int stride = n_threads;
    int seg_len = (input_size + n_threads - 1) / n_threads;

    // 1. Local scan for each thread's portion
    float local_sum = 0.0f;
    int base = row * input_size;

    // Store partial results for output
    int start = tid * seg_len;
    int end = min(start + seg_len, (int)input_size);

    // Temporary storage for local scan
    float* local_scan = shared + n_threads + tid * seg_len; // [THREADS_PER_BLOCK][seg_len] after partials

    for (int i = start, j = 0; i < end; ++i, ++j) {
        float v = 0.0f;
        if (mask[base + i]) v = __half2float(x[base + i]);
        local_sum += v;
        local_scan[j] = local_sum;
    }

    // 2. Save each thread's last scan value
    float thread_sum = local_sum;
    shared[tid] = (end > start) ? thread_sum : 0.0f;
    __syncthreads();

    // 3. Block-wide exclusive scan over thread sums to get the offset for each thread
    float thread_offset = 0.0f;
    for (int offset = 1; offset < n_threads; offset <<= 1) {
        float val = 0.0f;
        if (tid >= offset) val = shared[tid - offset];
        __syncthreads();
        shared[tid] += val;
        __syncthreads();
    }
    if (tid > 0) thread_offset = shared[tid - 1];
    __syncthreads();

    // 4. Write out the results, adding thread_offset to each local scan value
    for (int i = start, j = 0; i < end; ++i, ++j) {
        float out_val = local_scan[j] + thread_offset;
        output[base + i] = __float2half_rn(out_val);
    }
}

// Host launcher
void launch_gpu_implementation(
    void* output,
    void* x,
    void* mask,
    int64_t dim,
    int64_t batch_size,
    int64_t input_size
) {
    assert(dim == 1 && "Only dim==1 (row-wise cumsum) is supported.");

    int threads = THREADS_PER_BLOCK;
    int seg_len = (input_size + threads - 1) / threads;
    size_t smem = threads * sizeof(float) + threads * seg_len * sizeof(float);

    masked_cumsum_fp16_kernel<<<batch_size, threads, smem>>>(
        static_cast<half*>(output),
        static_cast<const half*>(x),
        static_cast<const bool*>(mask),
        batch_size,
        input_size
    );
    cudaDeviceSynchronize();
}
