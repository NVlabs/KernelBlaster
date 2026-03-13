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
// Efficient CUDA Argmin kernel for fp16 input, int64 output, supporting arbitrary dim (0,1,2).
// Handles shapes: (batch_size, dim1, dim2), dim in {0,1,2}. Input: at::Half*, Output: int64_t*.
// Accumulation is always in float32 for numerical stability.
// Designed for: batch_size=16, dim1=256, dim2=256, dim=1 (but works for all dims).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits>
#include <cassert>

// Utility: Loads fp16 as float
__device__ __forceinline__ float load_fp16(const __half* ptr, int idx) {
    return __half2float(ptr[idx]);
}

// Utility: Returns max float for initializing min search
__device__ __forceinline__ float kBigFloat() {
    return 65504.0f; // max representable half, safe for min search
}

// Kernel for argmin along dim=0
__global__ void argmin_dim0_kernel(
    int64_t* __restrict__ output,    // output: (dim1, dim2) int64_t
    const __half* __restrict__ input,// input: (batch, dim1, dim2) fp16
    int64_t batch_size, int64_t dim1, int64_t dim2)
{
    // Each thread processes one (i1, i2) in output
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim1 * dim2;
    if (tid >= total) return;

    int i1 = tid / dim2;
    int i2 = tid % dim2;

    float min_val = kBigFloat();
    int64_t min_idx = 0;

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t idx = b * dim1 * dim2 + i1 * dim2 + i2;
        float val = __half2float(input[idx]);
        if (val < min_val || (val == min_val && b < min_idx)) {
            min_val = val;
            min_idx = b;
        }
    }
    output[i1 * dim2 + i2] = min_idx;
}

// Kernel for argmin along dim=1
__global__ void argmin_dim1_kernel(
    int64_t* __restrict__ output,    // output: (batch, dim2) int64_t
    const __half* __restrict__ input,// input: (batch, dim1, dim2) fp16
    int64_t batch_size, int64_t dim1, int64_t dim2)
{
    // Each thread processes one (b, i2) in output
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * dim2;
    if (tid >= total) return;

    int b = tid / dim2;
    int i2 = tid % dim2;

    float min_val = kBigFloat();
    int64_t min_idx = 0;

    for (int64_t i1 = 0; i1 < dim1; ++i1) {
        int64_t idx = b * dim1 * dim2 + i1 * dim2 + i2;
        float val = __half2float(input[idx]);
        if (val < min_val || (val == min_val && i1 < min_idx)) {
            min_val = val;
            min_idx = i1;
        }
    }
    output[b * dim2 + i2] = min_idx;
}

// Kernel for argmin along dim=2
__global__ void argmin_dim2_kernel(
    int64_t* __restrict__ output,    // output: (batch, dim1) int64_t
    const __half* __restrict__ input,// input: (batch, dim1, dim2) fp16
    int64_t batch_size, int64_t dim1, int64_t dim2)
{
    // Each thread processes one (b, i1) in output
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * dim1;
    if (tid >= total) return;

    int b = tid / dim1;
    int i1 = tid % dim1;

    float min_val = kBigFloat();
    int64_t min_idx = 0;

    for (int64_t i2 = 0; i2 < dim2; ++i2) {
        int64_t idx = b * dim1 * dim2 + i1 * dim2 + i2;
        float val = __half2float(input[idx]);
        if (val < min_val || (val == min_val && i2 < min_idx)) {
            min_val = val;
            min_idx = i2;
        }
    }
    output[b * dim1 + i1] = min_idx;
}

void launch_gpu_implementation(
    void* output,          // int64_t*
    void* input,           // at::Half* (== __half*)
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2,
    int64_t dim
) {
    // Output shape and launch config
    int threads = 256;
    int blocks = 1;
    if (dim == 0) {
        int total = dim1 * dim2;
        blocks = (total + threads - 1) / threads;
        argmin_dim0_kernel<<<blocks, threads>>>(
            static_cast<int64_t*>(output),
            static_cast<const __half*>(input),
            batch_size, dim1, dim2
        );
    } else if (dim == 1) {
        int total = batch_size * dim2;
        blocks = (total + threads - 1) / threads;
        argmin_dim1_kernel<<<blocks, threads>>>(
            static_cast<int64_t*>(output),
            static_cast<const __half*>(input),
            batch_size, dim1, dim2
        );
    } else if (dim == 2) {
        int total = batch_size * dim1;
        blocks = (total + threads - 1) / threads;
        argmin_dim2_kernel<<<blocks, threads>>>(
            static_cast<int64_t*>(output),
            static_cast<const __half*>(input),
            batch_size, dim1, dim2
        );
    } else {
        // Invalid dim
        assert(0 && "Invalid dim for argmin");
    }
    cudaDeviceSynchronize();
}
