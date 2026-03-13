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
// CUDA kernel for fast sum reduction over a specified dimension for fp16 tensors.
//
// Implements: output = sum(input, dim=reduce_dim, keepdim=True)
// Input:  x [batch_size, dim1, dim2] (fp16)
// Output: y [batch_size, 1, dim2] (fp16), if reduce_dim==1
//
// This implementation is optimized for dim1 reduction (reduce_dim == 1) using shared memory and warp-level reductions in fp32 for accuracy.
// Accumulation is always performed in fp32 for stability, output is written in fp16.
// Supports batch processing (batch_size > 1).
//
// This kernel is specialized for reduce_dim == 1, as in the test case. For other dims, a generic kernel can be added.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include <cstdio>

// Utility: CUDA error check
#define CUDA_CHECK(stmt) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        assert(err == cudaSuccess); \
    } \
} while(0)

// Warp-level reduction (sum) in fp32
__inline__ __device__ float warp_reduce_sum(float val) {
    // Use warp shuffle for reduction
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction (sum) in fp32 using shared memory
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    // Each warp does partial reduction
    val = warp_reduce_sum(val);

    // Write reduced value to shared memory
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    // The first warp reduces all values
    float sum = 0.0f;
    if (wid == 0) {
        sum = (lane < blockDim.x / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    return sum;
}

// Kernel: sum reduction along dim1 (axis=1) for shape [batch_size, dim1, dim2]
// Each block processes one (batch, col) pair, reducing over dim1.
__global__ void reduce_sum_dim1_fp16_kernel(
    const half* __restrict__ input, // [batch_size, dim1, dim2]
    half* output,                   // [batch_size, 1, dim2]
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
) {
    // Each block computes one output element: (batch_idx, 0, col_idx)
    // 2D grid: (col_idx, batch_idx)
    int col_idx   = blockIdx.x;
    int batch_idx = blockIdx.y;

    if (col_idx >= dim2 || batch_idx >= batch_size) return;

    // Compute offset to start of reduction for this output
    const half* in_ptr = input + batch_idx * dim1 * dim2 + col_idx;
    // Output pointer (keepdim=1)
    half* out_ptr = output + batch_idx * dim2 + col_idx;

    // Each thread accumulates a partial sum over its strided elements in dim1
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim1; i += blockDim.x) {
        half v = in_ptr[i * dim2];
        sum += __half2float(v);
    }

    // Shared mem for block reduction (one float per warp)
    __shared__ float shared[32]; // up to 1024 threads (32 warps)

    float total = block_reduce_sum(sum, shared);

    // Write final result by thread 0
    if (threadIdx.x == 0) {
        *out_ptr = __float2half_rn(total);
    }
}

// Host launcher
void launch_gpu_implementation(
    void* output,        // Output tensor data pointer (fp16)
    const void* input,   // Input tensor data pointer (fp16)
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2,
    int64_t reduce_dim   // reduction dimension (in this case, 1)
) {
    // Only support reduce_dim==1 (sum over dim1) for this optimized kernel
    if (reduce_dim != 1) {
        printf("Error: Only reduce_dim == 1 is supported in this optimized kernel.\n");
        assert(reduce_dim == 1);
        return;
    }

    // Grid: one block per (col, batch)
    dim3 grid(dim2, batch_size);
    // Use 256 threads per block for good occupancy
    int threads = 256;

    reduce_sum_dim1_fp16_kernel<<<grid, threads>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        batch_size, dim1, dim2
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

