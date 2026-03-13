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
#include <cstdint>
#include <cstdio>
#include <algorithm>

// Utility for CUDA error checking
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(err));                               \
            return;                                                        \
        }                                                                  \
    } while (0)

// Warp-level min reduction for fp16
__inline__ __device__ half warp_min_fp16(half val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = __hlt(val, __shfl_down_sync(0xFFFFFFFF, val, offset)) ? val : __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level min reduction for fp16 using shared memory
template <unsigned int blockSize>
__inline__ __device__ half block_min_fp16(half val) {
    static __shared__ half shared[32]; // Enough for 1024 threads
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp reduce
    val = warp_min_fp16(val);

    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read reduced values from shared memory and reduce again
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : __float2half(65504.0f);
    if (wid == 0) {
        val = warp_min_fp16(val);
    }
    return val;
}

// Kernel for min reduction over an arbitrary dimension (dim = 0, 1, or 2)
__global__ void min_reduce_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t reduce_dim,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
) {
    // Determine output dimensions and reduction range
    int64_t out_dim0 = (reduce_dim == 0) ? dim1 : batch_size;
    int64_t out_dim1 = (reduce_dim == 2) ? dim1 : dim2;
    int64_t reduce_size = (reduce_dim == 0) ? batch_size : (reduce_dim == 1) ? dim1 : dim2;

    int64_t total_outputs = out_dim0 * out_dim1;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = out_idx; idx < total_outputs; idx += gridDim.x * blockDim.x) {
        // Map idx to output indices
        int i0 = idx / out_dim1;
        int i1 = idx % out_dim1;
        float minval = 65504.0f; // max value for half

        // Reduce along the specified dimension
        for (int r = 0; r < reduce_size; ++r) {
            int input_idx;
            if (reduce_dim == 0) {
                // Reduce over batch: [r, i0, i1]
                input_idx = r * dim1 * dim2 + i0 * dim2 + i1;
            } else if (reduce_dim == 1) {
                // Reduce over dim1: [i0, r, i1]
                input_idx = i0 * dim1 * dim2 + r * dim2 + i1;
            } else {
                // Reduce over dim2: [i0, i1, r]
                input_idx = i0 * dim1 * dim2 + i1 * dim2 + r;
            }
            float v = __half2float(input[input_idx]);
            minval = (v < minval) ? v : minval;
        }

        // Write result as half
        output[idx] = __float2half(minval);
    }
}

// Fast kernel: each output row/col is reduced in parallel using shared memory and block reduction
// Supports dim=1 or dim=2 efficiently (row or col reduction). dim=0 (batch) is also supported.
template <int REDUCE_DIM>
__global__ void min_reduce_fp16_block_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
) {
    // REDUCE_DIM: 0=batch, 1=dim1, 2=dim2
    // For dim=1 (reduce over dim1): output shape = [batch_size, dim2]
    // For dim=2 (reduce over dim2): output shape = [batch_size, dim1]
    // For dim=0 (reduce over batch): output shape = [dim1, dim2]

    // Set up output grid
    int out_i, out_j, reduce_size;
    if (REDUCE_DIM == 1) {
        // Reduce over dim1, output [batch_size, dim2]
        out_i = blockIdx.y; // batch idx
        out_j = blockIdx.x * blockDim.x + threadIdx.x;
        reduce_size = dim1;
        if (out_j >= dim2) return;
    } else if (REDUCE_DIM == 2) {
        // Reduce over dim2, output [batch_size, dim1]
        out_i = blockIdx.y; // batch idx
        out_j = blockIdx.x * blockDim.x + threadIdx.x;
        reduce_size = dim2;
        if (out_j >= dim1) return;
    } else {
        // Reduce over batch, output [dim1, dim2]
        out_i = blockIdx.y; // dim1 idx
        out_j = blockIdx.x * blockDim.x + threadIdx.x;
        reduce_size = batch_size;
        if (out_j >= dim2) return;
    }

    float minval = 65504.0f;
    if (REDUCE_DIM == 1) {
        // Reduce over dim1: for fixed [out_i, out_j], min over k in [0,dim1)
        for (int k = 0; k < dim1; ++k) {
            int idx = out_i * dim1 * dim2 + k * dim2 + out_j;
            float v = __half2float(input[idx]);
            minval = (v < minval) ? v : minval;
        }
        output[out_i * dim2 + out_j] = __float2half(minval);
    } else if (REDUCE_DIM == 2) {
        // Reduce over dim2: for fixed [out_i, out_j], min over k in [0,dim2)
        for (int k = 0; k < dim2; ++k) {
            int idx = out_i * dim1 * dim2 + out_j * dim2 + k;
            float v = __half2float(input[idx]);
            minval = (v < minval) ? v : minval;
        }
        output[out_i * dim1 + out_j] = __float2half(minval);
    } else {
        // Reduce over batch: for fixed [out_i, out_j], min over k in [0,batch_size)
        for (int k = 0; k < batch_size; ++k) {
            int idx = k * dim1 * dim2 + out_i * dim2 + out_j;
            float v = __half2float(input[idx]);
            minval = (v < minval) ? v : minval;
        }
        output[out_i * dim2 + out_j] = __float2half(minval);
    }
}

// Host function to launch the kernel
void launch_gpu_implementation(
    void* output_,
    void* input_,
    int64_t dim,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
) {
    half* output = static_cast<half*>(output_);
    const half* input = static_cast<const half*>(input_);

    // Tune block size for best occupancy and memory coalescing
    const int block_size = 256;
    cudaError_t err;

    if (dim == 1) {
        // Reduce over dim1: output [batch_size, dim2]
        dim3 grid((dim2 + block_size - 1) / block_size, batch_size);
        min_reduce_fp16_block_kernel<1><<<grid, block_size>>>(input, output, batch_size, dim1, dim2);
    } else if (dim == 2) {
        // Reduce over dim2: output [batch_size, dim1]
        dim3 grid((dim1 + block_size - 1) / block_size, batch_size);
        min_reduce_fp16_block_kernel<2><<<grid, block_size>>>(input, output, batch_size, dim1, dim2);
    } else if (dim == 0) {
        // Reduce over batch: output [dim1, dim2]
        dim3 grid((dim2 + block_size - 1) / block_size, dim1);
        min_reduce_fp16_block_kernel<0><<<grid, block_size>>>(input, output, batch_size, dim1, dim2);
    } else {
        // Fallback (should not happen for 3D input)
        int64_t out_dim0 = (dim == 0) ? dim1 : batch_size;
        int64_t out_dim1 = (dim == 2) ? dim1 : dim2;
        int64_t total_outputs = out_dim0 * out_dim1;
        int grid_size = (total_outputs + block_size - 1) / block_size;
        min_reduce_fp16_kernel<<<grid_size, block_size>>>(
            input, output, dim, batch_size, dim1, dim2
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
