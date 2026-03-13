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
// CUDA kernel for product reduction over a dimension for fp16 tensors.
// Handles arbitrary reduction dimension for 3D input (batch_size, dim1, dim2).
// Accumulation is performed in fp32 for numerical stability.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Utility: ceil division
inline int64_t div_up(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

// Kernel for reduction over dim1 (axis=1): shape [batch_size, dim1, dim2] -> [batch_size, dim2]
template<int REDUCE_DIM>
__global__ void prod_reduce_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
) {
    // Each thread block processes a tile of output elements.
    // We reduce along REDUCE_DIM.

    // Output shape: [batch_size, dim2] if REDUCE_DIM==1
    //               [dim1, dim2] if REDUCE_DIM==0
    //               [batch_size, dim1] if REDUCE_DIM==2

    // For REDUCE_DIM==1 (the most common case), we parallelize over (batch, col)

    int out_row, out_col, reduce_size;
    int stride0, stride1, stride2;

    if (REDUCE_DIM == 0) {
        // Reduce over batch: [batch_size, dim1, dim2] -> [dim1, dim2]
        out_row = blockIdx.y * blockDim.y + threadIdx.y; // dim1
        out_col = blockIdx.x * blockDim.x + threadIdx.x; // dim2
        reduce_size = batch_size;
        stride0 = dim1 * dim2;
        stride1 = dim2;
        stride2 = 1;
    } else if (REDUCE_DIM == 1) {
        // Reduce over dim1: [batch_size, dim1, dim2] -> [batch_size, dim2]
        out_row = blockIdx.y * blockDim.y + threadIdx.y; // batch_size
        out_col = blockIdx.x * blockDim.x + threadIdx.x; // dim2
        reduce_size = dim1;
        stride0 = dim1 * dim2;
        stride1 = dim2;
        stride2 = 1;
    } else {
        // Reduce over dim2: [batch_size, dim1, dim2] -> [batch_size, dim1]
        out_row = blockIdx.y * blockDim.y + threadIdx.y; // batch_size
        out_col = blockIdx.x * blockDim.x + threadIdx.x; // dim1
        reduce_size = dim2;
        stride0 = dim1 * dim2;
        stride1 = dim2;
        stride2 = 1;
    }

    // Compute output bounds
    int max_row, max_col;
    if (REDUCE_DIM == 0) {
        max_row = dim1;
        max_col = dim2;
    } else if (REDUCE_DIM == 1) {
        max_row = batch_size;
        max_col = dim2;
    } else {
        max_row = batch_size;
        max_col = dim1;
    }

    if (out_row >= max_row || out_col >= max_col)
        return;

    float acc = 1.0f;

    if (REDUCE_DIM == 0) {
        // Reduce over batch
        for (int k = 0; k < reduce_size; ++k) {
            int idx = k * stride0 + out_row * stride1 + out_col * stride2;
            acc *= __half2float(input[idx]);
        }
        output[out_row * max_col + out_col] = __float2half(acc);
    } else if (REDUCE_DIM == 1) {
        // Reduce over dim1
        for (int k = 0; k < reduce_size; ++k) {
            int idx = out_row * stride0 + k * stride1 + out_col * stride2;
            acc *= __half2float(input[idx]);
        }
        output[out_row * max_col + out_col] = __float2half(acc);
    } else {
        // Reduce over dim2
        for (int k = 0; k < reduce_size; ++k) {
            int idx = out_row * stride0 + out_col * stride1 + k * stride2;
            acc *= __half2float(input[idx]);
        }
        output[out_row * max_col + out_col] = __float2half(acc);
    }
}

// Warp-level reduction (product) in fp32, for up to 32 values
__inline__ __device__ float warp_allreduce_prod(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val *= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block reduction using shared memory for product, for up to 1024 elements
template <int BLOCK_SIZE>
__device__ float block_reduce_prod(float val) {
    __shared__ float shared[BLOCK_SIZE / 32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_allreduce_prod(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    float block_prod = 1.0f;
    if (threadIdx.x < BLOCK_SIZE / 32) block_prod = shared[threadIdx.x];
    if (wid == 0)
        block_prod = warp_allreduce_prod(block_prod);

    return block_prod;
}

// Optimized kernel for reducing over dim1 (axis 1) using parallel reduction
__global__ void prod_reduce_dim1_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
) {
    // Each block processes one (batch, col) output
    int col = blockIdx.x;
    int batch = blockIdx.y;

    if (col >= dim2 || batch >= batch_size)
        return;

    // Each thread reduces over a chunk of dim1
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float local_prod = 1.0f;
    for (int i = tid; i < dim1; i += num_threads) {
        int idx = batch * dim1 * dim2 + i * dim2 + col;
        local_prod *= __half2float(input[idx]);
    }

    // Block-wide reduction
    __shared__ float sdata[32]; // up to 1024 threads per block
    local_prod = warp_allreduce_prod(local_prod);

    if ((tid & 31) == 0)
        sdata[tid / 32] = local_prod;
    __syncthreads();

    float block_prod = 1.0f;
    if (tid < (blockDim.x / 32))
        block_prod = sdata[tid];

    if (tid < 32) {
        block_prod = warp_allreduce_prod(block_prod);
        if (tid == 0) {
            output[batch * dim2 + col] = __float2half(block_prod);
        }
    }
}

// Kernel dispatcher for arbitrary reduction dimension
void launch_gpu_implementation(
    void* output,           // output tensor (fp16)
    void* input,            // input tensor (fp16)
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2,
    int64_t reduction_dim   // dimension to reduce over
) {
    const half* in = static_cast<const half*>(input);
    half* out = static_cast<half*>(output);

    // For dim1 reduction (axis=1), use optimized reduction kernel
    if (reduction_dim == 1) {
        // Each block reduces one (batch, col)
        dim3 grid(dim2, batch_size);
        int threads = 256;
        prod_reduce_dim1_fp16_kernel<<<grid, threads>>>(in, out, batch_size, dim1, dim2);
        cudaDeviceSynchronize();
        return;
    }

    // For other axes, launch generic kernel
    // Choose block size
    int block_x = 32, block_y = 8;
    int max_row = 0, max_col = 0;
    if (reduction_dim == 0) {
        max_row = dim1;
        max_col = dim2;
    } else if (reduction_dim == 1) {
        max_row = batch_size;
        max_col = dim2;
    } else {
        max_row = batch_size;
        max_col = dim1;
    }
    dim3 grid(div_up(max_col, block_x), div_up(max_row, block_y));
    dim3 block(block_x, block_y);

    if (reduction_dim == 0) {
        prod_reduce_fp16_kernel<0><<<grid, block>>>(in, out, batch_size, dim1, dim2);
    } else if (reduction_dim == 1) {
        prod_reduce_fp16_kernel<1><<<grid, block>>>(in, out, batch_size, dim1, dim2);
    } else {
        prod_reduce_fp16_kernel<2><<<grid, block>>>(in, out, batch_size, dim1, dim2);
    }
    cudaDeviceSynchronize();
}
