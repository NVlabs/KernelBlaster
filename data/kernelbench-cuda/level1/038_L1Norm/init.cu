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
// l1_norm_cuda.cu
// CUDA implementation of L1 normalization along dim=1 for [batch_size, dim] fp16 tensors.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

// Utility: Warp-wide sum for fp32 using shuffle instructions
__inline__ __device__ float warpReduceSum(float val) {
    // For CUDA 9.0+, warpSize is 32
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-wide reduction for fp32; returns the sum for thread 0 in the block
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // max 32 warps per block
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val); // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0)
        val = warpReduceSum(val); // Final reduce within first warp

    return val;
}

// Kernel: Compute L1 norm for each row
// Each block handles one row (batch element)
__global__ void l1_norm_reduce_kernel(
    const half* __restrict__ x,
    float* __restrict__ l1_norm,
    int dim
) {
    // Each block handles one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Pointer to this row
    const half* row_ptr = x + row * dim;

    // Strided loop for this thread
    float sum = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        // Convert to fp32 and accumulate abs
        sum += fabsf(__half2float(row_ptr[i]));
    }

    // Block-wide reduction (fp32 for accum)
    float total = blockReduceSum(sum);

    // Write result for this row (by thread 0)
    if (threadIdx.x == 0) {
        l1_norm[row] = total;
    }
}

// Kernel: Normalize each row by its L1 norm
// Each block handles one row (batch element)
__global__ void l1_norm_apply_kernel(
    half* __restrict__ output,
    const half* __restrict__ x,
    const float* __restrict__ l1_norm,
    int dim
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float norm = l1_norm[row];

    // Avoid division by zero: If norm==0, set output to zero
    bool valid = norm > 1e-8f;
    float inv_norm = valid ? __frcp_rn(norm) : 0.0f;

    const half* row_in = x + row * dim;
    half* row_out = output + row * dim;

    for (int i = tid; i < dim; i += nthreads) {
        float v = __half2float(row_in[i]);
        float r = valid ? v * inv_norm : 0.0f;
        row_out[i] = __float2half_rn(r);
    }
}

// Host launcher for the L1 normalization CUDA implementation
//   output: [batch_size, dim], fp16
//   input:  [batch_size, dim], fp16
//   batch_size, dim: sizes
void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim) {
    half* x = static_cast<half*>(input);
    half* out = static_cast<half*>(output);

    // Temporary buffer for storing L1 norms (fp32 for accuracy)
    float* d_l1_norm = nullptr;
    cudaMalloc(&d_l1_norm, batch_size * sizeof(float));

    // Kernel config
    int threads = 256;
    int blocks = batch_size;

    // 1. Compute L1 norm for each row
    l1_norm_reduce_kernel<<<blocks, threads>>>(x, d_l1_norm, static_cast<int>(dim));

    // 2. Normalize each row
    l1_norm_apply_kernel<<<blocks, threads>>>(out, x, d_l1_norm, static_cast<int>(dim));

    cudaFree(d_l1_norm);
    cudaDeviceSynchronize();
}
