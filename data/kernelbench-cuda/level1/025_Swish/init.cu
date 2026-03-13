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
//
// Efficient CUDA Swish kernel for fp16 tensors (Swish: y = x * sigmoid(x)), supporting arbitrary batch_size, dim.
// All I/O in half (fp16), accumulation in float for numerical stability.
// Designed for large dims, memory coalescing, and warp-level efficiency.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include <cstdio>

// --- CUDA Swish kernel (fp16 I/O, fp32 math) ---
// Each thread processes multiple elements for best occupancy.

__device__ __forceinline__ float sigmoidf(float x) {
    // Numerically stable sigmoid for fp32
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ half swish_half(half xh) {
    float x = __half2float(xh);
    float sig = sigmoidf(x);
    float y = x * sig;
    return __float2half_rn(y);
}

__global__ void swish_fp16_kernel(const half* __restrict__ x,
                                  half* __restrict__ y,
                                  int64_t N, int64_t dim) {
    // Flattened 2D tensor (N, dim) into 1D array of N*dim elements
    int64_t total = N * dim;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Process 4 elements per thread for better memory coalescing
    const int VEC = 4;
    int64_t vec_start = tid * VEC;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int64_t idx = vec_start + i;
        if (idx < total) {
            half xh = x[idx];
            y[idx] = swish_half(xh);
        }
    }
}

// --- Host launcher ---
// output: pointer to device memory, half (fp16), shape (batch_size, dim)
// input:  pointer to device memory, half (fp16), shape (batch_size, dim)

void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim) {
    // Kernel launch parameters
    const int threads_per_block = 256;
    const int elements_per_thread = 4;
    int64_t total = batch_size * dim;
    int64_t num_threads = (total + elements_per_thread - 1) / elements_per_thread;
    int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    swish_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        batch_size, dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        assert(false);
    }
    cudaDeviceSynchronize();
}
