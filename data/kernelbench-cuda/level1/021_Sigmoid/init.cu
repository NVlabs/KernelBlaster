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
#include <cmath>
#include <cstdint>

// CUDA kernel for fast half-precision Sigmoid activation.
// - Processes data in float for accumulator precision, but IO is fp16.
// - Uses vectorized access (half2) where possible for throughput.
// - Each thread processes multiple elements for better occupancy.

__device__ __forceinline__ float sigmoidf(float x) {
    // Numerically stable sigmoid: 1 / (1 + exp(-x))
    return 1.0f / (1.0f + expf(-x));
}

// Vectorized sigmoid on half2, output half2.
__device__ __forceinline__ half2 sigmoid_half2(half2 h2) {
#if __CUDA_ARCH__ >= 530
    float2 f = __half22float2(h2);
    f.x = sigmoidf(f.x);
    f.y = sigmoidf(f.y);
    return __float22half2_rn(f);
#else
    // Fallback to scalar if half2 is not supported
    half h0 = __low2half(h2);
    half h1 = __high2half(h2);
    float f0 = sigmoidf(__half2float(h0));
    float f1 = sigmoidf(__half2float(h1));
    return __halves2half2(__float2half(f0), __float2half(f1));
#endif
}

__global__ void sigmoid_fp16_kernel(const half* __restrict__ x, half* __restrict__ y, int64_t N) {
    // Vectorized: each thread processes 4 half elements (2 x half2)
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    int64_t N4 = N & (~int64_t(3));
    if (idx < N4) {
        // Use half2 for vectorization
        const half2* xh2 = reinterpret_cast<const half2*>(x + idx);
        half2* yh2 = reinterpret_cast<half2*>(y + idx);

        half2 v0 = xh2[0];
        half2 v1 = xh2[1];
        yh2[0] = sigmoid_half2(v0);
        yh2[1] = sigmoid_half2(v1);
    }
    // Tail processing for remaining elements (up to 3)
    if (threadIdx.x == 0) {
        for (int tail = N4 + blockIdx.x * blockDim.x + threadIdx.x; tail < N; ++tail) {
            float xf = __half2float(x[tail]);
            y[tail] = __float2half(sigmoidf(xf));
        }
    }
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(
    void* output,           // Output tensor (float16, GPU memory)
    void* input,            // Input tensor (float16, GPU memory)
    int64_t batch_size,     // Batch size
    int64_t dim             // Dimension
) {
    const int64_t N = batch_size * dim;
    const int threads_per_block = 256;
    const int vec_elems_per_thread = 4; // Each thread computes 4 elements (2 half2 loads)
    const int elements_per_block = threads_per_block * vec_elems_per_thread;
    int blocks = (N + elements_per_block - 1) / elements_per_block;

    sigmoid_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        N
    );
    cudaDeviceSynchronize();
}
