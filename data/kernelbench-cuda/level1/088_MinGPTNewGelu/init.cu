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
#include <math_constants.h>
#include <cmath>
#include <stdint.h>

// GELU (OpenAI GPT/BERT flavor): 
// y = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Input/output: half (fp16), accumulation in float

// CUDA: Efficient thread-parallel implementation, with vectorization for fp16 loads/stores.

__device__ __forceinline__ float gelu_openai_fp32(float x) {
    // Constants for the OpenAI GELU
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = x + coeff * x3;
    float tanh_out = tanhf(sqrt_2_over_pi * inner);
    return 0.5f * x * (1.0f + tanh_out);
}

// Vectorized version for half2 (2 x fp16)
__device__ __forceinline__ half2 gelu_openai_half2(half2 x) {
    float2 x_f = __half22float2(x);
    float2 x3;
    x3.x = x_f.x * x_f.x * x_f.x;
    x3.y = x_f.y * x_f.y * x_f.y;
    float2 inner;
    inner.x = x_f.x + 0.044715f * x3.x;
    inner.y = x_f.y + 0.044715f * x3.y;
    float2 tanh_out;
    tanh_out.x = tanhf(0.7978845608028654f * inner.x);
    tanh_out.y = tanhf(0.7978845608028654f * inner.y);
    float2 out;
    out.x = 0.5f * x_f.x * (1.0f + tanh_out.x);
    out.y = 0.5f * x_f.y * (1.0f + tanh_out.y);
    return __floats2half2_rn(out.x, out.y);
}

__global__ void gelu_openai_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t numel
) {
    // Vectorize: 2 elements per thread using half2 if possible
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    int last = numel & ~1;  // last even index

    // Main path: process two elements at a time
    if (idx2 < last) {
        half2 x = reinterpret_cast<const half2*>(input)[idx];
        half2 y = gelu_openai_half2(x);
        reinterpret_cast<half2*>(output)[idx] = y;
    }

    // Epilogue: handle last odd element (if numel is odd)
    if ((idx2 + 1 == numel) && (numel & 1)) {
        float x = __half2float(input[numel-1]);
        float y = gelu_openai_fp32(x);
        output[numel-1] = __float2half_rn(y);
    }
}

// Host function to launch the CUDA kernel
void launch_gpu_implementation(
    void* output,      // [batch_size, dim] half (fp16)
    void* input,       // [batch_size, dim] half (fp16)
    int64_t batch_size,
    int64_t dim
) {
    int64_t numel = batch_size * dim;
    // Vectorize: 2 elements per thread (half2)
    int threads = 256;
    int blocks = ((numel + 1) / 2 + threads - 1) / threads;

    gelu_openai_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        numel
    );
    cudaDeviceSynchronize();
}
