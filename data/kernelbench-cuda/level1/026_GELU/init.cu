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
// Fast GELU activation for fp16 (half) input/output tensors on CUDA
// I/O: half* input, half* output, shape [batch_size, dim]
// Accumulation is always in float (fp32) for numerical stability

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <assert.h>

// GELU implementation in CUDA for half tensors, using fp32 accumulation
// Reference: torch.nn.functional.gelu (approximate: "tanh" version)
//
// gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ) )
// (see also: https://arxiv.org/pdf/1606.08415.pdf)
//
// This implementation is vectorized (float4/half2 when possible) for performance.

__device__ __forceinline__ float gelu_tanh(float x) {
    // Constants
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_inner = tanhf(inner);
    float result = 0.5f * x * (1.0f + tanh_inner);
    return result;
}

// Vectorized GELU for half2 (two halfs at a time)
__device__ __forceinline__ half2 gelu_tanh_half2(half2 xh2) {
    float2 x = __half22float2(xh2);
    float2 r;
    r.x = gelu_tanh(x.x);
    r.y = gelu_tanh(x.y);
    return __float22half2_rn(r);
}

// Kernel: Vectorized GELU (half2), fallback to scalar for odd elements
__global__ void gelu_fp16_kernel(
    half* __restrict__ out,
    const half* __restrict__ inp,
    int64_t total_elems
) {
    // Vectorize: process 2 elements per thread if possible
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < total_elems) {
        // Safe to read/write 2 elements
        half2 inp_h2 = reinterpret_cast<const half2*>(inp)[idx / 2];
        half2 out_h2 = gelu_tanh_half2(inp_h2);
        reinterpret_cast<half2*>(out)[idx / 2] = out_h2;
    } else if (idx < total_elems) {
        // Handle last element if odd
        float x = __half2float(inp[idx]);
        float y = gelu_tanh(x);
        out[idx] = __float2half_rn(y);
    }
}

// Host launcher: C-compatible
extern "C"
void launch_gpu_implementation(
    void* output,         // half* output, shape: [batch_size, dim]
    void* input,          // half* input, shape: [batch_size, dim]
    int64_t batch_size,
    int64_t dim
) {
    half* out = static_cast<half*>(output);
    const half* inp = static_cast<const half*>(input);
    int64_t N = batch_size * dim;
    // Tune block size for best performance on L40S
    constexpr int block = 256;
    int grid = (N + 2*block - 1) / (2*block); // 2 elements per thread
    gelu_fp16_kernel<<<grid, block>>>(out, inp, N);
    cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);
    cudaDeviceSynchronize();
}
