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
#include <stdint.h>
#include <assert.h>

// Fast LeakyReLU kernel for fp16 tensors.
// Each thread processes multiple elements for better memory throughput.

__global__ void leaky_relu_fp16_kernel(
    half* __restrict__ out,
    const half* __restrict__ in,
    float negative_slope,
    int64_t total_elems
) {
    // Use float for accumulation and computation for best accuracy (even on fp16 tensors)
    const int vec_size = 4; // Vectorized loads/stores for throughput
    int idx = blockIdx.x * blockDim.x * vec_size + threadIdx.x * vec_size;

    // Use vectorized access where possible
    if (idx + vec_size - 1 < total_elems) {
        // Load 4 elements at once
        half2* in_h2_ptr = (half2*)(in + idx);
        half2* out_h2_ptr = (half2*)(out + idx);

        half2 x0 = in_h2_ptr[0];
        half2 x1 = in_h2_ptr[1];

        // Convert to float2 for computation
        float2 fx0 = __half22float2(x0);
        float2 fx1 = __half22float2(x1);

        float2 fy0, fy1;
        // LeakyReLU: y = x if x >= 0 else x * negative_slope
        fy0.x = fx0.x >= 0.f ? fx0.x : fx0.x * negative_slope;
        fy0.y = fx0.y >= 0.f ? fx0.y : fx0.y * negative_slope;
        fy1.x = fx1.x >= 0.f ? fx1.x : fx1.x * negative_slope;
        fy1.y = fx1.y >= 0.f ? fx1.y : fx1.y * negative_slope;

        // Convert float2 back to half2
        out_h2_ptr[0] = __float22half2_rn(fy0);
        out_h2_ptr[1] = __float22half2_rn(fy1);
    }

    // Handle remaining elements (tail)
    int base = (total_elems / (vec_size * blockDim.x * gridDim.x)) * (vec_size * blockDim.x * gridDim.x);
    for (int i = idx; i < total_elems && i < base + blockDim.x * gridDim.x * vec_size; ++i) {
        float x = __half2float(in[i]);
        float y = x >= 0.f ? x : x * negative_slope;
        out[i] = __float2half_rn(y);
    }
}

// Host launcher
void launch_gpu_implementation(
    void* output,
    void* input,
    float negative_slope,
    int64_t batch_size,
    int64_t dim
) {
    // All tensors are fp16 (half)
    half* out = static_cast<half*>(output);
    const half* in = static_cast<const half*>(input);

    int64_t total_elems = batch_size * dim;

    // Tune for best throughput; vectorized loads/stores
    int block_size = 256;
    int elems_per_thread = 4;
    int threads = block_size;
    int blocks = (total_elems + (block_size * elems_per_thread - 1)) / (block_size * elems_per_thread);

    leaky_relu_fp16_kernel<<<blocks, threads>>>(
        out, in, negative_slope, total_elems
    );

    cudaDeviceSynchronize();
}

