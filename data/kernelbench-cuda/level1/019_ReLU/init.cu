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
#include <algorithm>

/**
 * Fast ReLU kernel for fp16 input/output.
 * Each thread processes multiple elements for high occupancy and memory throughput.
 * - Input:  [batch_size, dim] (contiguous, row-major), half
 * - Output: [batch_size, dim] (contiguous, row-major), half
 */
__global__ void relu_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t total_elements
) {
    // Process 4 elements per thread using half2 for vectorized memory access
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    if (idx >= total_elements) return;

    // Use half2 vectorization if possible
    half2* input_h2 = (half2*)(input + idx);
    half2* output_h2 = (half2*)(output + idx);

    // For the last few elements, may need to process scalar fallback
    int num_vec = min(2, (int)((total_elements - idx + 1) / 2));

#pragma unroll
    for (int i = 0; i < num_vec; ++i) {
        half2 val = input_h2[i];
        // ReLU: max(val, 0)
#if __CUDA_ARCH__ >= 530
        half2 zero = __float2half2_rn(0.0f);
        half2 relu_val = __hmax2(val, zero);
        output_h2[i] = relu_val;
#else
        // Fallback for older archs: process each half separately
        half lo = __low2half(val);
        half hi = __high2half(val);
        output[idx + 2*i + 0] = __hgt(lo, __float2half(0.0f)) ? lo : __float2half(0.0f);
        output[idx + 2*i + 1] = __hgt(hi, __float2half(0.0f)) ? hi : __float2half(0.0f);
#endif
    }

    // Handle the last element if total_elements is odd
    if ((idx + 2*num_vec) < total_elements) {
        half v = input[idx + 2*num_vec];
        output[idx + 2*num_vec] = __hgt(v, __float2half(0.0f)) ? v : __float2half(0.0f);
    }
}

/**
 * Host launcher for the fast fp16 ReLU kernel.
 * @param output Pointer to device memory for output (half*, shape: [batch_size, dim])
 * @param input  Pointer to device memory for input  (half*, shape: [batch_size, dim])
 * @param batch_size Number of rows (int64_t)
 * @param dim        Number of columns (int64_t)
 */
void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t batch_size,
    int64_t dim
) {
    // Number of elements to process
    int64_t total_elements = batch_size * dim;

    // Kernel configuration
    constexpr int threads_per_block = 256;
    // Each thread processes 4 elements (8 bytes), for coalesced access
    int64_t vec_elements = (total_elements + 3) / 4;
    int blocks = static_cast<int>((vec_elements + threads_per_block - 1) / threads_per_block);

    relu_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        total_elements
    );
    cudaDeviceSynchronize();
}
