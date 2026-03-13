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
#include <cassert>

// CUDA kernel for Softsign activation in fp16
// y = x / (1 + |x|)
// Input:  in  - pointer to input tensor (fp16), shape [batch_size, dim]
// Output: out - pointer to output tensor (fp16), shape [batch_size, dim]
// batch_size: number of rows
// dim: number of columns (elements per row)
__global__ void softsign_fp16_kernel(const half* __restrict__ in, half* __restrict__ out, int64_t batch_size, int64_t dim) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * dim;
    if (idx >= total) return;

    // Read input as fp16
    half xh = in[idx];

    // Convert to fp32 for math (for better accuracy)
    float x = __half2float(xh);
    float y = x / (1.0f + fabsf(x));

    // Convert back to fp16 for output
    out[idx] = __float2half_rn(y);
}

// Host launcher for the CUDA kernel
//   output: pointer to output buffer (void*, assumed fp16)
//   input: pointer to input buffer (void*, assumed fp16)
//   batch_size: number of rows
//   dim: number of columns (elements per row)
void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim) {
    assert(output != nullptr && input != nullptr);
    const int64_t total = batch_size * dim;

    // Use 256 threads per block for high occupancy
    const int threads_per_block = 256;
    const int64_t blocks = (total + threads_per_block - 1) / threads_per_block;

    // Cast to appropriate pointer types
    half* out_ptr = static_cast<half*>(output);
    const half* in_ptr = static_cast<const half*>(input);

    // Launch kernel
    softsign_fp16_kernel<<<blocks, threads_per_block>>>(in_ptr, out_ptr, batch_size, dim);

    // Synchronize to ensure completion before returning
    cudaDeviceSynchronize();
}
