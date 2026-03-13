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
#include <stdint.h>
#include <algorithm>

// CUDA kernel for ELU activation on fp16 tensors.
// Applies: y = x                 if x >= 0
//          y = alpha * (exp(x)-1) if x < 0
// All computation is done in fp16 (I/O), but uses float for intermediate values for numerical stability.
__global__ void elu_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    float alpha,
    int64_t n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    // Convert half to float for computation
    float x = __half2float(input[idx]);
    float y;
    if (x >= 0.0f) {
        y = x;
    } else {
        y = alpha * (expf(x) - 1.0f);
    }
    output[idx] = __float2half_rn(y);
}

// Host launcher for the ELU activation CUDA kernel.
// All pointers are device pointers to fp16 data.
// - output: output tensor (fp16, device memory)
// - input: input tensor (fp16, device memory)
// - alpha: ELU alpha parameter (float)
// - batch_size: number of batches (int64_t)
// - dim: feature dimension (int64_t)
void launch_gpu_implementation(
    void* output,
    void* input,
    float alpha,
    int64_t batch_size,
    int64_t dim
) {
    // Total number of elements in the input/output
    int64_t n_elements = batch_size * dim;

    // Use 256 threads per block for good occupancy
    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((n_elements + threads_per_block - 1) / threads_per_block);

    elu_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        alpha,
        n_elements
    );
    cudaDeviceSynchronize();
}
