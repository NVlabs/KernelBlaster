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
/*
 * Fast CUDA kernel for SELU activation on fp16 tensors.
 * Input:  (batch_size, dim) tensor of half (fp16)
 * Output: (batch_size, dim) tensor of half (fp16)
 * 
 * SELU: y = scale * (x if x > 0 else alpha * (exp(x) - 1))
 * where:
 *   scale = 1.0507009873554805f
 *   alpha = 1.6732632423543772f
 * 
 * Accumulation and math are performed in fp32 for numerical stability,
 * I/O tensors are fp16.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cassert>

// SELU constants as in PyTorch
constexpr float SELU_SCALE = 1.0507009873554805f;
constexpr float SELU_ALPHA = 1.6732632423543772f;

__device__ __forceinline__ float selu_fp32(float x) {
    // SELU activation function in fp32
    return SELU_SCALE * (x > 0.0f ? x : SELU_ALPHA * (expf(x) - 1.0f));
}

__global__ void selu_fp16_kernel(const half* __restrict__ input, half* __restrict__ output, int64_t total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // Read input as fp16, convert to fp32
        float x = __half2float(input[idx]);
        float y = selu_fp32(x);
        output[idx] = __float2half_rn(y);
    }
}

// Host function to launch the kernel
// output: pointer to device memory for output (half*)
// input:  pointer to device memory for input (half*)
// batch_size: number of rows
// dim: number of columns per row
void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim) {
    assert(output != nullptr && input != nullptr);
    int64_t total_elements = batch_size * dim;
    constexpr int threads_per_block = 256;
    int grid = (total_elements + threads_per_block - 1) / threads_per_block;
    selu_fp16_kernel<<<grid, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        total_elements
    );
    cudaDeviceSynchronize();
}
