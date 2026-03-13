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
Implements the following PyTorch code in CUDA:

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float16)

class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x)
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>
#include <cassert>

// CUDA kernel for Softplus activation with half-precision I/O and float32 accumulation
// Softplus(x) = log(1 + exp(x))
__global__ void softplus_fp16_kernel(const half* __restrict__ input,
                                     half* __restrict__ output,
                                     int64_t total_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    half x_h = input[idx];
    float x = __half2float(x_h);

    // Numerically stable softplus computation:
    // softplus(x) = max(0, x) + log1p(exp(-abs(x)))
    float max0x = fmaxf(0.0f, x);
    float z = expf(-fabsf(x));
    float softplus = max0x + log1pf(z);

    output[idx] = __float2half_rn(softplus);
}

// Host launcher for the CUDA kernel
// Arguments:
//   output: device pointer to output tensor (half*)
//   input: device pointer to input tensor (half*)
//   batch_size: number of rows
//   dim: number of columns
void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim) {
    int64_t total_elems = batch_size * dim;
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((total_elems + threads_per_block - 1) / threads_per_block);

    softplus_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        total_elems
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch or execution failed: "
                  << cudaGetErrorString(err) << std::endl;
        assert(false);
    }
}
