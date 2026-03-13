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
#include <iostream>
#include <stdint.h>

// CUDA kernel for HardTanh activation (fp16, with fp16 I/O)
__global__ void hardtanh_fp16_kernel(
    const half* __restrict__ x,
    half* __restrict__ y,
    int64_t numel,
    half min_val,
    half max_val
) {
    // Each thread processes one element
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    half val = x[idx];
    // Clamp in fp16
    if (__hlt(val, min_val)) val = min_val;
    if (__hgt(val, max_val)) val = max_val;
    y[idx] = val;
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim) {
    // Number of elements in the tensor
    int64_t numel = batch_size * dim;
    const int threads_per_block = 256;
    const int blocks = (numel + threads_per_block - 1) / threads_per_block;

    // Use fp16 min/max (-1, 1)
    half min_val = __float2half(-1.0f);
    half max_val = __float2half(1.0f);

    // Launch kernel
    hardtanh_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        numel,
        min_val,
        max_val
    );

    // Ensure kernel completion
    cudaDeviceSynchronize();
}

