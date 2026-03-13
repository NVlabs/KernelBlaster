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
// CUDA implementation of HardSigmoid activation for fp16 input/output tensors.
// Applies y = clamp(x * 0.2 + 0.5, 0, 1) elementwise for all elements in input.
// Host launch function: launch_gpu_implementation

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for HardSigmoid activation (fp16 I/O)
__global__ void hardsigmoid_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t numel
) {
    // Each thread processes one element
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // HardSigmoid: y = clamp(x * 0.2 + 0.5, 0, 1)
    // Use float for intermediate computation to avoid fp16 rounding errors
    float x = __half2float(input[idx]);
    float y = x * 0.2f + 0.5f;
    y = fminf(fmaxf(y, 0.0f), 1.0f);

    output[idx] = __float2half_rn(y);
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(
    void* output,     // Output tensor pointer (fp16)
    void* input,      // Input tensor pointer (fp16)
    int64_t batch_size,
    int64_t dim
) {
    // Calculate total number of elements
    int64_t numel = batch_size * dim;

    // Configure launch parameters
    int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    // Cast pointers to half*
    const half* input_fp16 = static_cast<const half*>(input);
    half* output_fp16 = static_cast<half*>(output);

    // Launch kernel
    hardsigmoid_fp16_kernel<<<grid_size, block_size>>>(
        input_fp16,
        output_fp16,
        numel
    );

    // Ensure kernel completion
    cudaDeviceSynchronize();
}
