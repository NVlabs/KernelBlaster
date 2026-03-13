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
#include <cassert>
#include <cstdio>

/*
Implements torch.cumprod(x, dim) for fp16 tensors.
Input:
    - input: (batch_size, input_size) for dim=1, or (input_size, batch_size) for dim=0
    - output: same shape, fp16
    - dim: 0 or 1 (other dims not supported)
    - batch_size: int64_t
    - input_size: int64_t

Numerical stability: accumulation is done in FP32, but I/O is FP16.
*/

__global__ void cumprod_dim1_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t batch_size,
    int64_t input_size
) {
    // Each block processes one row (batch)
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // All threads in block work on the row, each thread processes multiple elements
    for (int col = threadIdx.x; col < input_size; col += blockDim.x) {
        // Compute cumulative product up to 'col' in row 'row'
        float acc = 1.0f;
        for (int k = 0; k <= col; ++k) {
            acc *= __half2float(input[row * input_size + k]);
        }
        output[row * input_size + col] = __float2half_rn(acc);
    }
}

__global__ void cumprod_dim0_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t batch_size,
    int64_t input_size
) {
    // Each block processes one column (input_size)
    int col = blockIdx.x;
    if (col >= input_size) return;

    // All threads in block work on the column, each thread processes multiple rows
    for (int row = threadIdx.x; row < batch_size; row += blockDim.x) {
        float acc = 1.0f;
        for (int k = 0; k <= row; ++k) {
            acc *= __half2float(input[k * input_size + col]);
        }
        output[row * input_size + col] = __float2half_rn(acc);
    }
}

void launch_gpu_implementation(
    void* output,
    void* input,
    int dim,
    int64_t batch_size,
    int64_t input_size
) {
    assert(dim == 0 || dim == 1 && "Only dim=0 or dim=1 are supported for 2D input");

    const int threads_per_block = 256;

    if (dim == 1) {
        // Launch one block per row
        int blocks = static_cast<int>(batch_size);
        cumprod_dim1_fp16_kernel<<<blocks, threads_per_block>>>(
            static_cast<const half*>(input),
            static_cast<half*>(output),
            batch_size,
            input_size
        );
    } else if (dim == 0) {
        // Launch one block per column
        int blocks = static_cast<int>(input_size);
        cumprod_dim0_fp16_kernel<<<blocks, threads_per_block>>>(
            static_cast<const half*>(input),
            static_cast<half*>(output),
            batch_size,
            input_size
        );
    }
    cudaDeviceSynchronize();
}
