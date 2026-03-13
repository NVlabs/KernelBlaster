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
#include <stdint.h>
#include <stdio.h>

// MaxPool1d CUDA kernel for (N, C, L) layout, with fp16 I/O, fp32 accumulation
// Parameters: kernel_size, stride, padding, dilation
// No indices output (return_indices == false)
__global__ void maxpool1d_fp16_kernel(
    const half* __restrict__ input,   // [N, C, L]
    half* __restrict__ output,        // [N, C, L_out]
    int64_t N,
    int64_t C,
    int64_t L,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t L_out
) {
    // Each thread computes one output element (n, c, l_out)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * L_out;
    if (tid >= total) return;

    // Calculate (n, c, l_out) from tid
    int l_out = tid % L_out;
    int c = (tid / L_out) % C;
    int n = tid / (C * L_out);

    // Compute pooling window start/end in input coordinates
    int start = l_out * stride - padding;
    int max_val_idx = -1;
    float max_val = -65504.0f; // Smallest fp16 value

    // Visit kernel window
#pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int l_in = start + k * dilation;
        if (l_in >= 0 && l_in < L) {
            // Compute input flat index
            int idx = n * (C * L) + c * L + l_in;
            float val = __half2float(input[idx]);
            if (val > max_val || max_val_idx == -1) {
                max_val = val;
                max_val_idx = l_in;
            }
        }
    }

    // Write output
    int out_idx = n * (C * L_out) + c * L_out + l_out;
    output[out_idx] = __float2half_rn(max_val);
}

// Host function to launch the CUDA MaxPool1d kernel
void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t batch_size,
    int64_t features,
    int64_t sequence_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices
) {
    // Only support return_indices == false
    if (return_indices) {
        printf("Error: return_indices=true not supported in this implementation.\n");
        return;
    }

    // Output length calculation as per PyTorch's formula
    // L_out = floor((L + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    int64_t L = sequence_length;
    int64_t L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    const int threadsPerBlock = 256;
    const int total = batch_size * features * L_out;
    const int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    maxpool1d_fp16_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        batch_size, features, sequence_length,
        kernel_size, stride, padding, dilation, L_out
    );
    cudaDeviceSynchronize();
}
