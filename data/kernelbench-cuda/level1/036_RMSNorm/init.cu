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
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <stdint.h>

// RMSNorm kernel for (batch_size, num_features, dim1, dim2) layout, fp16 I/O, fp32 accumulation
// Each block processes one sample (batch, d1, d2) tuple across all features (channels)
// Optimized for memory coalescing, shared memory, and warp-wide reduction

__global__ void rmsnorm_forward_fp16(
    half* __restrict__ output,            // [B, F, D1, D2]
    const half* __restrict__ input,       // [B, F, D1, D2]
    int64_t batch_size,
    int64_t num_features,
    int64_t dim1,
    int64_t dim2,
    float eps
) {
    // Each block computes one (b, d1, d2) instance
    int d1 = blockIdx.y;
    int d2 = blockIdx.z;
    int b  = blockIdx.x;

    // Bounds check (should not be needed if launch config is correct)
    if (b >= batch_size || d1 >= dim1 || d2 >= dim2) return;

    // Compute input/output offset for this (b, d1, d2) "column"
    int64_t offset = ((b * num_features * dim1 + 0) * dim2 + d1 * dim2 + d2) - d1 * dim2 - d2;
    // offset = b * num_features * dim1 * dim2 + d1 * dim2 + d2

    // Shared memory for partial sums (blockDim.x <= 256)
    extern __shared__ float shmem[];
    float local_sum = 0.0f;

    // Loop over features in a strided way
    for (int f = threadIdx.x; f < num_features; f += blockDim.x) {
        int64_t idx = ((b * num_features + f) * dim1 + d1) * dim2 + d2;
        half hval = input[idx];
        float fval = __half2float(hval);
        local_sum += fval * fval;
    }

    // Reduction across block (sum of squares)
    shmem[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce within block (tree reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float sq_sum;
    if (threadIdx.x == 0) {
        sq_sum = shmem[0];
        // Compute mean
        float mean_sq = sq_sum / num_features;
        // Compute RMS
        float rms = sqrtf(mean_sq + eps);
        shmem[0] = rms;
    }
    __syncthreads();
    float rms = shmem[0];

    // Write output: x / rms
    for (int f = threadIdx.x; f < num_features; f += blockDim.x) {
        int64_t idx = ((b * num_features + f) * dim1 + d1) * dim2 + d2;
        half hval = input[idx];
        float fval = __half2float(hval);
        float normed = fval / rms;
        output[idx] = __float2half_rn(normed);
    }
}

// Host launch function for RMSNorm (fp16 I/O, fp32 accum)
void launch_gpu_implementation(
    void* output,                   // Output tensor pointer (float16 on CUDA)
    void* input,                    // Input tensor pointer (float16 on CUDA)
    int64_t batch_size,             // Number of batches
    int64_t num_features,           // Number of features (channels)
    int64_t dim1,                   // First spatial dimension
    int64_t dim2,                   // Second spatial dimension
    float eps                       // Epsilon for numerical stability
) {
    // Set up launch config: 1 block per (b, d1, d2)
    dim3 block(256, 1, 1); // up to 256 threads per block (efficient for fp16)
    dim3 grid(batch_size, dim1, dim2);

    size_t shmem_bytes = 256 * sizeof(float);

    rmsnorm_forward_fp16<<<grid, block, shmem_bytes>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        batch_size, num_features, dim1, dim2,
        eps
    );

    cudaDeviceSynchronize();
}
