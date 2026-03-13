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
#include <cstdint>
#include <cstdio>
#include <cassert>

// Utility: warp-level reduction (for fp32)
__inline__ __device__ float warpReduceSum(float val) {
    // For CUDA >= 9.0
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Kernel for mean reduction over a specified dimension of a 3D fp16 tensor.
// Accumulation is performed in fp32 for numerical stability.
// Supports dim == 0, 1, or 2.
__global__ void mean_reduce_fp16_3d_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t dim,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
) {
    // Input: [batch_size, dim1, dim2] (always 3D)
    // Output: shape is input shape with the dim-th dimension removed

    // Determine reduction configuration
    int64_t out_shape[2];
    int64_t reduce_len, out_elems;
    int64_t out_stride0, out_stride1;
    int64_t in_stride0, in_stride1, in_stride2;

    if (dim == 0) {
        // Reduce over batch_size
        reduce_len = batch_size;
        out_shape[0] = dim1;
        out_shape[1] = dim2;
        out_elems = dim1 * dim2;
        out_stride0 = dim2;
        out_stride1 = 1;
        in_stride0 = dim1 * dim2;
        in_stride1 = dim2;
        in_stride2 = 1;
    } else if (dim == 1) {
        // Reduce over dim1
        reduce_len = dim1;
        out_shape[0] = batch_size;
        out_shape[1] = dim2;
        out_elems = batch_size * dim2;
        out_stride0 = dim2;
        out_stride1 = 1;
        in_stride0 = dim1 * dim2;
        in_stride1 = dim2;
        in_stride2 = 1;
    } else if (dim == 2) {
        // Reduce over dim2
        reduce_len = dim2;
        out_shape[0] = batch_size;
        out_shape[1] = dim1;
        out_elems = batch_size * dim1;
        out_stride0 = dim1;
        out_stride1 = 1;
        in_stride0 = dim1 * dim2;
        in_stride1 = dim2;
        in_stride2 = 1;
    } else {
        // Invalid
        return;
    }

    // Each thread computes one output element (out_idx in [0, out_elems))
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_elems) return;

    // Compute output indices
    int64_t o0 = out_idx / out_stride0;
    int64_t o1 = out_idx % out_stride0;

    float sum = 0.0f;

    // For coalesced access, thread loops over reduction dim
    if (dim == 0) {
        // output[o0, o1] = mean over batch_size of input[b, o0, o1]
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t in_idx = b * in_stride0 + o0 * in_stride1 + o1 * in_stride2;
            sum += __half2float(input[in_idx]);
        }
    } else if (dim == 1) {
        // output[o0, o1] = mean over dim1 of input[o0, d1, o1]
        for (int64_t d1 = 0; d1 < dim1; ++d1) {
            int64_t in_idx = o0 * in_stride0 + d1 * in_stride1 + o1 * in_stride2;
            sum += __half2float(input[in_idx]);
        }
    } else if (dim == 2) {
        // output[o0, o1] = mean over dim2 of input[o0, o1, d2]
        for (int64_t d2 = 0; d2 < dim2; ++d2) {
            int64_t in_idx = o0 * in_stride0 + o1 * in_stride1 + d2 * in_stride2;
            sum += __half2float(input[in_idx]);
        }
    }

    // Compute mean
    float mean = sum / static_cast<float>(reduce_len);
    output[out_idx] = __float2half_rn(mean);
}

// Host launcher
void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t dim,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
) {
    // All pointers are device pointers, data is fp16 (half)
    const half* d_input = static_cast<const half*>(input);
    half* d_output = static_cast<half*>(output);

    // Output shape: input shape with dim-th dimension removed
    int64_t out_elems;
    if (dim == 0) {
        out_elems = dim1 * dim2;
    } else if (dim == 1) {
        out_elems = batch_size * dim2;
    } else if (dim == 2) {
        out_elems = batch_size * dim1;
    } else {
        // Invalid
        fprintf(stderr, "Invalid reduction dim: %lld\n", (long long)dim);
        return;
    }

    // Kernel launch config
    int threads = 256;
    int blocks = static_cast<int>((out_elems + threads - 1) / threads);

    mean_reduce_fp16_3d_kernel<<<blocks, threads>>>(
        d_input, d_output, dim, batch_size, dim1, dim2
    );

    // Synchronize to check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after mean_reduce_fp16_3d_kernel: %s\n", cudaGetErrorString(err));
        assert(false);
    }
}
