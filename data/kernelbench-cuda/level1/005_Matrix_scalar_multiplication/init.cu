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
// Matrix-Scalar Multiplication Kernel for FP16 (half) tensors
// Implements: C = A * s, where A is (M, N) in fp16, s is float, output C is (M, N) in fp16
// The scalar is promoted to float for computation and cast back to fp16 for output, to match PyTorch semantics.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CUDA kernel to perform elementwise matrix-scalar multiplication (fp16 I/O, fp32 scalar)
__global__ void mat_scalar_fp16_kernel(
    const half* __restrict__ A,
    half* __restrict__ C,
    float s,
    int64_t numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Load A[idx] as half, promote to float, multiply with s, cast back to half
    float a_fp32 = __half2float(A[idx]);
    float c_fp32 = a_fp32 * s;
    C[idx] = __float2half_rn(c_fp32);
}

// Host launcher for the kernel
void launch_gpu_implementation(
    void* output,    // half* (C)
    void* input,     // half* (A)
    float s,         // scalar
    int64_t M,
    int64_t N
) {
    const int64_t numel = M * N;
    const int threads_per_block = 256;
    const int blocks = (numel + threads_per_block - 1) / threads_per_block;

    mat_scalar_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        s,
        numel
    );

    cudaDeviceSynchronize();
}
