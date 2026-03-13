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
Implements CUDA kernel for:
    C = diag(A) @ B
Where:
    - A: (N,) float16, representing the diagonal of an (N,N) matrix
    - B: (N, M) float16
    - C: (N, M) float16 output

This is equivalent to:
    C[i, j] = A[i] * B[i, j]

The kernel is optimized for large N and M, using grid-stride loops and vectorized memory access for best throughput.
Input/output type: fp16
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <assert.h>

// Kernel for C = diag(A) @ B, i.e. C[i, j] = A[i] * B[i, j]
// All tensors are float16. Accumulation is not required, so use float16 throughout.
__global__ void diag_matmul_kernel(
    const half* __restrict__ A,   // (N), diagonal vector
    const half* __restrict__ B,   // (N, M), row-major
    half* __restrict__ C,         // (N, M), row-major
    int64_t N,
    int64_t M
) {
    // Use 128-bit vectorized loads/stores where possible (8xfp16 = 128b)
    constexpr int VEC = 8; // number of halfs in 128 bits
    int row = blockIdx.y * blockDim.y + threadIdx.y; // each thread processes one row
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * VEC; // start column for vectorized access

    if (row >= N) return;

    // Each thread processes VEC consecutive columns with vectorized access
    if (col + VEC <= M) {
        // Vectorized load/store
        // Load A[row] as half
        half a = A[row];
        // Get pointer to B and C
        const half* b_ptr = B + row * M + col;
        half* c_ptr = C + row * M + col;
        // Load VEC half elements from B
        half2 b_vec[VEC/2];
        #pragma unroll
        for (int i = 0; i < VEC/2; ++i) {
            b_vec[i] = reinterpret_cast<const half2*>(b_ptr)[i];
        }
        // Multiply and store
        #pragma unroll
        for (int i = 0; i < VEC/2; ++i) {
            half2 a2 = __halves2half2(a, a);
            half2 c2 = __hmul2(a2, b_vec[i]);
            reinterpret_cast<half2*>(c_ptr)[i] = c2;
        }
    } else if (col < M) {
        // Tail: do scalar for remaining columns
        half a = A[row];
        for (int j = col; j < M; ++j) {
            C[row * M + j] = __hmul(a, B[row * M + j]);
        }
    }
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(
    void* output,        // (N, M) float16, output tensor
    void* input_A,       // (N) float16, diagonal vector A
    void* input_B,       // (N, M) float16, matrix B
    int64_t N,           // size of A, first dim of B
    int64_t M            // second dim of B
) {
    const int VEC = 8; // Vector width (128 bits)
    // Use 16x16 thread blocks, each thread handles VEC columns
    dim3 block(32, 8); // 256 threads per block, for good occupancy
    dim3 grid(
        (M + VEC * block.x - 1) / (VEC * block.x), // columns
        (N + block.y - 1) / block.y                // rows
    );

    diag_matmul_kernel<<<grid, block>>>(
        static_cast<const half*>(input_A),
        static_cast<const half*>(input_B),
        static_cast<half*>(output),
        N, M
    );
    cudaDeviceSynchronize();
}

