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
#include <math.h>

// FP16 GEMM with FP32 accumulation and proper weight layout
__global__ void gemm_fp32_accum_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            // Weight matrix is accessed with transposed layout
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[col * K + k]);  // Transposed access
            acc += a * b;
        }
        C[row * N + col] = __float2half_rn(acc);
    }
}

// Double Mish with PyTorch-compatible computation
__device__ __forceinline__ float pytorch_mish(float x) {
    // PyTorch's exact computation: x * tanh(softplus(x))
    const float sp = log1pf(expf(x));
    return x * tanhf(sp);
}

__global__ void double_mish_kernel(const half* input, half* output, int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        float x = __half2float(input[idx]);
        x = pytorch_mish(x);  // First Mish
        x = pytorch_mish(x);  // Second Mish
        output[idx] = __float2half_rn(x);
    }
}

// Bias addition matching PyTorch's layout
__global__ void add_bias_kernel(half* matrix, const half* bias, int elements, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        float val = __half2float(matrix[idx]);
        if (bias) {
            // PyTorch adds bias per output feature
            val += __half2float(bias[idx % out_features]);
        }
        matrix[idx] = __float2half_rn(val);
    }
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, 
                              int batch_size, int in_features, int out_features) {
    // Configure kernel dimensions
    dim3 threads(16, 16);
    dim3 blocks(
        (out_features + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );

    // Perform GEMM with FP32 accumulation
    gemm_fp32_accum_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),  // Weight already in [out, in] layout
        static_cast<half*>(output),
        batch_size,
        out_features,
        in_features
    );

    // Add bias
    const int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    add_bias_kernel<<<grid_size, block_size>>>(
        static_cast<half*>(output),
        static_cast<const half*>(bias),
        total_elements,
        out_features
    );

    // Apply double Mish activation
    double_mish_kernel<<<grid_size, block_size>>>(
        static_cast<half*>(output),
        static_cast<half*>(output),
        total_elements
    );

    cudaDeviceSynchronize();
}
