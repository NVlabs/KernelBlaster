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

__global__ void fused_gemm_bias_scale_relu_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    float multiplier,
    float negative_slope,
    int M, int N, int K) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * N) return;

    const int row = tid / N;    // Batch index
    const int col = tid % N;    // Output feature index

    float acc = 0.0f;
    for(int k = 0; k < K; ++k) {
        acc += __half2float(input[row * K + k]) * 
               __half2float(weight[col * K + k]);
    }

    // Add bias if present
    if(bias) {
        acc += __half2float(bias[col]);
    }

    // Apply scaling
    acc *= multiplier;

    // Apply LeakyReLU
    acc = acc > 0.0f ? acc : acc * negative_slope;

    // Convert back to FP16 and store
    output[row * N + col] = __float2half_rn(acc);
}

void launch_gpu_implementation(void* output, void* input, void* weights, void* bias, 
                              float multiplier, float negative_slope) {
    const int M = 128;   // batch_size
    const int K = 1024;  // in_features
    const int N = 512;   // out_features

    const int num_output_elements = M * N;
    const int block_size = 256;
    const int grid_size = (num_output_elements + block_size - 1) / block_size;

    fused_gemm_bias_scale_relu_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weights),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        multiplier,
        negative_slope,
        M, N, K
    );
    
    cudaDeviceSynchronize();
}
