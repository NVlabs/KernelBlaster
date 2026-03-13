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

__global__ void matmul_activation_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ add_value,
    half* output,
    int batch_size, int in_features, int out_features) 
{
    // 2D grid for matrix multiplication
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        // Matrix multiplication with fp32 accumulation
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += __half2float(input[row * in_features + k]) * 
                   __half2float(weight[col * in_features + k]);
        }
        
        // Add bias and value
        sum += __half2float(bias[col]) + __half2float(add_value[col]);
        
        // Swish activation
        sum *= __frcp_rn(1.0f + expf(-sum));
        
        // Tanh activation
        sum = tanhf(sum);
        
        // GELU approximation
        sum *= 0.5f * (1.0f + erff(sum / 1.41421356237f));
        
        // Hardtanh clamping
        sum = fmaxf(fminf(sum, 1.0f), -1.0f);
        
        // Store final result
        output[row * out_features + col] = __float2half_rn(sum);
    }
}

void launch_gpu_implementation(void* output, void* input, 
                              void* matmul_weight, void* matmul_bias, void* add_value,
                              int64_t batch_size, int64_t in_features, int64_t out_features) 
{
    // Configure kernel dimensions
    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x,
              (batch_size + block.y - 1) / block.y);

    // Launch unified kernel
    matmul_activation_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(matmul_weight),
        static_cast<const half*>(matmul_bias),
        static_cast<const half*>(add_value),
        static_cast<half*>(output),
        batch_size,
        in_features,
        out_features
    );
    
    cudaDeviceSynchronize();
}
