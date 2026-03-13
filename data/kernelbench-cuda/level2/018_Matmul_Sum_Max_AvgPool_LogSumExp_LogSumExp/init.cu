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

__global__ void model_forward_kernel(half* output, const half* input, const half* weight, const half* bias, 
                                    int batch_size, int in_features, int out_features) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Load input for this batch element
    const half* x = input + idx * in_features;
    
    // Accumulate in FP32 for numerical stability
    float sum = 0.0f;
    
    // Compute linear layer output and sum reduction
    for (int o = 0; o < out_features; ++o) {
        float val = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            val += __half2float(x[i]) * __half2float(weight[o * in_features + i]);
        }
        val += __half2float(bias[o]);  // Add bias
        sum += val;
    }

    // Subsequent reductions are no-ops on single element, just convert back to FP16
    output[idx] = __float2half_rn(sum);
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, 
                              int batch_size, int in_features, int out_features) {
    const int threads_per_block = 256;
    const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    model_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        batch_size,
        in_features,
        out_features
    );
    
    cudaDeviceSynchronize();
}
