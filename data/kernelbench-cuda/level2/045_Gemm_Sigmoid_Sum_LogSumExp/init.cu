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
#include <cmath>

__global__ void model_kernel(const half* input,
                            const half* linear1_weight, const half* linear1_bias,
                            half* output,
                            int input_size, int hidden_size, int batch_size) {
    // Shared memory for logsumexp reduction
    __shared__ float shared_sums[128];
    
    const int tid = threadIdx.x;
    if (tid >= batch_size) return;

    // Each thread processes one batch element
    float element_sum = 0.0f;
    
    // Compute linear1 + sigmoid + sum for this batch element
    for (int j = 0; j < hidden_size; ++j) {
        float val = 0.0f;
        // Dot product for hidden unit j
        for (int k = 0; k < input_size; ++k) {
            val += __half2float(input[tid * input_size + k]) * 
                   __half2float(linear1_weight[j * input_size + k]);
        }
        val += __half2float(linear1_bias[j]);
        element_sum += 1.0f / (1.0f + expf(-val));  // Sigmoid and accumulate
    }

    // Store per-element sum in shared memory
    shared_sums[tid] = element_sum;
    __syncthreads();

    // LogSumExp across all batch elements (single block reduction)
    if (tid == 0) {
        float max_val = -INFINITY;
        // Find maximum value
        for (int i = 0; i < batch_size; ++i) {
            max_val = fmaxf(max_val, shared_sums[i]);
        }

        // Compute sum of exponentials
        float exp_sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            exp_sum += expf(shared_sums[i] - max_val);
        }

        // Final result
        output[0] = __float2half_rn(logf(exp_sum) + max_val);
    }
}

void launch_gpu_implementation(void* output, void* input,
                               void* linear1_weight, void* linear1_bias,
                               void* linear2_weight, void* linear2_bias,
                               int64_t input_size, int64_t hidden_size, int64_t output_size) {
    const int batch_size = 128;
    
    // Single block with batch_size threads
    model_kernel<<<1, batch_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(linear1_weight),
        static_cast<const half*>(linear1_bias),
        static_cast<half*>(output),
        input_size, hidden_size, batch_size
    );
    
    cudaDeviceSynchronize();
}
