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

__global__ void sum_weights_kernel(const half* weight, float* sum_weights, int hidden_size, int input_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= input_size) return;

    float sum = 0.0f;
    for (int j = 0; j < hidden_size; ++j) {
        sum += __half2float(weight[j * input_size + k]);
    }
    sum_weights[k] = sum;
}

__global__ void compute_output_kernel(const half* input, const float* sum_weights, half* output, float scaling_factor, int batch_size, int input_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    float sum = 0.0f;
    for (int k = 0; k < input_size; ++k) {
        sum += __half2float(input[i * input_size + k]) * sum_weights[k];
    }
    sum = sum * 0.5f * scaling_factor;
    output[i] = __float2half_rn(sum);
}

void launch_gpu_implementation(void* output, void* input, void* weight, float scaling_factor) {
    const int batch_size = 128;
    const int input_size = 10;
    const int hidden_size = 20;

    float* d_sum_weights;
    cudaMalloc(&d_sum_weights, input_size * sizeof(float));

    sum_weights_kernel<<<1, input_size>>>(
        static_cast<const half*>(weight),
        d_sum_weights,
        hidden_size,
        input_size
    );

    compute_output_kernel<<<1, batch_size>>>(
        static_cast<const half*>(input),
        d_sum_weights,
        static_cast<half*>(output),
        scaling_factor,
        batch_size,
        input_size
    );

    cudaFree(d_sum_weights);
    cudaDeviceSynchronize();
}
