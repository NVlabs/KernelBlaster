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
#include <mma.h>

__global__ void fused_linear_swish_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    const half scaling_factor,
    int batch_size,
    int in_features,
    int out_features
) {
    // Each thread handles one output element (batch, out_feature)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_features;
    if (tid >= total_elements) return;

    const int batch = tid / out_features;
    const int out_feat = tid % out_features;

    // FP32 accumulation for numerical stability
    float acc = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        const float x = __half2float(input[batch * in_features + i]);
        const float w = __half2float(weight[out_feat * in_features + i]);
        acc += x * w;
    }

    // Add bias using FP32
    acc += __half2float(bias[out_feat]);

    // Swish activation with FP32 math
    const float sigmoid = 1.0f / (1.0f + expf(-acc));
    acc *= sigmoid;

    // Apply scaling and convert back to FP16
    acc *= __half2float(scaling_factor);
    output[tid] = __float2half_rn(acc);
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, void* scaling_factor, 
                               int in_features, int out_features, int batch_size) {
    const int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    // Convert scalar factor once and pass by value
    half h_scaling_factor;
    cudaMemcpy(&h_scaling_factor, scaling_factor, sizeof(half), cudaMemcpyDeviceToHost);

    fused_linear_swish_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        h_scaling_factor,
        batch_size,
        in_features,
        out_features
    );
    
    cudaDeviceSynchronize();
}
