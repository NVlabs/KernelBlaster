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

__global__ void linear_min_sub_kernel(
    const half* input,   // [batch_size, in_features]
    const half* weight,  // [out_features, in_features]
    const half* bias,    // [out_features]
    const half* constant,// scalar
    half* output,        // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * out_features) return;

    int row = tid / out_features;
    int col = tid % out_features;

    const half* input_row = input + row * in_features;
    const half* weight_row = weight + col * in_features;

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 10; ++k) { // Compile-time known in_features=10
        sum += __half2float(input_row[k]) * __half2float(weight_row[k]);
    }
    sum += __half2float(bias[col]);

    half val = __float2half_rn(sum);
    half c = *constant;

    val = __hmin(val, c);
    val = __hsub(val, c);

    output[tid] = val;
}

void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    void* constant,
    int batch_size,
    int in_features,
    int out_features
) {
    int num_elements = batch_size * out_features;
    const int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    linear_min_sub_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<const half*>(constant),
        static_cast<half*>(output),
        batch_size,
        in_features,
        out_features
    );

    cudaDeviceSynchronize();
}
