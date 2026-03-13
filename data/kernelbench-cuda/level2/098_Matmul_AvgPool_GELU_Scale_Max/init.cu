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

__global__ void matmul_bias_kernel(
    const half* input,    // [batch_size, in_features]
    const half* weight,   // [out_features, in_features]
    const half* bias,     // [out_features]
    half* output,         // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= out_features) return;

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += __half2float(input[row * in_features + k]) * __half2float(weight[col * in_features + k]);
    }
    sum += __half2float(bias[col]);
    output[row * out_features + col] = __float2half_rn(sum);
}

__global__ void fused_avgpool_gelu_scale_kernel(
    const half* input,    // [batch_size, out_features]
    half* output,         // [batch_size, reduced_features]
    int batch_size,
    int out_features,
    int pool_kernel_size,
    float scale_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int reduced_features = out_features / pool_kernel_size;
    int total = batch_size * reduced_features;

    if (idx >= total) return;

    int batch = idx / reduced_features;
    int f_out = idx % reduced_features;

    float sum = 0.0f;
    for (int k = 0; k < pool_kernel_size; ++k) {
        int in_idx = batch * out_features + f_out * pool_kernel_size + k;
        sum += __half2float(input[in_idx]);
    }
    float avg = sum / pool_kernel_size;

    // GELU approximation
    float gelu = 0.5f * avg * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (avg + 0.044715f * avg * avg * avg)));
    gelu *= scale_factor;

    output[idx] = __float2half_rn(gelu);
}

__global__ void max_reduction_kernel(
    const half* input,  // [batch_size, reduced_features]
    half* output,       // [batch_size]
    int reduced_features
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int batch = blockIdx.x;

    float max_val = -INFINITY;
    for (int i = tid; i < reduced_features; i += blockDim.x) {
        float val = __half2float(input[batch * reduced_features + i]);
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch] = __float2half_rn(sdata[0]);
    }
}

void launch_gpu_implementation(
    void* output,
    void* input,
    const void* matmul_weight,
    const void* matmul_bias,
    int64_t in_features,
    int64_t out_features,
    int64_t pool_kernel_size,
    float scale_factor,
    int64_t batch_size
) {
    // Step 1: Matmul + Bias
    const half* d_input = static_cast<const half*>(input);
    const half* d_weight = static_cast<const half*>(matmul_weight);
    const half* d_bias = static_cast<const half*>(matmul_bias);
    half* d_matmul_output;
    cudaMalloc(&d_matmul_output, batch_size * out_features * sizeof(half));

    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    matmul_bias_kernel<<<grid, block>>>(d_input, d_weight, d_bias, d_matmul_output, batch_size, in_features, out_features);
    cudaDeviceSynchronize();

    // Step 2: AvgPool + GELU + Scale
    int reduced_features = out_features / pool_kernel_size;
    half* d_fused_output;
    cudaMalloc(&d_fused_output, batch_size * reduced_features * sizeof(half));

    int fused_threads = 256;
    int fused_blocks = (batch_size * reduced_features + fused_threads - 1) / fused_threads;
    fused_avgpool_gelu_scale_kernel<<<fused_blocks, fused_threads>>>(
        d_matmul_output, d_fused_output, batch_size, out_features, pool_kernel_size, scale_factor
    );
    cudaDeviceSynchronize();

    // Step 3: Max reduction
    half* d_output = static_cast<half*>(output);
    int max_threads = 256;
    int max_blocks = batch_size;
    max_reduction_kernel<<<max_blocks, max_threads, max_threads * sizeof(float)>>>(
        d_fused_output, d_output, reduced_features
    );
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_matmul_output);
    cudaFree(d_fused_output);
}
