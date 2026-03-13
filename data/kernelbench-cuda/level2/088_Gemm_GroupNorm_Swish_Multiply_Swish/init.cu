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

__global__ void gemm_bias_kernel(
    const half* input, int M, int K,
    const half* weight, int N,
    const half* bias,
    half* output
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __half2float(input[row * K + k]) * __half2float(weight[col * K + k]);
    }
    sum += __half2float(bias[col]);
    output[row * N + col] = __float2half_rn(sum);
}

__global__ void fused_groupnorm_swish_mul_swish_kernel(
    half* output,
    const half* gamma,
    const half* beta,
    const half* multiply_weight,
    int batch_size,
    int num_groups,
    int out_features
) {
    int group_id = blockIdx.x;
    int sample_id = group_id / num_groups;
    int group = group_id % num_groups;

    const int features_per_group = out_features / num_groups;
    const int start_feature = group * features_per_group;
    const int tid = threadIdx.x;

    if (tid >= features_per_group) return;

    const int feature_idx = start_feature + tid;
    const int output_idx = sample_id * out_features + feature_idx;
    
    const float x = __half2float(output[output_idx]);

    __shared__ float s_sum[64];
    __shared__ float s_sq_sum[64];

    // Initialize shared memory
    s_sum[tid] = x;
    s_sq_sum[tid] = x * x;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq_sum[tid] += s_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    const float mean = s_sum[0] / features_per_group;
    const float var = (s_sq_sum[0]/features_per_group) - (mean*mean);
    const float inv_std = rsqrtf(var + 1e-5f);

    // Normalize and apply gamma/beta
    float normalized = (x - mean) * inv_std;
    normalized = normalized * __half2float(gamma[feature_idx]) + __half2float(beta[feature_idx]);

    // Swish activation
    const float swish = normalized * __frcp_rn(1.0f + expf(-normalized));

    // Multiply with learned weight
    const float multiplied = swish * __half2float(multiply_weight[feature_idx]);

    // Final swish activation
    const float final_val = multiplied * __frcp_rn(1.0f + expf(-multiplied));

    output[output_idx] = __float2half_rn(final_val);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* gemm_weight, void* gemm_bias,
    void* group_norm_weight, void* group_norm_bias,
    void* multiply_weight,
    int num_groups,
    int batch_size, int in_features, int out_features
) {
    // GEMM configuration
    const dim3 gemm_block(16, 16);
    const dim3 gemm_grid(
        (out_features + gemm_block.x - 1) / gemm_block.x,
        (batch_size + gemm_block.y - 1) / gemm_block.y
    );

    gemm_bias_kernel<<<gemm_grid, gemm_block>>>(
        static_cast<const half*>(input),
        batch_size,
        in_features,
        static_cast<const half*>(gemm_weight),
        out_features,
        static_cast<const half*>(gemm_bias),
        static_cast<half*>(output)
    );

    // GroupNorm configuration
    const int features_per_group = out_features / num_groups;
    const dim3 groupnorm_block(features_per_group);
    const dim3 groupnorm_grid(batch_size * num_groups);

    fused_groupnorm_swish_mul_swish_kernel<<<groupnorm_grid, groupnorm_block>>>(
        static_cast<half*>(output),
        static_cast<const half*>(group_norm_weight),
        static_cast<const half*>(group_norm_bias),
        static_cast<const half*>(multiply_weight),
        batch_size,
        num_groups,
        out_features
    );

    cudaDeviceSynchronize();
}
