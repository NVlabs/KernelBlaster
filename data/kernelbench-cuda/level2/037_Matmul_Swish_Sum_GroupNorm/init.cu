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

const int BLOCK_SIZE = 256;

__global__ void matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,  // B is weight matrix (out_features x in_features)
    const half* __restrict__ bias,
    half* __restrict__ C,
    int M, int N, int K
) {
    // M = batch_size, N = out_features, K = in_features
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[row, k] * B[col, k] since B is transposed
            acc += __half2float(A[row * K + k]) * __half2float(B[col * K + k]);
        }
        // Add bias
        acc += __half2float(bias[col]);
        C[row * N + col] = __float2half_rn(acc);
    }
}

__global__ void swish_add_kernel(
    half* __restrict__ output,
    const half* __restrict__ extra_bias,
    int64_t out_features,
    int64_t batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elements = batch_size * out_features;
    if (tid >= elements) return;

    int bias_idx = tid % out_features;
    float val = __half2float(output[tid]);
    
    // Numerically stable Swish: x * sigmoid(x)
    float sigmoid = 1.0f / (1.0f + expf(-val));
    val *= sigmoid;
    
    // Add extra bias
    val += __half2float(extra_bias[bias_idx]);
    
    output[tid] = __float2half_rn(val);
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void group_norm_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    int64_t num_groups,
    int64_t batch_size,
    int64_t out_features
) {
    const int group_id = blockIdx.x;
    const int sample_id = group_id / num_groups;
    const int group = group_id % num_groups;
    const int channels_per_group = out_features / num_groups;

    if (sample_id >= batch_size) return;

    const int start_idx = sample_id * out_features + group * channels_per_group;
    const int end_idx = start_idx + channels_per_group;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Each thread processes multiple elements if needed
    for (int c = threadIdx.x; c < channels_per_group; c += blockDim.x) {
        float val = __half2float(input[start_idx + c]);
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction
    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    // Block-level reduction using shared memory
    __shared__ float s_sum[32];
    __shared__ float s_sum_sq[32];
    __shared__ float mean, inv_std;

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        s_sum[wid] = sum;
        s_sum_sq[wid] = sum_sq;
    }
    __syncthreads();

    // First warp reduces partial sums
    if (wid == 0) {
        sum = lane < blockDim.x / 32 ? s_sum[lane] : 0.0f;
        sum_sq = lane < blockDim.x / 32 ? s_sum_sq[lane] : 0.0f;

        // Final warp reduction
        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);

        if (lane == 0) {
            mean = sum / channels_per_group;
            float variance = (sum_sq / channels_per_group) - (mean * mean);
            inv_std = rsqrtf(variance + 1e-5f);
        }
    }
    __syncthreads();

    // Apply normalization
    for (int c = threadIdx.x; c < channels_per_group; c += blockDim.x) {
        int idx = start_idx + c;
        float val = __half2float(input[idx]);
        val = (val - mean) * inv_std;
        val = val * __half2float(gamma[c]) + __half2float(beta[c]);
        output[idx] = __float2half_rn(val);
    }
}

extern "C" void launch_gpu_implementation(
    void* output,
    void* input,
    void* matmul_weight,
    void* matmul_bias,
    void* extra_bias,
    void* gn_weight,
    void* gn_bias,
    int64_t in_features,
    int64_t out_features,
    int64_t num_groups,
    int64_t batch_size
) {
    // Matrix multiplication with transposed weight
    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x, 
              (batch_size + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(matmul_weight),  // Weight is (out_features x in_features)
        static_cast<const half*>(matmul_bias),
        static_cast<half*>(output),
        batch_size, out_features, in_features
    );
    cudaDeviceSynchronize();

    // Swish activation and extra bias addition
    const int elements = batch_size * out_features;
    dim3 swish_block(BLOCK_SIZE);
    dim3 swish_grid((elements + swish_block.x - 1) / swish_block.x);
    swish_add_kernel<<<swish_grid, swish_block>>>(
        static_cast<half*>(output),
        static_cast<const half*>(extra_bias),
        out_features,
        batch_size
    );
    cudaDeviceSynchronize();

    // Group normalization
    const int group_threads = 256;
    dim3 gn_grid(batch_size * num_groups);
    group_norm_kernel<<<gn_grid, group_threads>>>(
        static_cast<const half*>(output),
        static_cast<half*>(output),
        static_cast<const half*>(gn_weight),
        static_cast<const half*>(gn_bias),
        num_groups,
        batch_size,
        out_features
    );
    cudaDeviceSynchronize();
}
