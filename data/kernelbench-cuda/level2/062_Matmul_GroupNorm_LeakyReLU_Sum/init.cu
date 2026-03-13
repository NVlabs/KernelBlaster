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

// Matrix multiplication kernel with shared memory tiling
__global__ void fc_kernel(const half* __restrict__ input,
                          const half* __restrict__ weight,
                          const half* __restrict__ bias,
                          half* __restrict__ output,
                          int batch_size, int input_size, int hidden_size) {
    extern __shared__ __align__(sizeof(half)) unsigned char shared_buf[];
    half* shared_tile = reinterpret_cast<half*>(shared_buf);

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= hidden_size) return;

    float acc = 0.0f;
    const int tile_width = blockDim.x;
    
    for (int t = 0; t < input_size; t += tile_width) {
        const int load_col = t + threadIdx.x;
        if (load_col < input_size && row < batch_size) {
            shared_tile[threadIdx.y * (tile_width + 1) + threadIdx.x] = 
                input[row * input_size + load_col];
        }
        __syncthreads();

        const int k_max = min(tile_width, input_size - t);
        for (int k = 0; k < k_max; ++k) {
            const float w = __half2float(weight[col * input_size + t + k]);
            const float x = __half2float(shared_tile[threadIdx.y * (tile_width + 1) + k]);
            acc += x * w;
        }
        __syncthreads();
    }

    if (row < batch_size && col < hidden_size) {
        acc += __half2float(bias[col]);
        output[row * hidden_size + col] = __float2half_rn(acc);
    }
}

// Optimized group normalization kernel with warp-level reduction
__global__ void group_norm_kernel(half* __restrict__ data,
                                  const half* __restrict__ gamma,
                                  const half* __restrict__ beta,
                                  int batch_size, int hidden_size, 
                                  int num_groups, float eps) {
    const int group_id = blockIdx.x;
    const int sample_id = group_id / num_groups;
    const int group = group_id % num_groups;
    const int group_size = hidden_size / num_groups;
    const int idx = group * group_size + threadIdx.x;

    if (sample_id >= batch_size || idx >= hidden_size) return;

    // Warp-level reduction for mean and variance
    float x = __half2float(data[sample_id * hidden_size + idx]);
    
    float sum = x;
    float sum_sq = x * x;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    const float mean = __shfl_sync(0xffffffff, sum, 0) / group_size;
    const float var = (__shfl_sync(0xffffffff, sum_sq, 0) / group_size) - (mean * mean);
    const float inv_std = rsqrtf(var + eps);

    const float val = (x - mean) * inv_std;
    data[sample_id * hidden_size + idx] = __float2half_rn(
        val * __half2float(gamma[idx]) + __half2float(beta[idx])
    );
}

// Fused activation kernel with leaky ReLU and addition
__global__ void fused_activation_kernel(half* __restrict__ data, 
                                       int num_elements, 
                                       float negative_slope) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = __half2float(data[idx]);
    val = val < 0.0f ? val * negative_slope : val;
    data[idx] = __float2half_rn(val + val); // Fused add operation
}

extern "C" // Add extern "C" for proper linkage
void launch_gpu_implementation(void* output, void* input,
                               const void* fc_weight, const void* fc_bias,
                               const void* gn_weight, const void* gn_bias,
                               int batch_size, int input_size, int hidden_size,
                               int num_groups, float negative_slope, float eps) {
    // Launch fully connected layer
    const dim3 block_dim(32, 8);  // Optimal for A6000's 1024 threads/SM
    const dim3 grid_dim(
        (hidden_size + block_dim.x - 1) / block_dim.x,
        (batch_size + block_dim.y - 1) / block_dim.y
    );
    const size_t shared_mem = block_dim.y * (block_dim.x + 1) * sizeof(half);
    
    fc_kernel<<<grid_dim, block_dim, shared_mem>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(fc_weight),
        static_cast<const half*>(fc_bias),
        static_cast<half*>(output),
        batch_size, input_size, hidden_size
    );
    cudaDeviceSynchronize();

    // Launch group normalization
    const int group_size = hidden_size / num_groups;
    group_norm_kernel<<<batch_size * num_groups, group_size>>>(
        static_cast<half*>(output),
        static_cast<const half*>(gn_weight),
        static_cast<const half*>(gn_bias),
        batch_size, hidden_size, num_groups, eps
    );
    cudaDeviceSynchronize();

    // Launch fused activation + add
    const int num_elements = batch_size * hidden_size;
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    fused_activation_kernel<<<grid_size, block_size>>>(
        static_cast<half*>(output),
        num_elements,
        negative_slope
    );
    cudaDeviceSynchronize();
}
