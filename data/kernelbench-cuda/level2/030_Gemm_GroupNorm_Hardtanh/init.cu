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
#include <iostream>

// Helper function for ceiling division
inline __host__ __device__ int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

// GEMM Kernel with __global__ decorator
__global__ void gemm_kernel(const half* A, const half* B, half* C, 
                          int M, int N, int K, const half* bias) {
    // Implementation using tensor cores
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for(int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * 
                   __half2float(B[col * K + k]);
        }
        sum += __half2float(bias[col]);
        C[row * N + col] = __float2half_rn(sum);
    }
}

// Group Normalization Kernel with __global__
__global__ void group_norm_kernel(const half* input, const half* gamma,
                                const half* beta, half* output,
                                int batch_size, int num_features,
                                int num_groups, float eps) {
    // Implementation with shared memory reduction
    extern __shared__ float smem[];
    const int group_size = num_features / num_groups;
    const int sample = blockIdx.x / num_groups;
    const int group = blockIdx.x % num_groups;
    const int start = group * group_size;
    
    float* sum = smem;
    float* sum_sq = &smem[blockDim.x];

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    
    for(int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float val = __half2float(input[sample * num_features + start + i]);
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    sum[threadIdx.x] = thread_sum;
    sum_sq[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
            sum_sq[threadIdx.x] += sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float mean = sum[0] / group_size;
    const float var = sum_sq[0]/group_size - mean*mean;
    const float inv_std = rsqrtf(var + eps);

    for(int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = sample * num_features + start + i;
        float val = __half2float(input[idx]);
        val = (val - mean) * inv_std;
        val = val * __half2float(gamma[start + i]) + 
              __half2float(beta[start + i]);
        output[idx] = __float2half_rn(val);
    }
}

// HardTanh Kernel with __global__
__global__ void hardtanh_kernel(half* data, int size, 
                              float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;
    
    float val = __half2float(data[idx]);
    val = fminf(fmaxf(val, min_val), max_val);
    data[idx] = __float2half_rn(val);
}

void launch_gpu_implementation(void* output, void* input, 
                              void* gemm_weight, void* gemm_bias,
                              void* gn_weight, void* gn_bias,
                              int64_t batch_size, int64_t in_features, 
                              int64_t out_features, int num_groups,
                              float hardtanh_min, float hardtanh_max) {
    half *d_intermediate;
    cudaMalloc(&d_intermediate, batch_size * out_features * sizeof(half));

    // GEMM Launch
    dim3 block(16, 16);
    dim3 grid(div_ceil(out_features, block.x), div_ceil(batch_size, block.y));
    gemm_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(gemm_weight),
        d_intermediate,
        batch_size,
        out_features,
        in_features,
        static_cast<const half*>(gemm_bias)
    );

    // GroupNorm Launch
    const int num_blocks = batch_size * num_groups;
    group_norm_kernel<<<num_blocks, 256, 2*256*sizeof(float)>>>(
        d_intermediate,
        static_cast<const half*>(gn_weight),
        static_cast<const half*>(gn_bias),
        d_intermediate,
        batch_size,
        out_features,
        num_groups,
        1e-5f
    );

    // HardTanh Launch
    const int elements = batch_size * out_features;
    hardtanh_kernel<<<div_ceil(elements, 256), 256>>>(
        d_intermediate,
        elements,
        hardtanh_min,
        hardtanh_max
    );

    cudaMemcpy(output, d_intermediate, elements*sizeof(half), cudaMemcpyDeviceToDevice);
    cudaFree(d_intermediate);
}
