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

__global__ void conv_kernel(const half* __restrict__ input, const half* __restrict__ weight, const half* __restrict__ bias,
                            half* __restrict__ output,
                            int N, int C_in, int C_out, int H, int W, int K) {
    const int H_conv = H - K + 1;
    const int W_conv = W - K + 1;
    const int output_size = N * C_out * H_conv * W_conv;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    const int n = tid / (C_out * H_conv * W_conv);
    const int c_out = (tid % (C_out * H_conv * W_conv)) / (H_conv * W_conv);
    const int h_conv = (tid % (H_conv * W_conv)) / W_conv;
    const int w_conv = tid % W_conv;

    float acc = 0.0f;
    #pragma unroll
    for (int c_in = 0; c_in < C_in; ++c_in) {
        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                const int h_in = h_conv + kh;
                const int w_in = w_conv + kw;
                if (h_in < H && w_in < W) {
                    const int input_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                    const int weight_idx = ((c_out * C_in + c_in) * K + kh) * K + kw;
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }
    acc += __half2float(bias[c_out]);
    output[tid] = __float2half_rn(acc);
}

__global__ void avg_pool_kernel(const half* __restrict__ input, half* __restrict__ output,
                                int N, int C, int H, int W, int pool_size) {
    const int H_pool = H / pool_size;
    const int W_pool = W / pool_size;
    const int output_size = N * C * H_pool * W_pool;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    const int n = tid / (C * H_pool * W_pool);
    const int c = (tid % (C * H_pool * W_pool)) / (H_pool * W_pool);
    const int h_pool = (tid % (H_pool * W_pool)) / W_pool;
    const int w_pool = tid % W_pool;

    const int h_start = h_pool * pool_size;
    const int w_start = w_pool * pool_size;
    float sum = 0.0f;
    int count = 0;

    #pragma unroll
    for (int h = h_start; h < h_start + pool_size; ++h) {
        #pragma unroll
        for (int w = w_start; w < w_start + pool_size; ++w) {
            if (h < H && w < W) {
                const int input_idx = ((n * C + c) * H + h) * W + w;
                sum += __half2float(input[input_idx]);
                count++;
            }
        }
    }
    output[tid] = __float2half_rn(sum / count);
}

__global__ void sigmoid_kernel(half* __restrict__ data, int size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    const float val = __half2float(data[tid]);
    data[tid] = __float2half_rn(1.0f / (1.0f + expf(-val)));
}

__global__ void sum_reduce_kernel(const half* __restrict__ input, half* __restrict__ output, 
                                 int elements_per_batch, int num_batches) {
    extern __shared__ float smem[];
    const int batch = blockIdx.x;
    const int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < elements_per_batch; i += blockDim.x) {
        sum += __half2float(input[batch * elements_per_batch + i]);
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[batch] = __float2half_rn(smem[0]);
}

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, 
                              int batch_size, int in_channels, int out_channels, 
                              int input_height, int input_width, int conv_kernel_size, 
                              int pool_kernel_size) {
    const int H_conv = input_height - conv_kernel_size + 1;
    const int W_conv = input_width - conv_kernel_size + 1;
    const int H_pool = H_conv / pool_kernel_size;
    const int W_pool = W_conv / pool_kernel_size;

    half *d_conv, *d_pool;
    const size_t conv_size = batch_size * out_channels * H_conv * W_conv * sizeof(half);
    const size_t pool_size = batch_size * out_channels * H_pool * W_pool * sizeof(half);
    cudaMalloc(&d_conv, conv_size);
    cudaMalloc(&d_pool, pool_size);

    // Convolution
    const int conv_block = 256;
    const int conv_grid = (batch_size * out_channels * H_conv * W_conv + conv_block - 1) / conv_block;
    conv_kernel<<<conv_grid, conv_block>>>(
        static_cast<const half*>(input), 
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv,
        batch_size, in_channels, out_channels,
        input_height, input_width, conv_kernel_size
    );

    // Average Pooling
    const int pool_block = 256;
    const int pool_grid = (batch_size * out_channels * H_pool * W_pool + pool_block - 1) / pool_block;
    avg_pool_kernel<<<pool_grid, pool_block>>>(
        d_conv, d_pool,
        batch_size, out_channels, H_conv, W_conv, pool_kernel_size
    );

    // Sigmoid
    sigmoid_kernel<<<pool_grid, pool_block>>>(d_pool, batch_size * out_channels * H_pool * W_pool);

    // Sum Reduction
    const int elements_per_batch = out_channels * H_pool * W_pool;
    const int reduce_block = 256;
    const int reduce_shared = reduce_block * sizeof(float);
    sum_reduce_kernel<<<batch_size, reduce_block, reduce_shared>>>(
        d_pool, static_cast<half*>(output), elements_per_batch, batch_size
    );

    cudaFree(d_conv);
    cudaFree(d_pool);
    cudaDeviceSynchronize();
}
