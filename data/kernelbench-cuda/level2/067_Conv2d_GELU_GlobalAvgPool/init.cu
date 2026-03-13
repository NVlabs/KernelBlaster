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

__global__ void conv_gelu_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ conv_output,
    int in_channels, int out_channels, int kernel_size,
    int batch_size, int height, int width
) {
    const int OH = height - kernel_size + 1;
    const int OW = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * OH * OW;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    const int n = tid / (out_channels * OH * OW);
    const int c_out = (tid / (OH * OW)) % out_channels;
    const int h_out = (tid / OW) % OH;
    const int w_out = tid % OW;

    float acc = 0.0f;
    const int K = kernel_size;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int h_in = h_out + kh;
                const int w_in = w_out + kw;
                if (h_in < height && w_in < width) {
                    const int input_idx = n * in_channels * height * width + 
                                       c_in * height * width + h_in * width + w_in;
                    const int weight_idx = c_out * in_channels * K * K + 
                                       c_in * K * K + kh * K + kw;
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    acc += __half2float(bias[c_out]);
    
    // GELU implementation
    acc = 0.5f * acc * (1.0f + erff(acc / sqrtf(2.0f)));
    
    conv_output[tid] = __float2half_rn(acc);
}

__global__ void avg_pool_kernel(
    const half* __restrict__ conv_output,
    half* __restrict__ final_output,
    int batch_size, int out_channels,
    int OH, int OW
) {
    const int n = blockIdx.x / out_channels;
    const int c_out = blockIdx.x % out_channels;
    const int spatial_size = OH * OW;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
        const int h = idx / OW;
        const int w = idx % OW;
        const int conv_idx = n * out_channels * OH * OW + c_out * OH * OW + h * OW + w;
        sum += __half2float(conv_output[conv_idx]);
    }

    // Block-wide reduction
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        final_output[n * out_channels + c_out] = __float2half_rn(shared_sum[0] / spatial_size);
    }
}

void launch_gpu_implementation(void* output, void* input, const void* weight, const void* bias,
                               int in_channels, int out_channels, int kernel_size,
                               int batch_size, int height, int width) {
    const int OH = height - kernel_size + 1;
    const int OW = width - kernel_size + 1;
    const int conv_output_size = batch_size * out_channels * OH * OW;

    half* d_conv_output;
    cudaMalloc(&d_conv_output, conv_output_size * sizeof(half));

    // Launch convolution + GELU kernel
    const int block_size = 256;
    const int grid_size = (conv_output_size + block_size - 1) / block_size;
    conv_gelu_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        d_conv_output,
        in_channels, out_channels, kernel_size,
        batch_size, height, width
    );

    // Launch average pooling kernel
    const int pool_grid_size = batch_size * out_channels;
    avg_pool_kernel<<<pool_grid_size, block_size>>>(
        d_conv_output,
        static_cast<half*>(output),
        batch_size, out_channels, OH, OW
    );

    cudaFree(d_conv_output);
    cudaDeviceSynchronize();
}
