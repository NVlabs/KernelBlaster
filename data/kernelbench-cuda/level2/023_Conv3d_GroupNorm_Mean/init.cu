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
#include <iostream>
#include <cmath>

// Convolution parameters
#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 0

// GroupNorm parameters
#define EPSILON 1e-5f

// 3D Convolution Kernel using GEMM approach with im2col
__global__ void conv3d_im2col_kernel(
    const half* input, const half* weight, const half* bias, half* output,
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int D_out, int H_out, int W_out) {
    
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = batch_size * out_channels * D_out * H_out * W_out;
    if (output_idx >= output_size) return;

    // Calculate output indices
    int b = output_idx / (out_channels * D_out * H_out * W_out);
    int oc = (output_idx % (out_channels * D_out * H_out * W_out)) / (D_out * H_out * W_out);
    int d = (output_idx % (D_out * H_out * W_out)) / (H_out * W_out);
    int h = (output_idx % (H_out * W_out)) / W_out;
    int w = output_idx % W_out;

    float acc = 0.0f;
    for (int k = 0; k < in_channels * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE; ++k) {
        int ic = k / (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE);
        int kd = (k % (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)) / (KERNEL_SIZE * KERNEL_SIZE);
        int kh = (k % (KERNEL_SIZE * KERNEL_SIZE)) / KERNEL_SIZE;
        int kw = k % KERNEL_SIZE;

        int input_d = d * STRIDE + kd - PADDING;
        int input_h = h * STRIDE + kh - PADDING;
        int input_w = w * STRIDE + kw - PADDING;

        if (input_d >= 0 && input_d < D && input_h >= 0 && input_h < H && input_w >= 0 && input_w < W) {
            int input_idx = b * in_channels * D * H * W +
                            ic * D * H * W +
                            input_d * H * W +
                            input_h * W +
                            input_w;
            int weight_idx = oc * in_channels * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE + k;
            acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
        }
    }
    acc += __half2float(bias[oc]);
    output[output_idx] = __float2half_rn(acc);
}

// Group Normalization Kernel
__global__ void group_norm_kernel(
    const half* input, const half* gamma, const half* beta, half* output,
    int batch_size, int channels, int num_groups,
    int D, int H, int W) {
    
    const int group_size = channels / num_groups;
    const int elements_per_group = group_size * D * H * W;
    
    int batch = blockIdx.x;
    int group = blockIdx.y;
    int tid = threadIdx.x;
    int c_start = group * group_size;
    int c_end = c_start + group_size;

    extern __shared__ float smem[];
    float* sum = smem;
    float* sum_sq = &smem[blockDim.x];

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        int c = c_start + (i / (D * H * W));
        int d = (i % (D * H * W)) / (H * W);
        int h = (i % (H * W)) / W;
        int w = i % W;
        
        int idx = batch * channels * D * H * W +
                  c * D * H * W +
                  d * H * W +
                  h * W +
                  w;
        float val = __half2float(input[idx]);
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    sum[tid] = thread_sum;
    sum_sq[tid] = thread_sum_sq;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sum[tid] += sum[tid + s];
            sum_sq[tid] += sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = sum[0] / elements_per_group;
        float var = sum_sq[0] / elements_per_group - mean * mean;
        float inv_std = rsqrtf(var + EPSILON);

        // Store mean and variance for normalization
        sum[0] = mean;
        sum[1] = inv_std;
    }
    __syncthreads();

    float mean = sum[0];
    float inv_std = sum[1];

    // Apply normalization
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        int c = c_start + (i / (D * H * W));
        int d = (i % (D * H * W)) / (H * W);
        int h = (i % (H * W)) / W;
        int w = i % W;
        
        int idx = batch * channels * D * H * W +
                  c * D * H * W +
                  d * H * W +
                  h * W +
                  w;
        float val = (__half2float(input[idx]) - mean) * inv_std;
        val = val * __half2float(gamma[c]) + __half2float(beta[c]);
        output[idx] = __float2half_rn(val);
    }
}

// Mean Reduction Kernel
__global__ void mean_reduction_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int D, int H, int W) {
    
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int elements = channels * D * H * W;

    extern __shared__ float smem[];
    float* sum = smem;

    float thread_sum = 0.0f;
    for (int i = tid; i < elements; i += blockDim.x) {
        int c = i / (D * H * W);
        int d = (i % (D * H * W)) / (H * W);
        int h = (i % (H * W)) / W;
        int w = i % W;
        
        int idx = batch * channels * D * H * W +
                  c * D * H * W +
                  d * H * W +
                  h * W +
                  w;
        thread_sum += __half2float(input[idx]);
    }

    sum[tid] = thread_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sum[tid] += sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch] = __float2half_rn(sum[0] / elements);
    }
}

// Launch function
void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    void* gn_weight, void* gn_bias,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int num_groups,
    int D, int H, int W) {
    
    // Calculate output dimensions
    const int D_out = D - kernel_size + 1;
    const int H_out = H - kernel_size + 1;
    const int W_out = W - kernel_size + 1;
    const int conv_output_size = batch_size * out_channels * D_out * H_out * W_out;

    // Allocate intermediate buffers
    half *d_conv_output, *d_gn_output;
    cudaMalloc(&d_conv_output, conv_output_size * sizeof(half));
    cudaMalloc(&d_gn_output, conv_output_size * sizeof(half));

    // Launch convolution kernel
    dim3 block(256);
    dim3 grid((conv_output_size + block.x - 1) / block.x);
    conv3d_im2col_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast< const half*>(conv_bias),
        d_conv_output,
        batch_size, in_channels, out_channels,
        D, H, W, D_out, H_out, W_out
    );

    // Launch GroupNorm kernel
    dim3 gn_grid(batch_size, num_groups);
    dim3 gn_block(256);
    size_t smem_size = 2 * gn_block.x * sizeof(float);
    group_norm_kernel<<<gn_grid, gn_block, smem_size>>>(
        d_conv_output,
        static_cast<const half*>(gn_weight),
        static_cast<const half*>(gn_bias),
        d_gn_output,
        batch_size, out_channels, num_groups,
        D_out, H_out, W_out
    );

    // Launch Mean Reduction kernel
    dim3 mean_grid(batch_size);
    dim3 mean_block(256);
    mean_reduction_kernel<<<mean_grid, mean_block, mean_block.x * sizeof(float)>>>(
        d_gn_output,
        static_cast<half*>(output),
        batch_size, out_channels,
        D_out, H_out, W_out
    );

    // Cleanup
    cudaFree(d_conv_output);
    cudaFree(d_gn_output);
    cudaDeviceSynchronize();
}
