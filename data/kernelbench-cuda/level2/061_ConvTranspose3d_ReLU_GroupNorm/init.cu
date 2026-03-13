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

// ConvTranspose3D + ReLU kernel
__global__ void conv_transpose_relu_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W, int kernel_size
) {
    const int D_out = D + kernel_size - 1;
    const int H_out = H + kernel_size - 1;
    const int W_out = W + kernel_size - 1;
    const int output_size = batch_size * out_channels * D_out * H_out * W_out;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    // Unravel output indices
    int n = tid / (out_channels * D_out * H_out * W_out);
    int rem = tid % (out_channels * D_out * H_out * W_out);
    int c_out = rem / (D_out * H_out * W_out);
    rem %= (D_out * H_out * W_out);
    int d_out = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;

    float acc = 0.0f;
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int d_in = d_out - kd;
                int h_in = h_out - kh;
                int w_in = w_out - kw;

                if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        int input_idx = n * in_channels * D * H * W +
                                      c_in * D * H * W +
                                      d_in * H * W +
                                      h_in * W +
                                      w_in;
                        int weight_idx = c_in * out_channels * kernel_size * kernel_size * kernel_size +
                                       c_out * kernel_size * kernel_size * kernel_size +
                                       kd * kernel_size * kernel_size +
                                       kh * kernel_size +
                                       kw;

                        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }

    // Apply ReLU
    output[tid] = __float2half_rn(fmaxf(acc, 0.0f));
}

// GroupNorm kernel with parallel reduction
__global__ void group_norm_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    int batch_size, int channels, int groups,
    int D, int H, int W, float epsilon
) {
    extern __shared__ float smem[];
    const int group_id = blockIdx.x;
    const int group_size = (channels / groups) * D * H * W;
    const int n = group_id / groups;
    const int g = group_id % groups;

    const int c_start = g * (channels / groups);
    const int c_end = c_start + (channels / groups);

    // Phase 1: Compute sum and sum_sq
    float sum = 0.0f, sum_sq = 0.0f;
    for (int c = c_start; c < c_end; ++c) {
        for (int i = threadIdx.x; i < D*H*W; i += blockDim.x) {
            int idx = n * channels * D * H * W +
                      c * D * H * W +
                      i;
            float val = __half2float(input[idx]);
            sum += val;
            sum_sq += val * val;
        }
    }

    // Block reduction for sum and sum_sq
    smem[threadIdx.x] = sum;
    smem[threadIdx.x + blockDim.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
            smem[threadIdx.x + blockDim.x] += smem[threadIdx.x + s + blockDim.x];
        }
        __syncthreads();
    }

    const float mean = smem[0] / group_size;
    const float var = (smem[blockDim.x] / group_size) - (mean * mean);
    const float inv_std = rsqrtf(var + epsilon);

    // Phase 2: Apply normalization
    for (int c = c_start; c < c_end; ++c) {
        for (int i = threadIdx.x; i < D*H*W; i += blockDim.x) {
            int idx = n * channels * D * H * W +
                      c * D * H * W +
                      i;
            float val = (__half2float(input[idx]) - mean) * inv_std;
            val = val * __half2float(gamma[c]) + __half2float(beta[c]);
            output[idx] = __float2half_rn(val);
        }
    }
}

void launch_gpu_implementation(
    void* output, void* input, const void* conv_weight,
    const void* gn_weight, const void* gn_bias,
    int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t groups, bool bias
) {
    const int batch_size = 16;
    const int D = 8, H = 16, W = 16;
    const int D_out = D + kernel_size - 1;
    const int H_out = H + kernel_size - 1;
    const int W_out = W + kernel_size - 1;
    const int output_size = batch_size * out_channels * D_out * H_out * W_out;

    // Launch ConvTranspose + ReLU
    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;
    conv_transpose_relu_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        D, H, W, kernel_size
    );

    // Launch GroupNorm
    const int num_groups = batch_size * groups;
    const int threads_per_block = 256;
    const size_t smem_size = 2 * threads_per_block * sizeof(float);
    group_norm_kernel<<<num_groups, threads_per_block, smem_size>>>(
        static_cast<const half*>(output),
        static_cast<half*>(output),
        static_cast<const half*>(gn_weight),
        static_cast<const half*>(gn_bias),
        batch_size, out_channels, groups,
        D_out, H_out, W_out, 1e-5f
    );

    cudaDeviceSynchronize();
}
