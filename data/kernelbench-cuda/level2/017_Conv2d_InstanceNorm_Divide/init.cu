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

__global__ void conv_kernel(const half* __restrict__ input, const half* __restrict__ weight,
                            const half* __restrict__ bias, half* __restrict__ output,
                            int batch_size, int in_channels, int out_channels,
                            int H, int W, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * H * W;
    if (idx >= total_elements) return;

    const int H_out = H;
    const int W_out = W;

    const int n = idx / (out_channels * H_out * W_out);
    const int oc = (idx % (out_channels * H_out * W_out)) / (H_out * W_out);
    const int h = (idx % (H_out * W_out)) / W_out;
    const int w = idx % W_out;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int ih = h + kh - 1;  // padding=1
                const int iw = w + kw - 1;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    const int input_idx = n * in_channels * H * W + ic * H * W + ih * W + iw;
                    const int weight_idx = oc * in_channels * K * K + ic * K * K + kh * K + kw;
                    sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    sum += __half2float(bias[oc]);
    output[idx] = __float2half_rn(sum);
}

__global__ void instance_norm_mean_var_kernel(const half* __restrict__ conv_output,
                                              float* __restrict__ mean, float* __restrict__ variance,
                                              int batch_size, int out_channels, int H, int W) {
    const int n_oc = blockIdx.x;
    const int n = n_oc / out_channels;
    const int oc = n_oc % out_channels;

    const int num_elements = H * W;
    const int elements_per_thread = (num_elements + blockDim.x - 1) / blockDim.x;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < elements_per_thread; ++i) {
        const int idx = threadIdx.x + i * blockDim.x;
        if (idx < num_elements) {
            const int h = idx / W;
            const int w = idx % W;
            const int conv_idx = n * out_channels * H * W + oc * H * W + h * W + w;
            const float val = __half2float(conv_output[conv_idx]);
            sum += val;
            sum_sq += val * val;
        }
    }

    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    s_sum[threadIdx.x] = sum;
    s_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float mean_val = s_sum[0] / num_elements;
        const float var_val = (s_sum_sq[0] / num_elements) - (mean_val * mean_val) + 1e-5f;
        mean[n_oc] = mean_val;
        variance[n_oc] = var_val;
    }
}

__global__ void normalize_div_kernel(const half* __restrict__ conv_output,
                                     const float* __restrict__ mean, const float* __restrict__ variance,
                                     half* __restrict__ output, float divide_by,
                                     int batch_size, int out_channels, int H, int W) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * H * W;
    if (idx >= total_elements) return;

    const int n = idx / (out_channels * H * W);
    const int oc = (idx % (out_channels * H * W)) / (H * W);
    const int h = (idx % (H * W)) / W;
    const int w = idx % W;

    const int n_oc = n * out_channels + oc;
    const float mean_val = mean[n_oc];
    const float inv_std = rsqrtf(variance[n_oc]);

    const int conv_val_idx = n * out_channels * H * W + oc * H * W + h * W + w;
    const float val = __half2float(conv_output[conv_val_idx]);
    const float normalized = (val - mean_val) * inv_std / divide_by;
    output[idx] = __float2half_rn(normalized);
}

void launch_gpu_implementation(void* output, void* input, void* conv_weight, void* conv_bias, float divide_by) {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int H = 32, W = 32;
    const int K = 3;

    // Allocate temporary buffers
    half* d_conv_output;
    cudaMalloc(&d_conv_output, batch_size * out_channels * H * W * sizeof(half));

    // Launch convolution kernel
    const int conv_total = batch_size * out_channels * H * W;
    const dim3 conv_block(256);
    const dim3 conv_grid((conv_total + conv_block.x - 1) / conv_block.x);
    conv_kernel<<<conv_grid, conv_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv_output,
        batch_size, in_channels, out_channels, H, W, K
    );

    // Allocate mean/variance buffers
    float *d_mean, *d_variance;
    const int mean_var_size = batch_size * out_channels;
    cudaMalloc(&d_mean, mean_var_size * sizeof(float));
    cudaMalloc(&d_variance, mean_var_size * sizeof(float));

    // Compute mean and variance
    const dim3 mean_var_block(256);
    const dim3 mean_var_grid(mean_var_size);
    instance_norm_mean_var_kernel<<<mean_var_grid, mean_var_block>>>(
        d_conv_output, d_mean, d_variance, batch_size, out_channels, H, W
    );

    // Normalize and divide
    const dim3 norm_block(256);
    const dim3 norm_grid((conv_total + norm_block.x - 1) / norm_block.x);
    normalize_div_kernel<<<norm_grid, norm_block>>>(
        d_conv_output, d_mean, d_variance,
        static_cast<half*>(output),
        divide_by,
        batch_size, out_channels, H, W
    );

    // Cleanup
    cudaFree(d_conv_output);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaDeviceSynchronize();
}
