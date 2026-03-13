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

__global__ void conv3d_kernel(
    const half* input,
    const half* weight,
    const half* conv_bias,
    half* conv_output,
    float* sum,
    float* sum_sq,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth, int height, int width,
    int kernel_size,
    int groups,
    int D_out, int H_out, int W_out,
    int channels_per_group
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * D_out * H_out * W_out;
    if (tid >= total_elements) return;

    int idx = tid;
    int n = idx / (out_channels * D_out * H_out * W_out);
    idx %= out_channels * D_out * H_out * W_out;
    int c_out = idx / (D_out * H_out * W_out);
    idx %= D_out * H_out * W_out;
    int d = idx / (H_out * W_out);
    idx %= H_out * W_out;
    int h = idx / W_out;
    int w = idx % W_out;

    float acc = 0.0f;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int d_in = d + kd;
                    int h_in = h + kh;
                    int w_in = w + kw;

                    if (d_in < depth && h_in < height && w_in < width) {
                        int input_idx = n * in_channels * depth * height * width +
                                       c_in * depth * height * width +
                                       d_in * height * width +
                                       h_in * width +
                                       w_in;

                        int weight_idx = c_out * in_channels * kernel_size * kernel_size * kernel_size +
                                        c_in * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size +
                                        kh * kernel_size +
                                        kw;

                        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }

    acc += __half2float(conv_bias[c_out]);
    conv_output[tid] = __float2half_rn(acc);

    int group = c_out / channels_per_group;
    int sum_idx = n * groups + group;
    atomicAdd(&sum[sum_idx], acc);
    atomicAdd(&sum_sq[sum_idx], acc * acc);
}

__global__ void compute_mean_var_kernel(
    const float* sum,
    const float* sum_sq,
    float* mean,
    float* var,
    int batch_size,
    int groups,
    int elements_per_group
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * groups) return;

    float s = sum[tid];
    float s_sq = sum_sq[tid];
    float count = elements_per_group;

    float m = s / count;
    float v = (s_sq / count) - m * m;

    mean[tid] = m;
    var[tid] = v + 1e-5f; // Add epsilon for numerical stability
}

__global__ void norm_min_clamp_dropout_kernel(
    const half* conv_output,
    const float* mean,
    const float* var,
    const half* gn_weight,
    const half* gn_bias,
    half* output,
    float min_value,
    float max_value,
    float dropout_p,
    int batch_size,
    int out_channels,
    int D_out, int H_out, int W_out,
    int groups,
    int channels_per_group
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * D_out * H_out * W_out;
    if (tid >= total_elements) return;

    int idx = tid;
    int n = idx / (out_channels * D_out * H_out * W_out);
    idx %= out_channels * D_out * H_out * W_out;
    int c_out = idx / (D_out * H_out * W_out);
    idx %= D_out * H_out * W_out;
    int d = idx / (H_out * W_out);
    idx %= H_out * W_out;
    int h = idx / W_out;
    int w = idx % W_out;

    int group = c_out / channels_per_group;
    int sum_idx = n * groups + group;

    float x = __half2float(conv_output[tid]);
    float m = mean[sum_idx];
    float v = var[sum_idx];
    float gamma = __half2float(gn_weight[c_out]);
    float beta = __half2float(gn_bias[c_out]);

    x = (x - m) / sqrtf(v);
    x = x * gamma + beta;

    x = fminf(x, min_value);
    x = fmaxf(x, min_value);
    x = fminf(x, max_value);

    x *= (1.0f - dropout_p);

    output[tid] = __float2half_rn(x);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    void* gn_weight, void* gn_bias,
    int kernel_size, int groups,
    float min_value, float max_value, float dropout_p
) {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int D_out = depth - kernel_size + 1;
    const int H_out = height - kernel_size + 1;
    const int W_out = width - kernel_size + 1;
    const int channels_per_group = out_channels / groups;

    int conv_output_size = batch_size * out_channels * D_out * H_out * W_out;
    int sum_size = batch_size * groups;

    half* d_conv_output;
    float* d_sum, *d_sum_sq, *d_mean, *d_var;

    cudaMalloc(&d_conv_output, conv_output_size * sizeof(half));
    cudaMalloc(&d_sum, sum_size * sizeof(float));
    cudaMalloc(&d_sum_sq, sum_size * sizeof(float));
    cudaMalloc(&d_mean, sum_size * sizeof(float));
    cudaMalloc(&d_var, sum_size * sizeof(float));

    cudaMemset(d_sum, 0, sum_size * sizeof(float));
    cudaMemset(d_sum_sq, 0, sum_size * sizeof(float));

    dim3 block(256);
    dim3 grid((conv_output_size + block.x - 1) / block.x);
    conv3d_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv_output,
        d_sum,
        d_sum_sq,
        batch_size,
        in_channels,
        out_channels,
        depth, height, width,
        kernel_size,
        groups,
        D_out, H_out, W_out,
        channels_per_group
    );

    dim3 block_mv(256);
    dim3 grid_mv((sum_size + block_mv.x - 1) / block_mv.x);
    compute_mean_var_kernel<<<grid_mv, block_mv>>>(
        d_sum,
        d_sum_sq,
        d_mean,
        d_var,
        batch_size,
        groups,
        channels_per_group * D_out * H_out * W_out
    );

    norm_min_clamp_dropout_kernel<<<grid, block>>>(
        d_conv_output,
        d_mean,
        d_var,
        static_cast<const half*>(gn_weight),
        static_cast<const half*>(gn_bias),
        static_cast<half*>(output),
        min_value,
        max_value,
        dropout_p,
        batch_size,
        out_channels,
        D_out, H_out, W_out,
        groups,
        channels_per_group
    );

    cudaFree(d_conv_output);
    cudaFree(d_sum);
    cudaFree(d_sum_sq);
    cudaFree(d_mean);
    cudaFree(d_var);
}
