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

// Convolution kernel
__global__ void conv_kernel(const half* input, const half* weight, const half* bias, half* output,
                            int batch_size, int in_channels, int height, int width,
                            int out_channels, int kernel_size) {
    int OH = height - kernel_size + 1;
    int OW = width - kernel_size + 1;
    int output_size = batch_size * out_channels * OH * OW;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    int n = tid / (out_channels * OH * OW);
    int residual = tid % (out_channels * OH * OW);
    int oc = residual / (OH * OW);
    residual = residual % (OH * OW);
    int oh = residual / OW;
    int ow = residual % OW;

    float acc = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h = oh + kh;
                int w = ow + kw;
                if (h < height && w < width) {
                    int input_idx = n * in_channels * height * width + ic * height * width + h * width + w;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }
    acc += __half2float(bias[oc]);
    output[tid] = __float2half_rn(acc);
}

// Group normalization helper kernels
__global__ void compute_group_stats_kernel(const half* input, float* mean, float* var,
                                           int batch_size, int out_channels, int groups,
                                           int height, int width, float eps) {
    int group_size = out_channels / groups;
    int elements_per_group = group_size * height * width;
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int n = blockIdx.x / groups;
    int g = blockIdx.x % groups;

    const half* group_input = input + n * out_channels * height * width + g * group_size * height * width;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        int c = i / (height * width);
        int hw = i % (height * width);
        int h = hw / width;
        int w = hw % width;
        int idx = c * height * width + h * width + w;
        float val = __half2float(group_input[idx]);
        sum += val;
        sum_sq += val * val;
    }

    float* s_sum = shared;
    float* s_sum_sq = (float*)&s_sum[blockDim.x];

    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = s_sum[0];
        float total_sum_sq = s_sum_sq[0];
        float m = total_sum / elements_per_group;
        float v = (total_sum_sq / elements_per_group) - (m * m);
        mean[blockIdx.x] = m;
        var[blockIdx.x] = v + eps;
    }
}

__global__ void apply_group_norm_kernel(const half* input, const half* gamma, const half* beta,
                                        const float* mean, const float* var,
                                        half* output,
                                        int batch_size, int out_channels, int groups,
                                        int height, int width) {
    int group_size = out_channels / groups;
    int elements_per_group = group_size * height * width;

    int n = blockIdx.x / groups;
    int g = blockIdx.x % groups;

    const float m = mean[blockIdx.x];
    const float v = var[blockIdx.x];
    const float inv_std = rsqrtf(v);

    const half* group_input = input + n * out_channels * height * width + g * group_size * height * width;
    half* group_output = output + n * out_channels * height * width + g * group_size * height * width;

    for (int i = threadIdx.x; i < elements_per_group; i += blockDim.x) {
        int c = i / (height * width);
        int hw = i % (height * width);
        int h = hw / width;
        int w = hw % width;
        int idx = c * height * width + h * width + w;
        float val = __half2float(group_input[idx]);
        float normalized = (val - m) * inv_std;
        normalized = normalized * __half2float(gamma[g * group_size + c]) + __half2float(beta[g * group_size + c]);
        group_output[idx] = __float2half_rn(normalized);
    }
}

// Activation kernels
__global__ void tanh_kernel(const half* input, half* output, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return; 
    float val = __half2float(input[tid]);
    val = tanhf(val);
    output[tid] = __float2half_rn(val);
}

__global__ void hard_shrink_kernel(const half* input, half* output, int num_elements, float threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    float val = __half2float(input[tid]);
    output[tid] = (val > threshold || val < -threshold) ? input[tid] : __float2half(0.0f);
}

__global__ void add_kernel(const half* a, const half* b, half* output, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    output[tid] = __hadd(a[tid], b[tid]);
}

// LogSumExp kernel
__global__ void logsumexp_kernel(const half* input, half* output,
                                 int batch_size, int channels,
                                 int height, int width) {
    int n = blockIdx.x / (height * width);
    int hw = blockIdx.x % (height * width);
    int h = hw / width;
    int w = hw % width;

    const half* input_ptr = input + n * channels * height * width + h * width + w;
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    for (int c = 0; c < channels; ++c) {
        float val = __half2float(input_ptr[c * height * width]);
        if (val > max_val) max_val = val;
    }

    for (int c = 0; c < channels; ++c) {
        float val = __half2float(input_ptr[c * height * width]);
        sum_exp += expf(val - max_val);
    }

    output[n * height * width + h * width + w] = __float2half_rn(logf(sum_exp) + max_val);
}

// Main launch function
void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    void* group_norm_weight, void* group_norm_bias,
    int batch_size, int in_channels, int height, int width,
    int out_channels, int kernel_size,
    int groups, float eps) {

    int OH = height - kernel_size + 1;
    int OW = width - kernel_size + 1;
    int conv_output_size = batch_size * out_channels * OH * OW;

    // Allocate intermediate buffers
    half *d_conv, *d_norm, *d_tanh, *d_hard_swish, *d_res;
    cudaMalloc(&d_conv, conv_output_size * sizeof(half));
    cudaMalloc(&d_norm, conv_output_size * sizeof(half));
    cudaMalloc(&d_tanh, conv_output_size * sizeof(half));
    cudaMalloc(&d_hard_swish, conv_output_size * sizeof(half));
    cudaMalloc(&d_res, conv_output_size * sizeof(half));

    // Step 1: Convolution
    dim3 conv_block(256);
    dim3 conv_grid((conv_output_size + conv_block.x - 1) / conv_block.x);
    conv_kernel<<<conv_grid, conv_block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv,
        batch_size, in_channels, height, width,
        out_channels, kernel_size
    );

    // Step 2: Group normalization
    int group_elements = batch_size * groups;
    float *d_mean, *d_var;
    cudaMalloc(&d_mean, group_elements * sizeof(float));
    cudaMalloc(&d_var, group_elements * sizeof(float));

    dim3 stats_block(256);
    dim3 stats_grid(batch_size * groups);
    compute_group_stats_kernel<<<stats_grid, stats_block, 2*stats_block.x*sizeof(float)>>>(
        d_conv, d_mean, d_var,
        batch_size, out_channels, groups, OH, OW, eps
    );

    apply_group_norm_kernel<<<stats_grid, stats_block>>>(
        d_conv,
        static_cast<const half*>(group_norm_weight),
        static_cast<const half*>(group_norm_bias),
        d_mean, d_var,
        d_norm,
        batch_size, out_channels, groups, OH, OW
    );

    // Step 3: Tanh
    tanh_kernel<<<conv_grid, conv_block>>>(d_norm, d_tanh, conv_output_size);

    // Step 4: HardShrink (threshold 1/6)
    hard_shrink_kernel<<<conv_grid, conv_block>>>(d_tanh, d_hard_swish, conv_output_size, 1.0f/6.0f);

    // Step 5: Residual addition
    add_kernel<<<conv_grid, conv_block>>>(d_conv, d_hard_swish, d_res, conv_output_size);

    // Step 6: LogSumExp
    int logsumexp_size = batch_size * OH * OW;
    dim3 logsumexp_grid(logsumexp_size);
    logsumexp_kernel<<<logsumexp_grid, 1>>>(d_res, static_cast<half*>(output), batch_size, out_channels, OH, OW);

    // Cleanup
    cudaFree(d_conv);
    cudaFree(d_norm);
    cudaFree(d_tanh);
    cudaFree(d_hard_swish);
    cudaFree(d_res);
    cudaFree(d_mean);
    cudaFree(d_var);
}
