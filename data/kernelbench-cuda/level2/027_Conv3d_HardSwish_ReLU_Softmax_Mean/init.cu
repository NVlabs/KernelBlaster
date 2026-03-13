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

__global__ void conv3d_kernel(half *output, const half *input, const half *weight, const half *bias,
                              int N, int C_in, int C_out, int D, int H, int W,
                              int kernel_size, int D_out, int H_out, int W_out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;
    if (tid >= total_elements) return;

    int n = tid / (C_out * D_out * H_out * W_out);
    int remainder = tid % (C_out * D_out * H_out * W_out);
    int c_out = remainder / (D_out * H_out * W_out);
    remainder = remainder % (D_out * H_out * W_out);
    int d_out = remainder / (H_out * W_out);
    remainder = remainder % (H_out * W_out);
    int h_out = remainder / W_out;
    int w_out = remainder % W_out;

    float sum = 0.0f;

    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int d_in = d_out + kd;
                    int h_in = h_out + kh;
                    int w_in = w_out + kw;

                    if (d_in < D && h_in < H && w_in < W) {
                        int input_idx = n * C_in * D * H * W +
                                      c_in * D * H * W +
                                      d_in * H * W +
                                      h_in * W +
                                      w_in;
                        int weight_idx = c_out * C_in * kernel_size * kernel_size * kernel_size +
                                       c_in * kernel_size * kernel_size * kernel_size +
                                       kd * kernel_size * kernel_size +
                                       kh * kernel_size +
                                       kw;

                        sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }

    sum += __half2float(bias[c_out]);
    output[tid] = __float2half_rn(sum);
}

__global__ void hardswish_kernel(half *data, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;

    half x = data[tid];
    float x_f = __half2float(x);
    float y = x_f + 3.0f;
    y = fmaxf(y, 0.0f);
    y = fminf(y, 6.0f);
    y = x_f * y / 6.0f;
    data[tid] = __float2half_rn(y);
}

__global__ void relu_kernel(half *data, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;

    half x = data[tid];
    data[tid] = __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
}

__global__ void softmax_kernel(half *data, int N, int C_out, int D_out, int H_out, int W_out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = D_out * H_out * W_out;
    int total_spatial = N * spatial_size;
    if (tid >= total_spatial) return;

    int n = tid / spatial_size;
    int spatial_idx = tid % spatial_size;

    int base_idx = n * C_out * spatial_size + spatial_idx;

    // Find max value
    float max_val = -INFINITY;
    for (int c = 0; c < C_out; c++) {
        int idx = base_idx + c * spatial_size;
        float val = __half2float(data[idx]);
        if (val > max_val) max_val = val;
    }

    // Compute sum of exponentials
    float sum = 0.0f;
    for (int c = 0; c < C_out; c++) {
        int idx = base_idx + c * spatial_size;
        float val = __half2float(data[idx]) - max_val;
        float exp_val = expf(val);
        sum += exp_val;
        data[idx] = __float2half_rn(exp_val);
    }

    // Normalize
    for (int c = 0; c < C_out; c++) {
        int idx = base_idx + c * spatial_size;
        data[idx] = __float2half_rn(__half2float(data[idx]) / sum);
    }
}

__global__ void mean_reduce_kernel(half *output, const half *input, int N, int C_out, int D_out, int H_out, int W_out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = N * C_out;
    if (tid >= total_pairs) return;

    int n = tid / C_out;
    int c_out = tid % C_out;

    int spatial_size = D_out * H_out * W_out;
    float sum = 0.0f;

    for (int d = 0; d < D_out; d++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                int input_idx = n * C_out * spatial_size + c_out * spatial_size + d * H_out * W_out + h * W_out + w;
                sum += __half2float(input[input_idx]);
            }
        }
    }

    output[tid] = __float2half_rn(sum / (D_out * H_out * W_out));
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, 
                               int batch_size, int in_channels, int out_channels, 
                               int depth, int height, int width, int kernel_size) {
    // Calculate output dimensions
    int D_out = depth - kernel_size + 1;
    int H_out = height - kernel_size + 1;
    int W_out = width - kernel_size + 1;

    // Allocate temporary buffer for convolution output
    half *d_conv_output;
    cudaMalloc(&d_conv_output, batch_size * out_channels * D_out * H_out * W_out * sizeof(half));

    // Launch convolution kernel
    int block_size = 256;
    int conv_output_elements = batch_size * out_channels * D_out * H_out * W_out;
    int grid_size = (conv_output_elements + block_size - 1) / block_size;
    conv3d_kernel<<<grid_size, block_size>>>(d_conv_output,
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size,
        D_out, H_out, W_out
    );

    // Apply HardSwish
    hardswish_kernel<<<grid_size, block_size>>>(d_conv_output, conv_output_elements);

    // Apply ReLU
    relu_kernel<<<grid_size, block_size>>>(d_conv_output, conv_output_elements);

    // Apply Softmax
    int softmax_elements = batch_size * D_out * H_out * W_out;
    int softmax_grid_size = (softmax_elements + block_size - 1) / block_size;
    softmax_kernel<<<softmax_grid_size, block_size>>>(d_conv_output,
        batch_size, out_channels,
        D_out, H_out, W_out
    );

    // Launch mean reduction
    int mean_output_elements = batch_size * out_channels;
    int mean_grid_size = (mean_output_elements + block_size - 1) / block_size;
    mean_reduce_kernel<<<mean_grid_size, block_size>>>(
        static_cast<half*>(output),
        d_conv_output,
        batch_size, out_channels,
        D_out, H_out, W_out
    );

    cudaFree(d_conv_output);
}
