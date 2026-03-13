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

__global__ void conv3d_kernel(const half* input, const half* weight, const half* bias, half* output,
                              int batch_size, int in_channels, int out_channels,
                              int D, int H, int W, int kernel_size) {
    // Calculate output dimensions
    int D_out = D - kernel_size + 1;
    int H_out = H - kernel_size + 1;
    int W_out = W - kernel_size + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * D_out * H_out * W_out;
    if (idx >= total_elements) return;

    // Compute output indices
    int n = idx / (out_channels * D_out * H_out * W_out);
    int residual = idx % (out_channels * D_out * H_out * W_out);
    int c_out = residual / (D_out * H_out * W_out);
    residual %= (D_out * H_out * W_out);
    int d_out = residual / (H_out * W_out);
    residual %= (H_out * W_out);
    int h_out = residual / W_out;
    int w_out = residual % W_out;

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            int d_in = d_out + kd;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int h_in = h_out + kh;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int w_in = w_out + kw;

                    if (d_in < D && h_in < H && w_in < W) {
                        int input_idx = n * in_channels * D * H * W
                                      + c_in * D * H * W
                                      + d_in * H * W
                                      + h_in * W
                                      + w_in;
                        int weight_idx = c_out * in_channels * kernel_size * kernel_size * kernel_size
                                       + c_in * kernel_size * kernel_size * kernel_size
                                       + kd * kernel_size * kernel_size
                                       + kh * kernel_size
                                       + kw;

                        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }

    if (bias) {
        acc += __half2float(bias[c_out]);
    }

    output[idx] = __float2half_rn(acc);
}

__global__ void min_reduction_kernel(const half* input, half* output,
                                     int batch_size, int out_channels,
                                     int D_out, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * H_out * W_out;
    if (idx >= total_elements) return;

    int n = idx / (out_channels * H_out * W_out);
    int residual = idx % (out_channels * H_out * W_out);
    int c_out = residual / (H_out * W_out);
    residual %= (H_out * W_out);
    int h_out = residual / W_out;
    int w_out = residual % W_out;

    half min_val = __float2half(INFINITY);
    for (int d_out = 0; d_out < D_out; ++d_out) {
        int input_idx = n * out_channels * D_out * H_out * W_out
                      + c_out * D_out * H_out * W_out
                      + d_out * H_out * W_out
                      + h_out * W_out
                      + w_out;
        half val = input[input_idx];
        if (__hlt(val, min_val)) {
            min_val = val;
        }
    }

    output[idx] = min_val;
}

__global__ void softmax_kernel(half* output, int batch_size, int out_channels, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_positions = batch_size * H_out * W_out;
    if (idx >= total_positions) return;

    int n = idx / (H_out * W_out);
    int residual = idx % (H_out * W_out);
    int h_out = residual / W_out;
    int w_out = residual % W_out;

    // Find max value
    float max_val = -INFINITY;
    for (int c_out = 0; c_out < out_channels; ++c_out) {
        int output_idx = n * out_channels * H_out * W_out
                       + c_out * H_out * W_out
                       + h_out * W_out
                       + w_out;
        float val = __half2float(output[output_idx]);
        max_val = fmaxf(max_val, val);
    }

    // Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int c_out = 0; c_out < out_channels; ++c_out) {
        int output_idx = n * out_channels * H_out * W_out
                       + c_out * H_out * W_out
                       + h_out * W_out
                       + w_out;
        float val = __half2float(output[output_idx]);
        sum_exp += expf(val - max_val);
    }

    // Compute softmax values
    for (int c_out = 0; c_out < out_channels; ++c_out) {
        int output_idx = n * out_channels * H_out * W_out
                       + c_out * H_out * W_out
                       + h_out * W_out
                       + w_out;
        float val = __half2float(output[output_idx]);
        float softmax_val = expf(val - max_val) / sum_exp;
        output[output_idx] = __float2half_rn(softmax_val);
    }
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, 
                              int in_channels, int out_channels, int kernel_size, int dim) {
    const int batch_size = 128;
    const int D = 16, H = 32, W = 32;
    const int D_out = D - kernel_size + 1;
    const int H_out = H - kernel_size + 1;
    const int W_out = W - kernel_size + 1;

    // Allocate intermediate buffers
    half *d_conv, *d_min;
    size_t conv_size = batch_size * out_channels * D_out * H_out * W_out * sizeof(half);
    size_t min_size = batch_size * out_channels * H_out * W_out * sizeof(half);
    cudaMalloc(&d_conv, conv_size);
    cudaMalloc(&d_min, min_size);

    // Launch convolution
    int block_size = 256;
    int grid_size = (batch_size * out_channels * D_out * H_out * W_out + block_size - 1) / block_size;
    conv3d_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        d_conv,
        batch_size, in_channels, out_channels,
        D, H, W, kernel_size
    );

    // Launch min reduction
    grid_size = (batch_size * out_channels * H_out * W_out + block_size - 1) / block_size;
    min_reduction_kernel<<<grid_size, block_size>>>(
        d_conv, d_min,
        batch_size, out_channels,
        D_out, H_out, W_out
    );

    // Launch softmax
    grid_size = (batch_size * H_out * W_out + block_size - 1) / block_size;
    softmax_kernel<<<grid_size, block_size>>>(
        d_min,
        batch_size, out_channels,
        H_out, W_out
    );

    // Copy final result to output
    cudaMemcpy(output, d_min, min_size, cudaMemcpyDeviceToDevice);

    // Cleanup
    cudaFree(d_conv);
    cudaFree(d_min);
}
