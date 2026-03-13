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

__global__ void conv_transpose_3d_kernel(
    const half* __restrict__ input, const half* __restrict__ weight, const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_channels * output_depth * output_height * output_width;
    if (idx >= total_output) return;

    // Unflatten indices
    int n = idx / (out_channels * output_depth * output_height * output_width);
    int c_out = (idx / (output_depth * output_height * output_width)) % out_channels;
    int d_out = (idx / (output_height * output_width)) % output_depth;
    int h_out = (idx / output_width) % output_height;
    int w_out = idx % output_width;

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int d_in = (d_out - kd + padding) / stride;
                    int h_in = (h_out - kh + padding) / stride;
                    int w_in = (w_out - kw + padding) / stride;

                    if (d_in >= 0 && h_in >= 0 && w_in >= 0 &&
                        d_in < input_depth && h_in < input_height && w_in < input_width &&
                        (d_out - kd + padding) % stride == 0 &&
                        (h_out - kh + padding) % stride == 0 &&
                        (w_out - kw + padding) % stride == 0) 
                    {
                        int input_idx = ((n * in_channels + c_in) * input_depth + d_in) * input_height * input_width +
                                      h_in * input_width + w_in;
                        
                        // Correct weight indexing for PyTorch's ConvTranspose3d layout [in_channels, out_channels, kd, kh, kw]
                        int weight_idx = ((c_in * out_channels + c_out) * kernel_size + kd) * 
                                       kernel_size * kernel_size + kh * kernel_size + kw;

                        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }

    if (bias) acc += __half2float(bias[c_out]);
    output[idx] = __float2half_rn(acc);
}

__global__ void batch_norm_kernel(
    half* __restrict__ input,
    const half* __restrict__ weight, const half* __restrict__ bias,
    const half* __restrict__ mean, const half* __restrict__ var,
    int num_elements, int num_channels, int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int c = (idx / spatial_size) % num_channels;
    float x = __half2float(input[idx]);
    float w = __half2float(weight[c]);
    float b = __half2float(bias[c]);
    float m = __half2float(mean[c]);
    float v = __half2float(var[c]);

    x = (x - m) / sqrtf(v + 1e-5f) * w + b;
    input[idx] = __float2half_rn(x);
}

__global__ void avg_pool_3d_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int batch_size, int channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * channels * output_depth * output_height * output_width;
    if (idx >= total_output) return;

    int n = idx / (channels * output_depth * output_height * output_width);
    int c = (idx / (output_depth * output_height * output_width)) % channels;
    int d_out = (idx / (output_height * output_width)) % output_depth;
    int h_out = (idx / output_width) % output_height;
    int w_out = idx % output_width;

    // Correct pooling window calculation for PyTorch-compatible behavior
    int d_start = d_out * 2;
    int d_end = min(d_start + 2, input_depth);
    int h_start = h_out * 2;
    int h_end = min(h_start + 2, input_height);
    int w_start = w_out * 2;
    int w_end = min(w_start + 2, input_width);

    float sum = 0.0f;
    int count = 0;

    for (int d = d_start; d < d_end; ++d) {
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_idx = ((n * channels + c) * input_depth + d) * input_height * input_width +
                              h * input_width + w;
                sum += __half2float(input[input_idx]);
                count++;
            }
        }
    }

    output[idx] = __float2half_rn(sum / count);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    void* bn_weight, void* bn_bias, void* bn_running_mean, void* bn_running_var,
    int64_t batch_size, int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t input_depth, int64_t input_height, int64_t input_width
) {
    // Correct output dimension calculation for transposed convolution
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;

    // Intermediate buffers
    half *d_conv, *d_pool1;
    const size_t conv_size = batch_size * out_channels * output_depth * output_height * output_width;
    cudaMalloc(&d_conv, conv_size * sizeof(half));

    // Launch transposed convolution
    const int block_size = 256;
    const dim3 grid_size((conv_size + block_size - 1) / block_size);
    conv_transpose_3d_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv,
        batch_size, in_channels, out_channels,
        kernel_size, stride, padding,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width
    );
    cudaDeviceSynchronize();

    // Launch batch normalization
    const size_t bn_size = conv_size;
    batch_norm_kernel<<<(bn_size + block_size - 1) / block_size, block_size>>>(
        d_conv,
        static_cast<const half*>(bn_weight),
        static_cast<const half*>(bn_bias),
        static_cast<const half*>(bn_running_mean),
        static_cast<const half*>(bn_running_var),
        bn_size, out_channels, output_depth * output_height * output_width
    );
    cudaDeviceSynchronize();

    // First average pooling with CORRECTED output dimensions
    const int pool1_depth = output_depth / 2;
    const int pool1_height = output_height / 2;
    const int pool1_width = output_width / 2;
    const size_t pool1_size = batch_size * out_channels * pool1_depth * pool1_height * pool1_width;
    cudaMalloc(&d_pool1, pool1_size * sizeof(half));
    
    avg_pool_3d_kernel<<<(pool1_size + block_size - 1) / block_size, block_size>>>(
        d_conv, d_pool1,
        batch_size, out_channels,
        output_depth, output_height, output_width,
        pool1_depth, pool1_height, pool1_width
    );
    cudaDeviceSynchronize();

    // Second average pooling with CORRECTED output dimensions
    const int pool2_depth = pool1_depth / 2;
    const int pool2_height = pool1_height / 2;
    const int pool2_width = pool1_width / 2;
    avg_pool_3d_kernel<<<(pool1_size + block_size - 1) / block_size, block_size>>>(
        d_pool1, static_cast<half*>(output),
        batch_size, out_channels,
        pool1_depth, pool1_height, pool1_width,
        pool2_depth, pool2_height, pool2_width
    );
    cudaDeviceSynchronize();

    cudaFree(d_conv);
    cudaFree(d_pool1);
}
