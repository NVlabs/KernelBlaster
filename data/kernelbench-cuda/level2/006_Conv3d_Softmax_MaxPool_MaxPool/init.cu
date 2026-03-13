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

// Convolution kernel
__global__ void conv3d_kernel(
    const half* input, const half* weight, const half* bias,
    half* output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_size
) {
    int output_depth = input_depth - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_depth * output_height * output_width) return;

    int b = idx / (out_channels * output_depth * output_height * output_width);
    int rem = idx % (out_channels * output_depth * output_height * output_width);
    int oc = rem / (output_depth * output_height * output_width);
    rem = rem % (output_depth * output_height * output_width);
    int d = rem / (output_height * output_width);
    rem = rem % (output_height * output_width);
    int h = rem / output_width;
    int w = rem % output_width;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int input_d = d + kd;
                    int input_h = h + kh;
                    int input_w = w + kw;

                    if (input_d < input_depth && input_h < input_height && input_w < input_width) {
                        int input_idx = b * in_channels * input_depth * input_height * input_width +
                                      ic * input_depth * input_height * input_width +
                                      input_d * input_height * input_width +
                                      input_h * input_width +
                                      input_w;

                        int weight_idx = oc * in_channels * kernel_size * kernel_size * kernel_size +
                                       ic * kernel_size * kernel_size * kernel_size +
                                       kd * kernel_size * kernel_size +
                                       kh * kernel_size +
                                       kw;

                        sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }
    sum += __half2float(bias[oc]);
    output[idx] = __float2half_rn(sum);
}

// Softmax kernel
__global__ void softmax_kernel(
    half* output, const half* input,
    int batch_size, int channels,
    int depth, int height, int width
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int h = blockIdx.z;
    int w = threadIdx.x;

    if (b >= batch_size || d >= depth || h >= height || w >= width) return;

    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = &shared[blockDim.x];

    int spatial_size = d * height * width + h * width + w;
    int base_idx = b * channels * depth * height * width + spatial_size;

    float max_val = -INFINITY;
    for (int c = 0; c < channels; ++c) {
        int idx = base_idx + c * depth * height * width;
        max_val = fmaxf(max_val, __half2float(input[idx]));
    }

    s_max[w] = max_val;
    __syncthreads();

    max_val = s_max[w];
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        int idx = base_idx + c * depth * height * width;
        sum += expf(__half2float(input[idx]) - max_val);
    }

    s_sum[w] = sum;
    __syncthreads();

    for (int c = 0; c < channels; ++c) {
        int idx = base_idx + c * depth * height * width;
        output[idx] = __float2half_rn(expf(__half2float(input[idx]) - max_val) / sum);
    }
}

// MaxPool3d kernel
__global__ void maxpool3d_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int input_depth, int input_height, int input_width,
    int kernel_size, int stride
) {
    int output_depth = (input_depth - kernel_size) / stride + 1;
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_depth * output_height * output_width) return;

    int b = idx / (channels * output_depth * output_height * output_width);
    int rem = idx % (channels * output_depth * output_height * output_width);
    int c = rem / (output_depth * output_height * output_width);
    rem = rem % (output_depth * output_height * output_width);
    int d = rem / (output_height * output_width);
    rem = rem % (output_height * output_width);
    int h = rem / output_width;
    int w = rem % output_width;

    int input_d_start = d * stride;
    int input_h_start = h * stride;
    int input_w_start = w * stride;

    half max_val = __float2half(-INFINITY);
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int input_d = input_d_start + kd;
                int input_h = input_h_start + kh;
                int input_w = input_w_start + kw;

                if (input_d < input_depth && input_h < input_height && input_w < input_width) {
                    int input_idx = b * channels * input_depth * input_height * input_width +
                                  c * input_depth * input_height * input_width +
                                  input_d * input_height * input_width +
                                  input_h * input_width +
                                  input_w;

                    max_val = __hmax(max_val, input[input_idx]);
                }
            }
        }
    }
    output[idx] = max_val;
}

void launch_gpu_implementation(void* output, void* input, int in_channels, int out_channels, 
                              int kernel_size, int pool_kernel_size, void* conv_weight, void* conv_bias) {
    const int batch_size = 128;
    const int input_depth = 16, input_height = 32, input_width = 32;

    // Conv output dimensions
    const int conv_depth = input_depth - kernel_size + 1;
    const int conv_height = input_height - kernel_size + 1;
    const int conv_width = input_width - kernel_size + 1;
    const size_t conv_size = batch_size * out_channels * conv_depth * conv_height * conv_width;

    // Allocate intermediate buffers
    half *d_conv, *d_pool1;
    cudaMalloc(&d_conv, conv_size * sizeof(half));
    
    // Launch conv kernel
    const int block_size = 256;
    dim3 grid_conv((conv_size + block_size - 1) / block_size);
    conv3d_kernel<<<grid_conv, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_size
    );

    // Launch softmax kernel
    dim3 grid_softmax(batch_size, conv_depth, conv_height);
    size_t smem = 2 * conv_width * sizeof(float);
    softmax_kernel<<<grid_softmax, conv_width, smem>>>(
        d_conv, d_conv,
        batch_size, out_channels,
        conv_depth, conv_height, conv_width
    );

    // First maxpool
    const int pool1_depth = conv_depth / pool_kernel_size;
    const int pool1_height = conv_height / pool_kernel_size;
    const int pool1_width = conv_width / pool_kernel_size;
    const size_t pool1_size = batch_size * out_channels * pool1_depth * pool1_height * pool1_width;
    cudaMalloc(&d_pool1, pool1_size * sizeof(half));
    
    dim3 grid_pool1((pool1_size + block_size - 1) / block_size);
    maxpool3d_kernel<<<grid_pool1, block_size>>>(
        d_conv, d_pool1,
        batch_size, out_channels,
        conv_depth, conv_height, conv_width,
        pool_kernel_size, pool_kernel_size
    );

    // Second maxpool
    const int pool2_depth = pool1_depth / pool_kernel_size;
    const int pool2_height = pool1_height / pool_kernel_size;
    const int pool2_width = pool1_width / pool_kernel_size;
    dim3 grid_pool2((batch_size * out_channels * pool2_depth * pool2_height * pool2_width + block_size - 1) / block_size);
    maxpool3d_kernel<<<grid_pool2, block_size>>>(
        d_pool1, static_cast<half*>(output),
        batch_size, out_channels,
        pool1_depth, pool1_height, pool1_width,
        pool_kernel_size, pool_kernel_size
    );

    cudaFree(d_conv);
    cudaFree(d_pool1);
    cudaDeviceSynchronize();
}
