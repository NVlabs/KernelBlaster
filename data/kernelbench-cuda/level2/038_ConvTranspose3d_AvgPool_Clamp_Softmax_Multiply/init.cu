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
#include <cub/cub.cuh>
#include <cuda_fp16.hpp>
#include <cassert>

// Utility function to calculate output dimensions
__host__ __device__ int conv_transpose_output_size(int input_size, int stride, int padding, int output_padding, int kernel_size) {
    return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding;
}

// 3D Transposed Convolution Kernel
__global__ void conv_transpose_3d_kernel(
    const half* input, const half* weight, const half* bias,
    half* output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_size, int stride, int padding, int output_padding
) {
    const int output_depth = conv_transpose_output_size(input_depth, stride, padding, output_padding, kernel_size);
    const int output_height = conv_transpose_output_size(input_height, stride, padding, output_padding, kernel_size);
    const int output_width = conv_transpose_output_size(input_width, stride, padding, output_padding, kernel_size);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_channels * output_depth * output_height * output_width;
    if (idx >= total_output) return;

    // Unravel 1D index to 5D tensor indices
    int n = idx / (out_channels * output_depth * output_height * output_width);
    int remainder = idx % (out_channels * output_depth * output_height * output_width);
    int c_out = remainder / (output_depth * output_height * output_width);
    remainder = remainder % (output_depth * output_height * output_width);
    int d_out = remainder / (output_height * output_width);
    remainder = remainder % (output_height * output_width);
    int h_out = remainder / output_width;
    int w_out = remainder % output_width;

    float acc = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int d_in = (d_out - kd + padding) / stride;
                    int h_in = (h_out - kh + padding) / stride;
                    int w_in = (w_out - kw + padding) / stride;

                    if (d_in >= 0 && d_in < input_depth &&
                        h_in >= 0 && h_in < input_height &&
                        w_in >= 0 && w_in < input_width &&
                        (d_out - kd + padding) % stride == 0 &&
                        (h_out - kh + padding) % stride == 0 &&
                        (w_out - kw + padding) % stride == 0)
                    {
                        int input_idx = n * in_channels * input_depth * input_height * input_width +
                                        c_in * input_depth * input_height * input_width +
                                        d_in * input_height * input_width +
                                        h_in * input_width +
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

    if (bias) {
        acc += __half2float(bias[c_out]);
    }

    output[idx] = __float2half_rn(acc);
}

// 3D Average Pooling Kernel
__global__ void avg_pool_3d_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int input_depth, int input_height, int input_width,
    int pool_size
) {
    int output_depth = input_depth / pool_size;
    int output_height = input_height / pool_size;
    int output_width = input_width / pool_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * channels * output_depth * output_height * output_width;
    if (idx >= total_output) return;

    int n = idx / (channels * output_depth * output_height * output_width);
    int remainder = idx % (channels * output_depth * output_height * output_width);
    int c = remainder / (output_depth * output_height * output_width);
    remainder = remainder % (output_depth * output_height * output_width);
    int d_out = remainder / (output_height * output_width);
    remainder = remainder % (output_height * output_width);
    int h_out = remainder / output_width;
    int w_out = remainder % output_width;

    float sum = 0.0f;
    int count = 0;
    for (int dd = 0; dd < pool_size; ++dd) {
        for (int dh = 0; dh < pool_size; ++dh) {
            for (int dw = 0; dw < pool_size; ++dw) {
                int d_in = d_out * pool_size + dd;
                int h_in = h_out * pool_size + dh;
                int w_in = w_out * pool_size + dw;
                if (d_in < input_depth && h_in < input_height && w_in < input_width) {
                    int input_idx = n * channels * input_depth * input_height * input_width +
                                    c * input_depth * input_height * input_width +
                                    d_in * input_height * input_width +
                                    h_in * input_width +
                                    w_in;
                    sum += __half2float(input[input_idx]);
                    count++;
                }
            }
        }
    }
    output[idx] = __float2half_rn(sum / count);
}

// Clamp Kernel
__global__ void clamp_kernel(
    half* data, float min_val, float max_val,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float val = __half2float(data[idx]);
    val = fminf(fmaxf(val, min_val), max_val);
    data[idx] = __float2half_rn(val);
}

// Softmax Kernel
__global__ void softmax_kernel(
    half* input, half* output,
    int batch_size, int channels,
    int depth, int height, int width
) {
    extern __shared__ float sdata[];
    int spatial_size = depth * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spatial_size) return;

    int n = blockIdx.y;
    int d = idx / (height * width);
    int hw = idx % (height * width);
    int h = hw / width;
    int w = hw % width;

    float max_val = sdata[threadIdx.x];
    float sum = 0.0f;

    // Compute max for numerical stability
    for (int c = 0; c < channels; ++c) {
        int input_idx = n * channels * depth * height * width +
                        c * depth * height * width +
                        d * height * width +
                        h * width +
                        w;
        float val = __half2float(input[input_idx]);
        if (c == 0) max_val = val;
        else max_val = fmaxf(max_val, val);
    }

    // Compute exponentials and sum
    for (int c = 0; c < channels; ++c) {
        int input_idx = n * channels * depth * height * width +
                        c * depth * height * width +
                        d * height * width +
                        h * width +
                        w;
        float val = __half2float(input[input_idx]);
        val = expf(val - max_val);
        sum += val;
        sdata[threadIdx.x] = val;
    }

    // Normalize and write output
    for (int c = 0; c < channels; ++c) {
        int output_idx = n * channels * depth * height * width +
                         c * depth * height * width +
                         d * height * width +
                         h * width +
                         w;
        float val = sdata[threadIdx.x] / sum;
        output[output_idx] = __float2half_rn(val);
    }
}

// Multiply by Scalar Kernel
__global__ void multiply_kernel(
    half* data, float scalar,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    data[idx] = __float2half_rn(__half2float(data[idx]) * scalar);
}

// Host launch function
void launch_gpu_implementation(
    void* output, void* input, 
    void* conv_weight, void* conv_bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, 
    int padding, int output_padding,
    int pool_kernel_size,
    float clamp_min, float clamp_max
) {
    const int batch_size = 16;
    const int input_depth = 16, input_height = 32, input_width = 32;
    const int output_depth = conv_transpose_output_size(input_depth, stride, padding, output_padding, kernel_size);
    const int output_height = conv_transpose_output_size(input_height, stride, padding, output_padding, kernel_size);
    const int output_width = conv_transpose_output_size(input_width, stride, padding, output_padding, kernel_size);
    const int pooled_depth = output_depth / pool_kernel_size;
    const int pooled_height = output_height / pool_kernel_size;
    const int pooled_width = output_width / pool_kernel_size;

    // Allocate intermediate buffers
    half *d_conv_out, *d_pool_out;
    size_t conv_size = batch_size * out_channels * output_depth * output_height * output_width * sizeof(half);
    size_t pool_size = batch_size * out_channels * pooled_depth * pooled_height * pooled_width * sizeof(half);
    cudaMalloc(&d_conv_out, conv_size);
    cudaMalloc(&d_pool_out, pool_size);

    // Launch conv transpose
    int num_blocks = (batch_size * out_channels * output_depth * output_height * output_width + 255) / 256;
    conv_transpose_3d_kernel<<<num_blocks, 256>>>(
        static_cast<const half*>(input), static_cast<const half*>(conv_weight), static_cast<const half*>(conv_bias),
        d_conv_out,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_size, stride, padding, output_padding
    );

    // Launch average pool
    num_blocks = (batch_size * out_channels * pooled_depth * pooled_height * pooled_width + 255) / 256;
    avg_pool_3d_kernel<<<num_blocks, 256>>>(
        d_conv_out, d_pool_out,
        batch_size, out_channels,
        output_depth, output_height, output_width,
        pool_kernel_size
    );

    // Launch clamp
    int num_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    num_blocks = (num_elements + 255) / 256;
    clamp_kernel<<<num_blocks, 256>>>(d_pool_out, clamp_min, clamp_max, num_elements);

    // Launch softmax
    dim3 block(256);
    dim3 grid(pooled_depth * pooled_height * pooled_width, batch_size);
    size_t shared_mem = out_channels * sizeof(float);
    softmax_kernel<<<grid, block, shared_mem>>>(d_pool_out, d_pool_out, batch_size, out_channels, pooled_depth, pooled_height, pooled_width);

    // Launch multiply by 2
    multiply_kernel<<<num_blocks, 256>>>(d_pool_out, 2.0f, num_elements);

    // Copy final result to output
    cudaMemcpy(output, d_pool_out, pool_size, cudaMemcpyDeviceToDevice);

    // Cleanup
    cudaFree(d_conv_out);
    cudaFree(d_pool_out);
}
