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

__host__ __device__ int output_size(int input_size, int kernel_size, int stride, int padding) {
    return (input_size - 1) * stride + kernel_size - 2 * padding;
}

__global__ void conv_transpose_3d_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_size, int stride, int padding
) {
    const int output_depth = output_size(input_depth, kernel_size, stride, padding);
    const int output_height = output_size(input_height, kernel_size, stride, padding);
    const int output_width = output_size(input_width, kernel_size, stride, padding);
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    if (tid >= total_elements) return;

    // Unravel output index
    const int n = tid / (out_channels * output_depth * output_height * output_width);
    int remainder = tid % (out_channels * output_depth * output_height * output_width);
    const int c_out = remainder / (output_depth * output_height * output_width);
    remainder %= output_depth * output_height * output_width;
    const int d_out = remainder / (output_height * output_width);
    remainder %= output_height * output_width;
    const int h_out = remainder / output_width;
    const int w_out = remainder % output_width;

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int d_in = (d_out - kd + padding) / stride;
                    const int h_in = (h_out - kh + padding) / stride;
                    const int w_in = (w_out - kw + padding) / stride;

                    if (d_in >= 0 && h_in >= 0 && w_in >= 0 &&
                        d_in < input_depth && h_in < input_height && w_in < input_width &&
                        (d_out - kd + padding) % stride == 0 &&
                        (h_out - kh + padding) % stride == 0 &&
                        (w_out - kw + padding) % stride == 0) {

                        const int input_idx = n * in_channels * input_depth * input_height * input_width +
                                            c_in * input_depth * input_height * input_width +
                                            d_in * input_height * input_width +
                                            h_in * input_width +
                                            w_in;

                        // CORRECTED WEIGHT INDEXING
                        const int weight_idx = c_in * out_channels * kernel_size * kernel_size * kernel_size +
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

    acc += __half2float(bias[c_out]);
    output[tid] = __float2half_rn(acc);
}

__global__ void swish_kernel(half* data, int num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    const float x = __half2float(data[tid]);
    data[tid] = __float2half_rn(x * (1.0f / (1.0f + expf(-x))));
}

__global__ void group_norm_sum_kernel(
    const half* __restrict__ input,
    float* __restrict__ sum,
    float* __restrict__ sum_sq,
    int batch_size, int channels, int depth, int height, int width,
    int groups
) {
    const int channels_per_group = channels / groups;
    const int elements_per_group = batch_size * depth * height * width * channels_per_group;
    const int group = blockIdx.x;
    const int tid = threadIdx.x;

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        const int n = i / (depth * height * width * channels_per_group);
        int remainder = i % (depth * height * width * channels_per_group);
        const int d = remainder / (height * width * channels_per_group);
        remainder %= height * width * channels_per_group;
        const int h = remainder / (width * channels_per_group);
        remainder %= width * channels_per_group;
        const int w = remainder / channels_per_group;
        const int c = group * channels_per_group + (remainder % channels_per_group);

        const int idx = n * channels * depth * height * width +
                       c * depth * height * width +
                       d * height * width +
                       h * width +
                       w;

        const float val = __half2float(input[idx]);
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    // Block reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];
    shared_sum[tid] = thread_sum;
    shared_sum_sq[tid] = thread_sum_sq;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sum_sq[tid] += shared_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[group] = shared_sum[0];       // CORRECTED: Direct assignment
        sum_sq[group] = shared_sum_sq[0]; // instead of atomicAdd
    }
}

__global__ void group_norm_apply_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    const float* __restrict__ sum,
    const float* __restrict__ sum_sq,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    float eps, int batch_size, int channels,
    int depth, int height, int width, int groups
) {
    const int channels_per_group = channels / groups;
    const int elements_per_group = batch_size * depth * height * width * channels_per_group;
    const int group = blockIdx.x;
    const int tid = threadIdx.x;

    const float mean = sum[group] / elements_per_group;
    const float var = sum_sq[group] / elements_per_group - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        const int n = i / (depth * height * width * channels_per_group);
        int remainder = i % (depth * height * width * channels_per_group);
        const int d = remainder / (height * width * channels_per_group);
        remainder %= height * width * channels_per_group;
        const int h = remainder / (width * channels_per_group);
        remainder %= width * channels_per_group;
        const int w = remainder / channels_per_group;
        const int c = group * channels_per_group + (remainder % channels_per_group);

        const int idx = n * channels * depth * height * width +
                       c * depth * height * width +
                       d * height * width +
                       h * width +
                       w;

        const float val = __half2float(input[idx]);
        const float norm = (val - mean) * inv_std;
        const float scaled = norm * __half2float(gamma[c]) + __half2float(beta[c]);
        output[idx] = __float2half_rn(scaled);
    }
}

__global__ void hardswish_kernel(half* data, int num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    const float x = __half2float(data[tid]);
    data[tid] = __float2half_rn(x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f);
}

void launch_gpu_implementation(
    void* output, void* input,
    const void* conv_weight, const void* conv_bias,
    const void* gn_weight, const void* gn_bias,
    int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t groups, double eps,
    int64_t batch_size, int64_t input_depth,
    int64_t input_height, int64_t input_width
) {
    // Calculate output dimensions
    const int output_depth = output_size(input_depth, kernel_size, stride, padding);
    const int output_height = output_size(input_height, kernel_size, stride, padding);
    const int output_width = output_size(input_width, kernel_size, stride, padding);
    const int conv_output_size = batch_size * out_channels * output_depth * output_height * output_width;

    // Allocate device memory
    half* d_conv_output;
    cudaMalloc(&d_conv_output, conv_output_size * sizeof(half));

    // Launch transposed convolution
    const int block_size = 256;
    int grid_size = (conv_output_size + block_size - 1) / block_size;
    conv_transpose_3d_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv_output,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_size, stride, padding
    );

    // Apply Swish activation
    swish_kernel<<<grid_size, block_size>>>(d_conv_output, conv_output_size);

    // Prepare GroupNorm buffers
    float *d_sum, *d_sum_sq;
    cudaMalloc(&d_sum, groups * sizeof(float));
    cudaMalloc(&d_sum_sq, groups * sizeof(float));
    cudaMemset(d_sum, 0, groups * sizeof(float));
    cudaMemset(d_sum_sq, 0, groups * sizeof(float));

    // Calculate GroupNorm statistics
    group_norm_sum_kernel<<<groups, 256>>>(
        d_conv_output, d_sum, d_sum_sq,
        batch_size, out_channels,
        output_depth, output_height, output_width,
        groups
    );
    cudaDeviceSynchronize(); // Critical synchronization

    // Apply GroupNorm
    group_norm_apply_kernel<<<groups, 256>>>(
        d_conv_output, d_conv_output,
        d_sum, d_sum_sq,
        static_cast<const half*>(gn_weight),
        static_cast<const half*>(gn_bias),
        static_cast<float>(eps), batch_size, out_channels,
        output_depth, output_height, output_width,
        groups
    );

    // Apply HardSwish activation
    hardswish_kernel<<<grid_size, block_size>>>(d_conv_output, conv_output_size);

    // Copy final result
    cudaMemcpy(output, d_conv_output, conv_output_size * sizeof(half), cudaMemcpyDeviceToDevice);

    // Cleanup
    cudaFree(d_conv_output);
    cudaFree(d_sum);
    cudaFree(d_sum_sq);
}
