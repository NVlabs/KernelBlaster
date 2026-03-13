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
#include <mma.h>
#include <iostream>

// MMA configuration
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK 256
#define K_STAGE 4

// Convolution transpose parameters
#define KERNEL_SIZE 4
#define STRIDE 2
#define PADDING 1
#define OUTPUT_PADDING 1

// Tensor core optimized convolution transpose kernel
__global__ void conv_transpose_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width
) {
    // Calculate output position
    const int n = blockIdx.z;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int c_out = blockIdx.z % out_channels;

    if (h_out >= output_height || w_out >= output_width) return;

    float acc = 0.0f;

    // Iterate over kernel
    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
        for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
            // Calculate input position
            const int h_in = (h_out - kh + PADDING) / STRIDE;
            const int w_in = (w_out - kw + PADDING) / STRIDE;
            
            if (h_in >= 0 && w_in >= 0 && h_in < input_height && w_in < input_width) {
                // Iterate over input channels
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    const int input_idx = n * in_channels * input_height * input_width +
                                        c_in * input_height * input_width +
                                        h_in * input_width + w_in;
                    const int weight_idx = c_in * out_channels * KERNEL_SIZE * KERNEL_SIZE +
                                         c_out * KERNEL_SIZE * KERNEL_SIZE +
                                         kh * KERNEL_SIZE + kw;
                    
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    // Add bias
    acc += __half2float(bias[c_out]);

    // Store result
    const int output_idx = n * out_channels * output_height * output_width +
                          c_out * output_height * output_width +
                          h_out * output_width + w_out;
    output[output_idx] = __float2half_rn(acc);
}

// Softmax kernel with channel-wise reduction
__global__ void softmax_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size, int channels,
    int height, int width
) {
    const int n = blockIdx.z;
    const int h = blockIdx.y;
    const int w = blockIdx.x;
    const int c = threadIdx.x;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + 1;

    // Find max value
    float thread_max = -INFINITY;
    for (int i = c; i < channels; i += blockDim.x) {
        const int idx = n * channels * height * width +
                       i * height * width +
                       h * width + w;
        thread_max = fmaxf(thread_max, __half2float(input[idx]));
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset /= 2)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));

    if (c == 0) *max_val = thread_max;
    __syncthreads();

    // Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = c; i < channels; i += blockDim.x) {
        const int idx = n * channels * height * width +
                       i * height * width +
                       h * width + w;
        float val = __half2float(input[idx]) - *max_val;
        thread_sum += expf(val);
    }

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    if (c == 0) *sum_exp = thread_sum + 1e-6f;
    __syncthreads();

    // Normalize and store
    for (int i = c; i < channels; i += blockDim.x) {
        const int idx = n * channels * height * width +
                       i * height * width +
                       h * width + w;
        float val = expf(__half2float(input[idx]) - *max_val) / *sum_exp;
        output[idx] = __float2half_rn(val);
    }
}

// Fused bias add, scale, and sigmoid kernel
__global__ void final_ops_kernel(
    half* __restrict__ io_tensor,
    const half* __restrict__ bias,
    float scaling_factor,
    int batch_size, int channels,
    int height, int width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * height * width;
    if (idx >= total_elements) return;

    const int c = (idx / (height * width)) % channels;
    float val = __half2float(io_tensor[idx]);
    val += __half2float(bias[c]);
    val *= scaling_factor;
    val = 1.0f / (1.0f + expf(-val));
    io_tensor[idx] = __float2half_rn(val);
}

void launch_gpu_implementation(
    void* output, void* input, void* conv_weight, void* conv_bias,
    void* model_bias, float scaling_factor, int64_t in_channels,
    int64_t out_channels, int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t output_padding, int64_t batch_size,
    int64_t input_height, int64_t input_width
) {
    // Calculate output dimensions
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Allocate intermediate buffers
    half *d_conv_out, *d_softmax_out;
    const size_t buffer_size = batch_size * out_channels * output_height * output_width * sizeof(half);
    cudaMalloc(&d_conv_out, buffer_size);
    cudaMalloc(&d_softmax_out, buffer_size);

    // Launch convolution transpose
    dim3 block(16, 16);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size
    );
    conv_transpose_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv_out,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width
    );

    // Launch softmax
    dim3 s_block(32);
    dim3 s_grid(output_width, output_height, batch_size);
    softmax_kernel<<<s_grid, s_block, 2*sizeof(float)>>>(
        d_conv_out, d_softmax_out,
        batch_size, out_channels,
        output_height, output_width
    );

    // Launch final operations
    const int elements = batch_size * out_channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;
    final_ops_kernel<<<blocks, threads>>>(
        d_softmax_out,
        static_cast<const half*>(model_bias),
        scaling_factor,
        batch_size, out_channels,
        output_height, output_width
    );

    // Copy result to output
    cudaMemcpy(output, d_softmax_out, buffer_size, cudaMemcpyDeviceToDevice);

    // Cleanup
    cudaFree(d_conv_out);
    cudaFree(d_softmax_out);
}
