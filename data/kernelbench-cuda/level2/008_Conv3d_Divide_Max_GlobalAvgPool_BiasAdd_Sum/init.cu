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

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define BLOCK_ROWS 256
#define BLOCK_COLS 128
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK 256
#define K_STAGE 4
#define MAX_THREADS 256

// 3D Convolution with proper NDHWC data layout and bias addition
__global__ void conv3d_ndhwc_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ conv_bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_d, int input_h, int input_w,
    int kernel_d, int kernel_h, int kernel_w
) {
    const int output_d = input_d - kernel_d + 1;
    const int output_h = input_h - kernel_h + 1;
    const int output_w = input_w - kernel_w + 1;
    const int output_size = batch_size * output_d * output_h * output_w * out_channels;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    // NDHWC output indexing
    const int n = tid / (output_d * output_h * output_w * out_channels);
    int residual = tid % (output_d * output_h * output_w * out_channels);
    const int d_out = residual / (output_h * output_w * out_channels);
    residual %= (output_h * output_w * out_channels);
    const int h_out = residual / (output_w * out_channels);
    residual %= (output_w * out_channels);
    const int w_out = residual / out_channels;
    const int c_out = residual % out_channels;

    float acc = 0.0f;
    for (int kd = 0; kd < kernel_d; kd++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                for (int c_in = 0; c_in < in_channels; c_in++) {
                    const int d_in = d_out + kd;
                    const int h_in = h_out + kh;
                    const int w_in = w_out + kw;
                    
                    if (d_in < input_d && h_in < input_h && w_in < input_w) {
                        // NDHWC input indexing
                        const int input_idx = n * input_d * input_h * input_w * in_channels +
                                            d_in * input_h * input_w * in_channels +
                                            h_in * input_w * in_channels +
                                            w_in * in_channels +
                                            c_in;
                                            
                        // OIDHW weight indexing
                        const int weight_idx = c_out * in_channels * kernel_d * kernel_h * kernel_w +
                                            c_in * kernel_d * kernel_h * kernel_w +
                                            kd * kernel_h * kernel_w +
                                            kh * kernel_w +
                                            kw;

                        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    acc += __half2float(conv_bias[c_out]);
    output[tid] = __float2half_rn(acc);
}

// Element-wise division kernel
__global__ void divide_kernel(half* data, float divisor, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    data[tid] = __float2half_rn(__half2float(data[tid]) / divisor);
}

// 3D max pooling with NDHWC layout
__global__ void max_pool3d_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int input_d, int input_h, int input_w,
    int pool_d, int pool_h, int pool_w
) {
    const int output_d = input_d / pool_d;
    const int output_h = input_h / pool_h;
    const int output_w = input_w / pool_w;
    const int output_size = batch_size * output_d * output_h * output_w * channels;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    // NDHWC output indexing
    const int n = tid / (output_d * output_h * output_w * channels);
    int residual = tid % (output_d * output_h * output_w * channels);
    const int d_out = residual / (output_h * output_w * channels);
    residual %= (output_h * output_w * channels);
    const int h_out = residual / (output_w * channels);
    residual %= (output_w * channels);
    const int w_out = residual / channels;
    const int c_out = residual % channels;

    half max_val = __float2half(-INFINITY);
    for (int kd = 0; kd < pool_d; kd++) {
        for (int kh = 0; kh < pool_h; kh++) {
            for (int kw = 0; kw < pool_w; kw++) {
                const int d_in = d_out * pool_d + kd;
                const int h_in = h_out * pool_h + kh;
                const int w_in = w_out * pool_w + kw;
                
                if (d_in < input_d && h_in < input_h && w_in < input_w) {
                    const int input_idx = n * input_d * input_h * input_w * channels +
                                        d_in * input_h * input_w * channels +
                                        h_in * input_w * channels +
                                        w_in * channels +
                                        c_out;
                    max_val = __hmax(max_val, input[input_idx]);
                }
            }
        }
    }
    output[tid] = max_val;
}

// Global average pooling with NDHWC layout
__global__ void global_avg_pool3d_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int input_d, int input_h, int input_w
) {
    const int output_size = batch_size * channels;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    const int n = tid / channels;
    const int c = tid % channels;
    float sum = 0.0f;
    const int spatial_size = input_d * input_h * input_w;

    for (int d = 0; d < input_d; d++) {
        for (int h = 0; h < input_h; h++) {
            for (int w = 0; w < input_w; w++) {
                const int input_idx = n * input_d * input_h * input_w * channels +
                                    d * input_h * input_w * channels +
                                    h * input_w * channels +
                                    w * channels +
                                    c;
                sum += __half2float(input[input_idx]);
            }
        }
    }
    output[tid] = __float2half_rn(sum / spatial_size);
}

// Bias addition kernel
__global__ void add_bias_kernel(
    half* data, const half* bias,
    int batch_size, int channels
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * channels) return;

    const int c = tid % channels;
    data[tid] = __hadd(data[tid], bias[c]);
}

// Sum reduction kernel
__global__ void sum_reduction_kernel(
    const half* input, half* output,
    int batch_size, int channels
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    float sum = 0.0f;
    for (int c = 0; c < channels; c++) {
        sum += __half2float(input[tid * channels + c]);
    }
    output[tid] = __float2half_rn(sum);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* model_bias,
    float divisor,
    int kernel_d, int kernel_h, int kernel_w,
    int pool_d, int pool_h, int pool_w,
    int sum_dim,
    int in_channels, int out_channels,
    int batch_size, int depth, int height, int width
) {
    // Calculate intermediate dimensions
    const int conv_out_d = depth - kernel_d + 1;
    const int conv_out_h = height - kernel_h + 1;
    const int conv_out_w = width - kernel_w + 1;
    const int conv_out_size = batch_size * conv_out_d * conv_out_h * conv_out_w * out_channels;

    // Allocate device memory
    half *d_conv, *d_div, *d_pool, *d_global, *d_biased;
    cudaMalloc(&d_conv, conv_out_size * sizeof(half));
    
    // 1. Convolution with bias
    dim3 block(MAX_THREADS);
    dim3 grid((conv_out_size + block.x - 1) / block.x);
    conv3d_ndhwc_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_conv,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w
    );

    // 2. Division by scalar
    divide_kernel<<<grid, block>>>(d_conv, divisor, conv_out_size);

    // 3. Max pooling
    const int pool_out_d = conv_out_d / pool_d;
    const int pool_out_h = conv_out_h / pool_h;
    const int pool_out_w = conv_out_w / pool_w;
    const int pool_out_size = batch_size * pool_out_d * pool_out_h * pool_out_w * out_channels;
    cudaMalloc(&d_pool, pool_out_size * sizeof(half));
    max_pool3d_kernel<<<grid, block>>>(
        d_conv, d_pool,
        batch_size, out_channels,
        conv_out_d, conv_out_h, conv_out_w,
        pool_d, pool_h, pool_w
    );
    cudaFree(d_conv);

    // 4. Global average pooling
    const int global_size = batch_size * out_channels;
    cudaMalloc(&d_global, global_size * sizeof(half));
    global_avg_pool3d_kernel<<<grid, block>>>(
        d_pool, d_global,
        batch_size, out_channels,
        pool_out_d, pool_out_h, pool_out_w
    );
    cudaFree(d_pool);

    // 5. Add model bias
    cudaMalloc(&d_biased, global_size * sizeof(half));
    cudaMemcpy(d_biased, d_global, global_size * sizeof(half), cudaMemcpyDeviceToDevice);
    add_bias_kernel<<<grid, block>>>(d_biased, static_cast<const half*>(model_bias), batch_size, out_channels);
    cudaFree(d_global);

    // 6. Sum reduction
    sum_reduction_kernel<<<dim3((batch_size + 255)/256), dim3(256)>>>(
        d_biased, static_cast<half*>(output), batch_size, out_channels
    );
    
    // Cleanup
    cudaFree(d_biased);
    cudaDeviceSynchronize();
}
