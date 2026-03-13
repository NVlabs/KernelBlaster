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

// ConvTranspose3D Kernel
__global__ void conv_transpose_3d_kernel(
    const half* input, const half* weight,
    half* output,
    int batch_size, int in_channels, int out_channels,
    int D_in, int H_in, int W_in,
    int kernel_size, int stride, int padding
) {
    const int D_out = (D_in - 1) * stride - 2 * padding + kernel_size;
    const int H_out = (H_in - 1) * stride - 2 * padding + kernel_size;
    const int W_out = (W_in - 1) * stride - 2 * padding + kernel_size;
    const int total_output = batch_size * out_channels * D_out * H_out * W_out;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_output) return;

    // Unravel indices
    int n = tid / (out_channels * D_out * H_out * W_out);
    int oc = (tid / (D_out * H_out * W_out)) % out_channels;
    int d = (tid / (H_out * W_out)) % D_out;
    int h = (tid / W_out) % H_out;
    int w = tid % W_out;

    float acc = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int input_d = (d - kd + padding) / stride;
                    int input_h = (h - kh + padding) / stride;
                    int input_w = (w - kw + padding) / stride;
                    
                    if ((d - kd + padding) % stride == 0 &&
                        (h - kh + padding) % stride == 0 &&
                        (w - kw + padding) % stride == 0 &&
                        input_d >= 0 && input_d < D_in &&
                        input_h >= 0 && input_h < H_in &&
                        input_w >= 0 && input_w < W_in) {
                        
                        int input_idx = n * in_channels * D_in * H_in * W_in +
                                      ic * D_in * H_in * W_in +
                                      input_d * H_in * W_in +
                                      input_h * W_in +
                                      input_w;
                                      
                        int weight_idx = ic * out_channels * kernel_size * kernel_size * kernel_size +
                                       oc * kernel_size * kernel_size * kernel_size +
                                       kd * kernel_size * kernel_size +
                                       kh * kernel_size +
                                       kw;

                        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }
    
    output[tid] = __float2half_rn(acc);
}

// Scale Kernel
__global__ void scale_kernel(half* data, float scale, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    data[tid] = __float2half_rn(__half2float(data[tid]) * scale);
}

// MaxPool3D Kernel
__global__ void max_pool_3d_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int D_in, int H_in, int W_in,
    int kernel_size
) {
    const int D_out = (D_in - kernel_size) / kernel_size + 1;
    const int H_out = (H_in - kernel_size) / kernel_size + 1;
    const int W_out = (W_in - kernel_size) / kernel_size + 1;
    const int total_output = batch_size * channels * D_out * H_out * W_out;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_output) return;

    int n = tid / (channels * D_out * H_out * W_out);
    int c = (tid / (D_out * H_out * W_out)) % channels;
    int d = (tid / (H_out * W_out)) % D_out;
    int h = (tid / W_out) % H_out;
    int w = tid % W_out;

    float max_val = -INFINITY;
    
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_d = d * kernel_size + kd;
                int in_h = h * kernel_size + kh;
                int in_w = w * kernel_size + kw;
                
                if (in_d < D_in && in_h < H_in && in_w < W_in) {
                    int idx = n * channels * D_in * H_in * W_in +
                              c * D_in * H_in * W_in +
                              in_d * H_in * W_in +
                              in_h * W_in +
                              in_w;
                    max_val = fmaxf(max_val, __half2float(input[idx]));
                }
            }
        }
    }
    
    output[tid] = __float2half_rn(max_val);
}

// Global Average Pool Kernel
__global__ void global_avg_pool_3d_kernel(
    const half* input, half* output,
    int batch_size, int channels,
    int D, int H, int W
) {
    const int elements = D * H * W;
    const int bid = blockIdx.x;
    const int b = bid / channels;
    const int c = bid % channels;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < elements; i += blockDim.x) {
        int d = i / (H * W);
        int h = (i / W) % H;
        int w = i % W;
        int idx = b * channels * D * H * W +
                c * D * H * W +
                d * H * W +
                h * W +
                w;
        sum += __half2float(input[idx]);
    }
    
    __shared__ float shared[256];
    shared[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[b * channels + c] = __float2half_rn(shared[0] / elements);
    }
}

// Clamp Kernel
__global__ void clamp_kernel(half* data, float min_val, float max_val, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    float val = __half2float(data[tid]);
    val = fminf(fmaxf(val, min_val), max_val);
    data[tid] = __float2half_rn(val);
}

// Host Implementation
void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    int in_channels, int out_channels, int kernel_size,
    int stride, int padding, float scale,
    int maxpool_kernel_size, float clamp_min, float clamp_max
) {
    const int batch_size = 128;
    const int D_in = 16, H_in = 32, W_in = 32;

    // Calculate intermediate dimensions
    const int D_conv = (D_in - 1) * stride - 2 * padding + kernel_size;
    const int H_conv = (H_in - 1) * stride - 2 * padding + kernel_size;
    const int W_conv = (W_in - 1) * stride - 2 * padding + kernel_size;
    const int conv_elements = batch_size * out_channels * D_conv * H_conv * W_conv;

    const int D_pool = (D_conv - maxpool_kernel_size) / maxpool_kernel_size + 1;
    const int H_pool = (H_conv - maxpool_kernel_size) / maxpool_kernel_size + 1;
    const int W_pool = (W_conv - maxpool_kernel_size) / maxpool_kernel_size + 1;
    const int pool_elements = batch_size * out_channels * D_pool * H_pool * W_pool;

    const int avg_elements = batch_size * out_channels;

    // Allocate intermediate buffers
    half *d_conv, *d_pool, *d_avg;
    cudaMalloc(&d_conv, conv_elements * sizeof(half));
    cudaMalloc(&d_pool, pool_elements * sizeof(half));
    cudaMalloc(&d_avg, avg_elements * sizeof(half));

    // Launch ConvTranspose3D
    int block_size = 256;
    int grid_size = (conv_elements + block_size - 1) / block_size;
    conv_transpose_3d_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        d_conv,
        batch_size, in_channels, out_channels,
        D_in, H_in, W_in,
        kernel_size, stride, padding
    );

    // Apply scaling
    scale_kernel<<<grid_size, block_size>>>(d_conv, scale, conv_elements);

    // MaxPool3D
    grid_size = (pool_elements + block_size - 1) / block_size;
    max_pool_3d_kernel<<<grid_size, block_size>>>(
        d_conv, d_pool,
        batch_size, out_channels,
        D_conv, H_conv, W_conv,
        maxpool_kernel_size
    );
    cudaFree(d_conv);

    // Global Average Pool
    grid_size = batch_size * out_channels;
    global_avg_pool_3d_kernel<<<grid_size, block_size>>>(
        d_pool, d_avg,
        batch_size, out_channels,
        D_pool, H_pool, W_pool
    );
    cudaFree(d_pool);

    // Clamp and copy to output
    grid_size = (avg_elements + block_size - 1) / block_size;
    clamp_kernel<<<grid_size, block_size>>>(d_avg, clamp_min, clamp_max, avg_elements);
    cudaMemcpy(output, d_avg, avg_elements * sizeof(half), cudaMemcpyDeviceToDevice);
    cudaFree(d_avg);
}
