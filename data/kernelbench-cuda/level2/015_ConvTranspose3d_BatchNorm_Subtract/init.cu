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

// Block reduction helper functions
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ConvTranspose3D using GEMM
__global__ void conv_transpose_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width,
    int kernel_size, int stride, int padding
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements_per_sample = out_channels * output_depth * output_height * output_width;
    const int sample = tid / elements_per_sample;
    const int element = tid % elements_per_sample;
    if (sample >= batch_size || element >= elements_per_sample) return;

    const int oc = element / (output_depth * output_height * output_width);
    const int d_out = (element % (output_depth * output_height * output_width)) / (output_height * output_width);
    const int h_out = (element % (output_height * output_width)) / output_width;
    const int w_out = element % output_width;

    float acc = 0.0f;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int d_in = (d_out - kd + padding) / stride;
                const int h_in = (h_out - kh + padding) / stride;
                const int w_in = (w_out - kw + padding) / stride;
                
                if (d_in < 0 || h_in < 0 || w_in < 0) continue;
                if (d_in >= input_depth || h_in >= input_height || w_in >= input_width) continue;
                if ((d_out - kd + padding) % stride != 0) continue;
                if ((h_out - kh + padding) % stride != 0) continue;
                if ((w_out - kw + padding) % stride != 0) continue;

                for (int ic = 0; ic < in_channels; ++ic) {
                    const int input_idx = sample * in_channels * input_depth * input_height * input_width +
                                        ic * input_depth * input_height * input_width +
                                        d_in * input_height * input_width +
                                        h_in * input_width +
                                        w_in;
                    const int weight_idx = ic * out_channels * kernel_size * kernel_size * kernel_size +
                                         oc * kernel_size * kernel_size * kernel_size +
                                         kd * kernel_size * kernel_size +
                                         kh * kernel_size +
                                         kw;
                    acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }
    acc += __half2float(bias[oc]);
    output[tid] = __float2half_rn(acc);
}

// BatchNorm and Mean Subtraction
__global__ void batchnorm_subtract_mean_kernel(
    half* output,
    const half* input,
    const half* gamma,
    const half* beta,
    int batch_size, int channels,
    int depth, int height, int width
) {
    const int spatial_size = depth * height * width;
    const int c = blockIdx.x;
    const int n = blockIdx.y;
    const int tid = threadIdx.x;

    // BatchNorm: Compute mean and variance per channel
    __shared__ float mean_shared, var_shared, gamma_shared, beta_shared;
    if (tid == 0) {
        float sum = 0.0f, sum_sq = 0.0f;
        for (int i = 0; i < batch_size * spatial_size; ++i) {
            const int idx = n * channels * spatial_size + c * spatial_size + (i % spatial_size);
            float val = __half2float(input[idx]);
            sum += val;
            sum_sq += val * val;
        }
        mean_shared = sum / (batch_size * spatial_size);
        var_shared = sum_sq / (batch_size * spatial_size) - mean_shared * mean_shared;
        gamma_shared = __half2float(gamma[c]);
        beta_shared = __half2float(beta[c]);
    }
    __syncthreads();

    // Apply BatchNorm and compute spatial mean
    float spatial_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        const int idx = n * channels * spatial_size + c * spatial_size + i;
        float val = (__half2float(input[idx]) - mean_shared) * rsqrtf(var_shared + 1e-5f) * gamma_shared + beta_shared;
        spatial_sum += val;
        output[idx] = __float2half_rn(val);
    }

    // Compute spatial mean
    __shared__ float spatial_mean;
    float block_sum = blockReduceSum(spatial_sum);
    if (tid == 0) spatial_mean = block_sum / spatial_size;
    __syncthreads();

    // Subtract spatial mean
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        const int idx = n * channels * spatial_size + c * spatial_size + i;
        float val = __half2float(output[idx]) - spatial_mean;
        output[idx] = __float2half_rn(val);
    }
}

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    void* bn_weight, void* bn_bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int batch_size, int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width
) {
    // ConvTranspose
    const int conv_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int grid_size = (conv_elements + block_size - 1) / block_size;
    conv_transpose_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_size, stride, padding
    );

    // BatchNorm + Mean Subtraction
    dim3 bn_blocks(out_channels, batch_size);
    batchnorm_subtract_mean_kernel<<<bn_blocks, 256>>>(
        static_cast<half*>(output),
        static_cast<half*>(output),
        static_cast<const half*>(bn_weight),
        static_cast<const half*>(bn_bias),
        batch_size, out_channels,
        output_depth, output_height, output_width
    );

    cudaDeviceSynchronize();
}
