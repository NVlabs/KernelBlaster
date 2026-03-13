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

__global__ void conv_activation_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    float* __restrict__ activated_buffer,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size
) {
    const int K = kernel_size;
    const int padding = K / 2;
    const int OH = height;
    const int OW = width;
    const int HW = OH * OW;
    const int C = out_channels;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * C * HW;
    if (idx >= total_elements) return;

    // NHWC output index decomposition
    const int n = idx / (C * HW);
    const int c = (idx / HW) % C;
    const int hw = idx % HW;
    const int h = hw / OW;
    const int w = hw % OW;

    // Convolution accumulation in fp32
    float conv_val = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int h_in = h - padding + kh;
                const int w_in = w - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    const int input_idx = ((n * in_channels + c_in) * height + h_in) * width + w_in;
                    const int weight_idx = ((c * in_channels + c_in) * K + kh) * K + kw;
                    conv_val += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                }
            }
        }
    }

    // Add bias and apply activation
    conv_val += __half2float(conv_bias[c]);
    const float softplus = logf(1.0f + expf(conv_val));
    activated_buffer[idx] = conv_val * tanhf(softplus);
}

__global__ void compute_batch_stats_kernel(
    const float* __restrict__ activated_buffer,
    float* __restrict__ batch_sum,
    float* __restrict__ batch_sqsum,
    int batch_size, int out_channels,
    int height, int width
) {
    const int C = out_channels;
    const int HW = height * width;
    const int elements_per_channel = batch_size * HW;

    // Each block handles one channel
    const int c = blockIdx.x;
    float sum = 0.0f, sqsum = 0.0f;

    for (int idx = threadIdx.x; idx < elements_per_channel; idx += blockDim.x) {
        const int n = idx / HW;
        const int hw = idx % HW;
        const float val = activated_buffer[(n * C + c) * HW + hw];
        sum += val;
        sqsum += val * val;
    }

    // Block-wide reduction
    __shared__ float ssum[256], ssqsum[256];
    ssum[threadIdx.x] = sum;
    ssqsum[threadIdx.x] = sqsum;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
            ssqsum[threadIdx.x] += ssqsum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        batch_sum[c] = ssum[0];
        batch_sqsum[c] = ssqsum[0];
    }
}

__global__ void batch_norm_kernel(
    const float* __restrict__ activated_buffer,
    const half* __restrict__ bn_weight,
    const half* __restrict__ bn_bias,
    const float* __restrict__ batch_sum,
    const float* __restrict__ batch_sqsum,
    half* __restrict__ output,
    int batch_size, int out_channels,
    int height, int width, float eps
) {
    const int HW = height * width;
    const int C = out_channels;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * C * HW;
    if (idx >= total_elements) return;

    const int c = (idx / HW) % C;
    const float mean = batch_sum[c] / (batch_size * HW);
    const float var = (batch_sqsum[c] / (batch_size * HW)) - (mean * mean);
    const float weight = __half2float(bn_weight[c]);
    const float bias = __half2float(bn_bias[c]);

    const float normalized = (activated_buffer[idx] - mean) / sqrtf(var + eps);
    output[idx] = __float2half_rn(normalized * weight + bias);
}

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias,
    void* bn_weight, void* bn_bias,
    void* bn_running_mean, void* bn_running_var,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    float eps, float momentum
) {
    const int OH = height;
    const int OW = width;
    const int total_elements = batch_size * out_channels * OH * OW;

    // Allocate intermediate buffers
    float *d_activated, *d_batch_sum, *d_batch_sqsum;
    cudaMalloc(&d_activated, total_elements * sizeof(float));
    cudaMalloc(&d_batch_sum, out_channels * sizeof(float));
    cudaMalloc(&d_batch_sqsum, out_channels * sizeof(float));

    // Step 1: Compute convolution and activation
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    conv_activation_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        d_activated,
        batch_size, in_channels, out_channels,
        height, width, kernel_size
    );

    // Step 2: Compute batch statistics
    compute_batch_stats_kernel<<<out_channels, 256>>>(
        d_activated,
        d_batch_sum,
        d_batch_sqsum,
        batch_size, out_channels,
        OH, OW
    );

    // Step 3: Apply batch normalization
    batch_norm_kernel<<<grid_size, block_size>>>(
        d_activated,
        static_cast<const half*>(bn_weight),
        static_cast<const half*>(bn_bias),
        d_batch_sum,
        d_batch_sqsum,
        static_cast<half*>(output),
        batch_size, out_channels,
        height, width, eps
    );

    // Cleanup
    cudaFree(d_activated);
    cudaFree(d_batch_sum);
    cudaFree(d_batch_sqsum);
    cudaDeviceSynchronize();
}
