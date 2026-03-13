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

// ConvTranspose2d kernel with fused min, sum, GELU, and bias addition
__global__ void model_forward_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ model_bias,
    half* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int H_in, int W_in, int kernel_size,
    int stride, int padding, int output_padding,
    int OH, int OW
) {
    // Calculate output dimensions
    const int H_out = OH;
    const int W_out = OW;
    
    // Each thread handles one output element (n, c_out, h_out, w_out)
    const int n = blockIdx.z;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= batch_size || h_out >= H_out || w_out >= W_out) return;

    // Shared memory for intermediate reductions
    __shared__ float smem_min[16][32];
    __shared__ float smem_sum[32];

    // Phase 1: ConvTranspose2d and min reduction
    float channel_min = INFINITY;
    for (int c_out = 0; c_out < out_channels; ++c_out) {
        float conv_val = 0.0f;
        
        // ConvTranspose2d calculation
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int h_in = (h_out - kh + padding) / stride;
                    const int w_in = (w_out - kw + padding) / stride;
                    
                    if ((h_out - kh + padding) % stride != 0) continue;
                    if ((w_out - kw + padding) % stride != 0) continue;
                    
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        const int input_idx = n * in_channels * H_in * W_in +
                                            c_in * H_in * W_in + h_in * W_in + w_in;
                        const int weight_idx = c_in * out_channels * kernel_size * kernel_size +
                                             c_out * kernel_size * kernel_size + kh * kernel_size + kw;
                        
                        conv_val += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                    }
                }
            }
        }
        
        // Add conv_bias
        if (conv_bias) {
            conv_val += __half2float(conv_bias[c_out]);
        }
        
        // Track min across channels
        if (conv_val < channel_min) {
            channel_min = conv_val;
        }
    }

    // Phase 2: Sum reduction along height
    if (threadIdx.y == 0) {
        smem_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    atomicAdd(&smem_sum[threadIdx.x], channel_min);
    __syncthreads();

    // Phase 3: GELU and bias addition
    if (threadIdx.y == 0) {
        float sum_val = smem_sum[threadIdx.x];
        
        // GELU activation
        float gelu = sum_val * 0.5f * (1.0f + tanhf(0.7978845608f * (sum_val + 0.044715f * sum_val * sum_val * sum_val)));
        
        // Add model bias
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            const int output_idx = n * out_channels * W_out +
                                 c_out * W_out + w_out;
            output[output_idx] = __float2half_rn(gelu + __half2float(model_bias[c_out]));
        }
    }
}

void launch_gpu_implementation(
    void* output, void* input,
    void* conv_weight, void* conv_bias, void* model_bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int output_padding,
    int bias_c, int bias_h, int bias_w
) {
    const int batch_size = 128;
    const int H_in = 32, W_in = 32;
    const int OH = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int OW = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    dim3 block(32, 16);
    dim3 grid(
        (OW + block.x - 1) / block.x,
        (OH + block.y - 1) / block.y,
        batch_size
    );

    model_forward_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(model_bias),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        H_in, W_in, kernel_size,
        stride, padding, output_padding,
        OH, OW
    );
    
    cudaDeviceSynchronize();
}
