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

__global__ void fused_conv3d_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ sum_tensor,
    half* __restrict__ output,
    int batch_size, int in_channels,
    int depth, int height, int width,
    int kernel_size, int out_channels,
    float negative_slope
) {
    // Calculate output dimensions
    const int depth_out = depth - kernel_size + 1;
    const int height_out = height - kernel_size + 1;
    const int width_out = width - kernel_size + 1;
    const int output_elements = batch_size * out_channels * depth_out * height_out * width_out;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_elements) return;

    // Decompose index into output coordinates
    const int n = idx / (out_channels * depth_out * height_out * width_out);
    int rem = idx % (out_channels * depth_out * height_out * width_out);
    const int oc = rem / (depth_out * height_out * width_out);
    rem %= depth_out * height_out * width_out;
    const int d = rem / (height_out * width_out);
    rem %= height_out * width_out;
    const int h = rem / width_out;
    const int w = rem % width_out;

    float acc = 0.0f;

    // 3D convolution with fp32 accumulation
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int input_d = d + kd;
                const int input_h = h + kh;
                const int input_w = w + kw;
                
                if (input_d < depth && input_h < height && input_w < width) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const int input_idx = ((n * in_channels + ic) * depth + input_d) * height * width + input_h * width + input_w;
                        const int weight_idx = ((oc * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                        
                        acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                    }
                }
            }
        }
    }

    // Add bias and apply pointwise operations
    acc += __half2float(bias[oc]);
    acc = (acc < 0.0f) ? acc * negative_slope : acc;  // LeakyReLU
    acc += __half2float(sum_tensor[oc]);               // Add parameter tensor
    acc = fmaxf(fminf(acc, 1.0f), -1.0f);              // Clamp
    acc = 0.5f * acc * (1.0f + tanhf(0.7978845608028654f * (acc + 0.044715f * acc * acc * acc)));  // GELU

    output[idx] = __float2half_rn(acc);
}

void launch_gpu_implementation(
    void* output, void* input, 
    void* conv_weight, void* conv_bias,
    void* sum_tensor
) {
    const int batch_size = 128;
    const int in_channels = 3;
    const int out_channels = 16;
    const int depth = 16, height = 32, width = 32;
    const int kernel_size = 3;

    const int depth_out = depth - kernel_size + 1;
    const int height_out = height - kernel_size + 1;
    const int width_out = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * depth_out * height_out * width_out;

    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    fused_conv3d_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(sum_tensor),
        static_cast<half*>(output),
        batch_size, in_channels,
        depth, height, width,
        kernel_size, out_channels,
        0.2f  // leaky_relu negative slope
    );
    
    cudaDeviceSynchronize();
}
