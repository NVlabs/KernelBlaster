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

__global__ void conv2d_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int in_channels,
    int out_channels,
    int kernel_size,
    int batch_size,
    int height,
    int width
) {
    const int OH = height - kernel_size + 1;
    const int OW = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * OH * OW;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    const int n = tid / (out_channels * OH * OW);
    const int oc = (tid % (out_channels * OH * OW)) / (OH * OW);
    const int oh = (tid % (OH * OW)) / OW;
    const int ow = tid % OW;

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int ih = oh + kh;
            const int iw = ow + kw;

            if (ih < height && iw < width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    const int input_idx = n * in_channels * height * width + ic * height * width + ih * width + iw;
                    const int weight_idx = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;

                    sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
                }
            }
        }
    }

    if (bias) {
        sum += __half2float(bias[oc]);
    }

    output[tid] = __float2half_rn(sum);
}

__global__ void mish_kernel(half* in_out, int num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;

    float x = __half2float(in_out[tid]);
    float sp = log1pf(expf(x));
    float tanh_sp = tanhf(sp);
    float result = x * tanh_sp;

    in_out[tid] = __float2half_rn(result);
}

void launch_gpu_implementation(void* output, void* input, void* weight, void* bias, int in_channels, int out_channels, int kernel_size, int batch_size, int height, int width) {
    const int OH = height - kernel_size + 1;
    const int OW = width - kernel_size + 1;
    const int output_elements = batch_size * out_channels * OH * OW;

    const int block_size = 256;
    const int grid_size = (output_elements + block_size - 1) / block_size;

    // Launch convolution
    conv2d_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        in_channels,
        out_channels,
        kernel_size,
        batch_size,
        height,
        width
    );

    // Apply Mish twice
    mish_kernel<<<grid_size, block_size>>>(static_cast<half*>(output), output_elements);
    mish_kernel<<<grid_size, block_size>>>(static_cast<half*>(output), output_elements);

    cudaDeviceSynchronize();
}
