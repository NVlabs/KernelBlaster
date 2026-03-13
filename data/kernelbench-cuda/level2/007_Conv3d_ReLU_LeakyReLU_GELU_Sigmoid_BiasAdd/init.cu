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

__global__ void model_forward_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias,
    const half* __restrict__ model_bias,
    half* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_depth,
    int input_height,
    int input_width
) {
    const int output_depth = input_depth - kernel_size + 1;
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    const int output_size = out_channels * output_depth * output_height * output_width;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < batch_size * output_size; 
         idx += gridDim.x * blockDim.x) {
        
        const int b = idx / output_size;
        const int oc = (idx % output_size) / (output_depth * output_height * output_width);
        const int od = (idx % (output_depth * output_height * output_width)) / (output_height * output_width);
        const int oh = (idx % (output_height * output_width)) / output_width;
        const int ow = idx % output_width;

        float sum = 0.0f;
        
        // 3D convolution
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int id = od + kd;
                    const int ih = oh + kh;
                    const int iw = ow + kw;
                    
                    if (id < input_depth && ih < input_height && iw < input_width) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            const int input_idx = ((b * in_channels + ic) * input_depth + id) 
                                                * input_height * input_width + ih * input_width + iw;
                            const int weight_idx = ((oc * in_channels + ic) * kernel_size + kd) 
                                                * kernel_size * kernel_size + kh * kernel_size + kw;
                            
                            sum += __half2float(input[input_idx]) * __half2float(conv_weight[weight_idx]);
                        }
                    }
                }
            }
        }

        // Add convolution bias
        if (conv_bias) {
            sum += __half2float(conv_bias[oc]);
        }

        // Activation sequence
        sum = fmaxf(sum, 0.0f); // ReLU
        sum = fmaxf(0.01f * sum, sum); // LeakyReLU
        sum = 0.5f * sum * (1.0f + erff(sum / sqrtf(2.0f))); // GELU
        sum = 1.0f / (1.0f + expf(-sum)); // Sigmoid

        // Add model bias and store
        const int output_idx = ((b * out_channels + oc) * output_depth + od) 
                             * output_height * output_width + oh * output_width + ow;
        output[output_idx] = __hadd(__float2half_rn(sum), model_bias[oc]);
    }
}

void launch_gpu_implementation(
    void* output, void* input, void* conv_weight, void* conv_bias, void* model_bias,
    int in_channels, int out_channels, int kernel_size,
    const std::vector<int64_t>& bias_shape, const std::vector<int64_t>& input_shape
) {
    const int batch_size = input_shape[0];
    const int input_depth = input_shape[2];
    const int input_height = input_shape[3];
    const int input_width = input_shape[4];
    
    const int output_depth = input_depth - kernel_size + 1;
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    model_forward_kernel<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(model_bias),
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_depth,
        input_height,
        input_width
    );
    
    cudaDeviceSynchronize();
}
