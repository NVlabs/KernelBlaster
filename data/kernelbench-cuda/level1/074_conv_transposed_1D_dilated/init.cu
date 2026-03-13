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
// cuda_model.cuh

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cassert>
#include <stdint.h>
#include <cstdio>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/*
   Reference: PyTorch ConvTranspose1d docs

   For each output position l_out:
     For each n, c_out:
       output[n, c_out, l_out] = sum_{c_in} sum_{k=0}^{kernel_size-1} input[n, c_in, l_in] * weight[c_in, c_out, k]
         where l_in = (l_out + padding - k*dilation) // stride, and only if (l_out + padding - k*dilation) % stride == 0
         and l_in in [0, input_length-1]
*/

__global__ void conv1d_transpose_fp16_kernel(
    const half* __restrict__ input,    // [N, C_in, L_in]
    const half* __restrict__ weight,   // [C_in, C_out, K]
    half* __restrict__ output,         // [N, C_out, L_out]
    long batch_size,
    long in_channels,
    long out_channels,
    long input_length,
    long kernel_size,
    long stride,
    long padding,
    long dilation,
    long output_length
) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total = batch_size * out_channels * output_length;
    if (tid >= total) return;

    long l_out = tid % output_length;
    long c_out = (tid / output_length) % out_channels;
    long n = tid / (output_length * out_channels);

    float acc = 0.0f;
    // For each input channel and kernel position
    for (long c_in = 0; c_in < in_channels; ++c_in) {
        for (long k = 0; k < kernel_size; ++k) {
            long l_in_nom = l_out + padding - k * dilation;
            if (l_in_nom % stride != 0) continue;
            long l_in = l_in_nom / stride;
            if (l_in < 0 || l_in >= input_length) continue;
            // input[n, c_in, l_in]
            long input_idx = n * in_channels * input_length + c_in * input_length + l_in;
            // weight[c_in, c_out, k]
            long weight_idx = c_in * out_channels * kernel_size + c_out * kernel_size + k;
            acc += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
        }
    }
    output[n * out_channels * output_length + c_out * output_length + l_out] = __float2half(acc);
}

void launch_gpu_implementation(
    void* output,                // Output tensor (float16, GPU)
    void* input,                 // Input tensor (float16, GPU)
    void* weight,                // Weight tensor (float16, GPU)
    void* bias,                  // Bias tensor (nullptr, since bias=False)
    long batch_size,
    long in_channels,
    long out_channels,
    long input_length,
    long kernel_size,
    long stride,
    long padding,
    long dilation,
    long output_length
) {
    long total = batch_size * out_channels * output_length;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    conv1d_transpose_fp16_kernel<<<blocks, threads>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<half*>(output),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        output_length
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
