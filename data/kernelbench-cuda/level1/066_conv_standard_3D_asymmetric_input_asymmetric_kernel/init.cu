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
#include <cmath>
#include <cassert>
#include <stdint.h>
#include <stdio.h>

// Helper: integer division with ceiling
inline __host__ __device__ long div_ceil(long a, long b) { return (a + b - 1) / b; }

// Converts fp16 to fp32 (for accumulation)
__device__ inline float half2float_safe(const half x) { return __half2float(x); }
__device__ inline half float2half_safe(const float x) { return __float2half_rn(x); }

// Computes the output shape for a 3D convolution
__host__ __device__ inline void calc_conv3d_output_dims(
    long in_d, long in_h, long in_w,
    long k_d, long k_h, long k_w,
    long stride_d, long stride_h, long stride_w,
    long pad_d, long pad_h, long pad_w,
    long dil_d, long dil_h, long dil_w,
    long& out_d, long& out_h, long& out_w)
{
    out_d = div_ceil(in_d + 2*pad_d - dil_d*(k_d-1) - 1 + 1, stride_d);
    out_h = div_ceil(in_h + 2*pad_h - dil_h*(k_h-1) - 1 + 1, stride_h);
    out_w = div_ceil(in_w + 2*pad_w - dil_w*(k_w-1) - 1 + 1, stride_w);
}

// CUDA kernel for general 3D convolution (NCDHW layout, fp16 input/output, fp32 accumulation)
__global__ void conv3d_fp16_kernel(
    const half* __restrict__ input,   // [N, C_in, D, H, W]
    const half* __restrict__ weight,  // [C_out, C_in/groups, kD, kH, kW]
    const half* __restrict__ bias,    // [C_out] or nullptr
    half* __restrict__ output,        // [N, C_out, oD, oH, oW]
    long N, long C_in, long C_out,
    long D, long H, long W,
    long kD, long kH, long kW,
    long oD, long oH, long oW,
    long stride_d, long stride_h, long stride_w,
    long pad_d, long pad_h, long pad_w,
    long dil_d, long dil_h, long dil_w,
    long groups,
    bool bias_flag
)
{
    // Each thread computes one output element: (n, g, co, od, oh, ow)
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total = N * C_out * oD * oH * oW;
    if (tid >= total) return;

    // Decompose output coordinates
    long ow = tid % oW;
    long oh = (tid / oW) % oH;
    long od = (tid / (oW * oH)) % oD;
    long co = (tid / (oW * oH * oD)) % C_out;
    long n  = tid / (oW * oH * oD * C_out);

    // Grouped convolution: which group does this output channel belong to?
    long c_out_per_group = C_out / groups;
    long c_in_per_group = C_in / groups;
    long g = co / c_out_per_group;
    long co_in_group = co % c_out_per_group;

    // Compute input channel offset for this group
    long c_in_beg = g * c_in_per_group;
    long c_in_end = c_in_beg + c_in_per_group;

    // Accumulator in fp32
    float acc = 0.0f;

    // Iterate over input channels for this group
    for (long ci = c_in_beg; ci < c_in_end; ++ci) {
        long ci_in_group = ci - c_in_beg;
        // For each kernel position
        for (long kd = 0; kd < kD; ++kd) {
            long id = od * stride_d - pad_d + kd * dil_d;
            if (id < 0 || id >= D) continue;
            for (long kh = 0; kh < kH; ++kh) {
                long ih = oh * stride_h - pad_h + kh * dil_h;
                if (ih < 0 || ih >= H) continue;
                for (long kw = 0; kw < kW; ++kw) {
                    long iw = ow * stride_w - pad_w + kw * dil_w;
                    if (iw < 0 || iw >= W) continue;

                    // Input index: n, ci, id, ih, iw
                    long inp_idx = ((n * C_in + ci) * D + id) * H * W + ih * W + iw;

                    // Weight index: co, ci_in_group, kd, kh, kw
                    long w_idx = (((co * c_in_per_group + ci_in_group) * kD + kd) * kH + kh) * kW + kw;

                    float inp_val = half2float_safe(input[inp_idx]);
                    float w_val = half2float_safe(weight[w_idx]);
                    acc += inp_val * w_val;
                }
            }
        }
    }
    // Add bias if needed
    if (bias_flag && bias) {
        acc += half2float_safe(bias[co]);
    }

    // Store result as fp16
    long out_idx = ((n * C_out + co) * oD + od) * oH * oW + oh * oW + ow;
    output[out_idx] = float2half_safe(acc);
}

// Host function to launch the CUDA kernel
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    long batch_size,
    long in_channels,
    long out_channels,
    long depth,
    long height,
    long width,
    long kernel_size_d,
    long kernel_size_h,
    long kernel_size_w,
    long stride_d,
    long stride_h,
    long stride_w,
    long padding_d,
    long padding_h,
    long padding_w,
    long dilation_d,
    long dilation_h,
    long dilation_w,
    long groups,
    bool bias_flag
) {
    // Compute output shape
    long oD, oH, oW;
    calc_conv3d_output_dims(
        depth, height, width,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        oD, oH, oW
    );

    // Compute launch config
    long total = batch_size * out_channels * oD * oH * oW;
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>(div_ceil(total, threadsPerBlock));

    // Launch kernel
    conv3d_fp16_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        bias_flag ? static_cast<const half*>(bias) : nullptr,
        static_cast<half*>(output),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size_d, kernel_size_h, kernel_size_w,
        oD, oH, oW,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        bias_flag
    );
    cudaDeviceSynchronize();
}
