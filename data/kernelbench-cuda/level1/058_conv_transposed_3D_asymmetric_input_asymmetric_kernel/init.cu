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
#include <stdint.h>
#include <cstdio>
#include <cassert>

// Utility: convert __half to float, for accumulation
__device__ inline float load_as_float(const __half* ptr) {
    return __half2float(*ptr);
}

// Utility: convert float to __half (fp16), with rounding
__device__ inline __half store_as_half(float val) {
    return __float2half_rn(val);
}

/*
 * CUDA kernel for transposed 3D convolution (ConvTranspose3d)
 *
 * All tensors are in fp16 (__half) format.
 *
 * Layouts (PyTorch default):
 *   - input:  [N, C_in, D_in, H_in, W_in]         (NCHWD)
 *   - weight: [C_in, C_out/groups, kD, kH, kW]    (PyTorch ConvTranspose3d default)
 *   - output: [N, C_out, D_out, H_out, W_out]     (NCHWD)
 *   - bias:   [C_out]
 *
 * Parameters:
 *   - output: pointer to output tensor (N, C_out, D_out, H_out, W_out), fp16
 *   - input:  pointer to input tensor  (N, C_in, D_in, H_in, W_in), fp16
 *   - weight: pointer to weight tensor (C_in, C_out/groups, kD, kH, kW), fp16
 *   - bias:   pointer to bias tensor (C_out), fp16 or nullptr
 *
 * All dimensions are provided as int64_t.
 *
 * Accumulation is done in fp32 for better numerical stability.
 */
__global__ void convtranspose3d_fp16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    int64_t N,         // batch size
    int64_t C_in,      // input channels
    int64_t C_out,     // output channels
    int64_t D_in,      // input depth
    int64_t H_in,      // input height
    int64_t W_in,      // input width
    int64_t groups,    // number of groups
    int64_t kD,        // kernel depth
    int64_t kH,        // kernel height
    int64_t kW,        // kernel width
    int64_t strideD,   // stride depth
    int64_t strideH,   // stride height
    int64_t strideW,   // stride width
    int64_t padD,      // padding depth
    int64_t padH,      // padding height
    int64_t padW,      // padding width
    int64_t outpadD,   // output padding depth
    int64_t outpadH,   // output padding height
    int64_t outpadW    // output padding width
) {
    // Compute output dimensions
    const int64_t D_out = (D_in - 1) * strideD - 2 * padD + kD + outpadD;
    const int64_t H_out = (H_in - 1) * strideH - 2 * padH + kH + outpadH;
    const int64_t W_out = (W_in - 1) * strideW - 2 * padW + kW + outpadW;

    // Flat thread id for 5D output tensor (N, C_out, D_out, H_out, W_out)
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = N * C_out * D_out * H_out * W_out;
    if (tid >= total) return;

    // Decompose tid into output indices
    int64_t w_out = tid % W_out;
    int64_t h_out = (tid / W_out) % H_out;
    int64_t d_out = (tid / (W_out * H_out)) % D_out;
    int64_t c_out = (tid / (W_out * H_out * D_out)) % C_out;
    int64_t n = tid / (W_out * H_out * D_out * C_out);

    // Group info
    int64_t group = c_out / (C_out / groups);
    int64_t c_out_in_group = c_out % (C_out / groups);

    // Accumulate in fp32 for numerical stability
    float acc = 0.0f;

    // For each input channel in this group
    int64_t c_in_start = group * (C_in / groups);
    int64_t c_in_end = (group + 1) * (C_in / groups);

    // Loop over kernel volume
    for (int64_t kd = 0; kd < kD; ++kd) {
        for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
                // Compute input position corresponding to (d_out, h_out, w_out, kd, kh, kw)
                // Formula (PyTorch): 
                //   d_in = (d_out + padD - kd) / strideD
                //   h_in = (h_out + padH - kh) / strideH
                //   w_in = (w_out + padW - kw) / strideW
                int64_t d_in_nom = d_out + padD - kd;
                int64_t h_in_nom = h_out + padH - kh;
                int64_t w_in_nom = w_out + padW - kw;

                // Only if divisible by stride (otherwise not covered by any input)
                if ((d_in_nom % strideD != 0) || (h_in_nom % strideH != 0) || (w_in_nom % strideW != 0)) {
                    continue;
                }
                int64_t d_in = d_in_nom / strideD;
                int64_t h_in = h_in_nom / strideH;
                int64_t w_in = w_in_nom / strideW;

                // Bounds check
                if (d_in < 0 || d_in >= D_in) continue;
                if (h_in < 0 || h_in >= H_in) continue;
                if (w_in < 0 || w_in >= W_in) continue;

                // Loop over input channels in group
                for (int64_t c_in = c_in_start; c_in < c_in_end; ++c_in) {
                    // input:  [N, C_in, D_in, H_in, W_in]
                    int64_t input_offset =
                        n * (C_in * D_in * H_in * W_in) +
                        c_in * (D_in * H_in * W_in) +
                        d_in * (H_in * W_in) +
                        h_in * W_in +
                        w_in;

                    // weight: [C_in, C_out/groups, kD, kH, kW]
                    int64_t weight_offset =
                        c_in * ((C_out / groups) * kD * kH * kW) +
                        c_out_in_group * (kD * kH * kW) +
                        kd * (kH * kW) +
                        kh * kW +
                        kw;

                    float inp = load_as_float(&input[input_offset]);
                    float wgt = load_as_float(&weight[weight_offset]);
                    acc += inp * wgt;
                }
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        acc += load_as_float(&bias[c_out]);
    }

    // Write output as fp16
    int64_t output_offset =
        n * (C_out * D_out * H_out * W_out) +
        c_out * (D_out * H_out * W_out) +
        d_out * (H_out * W_out) +
        h_out * W_out +
        w_out;
    output[output_offset] = store_as_half(acc);
}

/*
 * Host launcher for the CUDA kernel.
 * All pointers are to GPU memory. All tensors are fp16 (__half).
 */
void launch_gpu_implementation(
    void* output,            // Output tensor, shape: (batch_size, out_channels, depth_out, height_out, width_out), fp16
    void* input,             // Input tensor, shape: (batch_size, in_channels, depth_in, height_in, width_in), fp16
    void* weight,            // Weight tensor, shape: (in_channels, out_channels/groups, kD, kH, kW), fp16
    void* bias,              // Bias tensor, shape: (out_channels), fp16 or nullptr if bias is not used
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t depth_in,
    int64_t height_in,
    int64_t width_in,
    int64_t groups,
    int64_t kD,
    int64_t kH,
    int64_t kW,
    int64_t strideD,
    int64_t strideH,
    int64_t strideW,
    int64_t padD,
    int64_t padH,
    int64_t padW,
    int64_t outpadD,
    int64_t outpadH,
    int64_t outpadW
) {
    // Compute output dimensions
    int64_t D_out = (depth_in - 1) * strideD - 2 * padD + kD + outpadD;
    int64_t H_out = (height_in - 1) * strideH - 2 * padH + kH + outpadH;
    int64_t W_out = (width_in - 1) * strideW - 2 * padW + kW + outpadW;

    int64_t total = batch_size * out_channels * D_out * H_out * W_out;

    // Kernel launch configuration
    const int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    convtranspose3d_fp16_kernel<<<blocks, threads_per_block>>>(
        static_cast<__half*>(output),
        static_cast<const __half*>(input),
        static_cast<const __half*>(weight),
        static_cast<const __half*>(bias),
        batch_size, in_channels, out_channels,
        depth_in, height_in, width_in,
        groups, kD, kH, kW,
        strideD, strideH, strideW,
        padD, padH, padW,
        outpadD, outpadH, outpadW
    );

    cudaDeviceSynchronize();
}
