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
// Efficient CUDA implementation of 3D transposed convolution (ConvTranspose3d) for fp16 tensors.
// - Handles arbitrary batch size, group count, stride, padding, output padding, and bias.
// - Accumulates in fp32 for numerical stability, outputs fp16.
// - Weight expected in PyTorch layout: [in_channels, out_channels/groups, kD, kH, kW] (as in torch.nn.ConvTranspose3d).
// - Input: [batch_size, in_channels, depth, height, width] (NCDHW).
// - Output: [batch_size, out_channels, depth_out, height_out, width_out] (NCDHW).
// - Groups are supported.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

// Utility: CUDA error checking for debugging
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Utility: ceil division
inline __host__ __device__ int div_up(int a, int b) { return (a + b - 1) / b; }

// CUDA kernel for 3D transposed convolution (fp16 input/output, fp32 accumulation)
__global__ void conv3d_transpose_kernel(
    const half* __restrict__ input,    // [N, IC, D, H, W]
    const half* __restrict__ weight,   // [IC, OC_per_g, kD, kH, kW] (PyTorch layout)
    const half* __restrict__ bias,     // [OC] or nullptr
    half* __restrict__ output,         // [N, OC, OD, OH, OW]
    int N, int IC, int OC, int D, int H, int W,
    int ksize, int stride, int padding, int output_padding, int groups, bool has_bias,
    int OD, int OH, int OW
) {
    // Output index mapping: [n, oc, od, oh, ow]
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OC * OD * OH * OW;
    if (tid >= total) return;

    // Unravel the flat output index
    int ow = tid % OW;
    int oh = (tid / OW) % OH;
    int od = (tid / (OW * OH)) % OD;
    int oc = (tid / (OW * OH * OD)) % OC;
    int n  = tid / (OW * OH * OD * OC);

    // Group mapping
    int OC_per_g = OC / groups;
    int IC_per_g = IC / groups;
    int g = oc / OC_per_g;          // which group this output channel belongs to
    int ocg = oc % OC_per_g;        // output channel within group

    // Accumulate in fp32 for stability
    float acc = 0.0f;

    // For each kernel point: loop over kD, kH, kW
    for (int kd = 0; kd < ksize; ++kd) {
        int id = od + padding - kd; // "input" depth index that would contribute here via this kd
        if (id % stride != 0) continue;
        id /= stride;
        if (id < 0 || id >= D) continue;
        for (int kh = 0; kh < ksize; ++kh) {
            int ih = oh + padding - kh;
            if (ih % stride != 0) continue;
            ih /= stride;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < ksize; ++kw) {
                int iw = ow + padding - kw;
                if (iw % stride != 0) continue;
                iw /= stride;
                if (iw < 0 || iw >= W) continue;

                // Loop over input channels in this group
                for (int icg = 0; icg < IC_per_g; ++icg) {
                    int ic = g * IC_per_g + icg;

                    // Input index: [n, ic, id, ih, iw]
                    int input_idx = (((n * IC + ic) * D + id) * H + ih) * W + iw;
                    float inp = __half2float(input[input_idx]);

                    // Weight index: [ic, ocg, kd, kh, kw]
                    int weight_idx = ((((ic) * OC_per_g + ocg) * ksize + kd) * ksize + kh) * ksize + kw;
                    float w = __half2float(weight[weight_idx]);

                    acc += inp * w;
                }
            }
        }
    }
    // Add bias if present
    if (has_bias) {
        acc += __half2float(bias[oc]);
    }
    // Convert to fp16 and store
    int out_idx = (((n * OC + oc) * OD + od) * OH + oh) * OW + ow;
    output[out_idx] = __float2half(acc);
}

// Host code to launch the kernel
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    bool has_bias
) {
    // Compute output spatial size (see PyTorch ConvTranspose3d formula)
    // output = (input - 1) * stride - 2*padding + kernel_size + output_padding
    int OD = (depth  - 1) * stride - 2 * padding + kernel_size + output_padding;
    int OH = (height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int OW = (width  - 1) * stride - 2 * padding + kernel_size + output_padding;

    int total = batch_size * out_channels * OD * OH * OW;
    int block = 256;
    int grid = div_up(total, block);

    conv3d_transpose_kernel<<<grid, block>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        static_cast<half*>(output),
        batch_size, in_channels, out_channels, depth, height, width,
        kernel_size, stride, padding, output_padding, groups, has_bias,
        OD, OH, OW
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
