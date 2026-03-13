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
// cuda_model.cuh (Self-contained CUDA implementation for fp16 ConvTranspose2d)
// Implements: launch_gpu_implementation(...)
// Input/output/bias/weight: half precision (fp16), accumulation in fp32.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

// Utility: convert __half to float (accumulator in fp32 for numerical stability)
__device__ __forceinline__ float half2float(const __half h) {
    return __half2float(h);
}
__device__ __forceinline__ __half float2half(const float f) {
    return __float2half_rn(f);
}

// CUDA kernel for fp16 ConvTranspose2d ("deconvolution") with asymmetric kernel
// NCHW format for all tensors, weight is (in_channels, out_channels/groups, kernel_h, kernel_w)
__global__ void conv_transpose2d_fp16_kernel(
    __half* __restrict__ output,          // [N, OutC, OH, OW]
    const __half* __restrict__ input,     // [N, InC, H, W]
    const __half* __restrict__ weight,    // [InC, OutC/groups, KH, KW]
    const __half* __restrict__ bias,      // [OutC] or nullptr
    int N, int InC, int OutC,
    int H, int W,
    int KH, int KW,
    int OH, int OW,
    int stride, int padding, int output_padding,
    int groups, bool has_bias
) {
    // Output index: (n, oc, oh, ow)
    int n = blockIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int oh = blockIdx.z * blockDim.x + threadIdx.x;

    if (n >= N || oc >= OutC || oh >= OH) return;

    // Compute group and local oc inside group
    int group_id = oc / (OutC / groups);
    int oc_in_group = oc % (OutC / groups);
    int in_c_start = group_id * (InC / groups);
    int in_c_end   = in_c_start + (InC / groups);

    for (int ow = 0; ow < OW; ++ow) {
        float acc = 0.0f;

        // For each input channel in group
        for (int ic = in_c_start; ic < in_c_end; ++ic) {
            // For each position in the kernel
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    // Compute input spatial location for this output
                    // See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
                    int ih = (oh + padding - kh) / stride;
                    int iw = (ow + padding - kw) / stride;

                    // Only consider if (oh + padding - kh) % stride == 0 and (ow + padding - kw) % stride == 0
                    if (((oh + padding - kh) % stride == 0) && ((ow + padding - kw) % stride == 0)) {
                        // Account for output_padding
                        if (output_padding > 0) {
                            // Output shape is: OH = (H-1)*stride - 2*padding + KH + output_padding
                            // So, output indices oh or ow in [0, OH)
                            // If oh or ow >= OH - output_padding, this is the result of output_padding
                            // See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
                            if (oh >= OH - output_padding || ow >= OW - output_padding)
                                continue;
                        }
                        // Bounds check
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            // Index calculation (NCHW)
                            int inp_idx = n * InC * H * W + ic * H * W + ih * W + iw;
                            int w_idx =
                                ic * (OutC / groups) * KH * KW +
                                oc_in_group * KH * KW +
                                kh * KW + kw;
                            acc += half2float(input[inp_idx]) * half2float(weight[w_idx]);
                        }
                    }
                }
            }
        }
        if (has_bias && bias != nullptr) {
            acc += half2float(bias[oc]);
        }
        // Write output (convert to half)
        int out_idx = n * OutC * OH * OW + oc * OH * OW + oh * OW + ow;
        output[out_idx] = float2half(acc);
    }
}

// Host launcher for the CUDA kernel
void launch_gpu_implementation(
    void* output, void* input,
    void* weight, void* bias,
    int batch_size, int in_channels, int out_channels,
    int height, int width,
    int kernel_h, int kernel_w,
    int stride, int padding, int output_padding, int groups, bool has_bias)
{
    using namespace std;
    assert(stride > 0 && groups > 0);

    // Calculate output spatial size (PyTorch formula)
    // OH = (H - 1) * stride - 2*padding + KH + output_padding
    // OW = (W - 1) * stride - 2*padding + KW + output_padding
    int OH = (height - 1) * stride - 2 * padding + kernel_h + output_padding;
    int OW = (width  - 1) * stride - 2 * padding + kernel_w + output_padding;

    // Thread/block config
    dim3 blockDim(16, 16, 1); // threadIdx.x: oh, threadIdx.y: oc
    dim3 gridDim(batch_size,
                 (out_channels + blockDim.y - 1) / blockDim.y,
                 (OH + blockDim.x - 1) / blockDim.x);

    conv_transpose2d_fp16_kernel<<<gridDim, blockDim>>>(
        static_cast<__half*>(output),
        static_cast<const __half*>(input),
        static_cast<const __half*>(weight),
        static_cast<const __half*>(bias),
        batch_size, in_channels, out_channels,
        height, width,
        kernel_h, kernel_w,
        OH, OW,
        stride, padding, output_padding,
        groups, has_bias
    );

    cudaDeviceSynchronize();
}
