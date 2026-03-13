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
#include <cstdio>

// Utility for checking CUDA errors
#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

inline __host__ __device__ int div_up(int a, int b) { return (a + b - 1) / b; }

// FP16 to FP32 conversion helpers
__device__ __forceinline__ float half2float_safe(half h) { return __half2float(h); }
__device__ __forceinline__ half float2half_safe(float f) { return __float2half_rn(f); }

// 3D transposed convolution kernel, fp16 I/O, fp32 accumulation
__global__ void conv_transpose3d_fp16_kernel(
    half* __restrict__ output,          // [B, OC, D_out, H_out, W_out]
    const half* __restrict__ input,     // [B, IC, D_in, H_in, W_in]
    const half* __restrict__ weight,    // [IC, OC, KD, KH, KW] (PyTorch format for ConvTranspose3d)
    const half* __restrict__ bias,      // [OC] or nullptr
    int B, int IC, int OC,
    int D_in, int H_in, int W_in,
    int KD, int KH, int KW,
    int stride, int padding, int dilation,
    int D_out, int H_out, int W_out,
    bool has_bias
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * D_out * H_out * W_out;
    if (tid >= total) return;

    int ow = tid % W_out;
    int oh = (tid / W_out) % H_out;
    int od = (tid / (W_out * H_out)) % D_out;
    int oc = (tid / (W_out * H_out * D_out)) % OC;
    int b  = tid / (W_out * H_out * D_out * OC);

    float acc = 0.0f;

    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            int id = od + padding - kd * dilation;
            if (id % stride != 0) continue;
            id /= stride;
            if (id < 0 || id >= D_in) continue;
            for (int kh = 0; kh < KH; ++kh) {
                int ih = oh + padding - kh * dilation;
                if (ih % stride != 0) continue;
                ih /= stride;
                if (ih < 0 || ih >= H_in) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    int iw = ow + padding - kw * dilation;
                    if (iw % stride != 0) continue;
                    iw /= stride;
                    if (iw < 0 || iw >= W_in) continue;

                    int in_idx = (((b * IC + ic) * D_in + id) * H_in + ih) * W_in + iw;
                    int w_idx = ((((ic * OC + oc) * KD + kd) * KH + kh) * KW + kw);

                    float in_val = half2float_safe(input[in_idx]);
                    float w_val = half2float_safe(weight[w_idx]);
                    acc += in_val * w_val;
                }
            }
        }
    }
    if (has_bias && bias != nullptr) {
        acc += half2float_safe(bias[oc]);
    }
    output[tid] = float2half_safe(acc);
}

// Host launcher, must be non-inline and non-static for linker
void launch_gpu_implementation(
    void* output, 
    void* input,
    void* weight,
    void* bias,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool has_bias,
    int64_t batch_size,
    int64_t depth,
    int64_t height,
    int64_t width
) {
    const int B = (int)batch_size;
    const int IC = (int)in_channels;
    const int OC = (int)out_channels;
    const int KD = (int)kernel_size, KH = (int)kernel_size, KW = (int)kernel_size;
    const int stride_ = (int)stride;
    const int padding_ = (int)padding;
    const int dilation_ = (int)dilation;
    const int D_in = (int)depth;
    const int H_in = (int)height;
    const int W_in = (int)width;

    int D_out = (D_in - 1) * stride_ - 2 * padding_ + dilation_ * (KD - 1) + 1;
    int H_out = (H_in - 1) * stride_ - 2 * padding_ + dilation_ * (KH - 1) + 1;
    int W_out = (W_in - 1) * stride_ - 2 * padding_ + dilation_ * (KW - 1) + 1;

    const int threads = 256;
    const int total = B * OC * D_out * H_out * W_out;
    const int blocks = div_up(total, threads);

    conv_transpose3d_fp16_kernel<<<blocks, threads>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias), // may be nullptr
        B, IC, OC,
        D_in, H_in, W_in,
        KD, KH, KW,
        stride_, padding_, dilation_,
        D_out, H_out, W_out,
        has_bias
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
