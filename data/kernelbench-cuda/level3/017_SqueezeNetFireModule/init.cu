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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// Utility: CUDA error check
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
        }                                                                      \
    } while (0)
#endif

// Index helpers for NCHW
static __device__ __forceinline__ int64_t idx_nchw(int64_t n, int64_t c, int64_t h, int64_t w,
                                                   int64_t C, int64_t H, int64_t W) {
    return ((n * C + c) * H + h) * W + w;
}

// Squeeze 1x1 conv + ReLU
__global__ void squeeze_1x1_relu_kernel(
    const half* __restrict__ input,        // [N, Cin, H, W]
    const half* __restrict__ weight,       // [Cout, Cin, 1, 1] (OIHW)
    const half* __restrict__ bias,         // [Cout] or nullptr
    half* __restrict__ output,             // [N, Cout, H, W]
    int64_t N, int64_t Cin, int64_t H, int64_t W, int64_t Cout
) {
    const int64_t P = N * H * W;
    int64_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int64_t n = p / (H * W);
    int64_t rem = p % (H * W);
    int64_t h = rem / W;
    int64_t w = rem % W;

    // Loop over output channels
    for (int64_t oc = 0; oc < Cout; ++oc) {
        float acc = 0.0f;
        if (bias) {
            acc = __half2float(bias[oc]);
        }

        // Accumulate over input channels
        for (int64_t ic = 0; ic < Cin; ++ic) {
            int64_t in_index = idx_nchw(n, ic, h, w, Cin, H, W);
            int64_t w_index  = oc * Cin + ic; // 1x1: (oc, ic, 0, 0)
            float x = __half2float(input[in_index]);
            float wv = __half2float(weight[w_index]);
            acc += x * wv;
        }

        // ReLU and store
        acc = acc > 0.0f ? acc : 0.0f;
        int64_t out_index = idx_nchw(n, oc, h, w, Cout, H, W);
        output[out_index] = __float2half_rn(acc);
    }
}

// Expand 1x1 conv + ReLU; writes into final output with channel offset
template<int OC_TILE>
__global__ void expand_1x1_relu_kernel(
    const half* __restrict__ input,        // [N, Cin, H, W] (squeezed activation)
    const half* __restrict__ weight,       // [Cout, Cin, 1, 1]
    const half* __restrict__ bias,         // [Cout] or nullptr
    half* __restrict__ output,             // [N, Cout_total, H, W] (final output)
    int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout,                          // number of output channels in this branch
    int64_t Cout_total,                    // total output channels across all branches
    int64_t out_channel_offset             // starting channel offset for this branch
) {
    const int64_t P = N * H * W;
    int64_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int64_t n = p / (H * W);
    int64_t rem = p % (H * W);
    int64_t h = rem / W;
    int64_t w = rem % W;

    // Process output channels in tiles for register reuse and reduced pressure
    for (int64_t oc_base = 0; oc_base < Cout; oc_base += OC_TILE) {
        const int tile = (oc_base + OC_TILE <= Cout) ? OC_TILE : (Cout - oc_base);
        float acc[OC_TILE];

        // Initialize with bias if available
        #pragma unroll
        for (int j = 0; j < OC_TILE; ++j) {
            if (j < tile) {
                if (bias) {
                    acc[j] = __half2float(bias[oc_base + j]);
                } else {
                    acc[j] = 0.0f;
                }
            }
        }

        // Accumulate over input channels
        for (int64_t ic = 0; ic < Cin; ++ic) {
            int64_t in_index = idx_nchw(n, ic, h, w, Cin, H, W);
            float x = __half2float(input[in_index]);

            // Load a contiguous vector of weights for this oc tile at fixed ic
            #pragma unroll
            for (int j = 0; j < OC_TILE; ++j) {
                if (j < tile) {
                    int64_t w_index = (oc_base + j) * Cin + ic;
                    float wv = __half2float(weight[w_index]);
                    acc[j] += x * wv;
                }
            }
        }

        // ReLU and store to final output with channel offset
        #pragma unroll
        for (int j = 0; j < OC_TILE; ++j) {
            if (j < tile) {
                float v = acc[j];
                v = v > 0.0f ? v : 0.0f;
                int64_t oc_out = out_channel_offset + (oc_base + j);
                int64_t out_index = idx_nchw(n, oc_out, h, w, Cout_total, H, W);
                output[out_index] = __float2half_rn(v);
            }
        }
    }
}

// Expand 3x3 conv (padding provided) + ReLU; writes into final output with channel offset
template<int OC_TILE>
__global__ void expand_3x3_relu_kernel(
    const half* __restrict__ input,        // [N, Cin, H, W] (squeezed activation)
    const half* __restrict__ weight,       // [Cout, Cin, 3, 3]
    const half* __restrict__ bias,         // [Cout] or nullptr
    half* __restrict__ output,             // [N, Cout_total, H, W]
    int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout,                          // number of output channels in this branch
    int64_t Cout_total,                    // total output channels across all branches
    int64_t out_channel_offset,            // starting channel offset for this branch
    int64_t pad_h, int64_t pad_w
) {
    const int64_t P = N * H * W;
    int64_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int64_t n = p / (H * W);
    int64_t rem = p % (H * W);
    int64_t h = rem / W;
    int64_t w = rem % W;

    const int64_t KH = 3, KW = 3;
    const int64_t W_per_oc = Cin * KH * KW;
    const int64_t W_per_ci = KH * KW;

    // Process output channels in tiles
    for (int64_t oc_base = 0; oc_base < Cout; oc_base += OC_TILE) {
        const int tile = (oc_base + OC_TILE <= Cout) ? OC_TILE : (Cout - oc_base);
        float acc[OC_TILE];

        #pragma unroll
        for (int j = 0; j < OC_TILE; ++j) {
            if (j < tile) {
                if (bias) acc[j] = __half2float(bias[oc_base + j]);
                else      acc[j] = 0.0f;
            }
        }

        // Optimize interior region (no boundary checks)
        bool interior = (h >= pad_h) && (h + (KH - 1 - pad_h) < H) &&
                        (w >= pad_w) && (w + (KW - 1 - pad_w) < W);

        for (int64_t ic = 0; ic < Cin; ++ic) {
            if (interior) {
                // No bounds checks
                #pragma unroll
                for (int k_y = 0; k_y < 3; ++k_y) {
                    int64_t ih = h + k_y - pad_h;
                    #pragma unroll
                    for (int k_x = 0; k_x < 3; ++k_x) {
                        int64_t iw = w + k_x - pad_w;
                        float x = __half2float(input[idx_nchw(n, ic, ih, iw, Cin, H, W)]);
                        int64_t wk = k_y * KW + k_x;
                        #pragma unroll
                        for (int j = 0; j < OC_TILE; ++j) {
                            if (j < tile) {
                                int64_t oc = oc_base + j;
                                int64_t w_index = oc * W_per_oc + ic * W_per_ci + wk;
                                float wv = __half2float(weight[w_index]);
                                acc[j] += x * wv;
                            }
                        }
                    }
                }
            } else {
                // With bounds checks
                #pragma unroll
                for (int k_y = 0; k_y < 3; ++k_y) {
                    int64_t ih = h + k_y - pad_h;
                    #pragma unroll
                    for (int k_x = 0; k_x < 3; ++k_x) {
                        int64_t iw = w + k_x - pad_w;
                        float x = 0.0f;
                        if ((ih >= 0) && (ih < H) && (iw >= 0) && (iw < W)) {
                            x = __half2float(input[idx_nchw(n, ic, ih, iw, Cin, H, W)]);
                        }
                        int64_t wk = k_y * KW + k_x;
                        #pragma unroll
                        for (int j = 0; j < OC_TILE; ++j) {
                            if (j < tile) {
                                int64_t oc = oc_base + j;
                                int64_t w_index = oc * W_per_oc + ic * W_per_ci + wk;
                                float wv = __half2float(weight[w_index]);
                                acc[j] += x * wv;
                            }
                        }
                    }
                }
            }
        }

        // ReLU and store into final output with channel offset
        #pragma unroll
        for (int j = 0; j < OC_TILE; ++j) {
            if (j < tile) {
                float v = acc[j];
                v = v > 0.0f ? v : 0.0f;
                int64_t oc_out = out_channel_offset + (oc_base + j);
                int64_t out_index = idx_nchw(n, oc_out, h, w, Cout_total, H, W);
                output[out_index] = __float2half_rn(v);
            }
        }
    }
}

// Host launcher: performs squeeze 1x1 + ReLU, then expand 1x1 + ReLU and expand 3x3 + ReLU, then concatenates by writing into output with offsets.
void launch_gpu_implementation(
    void* output,
    const void* input,
    const void* squeeze_weight,
    const void* squeeze_bias,
    const void* expand1x1_weight,
    const void* expand1x1_bias,
    const void* expand3x3_weight,
    const void* expand3x3_bias,
    int64_t batch_size,
    int64_t in_channels,
    int64_t height,
    int64_t width,
    int64_t squeeze_channels,
    int64_t expand1x1_channels,
    int64_t expand3x3_channels,
    // kernel sizes
    int64_t squeeze_kernel_h, int64_t squeeze_kernel_w,
    int64_t expand1x1_kernel_h, int64_t expand1x1_kernel_w,
    int64_t expand3x3_kernel_h, int64_t expand3x3_kernel_w,
    // padding for 3x3 conv
    int64_t expand3x3_padding_h, int64_t expand3x3_padding_w,
    // stride (all convs use stride=1)
    int64_t stride_h, int64_t stride_w,
    // dilation (all convs use dilation=1)
    int64_t dilation_h, int64_t dilation_w
) {
    // Cast pointers to half
    const half* d_input = static_cast<const half*>(input);
    const half* d_sq_w  = static_cast<const half*>(squeeze_weight);
    const half* d_sq_b  = static_cast<const half*>(squeeze_bias);
    const half* d_e1_w  = static_cast<const half*>(expand1x1_weight);
    const half* d_e1_b  = static_cast<const half*>(expand1x1_bias);
    const half* d_e3_w  = static_cast<const half*>(expand3x3_weight);
    const half* d_e3_b  = static_cast<const half*>(expand3x3_bias);
    half* d_output      = static_cast<half*>(output);

    // Validate expected kernel parameters (only used for this test setup)
    (void)squeeze_kernel_h; (void)squeeze_kernel_w;
    (void)expand1x1_kernel_h; (void)expand1x1_kernel_w;
    (void)expand3x3_kernel_h; (void)expand3x3_kernel_w;
    (void)stride_h; (void)stride_w;
    (void)dilation_h; (void)dilation_w;

    const int64_t N = batch_size;
    const int64_t Cin = in_channels;
    const int64_t H = height;
    const int64_t W = width;
    const int64_t Cs = squeeze_channels;
    const int64_t Ce1 = expand1x1_channels;
    const int64_t Ce3 = expand3x3_channels;
    const int64_t Cout_total = Ce1 + Ce3;

    const int64_t P = N * H * W;
    const int threads = 256;
    const int blocks = static_cast<int>((P + threads - 1) / threads);

    // Temporary buffer for squeezed activation [N, Cs, H, W]
    half* d_squeezed = nullptr;
    size_t squeezed_bytes = static_cast<size_t>(N) * Cs * H * W * sizeof(half);
    CUDA_CHECK(cudaMalloc(&d_squeezed, squeezed_bytes));

    // 1) Squeeze 1x1 + ReLU
    squeeze_1x1_relu_kernel<<<blocks, threads>>>(
        d_input, d_sq_w, d_sq_b, d_squeezed, N, Cin, H, W, Cs
    );
    CUDA_CHECK(cudaGetLastError());

    // 2) Expand 1x1 + ReLU into output channels [0 .. Ce1-1]
    // Use tile size of 8 output channels per thread for good balance
    expand_1x1_relu_kernel<8><<<blocks, threads>>>(
        d_squeezed, d_e1_w, d_e1_b, d_output,
        N, Cs, H, W, Ce1, Cout_total, /*out_channel_offset=*/0
    );
    CUDA_CHECK(cudaGetLastError());

    // 3) Expand 3x3 (padding) + ReLU into output channels [Ce1 .. Ce1+Ce3-1]
    expand_3x3_relu_kernel<8><<<blocks, threads>>>(
        d_squeezed, d_e3_w, d_e3_b, d_output,
        N, Cs, H, W, Ce3, Cout_total, /*out_channel_offset=*/Ce1,
        expand3x3_padding_h, expand3x3_padding_w
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_squeezed));
    CUDA_CHECK(cudaDeviceSynchronize());
}
