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
// Depthwise 2D Convolution (groups=in_channels, out_channels=in_channels, no bias, fp16 I/O, fp32 accum)
// Input/output NCHW, weight [out_channels, 1, k, k], bias==nullptr
// Kernel: Fast CUDA, coalesced, shared memory tiles for input, fp16 I/O, fp32 accum

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

// Utility for checking CUDA errors
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// CUDA kernel for depthwise 2D convolution, NCHW, weight [C,1,k,k], fp16 I/O, fp32 accum
template <int TILE_H, int TILE_W>
__global__ void depthwise_conv2d_nchw_fp16(
    const half* __restrict__ input,      // [N, C, Hin, Win]
    const half* __restrict__ weight,     // [C, 1, K, K]
    half* __restrict__ output,           // [N, C, Hout, Wout]
    int N, int C, int Hin, int Win,
    int K, int stride, int padding,
    int Hout, int Wout
) {
    // Each block computes a tile of (TILE_H, TILE_W) in output
    // Shared memory size for input tile: (TILE_H-1)*stride+K, (TILE_W-1)*stride+K
    extern __shared__ half smem[];
    half* tile = smem;

    // Block indices
    int n = blockIdx.z;
    int c = blockIdx.y;
    int tile_oh = blockIdx.x / ((Wout + TILE_W - 1) / TILE_W);
    int tile_ow = blockIdx.x % ((Wout + TILE_W - 1) / TILE_W);

    int oh0 = tile_oh * TILE_H;
    int ow0 = tile_ow * TILE_W;

    // Input tile covers [ih0, ih0+tile_H_eff), [iw0, iw0+tile_W_eff)
    int ih0 = oh0 * stride - padding;
    int iw0 = ow0 * stride - padding;

    // Effective tile size (may be truncated at right/bottom edges)
    int tile_H_eff = min(TILE_H, Hout - oh0);
    int tile_W_eff = min(TILE_W, Wout - ow0);

    // Shared memory shape
    constexpr int S_TILE_H = TILE_H + 16; // pad up for K up to 15
    constexpr int S_TILE_W = TILE_W + 16;

    int s_tile_h = (tile_H_eff - 1) * stride + K;
    int s_tile_w = (tile_W_eff - 1) * stride + K;

    // Cooperative loading of shared input tile
    for (int th = threadIdx.y; th < s_tile_h; th += blockDim.y) {
        for (int tw = threadIdx.x; tw < s_tile_w; tw += blockDim.x) {
            int ih = ih0 + th;
            int iw = iw0 + tw;
            half val = __float2half(0.f);
            if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                // NCHW
                val = input[((n * C + c) * Hin + ih) * Win + iw];
            }
            tile[th * s_tile_w + tw] = val;
        }
    }
    __syncthreads();

    // Each thread computes one output pixel in the tile
    for (int local_oh = threadIdx.y; local_oh < tile_H_eff; local_oh += blockDim.y) {
        for (int local_ow = threadIdx.x; local_ow < tile_W_eff; local_ow += blockDim.x) {
            int oh = oh0 + local_oh;
            int ow = ow0 + local_ow;
            if (oh < Hout && ow < Wout) {
                float acc = 0.f;
                // Depthwise: weight is [c, 1, K, K]
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int sh = local_oh * stride + kh;
                        int sw = local_ow * stride + kw;
                        half ival = tile[sh * s_tile_w + sw];
                        half wval = weight[((c * 1 + 0) * K + kh) * K + kw];
                        acc += __half2float(ival) * __half2float(wval);
                    }
                }
                // No bias
                output[((n * C + c) * Hout + oh) * Wout + ow] = __float2half(acc);
            }
        }
    }
}

// Host launcher
void launch_gpu_implementation(
    void* output,                  // output tensor (float16, GPU)
    void* input,                   // input tensor (float16, GPU)
    void* weight,                  // conv2d weight (float16, GPU)
    void* bias,                    // nullptr, since bias=False
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,          // == in_channels for depthwise
    int64_t height_in,
    int64_t width_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t height_out,
    int64_t width_out
) {
    assert(bias == nullptr); // No bias supported in this implementation
    assert(in_channels == out_channels);

    constexpr int TILE_H = 8;
    constexpr int TILE_W = 16;
    dim3 block(TILE_W, TILE_H); // 16*8=128 threads per block
    int num_tiles_h = (height_out + TILE_H - 1) / TILE_H;
    int num_tiles_w = (width_out + TILE_W - 1) / TILE_W;
    dim3 grid(num_tiles_h * num_tiles_w, in_channels, batch_size);

    // Shared memory per block: enough for input tile
    int s_tile_h = (TILE_H - 1) * stride + kernel_size;
    int s_tile_w = (TILE_W - 1) * stride + kernel_size;
    size_t smem_bytes = s_tile_h * s_tile_w * sizeof(half);

    depthwise_conv2d_nchw_fp16<TILE_H, TILE_W>
        <<<grid, block, smem_bytes>>>(
            static_cast<const half*>(input),
            static_cast<const half*>(weight),
            static_cast<half*>(output),
            (int)batch_size, (int)in_channels,
            (int)height_in, (int)width_in,
            (int)kernel_size, (int)stride, (int)padding,
            (int)height_out, (int)width_out
        );
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
