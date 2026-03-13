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

// Utility: fast atomic add for half on recent GPUs (volta+)
__device__ inline half atomicAdd_half(half* address, half val) {
#if __CUDA_ARCH__ >= 700
    return atomicAdd(address, val);
#else
    float old = __half2float(*address);
    float assumed;
    do {
        assumed = old;
        old = atomicCAS((unsigned int*)address, __float_as_uint(assumed), __float_as_uint(assumed + __half2float(val)));
    } while (assumed != old);
    return __float2half(old);
#endif
}

// Utility: fast atomic add for fp32 accumulation on output buffer (accumulate in fp32, cast to fp16 at store)
__device__ inline void atomicAdd_half_fp32(half* address, float val) {
#if __CUDA_ARCH__ >= 700
    // Atomically update as fp16
    unsigned int* addr_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *addr_as_ui;
    half hval = __float2half(val);
    half old_half;
    unsigned int assumed;
    do {
        assumed = old;
        if (((size_t)address & 2) == 0) {
            old_half = __ushort_as_half(old & 0xFFFF);
            hval = __float2half(__half2float(old_half) + val);
            old = atomicCAS(addr_as_ui, assumed, (old & 0xFFFF0000) | __half_as_ushort(hval));
        } else {
            old_half = __ushort_as_half((old >> 16) & 0xFFFF);
            hval = __float2half(__half2float(old_half) + val);
            old = atomicCAS(addr_as_ui, assumed, (old & 0xFFFF) | (__half_as_ushort(hval) << 16));
        }
    } while (assumed != old);
#else
    // Fallback: atomicAdd on float
    float* out_f = (float*)address;
    atomicAdd(out_f, val);
#endif
}

// Utility: convert fp16 to fp32
__device__ __forceinline__ float __half2float_safe(half h) {
#if __CUDA_ARCH__ >= 530
    return __half2float(h);
#else
    return float(h);
#endif
}

// Host utility for division with ceiling
inline int div_up(int x, int y) { return (x + y - 1) / y; }

// CUDA Kernel for 2D Transposed Convolution (Deconvolution)
// All tensors are assumed to be fp16, NCHW contiguous
// Weight shape: (in_channels, out_channels/groups, kernel_size, kernel_size)
// Input shape:  (batch_size, in_channels, height_in, width_in)
// Output shape: (batch_size, out_channels, height_out, width_out)
// Accumulation is in fp32 for numerical stability
template <int TILE_B = 2, int TILE_OC = 8, int TILE_OH = 4, int TILE_OW = 32>
__global__ void conv_transpose2d_nchw_fp16_kernel(
    half* __restrict__ output,           // (N, OC, H_out, W_out)
    const half* __restrict__ input,      // (N, IC, H_in, W_in)
    const half* __restrict__ weight,     // (IC, OC_per_group, K, K)
    const half* __restrict__ bias,       // (OC) or nullptr
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    // Tiling: Each block computes TILE_B batches, TILE_OC output channels, TILE_OH output rows, TILE_OW output cols
    int n0 = blockIdx.z * TILE_B;
    int oc0 = blockIdx.y * TILE_OC;
    int oh0 = blockIdx.x * TILE_OH;
    // Each thread computes several output columns (ow)
    int tid = threadIdx.x;
    // Each thread computes a subset of (b, oc, oh, ow)
    // For simplicity, we use 1D thread block, 128 or 256 threads
    constexpr int VEC_OW = TILE_OW; // Each thread computes 1 output element along width

    // Shared memory for input tile and weights (optional, not used in this baseline)

    // Loop over batch, output channel, output row
    for (int b = n0; b < min(n0 + TILE_B, batch_size); ++b) {
        for (int oc = oc0; oc < min(oc0 + TILE_OC, out_channels); ++oc) {
            for (int oh = oh0; oh < min(oh0 + TILE_OH, height_out); ++oh) {
                for (int ow = tid; ow < width_out; ow += blockDim.x) {
                    // Deconv: output[b, oc, oh, ow] = sum_ic sum_ky sum_kx input[b, ic, ih, iw] * weight[ic, oc, ky, kx]
                    float acc = 0.0f;
                    int group = oc / (out_channels / groups);
                    int ocg = oc % (out_channels / groups);
                    for (int icg = 0; icg < in_channels / groups; ++icg) {
                        int ic = group * (in_channels / groups) + icg;
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                // Compute corresponding input coordinates
                                int ih = (oh + padding - ky) / stride;
                                int iw = (ow + padding - kx) / stride;
                                // Check if input pixel contributes to this output pixel
                                if (((oh + padding - ky) % stride == 0) && ((ow + padding - kx) % stride == 0)) {
                                    if (ih >= 0 && ih < height_in && iw >= 0 && iw < width_in) {
                                        int in_idx = ((b * in_channels + ic) * height_in + ih) * width_in + iw;
                                        int w_idx = (((icg * (out_channels / groups) + ocg) * kernel_size + ky) * kernel_size) + kx;
                                        float inp = __half2float_safe(input[in_idx]);
                                        float wgt = __half2float_safe(weight[w_idx]);
                                        acc += inp * wgt;
                                    }
                                }
                            }
                        }
                    }
                    // Add bias if present
                    if (bias != nullptr) {
                        acc += __half2float_safe(bias[oc]);
                    }
                    // Write result (fp32->fp16)
                    int out_idx = ((b * out_channels + oc) * height_out + oh) * width_out + ow;
                    output[out_idx] = __float2half_rn(acc);
                }
            }
        }
    }
}

// Host launcher
void launch_gpu_implementation(
    void* output,                 // Output tensor (fp16, CUDA)
    void* input,                  // Input tensor (fp16, CUDA)
    void* weight,                 // Weight tensor (fp16, CUDA)
    void* bias,                   // Bias tensor (nullptr if not used, fp16, CUDA)
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int height_in,
    int width_in,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    // Compute output shape (see torch.nn.ConvTranspose2d calculation)
    int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int width_out  = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Block and grid configuration for tiling
    constexpr int TILE_B = 2, TILE_OC = 8, TILE_OH = 4, TILE_OW = 32; // These are tunable
    constexpr int THREADS = 128;

    dim3 block(THREADS);
    dim3 grid(
        div_up(height_out, TILE_OH),                       // output height tiles
        div_up(out_channels, TILE_OC),                     // output channel tiles
        div_up(batch_size, TILE_B)                         // batch tiles
    );

    conv_transpose2d_nchw_fp16_kernel<TILE_B, TILE_OC, TILE_OH, TILE_OW>
        <<<grid, block>>>(
            static_cast<half*>(output),
            static_cast<const half*>(input),
            static_cast<const half*>(weight),
            static_cast<const half*>(bias),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            height_in,
            width_in,
            height_out,
            width_out,
            stride,
            padding,
            output_padding,
            groups
        );

    cudaDeviceSynchronize();
}

