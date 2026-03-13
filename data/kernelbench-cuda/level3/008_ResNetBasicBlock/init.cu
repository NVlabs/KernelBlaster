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
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

// Simple CUDA error checking
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t err = (expr);                                                        \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err),         \
                    __FILE__, __LINE__);                                                 \
        }                                                                                \
    } while (0)
#endif

// Index helper for NCHW contiguous layout
__host__ __device__ __forceinline__ size_t idx_nchw(int n, int c, int h, int w, int C, int H, int W) {
    return ((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)h * W) + (size_t)w;
}

// Compute BN fused scale and bias: y = scale * x + bias, where
// scale = gamma / sqrt(var + eps), bias = beta - gamma * mean / sqrt(var + eps)
__global__ void compute_bn_scale_bias_kernel(
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    const half* __restrict__ running_mean,
    const half* __restrict__ running_var,
    double eps,
    float* __restrict__ out_scale,
    float* __restrict__ out_bias,
    int C)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float g = __half2float(gamma[c]);
    float b = __half2float(beta[c]);
    float m = __half2float(running_mean[c]);
    float v = __half2float(running_var[c]);
    float s = rsqrtf(v + (float)eps);
    float scale = g * s;
    float bias  = b - g * m * s;
    out_scale[c] = scale;
    out_bias[c]  = bias;
}

// Fused Conv2D (KxK) + BN (+ optional ReLU) for NCHW tensors.
// Supports general stride/padding. Special shared-memory path for K=3, stride=1, padding=1.
template<bool FuseRelu>
__global__ void conv_kxk_bn_kernel(
    const half* __restrict__ input,          // [N, C_in, H, W]
    const half* __restrict__ weight,         // [C_out, C_in, K, K]
    half* __restrict__ output,               // [N, C_out, OH, OW]
    const float* __restrict__ bn_scale,      // [C_out]
    const float* __restrict__ bn_bias,       // [C_out]
    int N, int C_in, int H, int W,
    int C_out,
    int K,
    int stride,
    int padding,
    int OH, int OW)
{
    // Map blockIdx.z into (n, co)
    int co = blockIdx.z % C_out;
    int n  = blockIdx.z / C_out;

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    bool valid = (ox < OW) && (oy < OH);

    float acc = 0.0f;

    // Specialized shared memory path for common 3x3 stride1 pad1
    if (K == 3 && stride == 1 && padding == 1) {
        const int tileH = blockDim.y + 2;
        const int tileW = blockDim.x + 2;
        extern __shared__ half smem[]; // size: tileH * tileW
        // Base coordinates of this output tile in input space (since stride=1, padding=1)
        int base_iy = blockIdx.y * blockDim.y - padding;
        int base_ix = blockIdx.x * blockDim.x - padding;

        // Loop over input channels
        for (int ci = 0; ci < C_in; ++ci) {
            // Cooperative load of input tile for this channel into shared memory
            for (int sy = threadIdx.y; sy < tileH; sy += blockDim.y) {
                int iy = base_iy + sy;
                bool in_y_ok = (iy >= 0) && (iy < H);
                for (int sx = threadIdx.x; sx < tileW; sx += blockDim.x) {
                    int ix = base_ix + sx;
                    half val = __float2half(0.0f);
                    if (in_y_ok && ix >= 0 && ix < W) {
                        size_t in_idx = idx_nchw(n, ci, iy, ix, C_in, H, W);
                        val = input[in_idx];
                    }
                    smem[sy * tileW + sx] = val;
                }
            }
            __syncthreads();

            if (valid) {
                // Each thread computes one output at (oy, ox)
                // For K=3, window [ty:ty+2], [tx:tx+2] in shared memory
                int ty = threadIdx.y;
                int tx = threadIdx.x;

                // weight layout: [co, ci, ky, kx]
                // Unroll the 3x3 MACs
#pragma unroll
                for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
                    for (int kx = 0; kx < 3; ++kx) {
                        half in_h = smem[(ty + ky) * tileW + (tx + kx)];
                        size_t w_idx = (((co * C_in + ci) * K + ky) * K + kx);
                        half w_h = weight[w_idx];
                        acc += __half2float(in_h) * __half2float(w_h);
                    }
                }
            }
            __syncthreads();
        }
    } else {
        // Generic path (no shared memory)
        if (valid) {
#pragma unroll 1
            for (int ci = 0; ci < C_in; ++ci) {
#pragma unroll 1
                for (int ky = 0; ky < K; ++ky) {
#pragma unroll 1
                    for (int kx = 0; kx < K; ++kx) {
                        int iy = oy * stride + ky - padding;
                        int ix = ox * stride + kx - padding;
                        if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
                            size_t in_idx = idx_nchw(n, ci, iy, ix, C_in, H, W);
                            size_t w_idx  = (((co * C_in + ci) * K + ky) * K + kx);
                            acc += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
                        }
                    }
                }
            }
        }
    }

    if (valid) {
        // Apply BN affine transform
        float y = acc * bn_scale[co] + bn_bias[co];
        if (FuseRelu) {
            y = fmaxf(y, 0.0f);
        }
        size_t out_idx = idx_nchw(n, co, oy, ox, C_out, OH, OW);
        output[out_idx] = __float2half_rn(y);
    }
}

// 1x1 Conv + BN (stride/padding supported). No ReLU.
__global__ void conv1x1_bn_kernel(
    const half* __restrict__ input,          // [N, C_in, H, W]
    const half* __restrict__ weight,         // [C_out, C_in, 1, 1]
    half* __restrict__ output,               // [N, C_out, OH, OW]
    const float* __restrict__ bn_scale,      // [C_out]
    const float* __restrict__ bn_bias,       // [C_out]
    int N, int C_in, int H, int W,
    int C_out,
    int stride,
    int padding,
    int OH, int OW)
{
    int co = blockIdx.z % C_out;
    int n  = blockIdx.z / C_out;

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= OW || oy >= OH) return;

    int iy = oy * stride - padding;
    int ix = ox * stride - padding;

    float acc = 0.0f;
    if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
#pragma unroll 1
        for (int ci = 0; ci < C_in; ++ci) {
            size_t in_idx = idx_nchw(n, ci, iy, ix, C_in, H, W);
            size_t w_idx  = (((co * C_in + ci) * 1 + 0) * 1 + 0);
            acc += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
        }
    }
    float y = acc * bn_scale[co] + bn_bias[co];
    size_t out_idx = idx_nchw(n, co, oy, ox, C_out, OH, OW);
    output[out_idx] = __float2half_rn(y);
}

// Elementwise add + ReLU: out = ReLU(out + id)
__global__ void add_relu_kernel(half* __restrict__ out, const half* __restrict__ id, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float y = __half2float(out[idx]) + __half2float(id[idx]);
    out[idx] = __float2half_rn(fmaxf(y, 0.0f));
}

// Launcher implementing the specified model forward pass.
void launch_gpu_implementation(
    void* output,
    const void* input,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out,
    // conv1 (3x3)
    const void* conv1_weight,
    int64_t conv1_stride,
    int64_t conv1_padding,
    int64_t conv1_kernel,
    // bn1
    const void* bn1_weight,
    const void* bn1_bias,
    const void* bn1_running_mean,
    const void* bn1_running_var,
    double bn1_eps,
    // conv2 (3x3)
    const void* conv2_weight,
    int64_t conv2_stride,
    int64_t conv2_padding,
    int64_t conv2_kernel,
    // bn2
    const void* bn2_weight,
    const void* bn2_bias,
    const void* bn2_running_mean,
    const void* bn2_running_var,
    double bn2_eps,
    // downsample conv (1x1)
    const void* downsample_conv_weight,
    int64_t downsample_conv_stride,
    int64_t downsample_conv_padding,
    int64_t downsample_conv_kernel,
    // downsample bn
    const void* downsample_bn_weight,
    const void* downsample_bn_bias,
    const void* downsample_bn_running_mean,
    const void* downsample_bn_running_var,
    double downsample_bn_eps
) {
    // Cast pointers
    const half* in_ptr  = static_cast<const half*>(input);
    half* out_ptr       = static_cast<half*>(output);

    const half* conv1_w = static_cast<const half*>(conv1_weight);
    const half* conv2_w = static_cast<const half*>(conv2_weight);
    const half* ds_w    = static_cast<const half*>(downsample_conv_weight);

    const half* bn1_gamma = static_cast<const half*>(bn1_weight);
    const half* bn1_beta  = static_cast<const half*>(bn1_bias);
    const half* bn1_mean  = static_cast<const half*>(bn1_running_mean);
    const half* bn1_var   = static_cast<const half*>(bn1_running_var);

    const half* bn2_gamma = static_cast<const half*>(bn2_weight);
    const half* bn2_beta  = static_cast<const half*>(bn2_bias);
    const half* bn2_mean  = static_cast<const half*>(bn2_running_mean);
    const half* bn2_var   = static_cast<const half*>(bn2_running_var);

    const half* dsg_gamma = static_cast<const half*>(downsample_bn_weight);
    const half* dsg_beta  = static_cast<const half*>(downsample_bn_bias);
    const half* dsg_mean  = static_cast<const half*>(downsample_bn_running_mean);
    const half* dsg_var   = static_cast<const half*>(downsample_bn_running_var);

    // Derived dimensions
    auto outDim = [&](int64_t H_in, int64_t K, int64_t pad, int64_t stride) -> int64_t {
        return (H_in + 2 * pad - K) / stride + 1;
    };

    const int64_t OH1 = outDim(H, conv1_kernel, conv1_padding, conv1_stride);
    const int64_t OW1 = outDim(W, conv1_kernel, conv1_padding, conv1_stride);

    const int64_t OH2 = outDim(OH1, conv2_kernel, conv2_padding, conv2_stride);
    const int64_t OW2 = outDim(OW1, conv2_kernel, conv2_padding, conv2_stride);

    const int64_t OHD = outDim(H, downsample_conv_kernel, downsample_conv_padding, downsample_conv_stride);
    const int64_t OWD = outDim(W, downsample_conv_kernel, downsample_conv_padding, downsample_conv_stride);

    // Allocate intermediate buffers:
    // act1: after conv1+bn1+relu, shape [N, C_out, OH1, OW1] (input to conv2)
    // identity: output of downsample path, shape [N, C_out, OHD, OWD]
    half* act1 = nullptr;
    half* identity = nullptr;
    size_t act1_elems = (size_t)N * C_out * OH1 * OW1;
    size_t id_elems   = (size_t)N * C_out * OHD * OWD;
    CUDA_CHECK(cudaMalloc(&act1, act1_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&identity, id_elems * sizeof(half)));

    // Allocate BN fused params (scale/bias) in float for each of the 3 BNs
    float *bn1_scale = nullptr, *bn1_bias_f = nullptr;
    float *bn2_scale = nullptr, *bn2_bias_f = nullptr;
    float *ds_scale = nullptr, *ds_bias_f = nullptr;
    CUDA_CHECK(cudaMalloc(&bn1_scale, C_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bn1_bias_f, C_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bn2_scale, C_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bn2_bias_f, C_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds_scale,  C_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds_bias_f, C_out * sizeof(float)));

    // Compute BN fused parameters
    {
        int threads = 256;
        int blocks = (int)((C_out + threads - 1) / threads);
        compute_bn_scale_bias_kernel<<<blocks, threads>>>(
            bn1_gamma, bn1_beta, bn1_mean, bn1_var, bn1_eps, bn1_scale, bn1_bias_f, (int)C_out);
        compute_bn_scale_bias_kernel<<<blocks, threads>>>(
            bn2_gamma, bn2_beta, bn2_mean, bn2_var, bn2_eps, bn2_scale, bn2_bias_f, (int)C_out);
        compute_bn_scale_bias_kernel<<<blocks, threads>>>(
            dsg_gamma, dsg_beta, dsg_mean, dsg_var, downsample_bn_eps, ds_scale, ds_bias_f, (int)C_out);
    }

    // Launch conv1 (K=3) + BN1 + ReLU into act1
    {
        dim3 block(16, 16, 1);
        dim3 grid((unsigned)((OW1 + block.x - 1) / block.x),
                  (unsigned)((OH1 + block.y - 1) / block.y),
                  (unsigned)(N * C_out));
        size_t smem_bytes = 0;
        // Use smem only if 3x3, stride=1, pad=1
        if (conv1_kernel == 3 && conv1_stride == 1 && conv1_padding == 1) {
            smem_bytes = (block.y + 2) * (block.x + 2) * sizeof(half);
        }
        conv_kxk_bn_kernel<true><<<grid, block, smem_bytes>>>(
            in_ptr, conv1_w, act1, bn1_scale, bn1_bias_f,
            (int)N, (int)C_in, (int)H, (int)W, (int)C_out,
            (int)conv1_kernel, (int)conv1_stride, (int)conv1_padding, (int)OH1, (int)OW1);
    }

    // Launch conv2 (K=3) + BN2 (no ReLU) into out_ptr
    {
        dim3 block(16, 16, 1);
        dim3 grid((unsigned)((OW2 + block.x - 1) / block.x),
                  (unsigned)((OH2 + block.y - 1) / block.y),
                  (unsigned)(N * C_out));
        size_t smem_bytes = 0;
        if (conv2_kernel == 3 && conv2_stride == 1 && conv2_padding == 1) {
            smem_bytes = (block.y + 2) * (block.x + 2) * sizeof(half);
        }
        conv_kxk_bn_kernel<false><<<grid, block, smem_bytes>>>(
            act1, conv2_w, out_ptr, bn2_scale, bn2_bias_f,
            (int)N, (int)C_out, (int)OH1, (int)OW1, (int)C_out,
            (int)conv2_kernel, (int)conv2_stride, (int)conv2_padding, (int)OH2, (int)OW2);
    }

    // Downsample path: 1x1 conv + BN into identity
    {
        dim3 block(16, 16, 1);
        dim3 grid((unsigned)((OWD + block.x - 1) / block.x),
                  (unsigned)((OHD + block.y - 1) / block.y),
                  (unsigned)(N * C_out));
        conv1x1_bn_kernel<<<grid, block>>>(
            in_ptr, ds_w, identity, ds_scale, ds_bias_f,
            (int)N, (int)C_in, (int)H, (int)W, (int)C_out,
            (int)downsample_conv_stride, (int)downsample_conv_padding, (int)OHD, (int)OWD);
    }

    // Add + ReLU: out_ptr = ReLU(out_ptr + identity)
    {
        size_t numel = (size_t)N * C_out * OH2 * OW2;
        int threads = 256;
        int blocks = (int)((numel + threads - 1) / threads);
        add_relu_kernel<<<blocks, threads>>>(out_ptr, identity, numel);
    }

    // Cleanup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(act1));
    CUDA_CHECK(cudaFree(identity));
    CUDA_CHECK(cudaFree(bn1_scale));
    CUDA_CHECK(cudaFree(bn1_bias_f));
    CUDA_CHECK(cudaFree(bn2_scale));
    CUDA_CHECK(cudaFree(bn2_bias_f));
    CUDA_CHECK(cudaFree(ds_scale));
    CUDA_CHECK(cudaFree(ds_bias_f));
}
