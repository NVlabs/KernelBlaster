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
#include <stdint.h>
#include <cstdio>
#include <cmath>

// Simple CUDA error checker (debugging aid)
#ifndef NDEBUG
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); abort(); } } while (0)
#else
#define CUDA_CHECK(x) x
#endif

inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

// Conv2d NCHW with bias, stride, padding; accumulate in FP32, store in FP16
__global__ void conv2d_nchw_bias_kernel(
    const half* __restrict__ input,   // [N, C_in, H, W]
    const half* __restrict__ weight,  // [C_out, C_in, kH, kW]
    const half* __restrict__ bias,    // [C_out]
    half* __restrict__ output,        // [N, C_out, OH, OW]
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int OH, int OW
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * OH * OW;
    if (tid >= total) return;

    // Map linear tid to (n, oc, oh, ow)
    int ow = tid % OW;
    int tmp1 = tid / OW;
    int oh = tmp1 % OH;
    int tmp2 = tmp1 / OH;
    int oc = tmp2 % C_out;
    int n  = tmp2 / C_out;

    float acc = __half2float(bias[oc]);

    int in_h0 = oh * stride_h - pad_h;
    int in_w0 = ow * stride_w - pad_w;

    int in_n_stride = C_in * H * W;
    int in_c_stride = H * W;
    int w_oc_stride = C_in * kH * kW;
    int w_ic_stride = kH * kW;

    #pragma unroll
    for (int ic = 0; ic < C_in; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kH; ++kh) {
            int h_in = in_h0 + kh;
            if ((unsigned)h_in >= (unsigned)H) continue;
            #pragma unroll
            for (int kw = 0; kw < kW; ++kw) {
                int w_in = in_w0 + kw;
                if ((unsigned)w_in >= (unsigned)W) continue;

                int in_idx = n * in_n_stride + ic * in_c_stride + h_in * W + w_in;
                int w_idx  = oc * w_oc_stride + ic * w_ic_stride + kh * kW + kw;

                acc += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
            }
        }
    }

    int out_idx = n * (C_out * OH * OW) + oc * (OH * OW) + oh * OW + ow;
    output[out_idx] = __float2half_rn(acc);
}

// ReLU kernel (can be used in-place: x and y can alias)
__global__ void relu_kernel(const half* __restrict__ x, half* __restrict__ y, int64_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    float v = __half2float(x[tid]);
    if (v < 0.f) v = 0.f;
    y[tid] = __float2half_rn(v);
}

// MaxPool2d NCHW without padding, general kernel/stride. Accumulate in FP32.
__global__ void maxpool2d_nchw_kernel(
    const half* __restrict__ input,  // [N, C, H, W]
    half* __restrict__ output,       // [N, C, OH, OW]
    int N, int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int OH, int OW
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OH * OW;
    if (tid >= total) return;

    int ow = tid % OW;
    int t1 = tid / OW;
    int oh = t1 % OH;
    int t2 = t1 / OH;
    int c  = t2 % C;
    int n  = t2 / C;

    int h_start = oh * stride_h;
    int w_start = ow * stride_w;

    int in_n_stride = C * H * W;
    int in_c_stride = H * W;

    float max_val = -INFINITY;

    #pragma unroll
    for (int kh = 0; kh < kH; ++kh) {
        int h_in = h_start + kh;
        if (h_in >= H) break;
        #pragma unroll
        for (int kw = 0; kw < kW; ++kw) {
            int w_in = w_start + kw;
            if (w_in >= W) break;
            int in_idx = n * in_n_stride + c * in_c_stride + h_in * W + w_in;
            max_val = fmaxf(max_val, __half2float(input[in_idx]));
        }
    }

    int out_idx = n * (C * OH * OW) + c * (OH * OW) + oh * OW + ow;
    output[out_idx] = __float2half_rn(max_val);
}

// Linear: Y = X * W^T + b; X [N,K], W [M,K], b [M], Y [N,M]; accumulate in FP32, store FP16
__global__ void linear_gemm_bias_kernel(
    const half* __restrict__ X,      // [N, K]
    const half* __restrict__ W,      // [M, K]
    const half* __restrict__ bias,   // [M]
    half* __restrict__ Y,            // [N, M]
    int N, int M, int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * M;
    if (tid >= total) return;

    int m = tid % M;
    int n = tid / M;

    float acc = __half2float(bias[m]);

    int x_row = n * K;
    int w_row = m * K;

    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        acc += __half2float(X[x_row + k]) * __half2float(W[w_row + k]);
    }

    Y[tid] = __float2half_rn(acc);
}

// Public entry point used by the harness
void launch_gpu_implementation(
    void* output,
    const void* input,
    const void* conv1_weight,
    const void* conv1_bias,
    const void* conv2_weight,
    const void* conv2_bias,
    const void* fc1_weight,
    const void* fc1_bias,
    const void* fc2_weight,
    const void* fc2_bias,
    const void* fc3_weight,
    const void* fc3_bias,
    int64_t batch_size,
    int64_t in_channels,
    int64_t in_h,
    int64_t in_w,
    // Conv1 params
    int64_t conv1_out_channels,
    int64_t conv1_kernel_h,
    int64_t conv1_kernel_w,
    int64_t conv1_stride_h,
    int64_t conv1_stride_w,
    int64_t conv1_pad_h,
    int64_t conv1_pad_w,
    // Pool params
    int64_t pool_kernel_h,
    int64_t pool_kernel_w,
    int64_t pool_stride_h,
    int64_t pool_stride_w,
    // Conv2 params
    int64_t conv2_out_channels,
    int64_t conv2_kernel_h,
    int64_t conv2_kernel_w,
    int64_t conv2_stride_h,
    int64_t conv2_stride_w,
    int64_t conv2_pad_h,
    int64_t conv2_pad_w,
    // Linear params
    int64_t fc1_in_features,
    int64_t fc1_out_features,
    int64_t fc2_out_features,
    int64_t fc3_out_features
) {
    // Cast inputs to half pointers
    const half* x_in  = static_cast<const half*>(input);
    const half* w1    = static_cast<const half*>(conv1_weight);
    const half* b1    = static_cast<const half*>(conv1_bias);
    const half* w2    = static_cast<const half*>(conv2_weight);
    const half* b2    = static_cast<const half*>(conv2_bias);
    const half* wfc1  = static_cast<const half*>(fc1_weight);
    const half* bfc1  = static_cast<const half*>(fc1_bias);
    const half* wfc2  = static_cast<const half*>(fc2_weight);
    const half* bfc2  = static_cast<const half*>(fc2_bias);
    const half* wfc3  = static_cast<const half*>(fc3_weight);
    const half* bfc3  = static_cast<const half*>(fc3_bias);
    half* y_out       = static_cast<half*>(output);

    // Shapes and params
    const int N  = static_cast<int>(batch_size);
    const int C0 = static_cast<int>(in_channels);
    const int H0 = static_cast<int>(in_h);
    const int W0 = static_cast<int>(in_w);

    const int C1  = static_cast<int>(conv1_out_channels);
    const int K1H = static_cast<int>(conv1_kernel_h);
    const int K1W = static_cast<int>(conv1_kernel_w);
    const int S1H = static_cast<int>(conv1_stride_h);
    const int S1W = static_cast<int>(conv1_stride_w);
    const int P1H = static_cast<int>(conv1_pad_h);
    const int P1W = static_cast<int>(conv1_pad_w);

    const int PKH = static_cast<int>(pool_kernel_h);
    const int PKW = static_cast<int>(pool_kernel_w);
    const int PSH = static_cast<int>(pool_stride_h);
    const int PSW = static_cast<int>(pool_stride_w);

    const int C2  = static_cast<int>(conv2_out_channels);
    const int K2H = static_cast<int>(conv2_kernel_h);
    const int K2W = static_cast<int>(conv2_kernel_w);
    const int S2H = static_cast<int>(conv2_stride_h);
    const int S2W = static_cast<int>(conv2_stride_w);
    const int P2H = static_cast<int>(conv2_pad_h);
    const int P2W = static_cast<int>(conv2_pad_w);

    const int FC1_K = static_cast<int>(fc1_in_features);   // Expect 16*5*5
    const int FC1_M = static_cast<int>(fc1_out_features);  // 120
    const int FC2_M = static_cast<int>(fc2_out_features);  // 84
    const int FC3_M = static_cast<int>(fc3_out_features);  // num_classes

    // Derived dims
    const int H1 = (H0 + 2 * P1H - K1H) / S1H + 1;
    const int W1 = (W0 + 2 * P1W - K1W) / S1W + 1;

    const int H1p = (H1 - PKH) / PSH + 1;
    const int W1p = (W1 - PKW) / PSW + 1;

    const int H2 = (H1p + 2 * P2H - K2H) / S2H + 1;
    const int W2 = (W1p + 2 * P2W - K2W) / S2W + 1;

    const int H2p = (H2 - PKH) / PSH + 1;
    const int W2p = (W2 - PKW) / PSW + 1;

    // Buffers for intermediates
    half *conv1_out = nullptr, *pool1_out = nullptr;
    half *conv2_out = nullptr, *pool2_out = nullptr;
    half *fc1_out   = nullptr, *fc2_out   = nullptr;

    size_t conv1_bytes = static_cast<size_t>(N) * C1 * H1 * W1 * sizeof(half);
    size_t pool1_bytes = static_cast<size_t>(N) * C1 * H1p * W1p * sizeof(half);
    size_t conv2_bytes = static_cast<size_t>(N) * C2 * H2 * W2 * sizeof(half);
    size_t pool2_bytes = static_cast<size_t>(N) * C2 * H2p * W2p * sizeof(half);
    size_t fc1_bytes   = static_cast<size_t>(N) * FC1_M * sizeof(half);
    size_t fc2_bytes   = static_cast<size_t>(N) * FC2_M * sizeof(half);

    CUDA_CHECK(cudaMalloc(&conv1_out, conv1_bytes));
    CUDA_CHECK(cudaMalloc(&pool1_out, pool1_bytes));
    CUDA_CHECK(cudaMalloc(&conv2_out, conv2_bytes));
    CUDA_CHECK(cudaMalloc(&pool2_out, pool2_bytes));
    CUDA_CHECK(cudaMalloc(&fc1_out,   fc1_bytes));
    CUDA_CHECK(cudaMalloc(&fc2_out,   fc2_bytes));

    const int threads = 256;

    // Conv1
    {
        int total = N * C1 * H1 * W1;
        int blocks = div_up_int(total, threads);
        conv2d_nchw_bias_kernel<<<blocks, threads>>>(
            x_in, w1, b1, conv1_out,
            N, C0, H0, W0,
            C1, K1H, K1W,
            S1H, S1W, P1H, P1W,
            H1, W1
        );
    }
    // ReLU after Conv1 (in-place)
    {
        int64_t numel = static_cast<int64_t>(N) * C1 * H1 * W1;
        int blocks = div_up_int((int)numel, threads);
        relu_kernel<<<blocks, threads>>>(conv1_out, conv1_out, numel);
    }
    // MaxPool1
    {
        int total = N * C1 * H1p * W1p;
        int blocks = div_up_int(total, threads);
        maxpool2d_nchw_kernel<<<blocks, threads>>>(
            conv1_out, pool1_out,
            N, C1, H1, W1,
            PKH, PKW, PSH, PSW,
            H1p, W1p
        );
    }

    // Conv2
    {
        int total = N * C2 * H2 * W2;
        int blocks = div_up_int(total, threads);
        conv2d_nchw_bias_kernel<<<blocks, threads>>>(
            pool1_out, w2, b2, conv2_out,
            N, C1, H1p, W1p,
            C2, K2H, K2W,
            S2H, S2W, P2H, P2W,
            H2, W2
        );
    }
    // ReLU after Conv2 (in-place)
    {
        int64_t numel = static_cast<int64_t>(N) * C2 * H2 * W2;
        int blocks = div_up_int((int)numel, threads);
        relu_kernel<<<blocks, threads>>>(conv2_out, conv2_out, numel);
    }
    // MaxPool2
    {
        int total = N * C2 * H2p * W2p;
        int blocks = div_up_int(total, threads);
        maxpool2d_nchw_kernel<<<blocks, threads>>>(
            conv2_out, pool2_out,
            N, C2, H2, W2,
            PKH, PKW, PSH, PSW,
            H2p, W2p
        );
    }

    // Flatten pool2_out [N, C2, H2p, W2p] to [N, K_flat] by pointer alias
    const int K_flat = C2 * H2p * W2p;

    // FC1
    {
        int total = N * FC1_M;
        int blocks = div_up_int(total, threads);
        linear_gemm_bias_kernel<<<blocks, threads>>>(
            pool2_out, wfc1, bfc1, fc1_out,
            N, FC1_M, K_flat
        );
    }
    // ReLU after FC1 (in-place)
    {
        int64_t numel = static_cast<int64_t>(N) * FC1_M;
        int blocks = div_up_int((int)numel, threads);
        relu_kernel<<<blocks, threads>>>(fc1_out, fc1_out, numel);
    }

    // FC2
    {
        int total = N * FC2_M;
        int blocks = div_up_int(total, threads);
        linear_gemm_bias_kernel<<<blocks, threads>>>(
            fc1_out, wfc2, bfc2, fc2_out,
            N, FC2_M, FC1_M
        );
    }
    // ReLU after FC2 (in-place)
    {
        int64_t numel = static_cast<int64_t>(N) * FC2_M;
        int blocks = div_up_int((int)numel, threads);
        relu_kernel<<<blocks, threads>>>(fc2_out, fc2_out, numel);
    }

    // FC3 -> output (no activation)
    {
        int total = N * FC3_M;
        int blocks = div_up_int(total, threads);
        linear_gemm_bias_kernel<<<blocks, threads>>>(
            fc2_out, wfc3, bfc3, y_out,
            N, FC3_M, FC2_M
        );
    }

    // Ensure kernels finish before freeing
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporaries
    CUDA_CHECK(cudaFree(conv1_out));
    CUDA_CHECK(cudaFree(pool1_out));
    CUDA_CHECK(cudaFree(conv2_out));
    CUDA_CHECK(cudaFree(pool2_out));
    CUDA_CHECK(cudaFree(fc1_out));
    CUDA_CHECK(cudaFree(fc2_out));
}
