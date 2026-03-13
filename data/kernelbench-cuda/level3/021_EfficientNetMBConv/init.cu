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
#include <cmath>
#include <iostream>

// Utility: CUDA error check
#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; return; } } while(0)
#endif

// Clamp to [0, 6] in float
__device__ __forceinline__ float relu6f(float x) {
    return fminf(fmaxf(x, 0.0f), 6.0f);
}

// Indexing helper for NCHW
__device__ __forceinline__ int idx_nchw(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

// Expand 1x1 convolution + BatchNorm (eval) + ReLU6
// input:  [N, Cin, H, W]
// weight: [Cout, Cin, 1, 1] (contiguous)
// BN params (per Cout): gamma(weight), beta(bias), running_mean, running_var, eps
// output: [N, Cout, H, W]
__global__ void expand_1x1_bn_relu_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bn_gamma,
    const half* __restrict__ bn_beta,
    const half* __restrict__ bn_mean,
    const half* __restrict__ bn_var,
    float eps,
    half* __restrict__ output,
    int N, int Cin, int H, int W, int Cout
) {
    const int total = N * Cout * H * W;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += blockDim.x * gridDim.x) {
        int tmp = tid;
        const int w = tmp % W;          tmp /= W;
        const int h = tmp % H;          tmp /= H;
        const int oc = tmp % Cout;      tmp /= Cout;
        const int n = tmp;

        // Dot over Cin
        float acc = 0.0f;
        // weight index base for this oc
        const int w_base = oc * Cin;
#pragma unroll 4
        for (int ic = 0; ic < Cin; ++ic) {
            const half in_h = input[idx_nchw(n, ic, h, w, Cin, H, W)];
            const half w_h  = weight[w_base + ic];
            acc += __half2float(in_h) * __half2float(w_h);
        }

        // BN (eval): y = gamma * (x - mean) / sqrt(var + eps) + beta
        const float gamma = __half2float(bn_gamma[oc]);
        const float beta  = __half2float(bn_beta[oc]);
        const float mean  = __half2float(bn_mean[oc]);
        const float var   = __half2float(bn_var[oc]);

        float y = gamma * (acc - mean) * rsqrtf(var + eps) + beta;
        y = relu6f(y);

        output[idx_nchw(n, oc, h, w, Cout, H, W)] = __float2half_rn(y);
    }
}

// Depthwise KxK convolution with stride/padding + BatchNorm (eval) + ReLU6
// input:  [N, C, H, W]
// weight: [C, 1, K, K] (contiguous)
// BN params (per C): gamma, beta, running_mean, running_var, eps
// output: [N, C, OH, OW]
__global__ void depthwise_kxk_bn_relu_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bn_gamma,
    const half* __restrict__ bn_beta,
    const half* __restrict__ bn_mean,
    const half* __restrict__ bn_var,
    float eps,
    half* __restrict__ output,
    int N, int C, int H, int W,
    int K, int stride, int padding,
    int OH, int OW
) {
    const int total = N * C * OH * OW;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += blockDim.x * gridDim.x) {
        int tmp = tid;
        const int ow = tmp % OW;    tmp /= OW;
        const int oh = tmp % OH;    tmp /= OH;
        const int c  = tmp % C;     tmp /= C;
        const int n  = tmp;

        const int h_start = oh * stride - padding;
        const int w_start = ow * stride - padding;

        float acc = 0.0f;
        const int wbase_c = c * K * K;
#pragma unroll 1
        for (int kh = 0; kh < K; ++kh) {
            const int ih = h_start + kh;
            if (ih < 0 || ih >= H) continue;
#pragma unroll 1
            for (int kw = 0; kw < K; ++kw) {
                const int iw = w_start + kw;
                if (iw < 0 || iw >= W) continue;

                const half in_h = input[idx_nchw(n, c, ih, iw, C, H, W)];
                const half w_h  = weight[wbase_c + kh * K + kw];
                acc += __half2float(in_h) * __half2float(w_h);
            }
        }

        // BN + ReLU6
        const float gamma = __half2float(bn_gamma[c]);
        const float beta  = __half2float(bn_beta[c]);
        const float mean  = __half2float(bn_mean[c]);
        const float var   = __half2float(bn_var[c]);

        float y = gamma * (acc - mean) * rsqrtf(var + eps) + beta;
        y = relu6f(y);

        output[((n * C + c) * OH + oh) * OW + ow] = __float2half_rn(y);
    }
}

// Project 1x1 convolution + BatchNorm (eval)
// input:  [N, Cin, H, W]
// weight: [Cout, Cin, 1, 1]
// BN params (per Cout)
// output: [N, Cout, H, W]
__global__ void project_1x1_bn_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bn_gamma,
    const half* __restrict__ bn_beta,
    const half* __restrict__ bn_mean,
    const half* __restrict__ bn_var,
    float eps,
    half* __restrict__ output,
    int N, int Cin, int H, int W, int Cout
) {
    const int total = N * Cout * H * W;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += blockDim.x * gridDim.x) {
        int tmp = tid;
        const int w = tmp % W;          tmp /= W;
        const int h = tmp % H;          tmp /= H;
        const int oc = tmp % Cout;      tmp /= Cout;
        const int n = tmp;

        float acc = 0.0f;
        const int w_base = oc * Cin;
#pragma unroll 4
        for (int ic = 0; ic < Cin; ++ic) {
            const half in_h = input[idx_nchw(n, ic, h, w, Cin, H, W)];
            const half w_h  = weight[w_base + ic];
            acc += __half2float(in_h) * __half2float(w_h);
        }

        // BN (eval)
        const float gamma = __half2float(bn_gamma[oc]);
        const float beta  = __half2float(bn_beta[oc]);
        const float mean  = __half2float(bn_mean[oc]);
        const float var   = __half2float(bn_var[oc]);

        float y = gamma * (acc - mean) * rsqrtf(var + eps) + beta;

        output[idx_nchw(n, oc, h, w, Cout, H, W)] = __float2half_rn(y);
    }
}

// Optional residual add: out += input (elementwise), both NCHW and same shape
__global__ void residual_add_kernel(
    half* __restrict__ out, const half* __restrict__ in,
    int N, int C, int H, int W
) {
    const int total = N * C * H * W;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += blockDim.x * gridDim.x) {
        float a = __half2float(out[tid]);
        float b = __half2float(in[tid]);
        out[tid] = __float2half_rn(a + b);
    }
}

// Host entrypoint (called by provided test harness)
void launch_gpu_implementation(
    void* output,
    const void* input,
    int64_t batch,
    int64_t in_channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_channels,
    int64_t hidden_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool use_residual,
    // expand conv params
    const void* expand_conv_weight,
    // expand BN params
    const void* expand_bn_weight,
    const void* expand_bn_bias,
    const void* expand_bn_running_mean,
    const void* expand_bn_running_var,
    double expand_bn_eps,
    // depthwise conv params
    const void* depthwise_conv_weight,
    const void* depthwise_bn_weight,
    const void* depthwise_bn_bias,
    const void* depthwise_bn_running_mean,
    const void* depthwise_bn_running_var,
    double depthwise_bn_eps,
    // project conv params
    const void* project_conv_weight,
    const void* project_bn_weight,
    const void* project_bn_bias,
    const void* project_bn_running_mean,
    const void* project_bn_running_var,
    double project_bn_eps
) {
    // Cast pointers
    const half* in_ptr = static_cast<const half*>(input);
    half* out_ptr = static_cast<half*>(output);

    const half* expand_w = static_cast<const half*>(expand_conv_weight);
    const half* expand_bn_gamma = static_cast<const half*>(expand_bn_weight);
    const half* expand_bn_beta  = static_cast<const half*>(expand_bn_bias);
    const half* expand_bn_mean  = static_cast<const half*>(expand_bn_running_mean);
    const half* expand_bn_var   = static_cast<const half*>(expand_bn_running_var);

    const half* depthwise_w = static_cast<const half*>(depthwise_conv_weight);
    const half* depth_bn_gamma = static_cast<const half*>(depthwise_bn_weight);
    const half* depth_bn_beta  = static_cast<const half*>(depthwise_bn_bias);
    const half* depth_bn_mean  = static_cast<const half*>(depthwise_bn_running_mean);
    const half* depth_bn_var   = static_cast<const half*>(depthwise_bn_running_var);

    const half* project_w = static_cast<const half*>(project_conv_weight);
    const half* project_bn_gamma = static_cast<const half*>(project_bn_weight);
    const half* project_bn_beta  = static_cast<const half*>(project_bn_bias);
    const half* project_bn_mean  = static_cast<const half*>(project_bn_running_mean);
    const half* project_bn_var   = static_cast<const half*>(project_bn_running_var);

    const int N  = static_cast<int>(batch);
    const int Cin = static_cast<int>(in_channels);
    const int H  = static_cast<int>(in_h);
    const int W  = static_cast<int>(in_w);
    const int Cmid = static_cast<int>(hidden_channels);
    const int K  = static_cast<int>(kernel_size);
    const int S  = static_cast<int>(stride);
    const int P  = static_cast<int>(padding);
    const int Cout = static_cast<int>(out_channels);

    // Output dims after depthwise (and thus for project)
    const int OH = (H + 2 * P - K) / S + 1;
    const int OW = (W + 2 * P - K) / S + 1;

    // Allocate intermediate buffers
    half* expand_out = nullptr;   // [N, Cmid, H, W]
    half* depth_out = nullptr;    // [N, Cmid, OH, OW]

    size_t bytes_expand = static_cast<size_t>(N) * Cmid * H * W * sizeof(half);
    size_t bytes_depth  = static_cast<size_t>(N) * Cmid * OH * OW * sizeof(half);

    CHECK_CUDA(cudaMalloc(&expand_out, bytes_expand));
    CHECK_CUDA(cudaMalloc(&depth_out,  bytes_depth));

    // Launch settings
    const int block_size = 256;

    // 1) Expand 1x1 + BN + ReLU6
    {
        const int total = N * Cmid * H * W;
        const int grid_size = (total + block_size - 1) / block_size;
        const int max_blocks = 65535;
        const int launch_blocks = grid_size > max_blocks ? max_blocks : grid_size;

        expand_1x1_bn_relu_kernel<<<launch_blocks, block_size>>>(
            in_ptr,
            expand_w,
            expand_bn_gamma, expand_bn_beta, expand_bn_mean, expand_bn_var,
            static_cast<float>(expand_bn_eps),
            expand_out,
            N, Cin, H, W, Cmid
        );
        CHECK_CUDA(cudaGetLastError());
    }

    // 2) Depthwise KxK + BN + ReLU6
    {
        const int total = N * Cmid * OH * OW;
        const int grid_size = (total + block_size - 1) / block_size;
        const int max_blocks = 65535;
        const int launch_blocks = grid_size > max_blocks ? max_blocks : grid_size;

        depthwise_kxk_bn_relu_kernel<<<launch_blocks, block_size>>>(
            expand_out,
            depthwise_w,
            depth_bn_gamma, depth_bn_beta, depth_bn_mean, depth_bn_var,
            static_cast<float>(depthwise_bn_eps),
            depth_out,
            N, Cmid, H, W,
            K, S, P,
            OH, OW
        );
        CHECK_CUDA(cudaGetLastError());
    }

    // 3) Project 1x1 + BN
    {
        const int total = N * Cout * OH * OW;
        const int grid_size = (total + block_size - 1) / block_size;
        const int max_blocks = 65535;
        const int launch_blocks = grid_size > max_blocks ? max_blocks : grid_size;

        project_1x1_bn_kernel<<<launch_blocks, block_size>>>(
            depth_out,
            project_w,
            project_bn_gamma, project_bn_beta, project_bn_mean, project_bn_var,
            static_cast<float>(project_bn_eps),
            out_ptr,
            N, Cmid, OH, OW, Cout
        );
        CHECK_CUDA(cudaGetLastError());
    }

    // 4) Optional residual add (only valid when shapes match)
    if (use_residual) {
        // By definition of MBConv: use_residual == (stride == 1 && in_channels == out_channels)
        // Also requires same spatial sizes (H == OH and W == OW)
        if (S == 1 && Cin == Cout && H == OH && W == OW) {
            const int total = N * Cout * OH * OW;
            const int grid_size = (total + block_size - 1) / block_size;
            const int max_blocks = 65535;
            const int launch_blocks = grid_size > max_blocks ? max_blocks : grid_size;

            residual_add_kernel<<<launch_blocks, block_size>>>(
                out_ptr, in_ptr, N, Cout, OH, OW
            );
            CHECK_CUDA(cudaGetLastError());
        }
    }

    // Cleanup
    CHECK_CUDA(cudaFree(expand_out));
    CHECK_CUDA(cudaFree(depth_out));

    // Ensure completion
    CHECK_CUDA(cudaDeviceSynchronize());
}
