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
#include <iostream>
#include <cmath>

// Convolution + Multiply kernel
__global__ void conv3d_multiply_kernel(
    const half* input, const half* weight, const half* bias, const half* multiplier,
    half* temp1, int B, int IC, int OC, int D, int H, int W, int K, int OD, int OH, int OW
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elements = B * OC * OD * OH * OW;
    if (tid >= elements) return;

    int idx = tid;
    int b = idx / (OC * OD * OH * OW);
    int oc = (idx / (OD * OH * OW)) % OC;
    int od = (idx / (OH * OW)) % OD;
    int oh = (idx / OW) % OH;
    int ow = idx % OW;

    float acc = 0.0f;
    for (int kd = 0; kd < K; ++kd) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                for (int ic = 0; ic < IC; ++ic) {
                    int id = od + kd;
                    int ih = oh + kh;
                    int iw = ow + kw;
                    if (id < D && ih < H && iw < W) {
                        int inp_idx = b * IC * D * H * W + ic * D * H * W + id * H * W + ih * W + iw;
                        int w_idx = oc * IC * K * K * K + ic * K * K * K + kd * K * K + kh * K + kw;
                        acc += __half2float(__ldg(input + inp_idx)) * __half2float(__ldg(weight + w_idx));
                    }
                }
            }
        }
    }
    acc += __half2float(__ldg(bias + oc));
    acc *= __half2float(__ldg(multiplier + oc));
    temp1[tid] = __float2half_rn(acc);
}

// InstanceNorm Reduce kernel
__global__ void instance_norm_reduce_kernel(
    const half* temp1, float* mean, float* var, 
    int B, int OC, int OD, int OH, int OW
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elements = B * OC;
    if (tid >= elements) return;

    int b = tid / OC;
    int oc = tid % OC;
    int N = OD * OH * OW;

    float sum = 0.0f, sum_sq = 0.0f;
    for (int od = 0; od < OD; ++od) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int idx = b * OC * OD * OH * OW + oc * OD * OH * OW + od * OH * OW + oh * OW + ow;
                float val = __half2float(temp1[idx]);
                sum += val;
                sum_sq += val * val;
            }
        }
    }
    mean[tid] = sum / N;
    var[tid] = (sum_sq / N) - (mean[tid] * mean[tid]);
}

// Normalize + Clamp + Multiply kernel
__global__ void normalize_clamp_multiply_kernel(
    const half* temp1, const float* mean, const float* var, const half* multiplier,
    half* temp2, float clamp_min, float clamp_max, int B, int OC, int OD, int OH, int OW
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elements = B * OC * OD * OH * OW;
    if (tid >= elements) return;

    int idx = tid;
    int b = idx / (OC * OD * OH * OW);
    int oc = (idx / (OD * OH * OW)) % OC;
    int m_idx = b * OC + oc;

    float m = mean[m_idx];
    float v = var[m_idx];
    float eps = 1e-5f;
    float val = __half2float(temp1[idx]);
    val = (val - m) / sqrtf(v + eps);
    val = fminf(fmaxf(val, clamp_min), clamp_max);
    val *= __half2float(__ldg(multiplier + oc));
    temp2[idx] = __float2half_rn(val);
}

// Max reduction kernel
__global__ void max_reduce_kernel(
    const half* temp2, half* output, int B, int OC, int OD, int OH, int OW
) {
    int spatial = OD * OH * OW;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= B * spatial) return;

    int b = gidx / spatial;
    int s = gidx % spatial;

    float max_val = -INFINITY;
    for (int oc = 0; oc < OC; ++oc) {
        int idx = b * OC * spatial + oc * spatial + s;
        float val = __half2float(temp2[idx]);
        max_val = fmaxf(max_val, val);
    }
    output[gidx] = __float2half_rn(max_val);
}

void launch_gpu_implementation(void* output, void* input, 
                              void* conv_weight, void* conv_bias, void* multiplier,
                              float clamp_min, float clamp_max) {
    const int B = 128, IC = 3, OC = 16, D = 16, H = 32, W = 32, K = 3;
    const int OD = D - K + 1, OH = H - K + 1, OW = W - K + 1;

    // Allocate temporary buffers
    half *d_temp1, *d_temp2;
    float *d_mean, *d_var;
    size_t temp_size = B * OC * OD * OH * OW * sizeof(half);
    size_t mean_var_size = B * OC * sizeof(float);
    
    cudaMalloc(&d_temp1, temp_size);
    cudaMalloc(&d_temp2, temp_size);
    cudaMalloc(&d_mean, mean_var_size);
    cudaMalloc(&d_var, mean_var_size);

    // Launch conv+multiply kernel
    int conv_threads = 256;
    int conv_blocks = (B * OC * OD * OH * OW + conv_threads - 1) / conv_threads;
    conv3d_multiply_kernel<<<conv_blocks, conv_threads>>>(
        static_cast<const half*>(input),
        static_cast<const half*>(conv_weight),
        static_cast<const half*>(conv_bias),
        static_cast<const half*>(multiplier),
        d_temp1, B, IC, OC, D, H, W, K, OD, OH, OW
    );

    // Launch instance norm reduce
    int reduce_threads = 256;
    int reduce_blocks = (B * OC + reduce_threads - 1) / reduce_threads;
    instance_norm_reduce_kernel<<<reduce_blocks, reduce_threads>>>(
        d_temp1, d_mean, d_var, B, OC, OD, OH, OW
    );

    // Launch normalize+clamp+multiply
    int norm_threads = 256;
    int norm_blocks = (B * OC * OD * OH * OW + norm_threads - 1) / norm_threads;
    normalize_clamp_multiply_kernel<<<norm_blocks, norm_threads>>>(
        d_temp1, d_mean, d_var, static_cast<const half*>(multiplier),
        d_temp2, clamp_min, clamp_max, B, OC, OD, OH, OW
    );

    // Launch max reduction
    int max_threads = 256;
    int max_elements = B * OD * OH * OW;
    int max_blocks = (max_elements + max_threads - 1) / max_threads;
    max_reduce_kernel<<<max_blocks, max_threads>>>(
        d_temp2, static_cast<half*>(output), B, OC, OD, OH, OW
    );

    // Cleanup
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    cudaFree(d_mean);
    cudaFree(d_var);
}
