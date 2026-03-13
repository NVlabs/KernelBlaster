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
#include <cstdint>
#include <iostream>

// Utility: CUDA error check (lightweight)
#define CUDA_CHECK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return; } } while (0)

// Convert half to float
__device__ inline float h2f(half h) { return __half2float(h); }
__device__ inline half f2h(float f) { return __float2half_rn(f); }

// 1) Conv2D kernel: NCHW input, OIHW weights, stride = kernel_size = patch_size, no padding.
__global__ void conv2d_stride_eq_kernel(
    const half* __restrict__ input,   // [B, C, H, W]
    const half* __restrict__ weight,  // [E, C, K, K]
    const half* __restrict__ bias,    // [E] or nullptr
    half* __restrict__ output,        // [B, E, H/K, W/K]
    int B, int C, int H, int W, int E, int K, int stride)
{
    int OH = H / stride;
    int OW = W / stride;

    int total = B * E * OH * OW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % OW;
    int tmp = idx / OW;
    int oh = tmp % OH;
    tmp /= OH;
    int oc = tmp % E;
    int b  = tmp / E;

    float acc = 0.0f;

    int h_start = oh * stride;
    int w_start = ow * stride;

    const half* in_ptr = input + b * (C * H * W);
    const half* w_ptr  = weight + oc * (C * K * K);

    // Compute convolution over KxK and C
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            int h_in = h_start + kh;
            for (int kw = 0; kw < K; ++kw) {
                int w_in = w_start + kw;
                int in_index = ic * (H * W) + h_in * W + w_in;
                int w_index  = ic * (K * K) + kh * K + kw;
                acc += h2f(in_ptr[in_index]) * h2f(w_ptr[w_index]);
            }
        }
    }

    if (bias) acc += h2f(bias[oc]);

    output[idx] = f2h(acc);
}

// 2) Flatten kernel: from [B, E, PH, PW] -> [B, E*PH*PW]
__global__ void flatten_kernel(
    const half* __restrict__ input, // [B, E, PH, PW]
    half* __restrict__ output,      // [B, E*PH*PW]
    int B, int E, int PH, int PW)
{
    int flat_per_b = E * PH * PW;
    int total = B * flat_per_b;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int b = idx / flat_per_b;
    int rem = idx % flat_per_b;
    int c = rem / (PH * PW);
    int rem2 = rem % (PH * PW);
    int ph = rem2 / PW;
    int pw = rem2 % PW;

    int in_index = b * (E * PH * PW) + c * (PH * PW) + ph * PW + pw;
    output[idx] = input[in_index];
}

// 3) GEMM kernel: C = A * B^T + bias (A[M,K], B[N,K]), row-major, output half, accumulate in float.
// Optional ReLU activation if act_relu != 0
__global__ void gemm_bias_act_kernel(
    const half* __restrict__ A,        // [M, K]
    const half* __restrict__ B,        // [N, K] (to be transposed on-the-fly)
    const half* __restrict__ bias,     // [N] or nullptr
    half* __restrict__ C,              // [M, N]
    int M, int N, int K,
    int act_relu)
{
    int total = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int n = idx % N;
    int m = idx / N;

    float acc = 0.0f;
    const half* a_row = A + m * K;
    const half* b_row = B + n * K;

    // Unrolled loop for small K can help; leave as a simple loop for generality
    for (int k = 0; k < K; ++k) {
        acc += h2f(a_row[k]) * h2f(b_row[k]);
    }

    if (bias) acc += h2f(bias[n]);
    if (act_relu) acc = fmaxf(acc, 0.0f);

    C[idx] = f2h(acc);
}

// 4) Kernel to build sequence with CLS token: seq[b,0,:] = cls_token[0,0,:], seq[b,1,:] = embed[b,:]
__global__ void build_seq_with_cls_kernel(
    const half* __restrict__ embed,       // [B, E]
    const half* __restrict__ cls_token,   // [1, 1, E]
    half* __restrict__ seq,               // [B, 2, E]
    int B, int E)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * E;
    if (idx >= total) return;

    int e = idx % E;
    int b = idx / E;

    // CLS at position 0
    seq[b * (2 * E) + 0 * E + e] = cls_token[e]; // cls_token[0,0,e]
    // Token at position 1
    seq[b * (2 * E) + 1 * E + e] = embed[b * E + e];
}

// 5) Residual add: out = a + b
__global__ void residual_add_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = f2h(h2f(a[idx]) + h2f(b[idx]));
}

// 6) LayerNorm over last dim E; input [M, E] where M=B*S. y = (x-mean)/sqrt(var+eps)*gamma + beta
__global__ void layernorm_kernel(
    const half* __restrict__ x,       // [M, E]
    const half* __restrict__ gamma,   // [E] or nullptr
    const half* __restrict__ beta,    // [E] or nullptr
    half* __restrict__ y,             // [M, E]
    int M, int E, float eps)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const half* x_row = x + m * E;
    float mean = 0.0f;
    float var = 0.0f;

    // Compute mean
    for (int i = 0; i < E; ++i) mean += h2f(x_row[i]);
    mean /= (float)E;

    // Compute variance
    for (int i = 0; i < E; ++i) {
        float diff = h2f(x_row[i]) - mean;
        var += diff * diff;
    }
    var /= (float)E;
    float inv_std = rsqrtf(var + eps);

    for (int i = 0; i < E; ++i) {
        float xv = (h2f(x_row[i]) - mean) * inv_std;
        float g = gamma ? h2f(gamma[i]) : 1.0f;
        float b = beta  ? h2f(beta[i])  : 0.0f;
        y[m * E + i] = f2h(xv * g + b);
    }
}

// 7) Pointwise ReLU inplace for [M, N]
__global__ void relu_inplace_kernel(half* __restrict__ x, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float v = h2f(x[idx]);
    x[idx] = f2h(fmaxf(v, 0.0f));
}

// 8) Attention context compute for small S (here S=2).
// Input qkv: [B*S, 3E] (half). Output ctx: [B*S, E] (half).
// num_heads H, head_dim = E/H. Softmax over sequence length S for each head and query token.
__global__ void attention_context_kernel(
    const half* __restrict__ qkv,   // [B*S, 3E]
    half* __restrict__ ctx,         // [B*S, E]
    int B, int S, int E, int H)
{
    int head_dim = E / H;
    int total_threads = B * S * H;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    int h = tid % H;
    int rem = tid / H;
    int t_q = rem % S;
    int b = rem / S;

    int q_base = (b * S + t_q) * (3 * E) + 0;
    // Compute attention logits for k over S
    float scale = 1.0f / sqrtf((float)head_dim);

    float logits_max = -INFINITY;
    float logits_sum = 0.0f;
    float logits[8]; // S is small (2 here), allocate small buffer
#pragma unroll
    for (int tk = 0; tk < S; ++tk) {
        int k_base = (b * S + tk) * (3 * E) + E; // K segment offset
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < 1024; ++d) { // upper bound unroll guard; real loop bounds below
            if (d >= head_dim) break;
            float qv = h2f(qkv[q_base + h * head_dim + d]);
            float kv = h2f(qkv[k_base + h * head_dim + d]);
            dot += qv * kv;
        }
        float logit = dot * scale;
        logits[tk] = logit;
        logits_max = fmaxf(logits_max, logit);
    }

    // Softmax with stability
#pragma unroll
    for (int tk = 0; tk < S; ++tk) {
        logits[tk] = expf(logits[tk] - logits_max);
        logits_sum += logits[tk];
    }
    float inv_denom = 1.0f / logits_sum;

    // Compute context vector: sum_k softmax * V_k
    int out_base = (b * S + t_q) * E + h * head_dim;
#pragma unroll
    for (int d = 0; d < 1024; ++d) {
        if (d >= head_dim) break;
        float acc = 0.0f;
#pragma unroll
        for (int tk = 0; tk < S; ++tk) {
            int v_base = (b * S + tk) * (3 * E) + 2 * E; // V segment offset
            float vv = h2f(qkv[v_base + h * head_dim + d]);
            acc += (logits[tk] * inv_denom) * vv;
        }
        ctx[out_base + d] = f2h(acc);
    }
}

// 9) Gather CLS token (index 0) from seq [B, S, E] into out [B, E]
__global__ void gather_cls_kernel(
    const half* __restrict__ seq, half* __restrict__ out, int B, int S, int E)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * E;
    if (idx >= total) return;
    int e = idx % E;
    int b = idx / E;
    out[b * E + e] = seq[b * (S * E) + 0 * E + e];
}

// Host launcher
void launch_gpu_implementation(
    void* output,                       // [B, num_classes], dtype = fp16
    void* input,                        // [B, C, H, W], dtype = fp16
    // conv1
    void* conv1_weight,                 // [embed_dim, in_channels, patch, patch], fp16
    void* conv1_bias,                   // [embed_dim], fp16
    // linear projection
    void* linear_proj_weight,           // [embed_dim, embed_dim*(H/patch)*(W/patch)], fp16
    void* linear_proj_bias,             // [embed_dim], fp16
    // cls token
    void* cls_token,                    // [1, 1, embed_dim], fp16
    // Transformer layers parameter pointer arrays (size = num_layers)
    void** attn_in_proj_weight,         // each [3*embed_dim, embed_dim], fp16
    void** attn_in_proj_bias,           // each [3*embed_dim], fp16
    void** attn_out_proj_weight,        // each [embed_dim, embed_dim], fp16
    void** attn_out_proj_bias,          // each [embed_dim], fp16
    void** ff1_weight,                  // each [ff_dim, embed_dim], fp16
    void** ff1_bias,                    // each [ff_dim], fp16
    void** ff2_weight,                  // each [embed_dim, ff_dim], fp16
    void** ff2_bias,                    // each [embed_dim], fp16
    void** norm1_weight,                // each [embed_dim], fp16
    void** norm1_bias,                  // each [embed_dim], fp16
    void** norm2_weight,                // each [embed_dim], fp16
    void** norm2_bias,                  // each [embed_dim], fp16
    // final classifier
    void* fc_weight,                    // [num_classes, embed_dim], fp16
    void* fc_bias,                      // [num_classes], fp16
    // problem sizes / hyperparameters
    int64_t batch_size,
    int64_t in_channels,
    int64_t image_h,
    int64_t image_w,
    int64_t patch_size,
    int64_t embed_dim,
    int64_t num_heads,
    int64_t num_layers,
    int64_t ff_dim,
    int64_t num_classes
) {
    // Cast pointers
    half* d_output = static_cast<half*>(output);
    const half* d_input = static_cast<const half*>(input);
    const half* d_conv_w = static_cast<const half*>(conv1_weight);
    const half* d_conv_b = static_cast<const half*>(conv1_bias);

    const half* d_lin_w = static_cast<const half*>(linear_proj_weight);
    const half* d_lin_b = static_cast<const half*>(linear_proj_bias);

    const half* d_cls_token = static_cast<const half*>(cls_token);

    const int B = static_cast<int>(batch_size);
    const int C = static_cast<int>(in_channels);
    const int H = static_cast<int>(image_h);
    const int W = static_cast<int>(image_w);
    const int K = static_cast<int>(patch_size);
    const int E = static_cast<int>(embed_dim);
    const int Hh = static_cast<int>(num_heads);
    const int L = static_cast<int>(num_layers);
    const int FF = static_cast<int>(ff_dim);
    const int NC = static_cast<int>(num_classes);

    const int OH = H / K;
    const int OW = W / K;
    const int PH = OH;
    const int PW = OW;

    // Allocate intermediate buffers
    half* d_conv_out = nullptr;   // [B, E, PH, PW]
    half* d_flat = nullptr;       // [B, E*PH*PW]
    half* d_embed = nullptr;      // [B, E]
    half* d_seq = nullptr;        // [B, 2, E]

    // For transformer pipeline
    const int S = 2; // sequence length = 2 (CLS + single token)
    const int M_seq = B * S; // flattened sequences

    half* d_qkv = nullptr;        // [B*S, 3E]
    half* d_ctx = nullptr;        // [B*S, E]
    half* d_attn_out = nullptr;   // [B*S, E]
    half* d_ln1_out = nullptr;    // [B*S, E]
    half* d_ff1 = nullptr;        // [B*S, FF]
    half* d_ff2 = nullptr;        // [B*S, E]
    half* d_ln2_out = nullptr;    // [B*S, E]
    half* d_cls = nullptr;        // [B, E]

    size_t conv_out_elems = (size_t)B * E * PH * PW;
    size_t flat_elems = (size_t)B * E * PH * PW;
    size_t embed_elems = (size_t)B * E;
    size_t seq_elems = (size_t)B * S * E;

    CUDA_CHECK(cudaMalloc(&d_conv_out, conv_out_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_flat, flat_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_embed, embed_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_seq, seq_elems * sizeof(half)));

    CUDA_CHECK(cudaMalloc(&d_qkv, (size_t)M_seq * 3 * E * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ctx, (size_t)M_seq * E * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_attn_out, (size_t)M_seq * E * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ln1_out, (size_t)M_seq * E * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ff1, (size_t)M_seq * FF * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ff2, (size_t)M_seq * E * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ln2_out, (size_t)M_seq * E * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_cls, (size_t)B * E * sizeof(half)));

    // 1) conv1
    {
        int threads = 256;
        int blocks = (int)((conv_out_elems + threads - 1) / threads);
        conv2d_stride_eq_kernel<<<blocks, threads>>>(
            d_input, d_conv_w, d_conv_b, d_conv_out,
            B, C, H, W, E, K, K
        );
    }

    // 2) flatten
    {
        int threads = 256;
        int blocks = (int)((flat_elems + threads - 1) / threads);
        flatten_kernel<<<blocks, threads>>>(d_conv_out, d_flat, B, E, PH, PW);
    }

    // 3) linear_proj: [B, E*PH*PW] x [E, E*PH*PW]^T -> [B,E]
    {
        int M = B;
        int N = E;
        int KK = E * PH * PW;
        int threads = 256;
        int blocks = (M * N + threads - 1) / threads;
        gemm_bias_act_kernel<<<blocks, threads>>>(
            d_flat, d_lin_w, d_lin_b, d_embed, M, N, KK, 0
        );
    }

    // 4) Build sequence with CLS token
    {
        int threads = 256;
        int blocks = (int)((embed_elems + threads - 1) / threads);
        build_seq_with_cls_kernel<<<blocks, threads>>>(d_embed, d_cls_token, d_seq, B, E);
    }

    // 5) Transformer layers (post-norm: x = LN(x + Attn(x)); x = LN(x + FF(x)))
    for (int li = 0; li < L; ++li) {
        const half* w_inproj = static_cast<const half*>(attn_in_proj_weight[li]); // [3E, E]
        const half* b_inproj = attn_in_proj_bias[li] ? static_cast<const half*>(attn_in_proj_bias[li]) : nullptr;
        const half* w_outproj = static_cast<const half*>(attn_out_proj_weight[li]); // [E, E]
        const half* b_outproj = attn_out_proj_bias[li] ? static_cast<const half*>(attn_out_proj_bias[li]) : nullptr;

        const half* w_ff1 = static_cast<const half*>(ff1_weight[li]); // [FF, E]
        const half* b_ff1 = ff1_bias[li] ? static_cast<const half*>(ff1_bias[li]) : nullptr;
        const half* w_ff2 = static_cast<const half*>(ff2_weight[li]); // [E, FF]
        const half* b_ff2 = ff2_bias[li] ? static_cast<const half*>(ff2_bias[li]) : nullptr;

        const half* ln1_w = norm1_weight[li] ? static_cast<const half*>(norm1_weight[li]) : nullptr;
        const half* ln1_b = norm1_bias[li] ? static_cast<const half*>(norm1_bias[li]) : nullptr;
        const half* ln2_w = norm2_weight[li] ? static_cast<const half*>(norm2_weight[li]) : nullptr;
        const half* ln2_b = norm2_bias[li] ? static_cast<const half*>(norm2_bias[li]) : nullptr;

        // Flatten seq [B, S, E] -> A [M_seq, E]
        half* A_ptr = d_seq; // treat as [M_seq, E]

        // 5.1 In-proj QKV: [M_seq, E] x [3E, E]^T -> [M_seq, 3E]
        {
            int M = M_seq;
            int N = 3 * E;
            int KK = E;
            int threads = 256;
            int blocks = (M * N + threads - 1) / threads;
            gemm_bias_act_kernel<<<blocks, threads>>>(A_ptr, w_inproj, b_inproj, d_qkv, M, N, KK, 0);
        }

        // 5.2 Attention context from QKV -> ctx [M_seq, E]
        {
            int threads = 128;
            int total = B * S * Hh;
            int blocks = (total + threads - 1) / threads;
            attention_context_kernel<<<blocks, threads>>>(d_qkv, d_ctx, B, S, E, Hh);
        }

        // 5.3 Out-proj: attn_out = ctx * W_o^T + b_o -> [M_seq, E]
        {
            int M = M_seq, N = E, KK = E;
            int threads = 256;
            int blocks = (M * N + threads - 1) / threads;
            gemm_bias_act_kernel<<<blocks, threads>>>(d_ctx, w_outproj, b_outproj, d_attn_out, M, N, KK, 0);
        }

        // 5.4 Residual add: x + attn_out -> tmp, then LN1
        {
            int numel = M_seq * E;
            int threads = 256;
            int blocks = (numel + threads - 1) / threads;
            residual_add_kernel<<<blocks, threads>>>(d_seq, d_attn_out, d_ln1_out, numel);
        }
        {
            int M = M_seq, E_ = E;
            int threads = 128;
            int blocks = (M + threads - 1) / threads;
            layernorm_kernel<<<blocks, threads>>>(d_ln1_out, ln1_w, ln1_b, d_ln1_out, M, E_, 1e-5f);
        }

        // 5.5 FFN: FF1 = ReLU( LN1 * W1^T + b1 ) -> [M_seq, FF]
        {
            int M = M_seq, N = FF, KK = E;
            int threads = 256;
            int blocks = (M * N + threads - 1) / threads;
            gemm_bias_act_kernel<<<blocks, threads>>>(d_ln1_out, w_ff1, b_ff1, d_ff1, M, N, KK, 0);
            int numel = M_seq * FF;
            int blocks_relu = (numel + threads - 1) / threads;
            relu_inplace_kernel<<<blocks_relu, threads>>>(d_ff1, numel);
        }

        // 5.6 FF2 = FF1 * W2^T + b2 -> [M_seq, E]
        {
            int M = M_seq, N = E, KK = FF;
            int threads = 256;
            int blocks = (M * N + threads - 1) / threads;
            gemm_bias_act_kernel<<<blocks, threads>>>(d_ff1, w_ff2, b_ff2, d_ff2, M, N, KK, 0);
        }

        // 5.7 Residual add: LN1 + FF2 -> LN2_out = LN(...)
        {
            int numel = M_seq * E;
            int threads = 256;
            int blocks = (numel + threads - 1) / threads;
            residual_add_kernel<<<blocks, threads>>>(d_ln1_out, d_ff2, d_ln2_out, numel);
        }
        {
            int M = M_seq, E_ = E;
            int threads = 128;
            int blocks = (M + threads - 1) / threads;
            layernorm_kernel<<<blocks, threads>>>(d_ln2_out, ln2_w, ln2_b, d_seq, M, E_, 1e-5f);
        }
    }

    // 6) Gather CLS token and final FC
    {
        int threads = 256;
        int total = B * E;
        int blocks = (total + threads - 1) / threads;
        gather_cls_kernel<<<blocks, threads>>>(d_seq, d_cls, B, S, E);
    }
    {
        int M = B, N = NC, KK = E;
        int threads = 256;
        int blocks = (M * N + threads - 1) / threads;
        const half* w_fc = static_cast<const half*>(fc_weight);
        const half* b_fc = static_cast<const half*>(fc_bias);
        gemm_bias_act_kernel<<<blocks, threads>>>(d_cls, w_fc, b_fc, d_output, M, N, KK, 0);
    }

    cudaDeviceSynchronize();

    // Free temporaries
    cudaFree(d_conv_out);
    cudaFree(d_flat);
    cudaFree(d_embed);
    cudaFree(d_seq);
    cudaFree(d_qkv);
    cudaFree(d_ctx);
    cudaFree(d_attn_out);
    cudaFree(d_ln1_out);
    cudaFree(d_ff1);
    cudaFree(d_ff2);
    cudaFree(d_ln2_out);
    cudaFree(d_cls);
}
