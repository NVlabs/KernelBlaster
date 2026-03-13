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
#include <cstdio>
#include <cmath>
#include <algorithm>

// Utility: CUDA error check (lightweight)
#define CUDA_CHECK(expr) do { cudaError_t __err = (expr); if (__err != cudaSuccess) { \
    printf("CUDA Error %s at %s:%d\n", cudaGetErrorString(__err), __FILE__, __LINE__); return; } } while(0)

// Warp reduction sum for float
__inline__ __device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffffu;
    // Assumes warpSize == 32
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

// -------- Kernel 1: pack input [B,C,H,W] (NCHW) -> Xmat [M=S*B, E=C] row-major ----------
__global__ void pack_input_to_mat(
    const half* __restrict__ x, // [B,C,H,W]
    half* __restrict__ xmat,    // [M, E]
    int B, int C, int H, int W
) {
    int S = H * W;
    int M = B * S;
    int E = C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * E;
    if (idx >= total) return;

    int m = idx / E;       // 0..M-1
    int e = idx % E;       // 0..E-1
    int b = m / S;         // batch
    int s = m % S;         // spatial index
    int y = s / W;
    int xw = s % W;

    // NCHW contiguous: offset = ((b*C + e)*H + y)*W + xw
    int src_off = ((b * C + e) * H + y) * W + xw;
    xmat[m * E + e] = x[src_off];
}

// -------- Tiled GEMM: C[M,N] = A[M,K] * (B[N,K])^T; A row-major MxK, B row-major NxK; accumulate FP32, output FP16 ----------
template<int TM, int TN, int TK>
__global__ void gemm_a_row_b_rowT_fp16fp32acc(
    const half* __restrict__ A, // [M,K]
    const half* __restrict__ B, // [N,K] (row-major), but used as B^T in matmul
    half* __restrict__ C,       // [M,N]
    int M, int N, int K
) {
    __shared__ half As[TM][TK];
    __shared__ half Bs[TN][TK]; // store B tile (N, Ktile)

    int block_m = blockIdx.y * TM;
    int block_n = blockIdx.x * TN;

    int ty = threadIdx.y; // 0..(TM/8 - 1) if we choose block dims accordingly
    int tx = threadIdx.x; // 0..(TN/8 - 1)
    // We assume blockDim = (16, 16) and TM=64, TN=64, TK=32 -> each thread computes 4x4 C elements
    // Map thread to 4x4 tile in C
    int row_base = block_m + ty * 4;
    int col_base = block_n + tx * 4;

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // Load A tile: TM x TK
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = block_m + ty * 4 + i;
#pragma unroll
            for (int kk = 0; kk < TK; kk += 4) {
                int colk = k0 + kk + (tx % 4); // help distribute
                int smem_col = kk + (tx % 4);
                if (row < M && colk < K) {
                    As[ty * 4 + i][smem_col] = A[row * K + colk];
                } else {
                    As[ty * 4 + i][smem_col] = __float2half(0.0f);
                }
            }
        }
        // Load B tile: TN x TK (note: B is [N,K])
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int coln = block_n + tx * 4 + j;
#pragma unroll
            for (int kk = 0; kk < TK; kk += 4) {
                int colk = k0 + kk + (ty % 4);
                int smem_col = kk + (ty % 4);
                if (coln < N && colk < K) {
                    Bs[tx * 4 + j][smem_col] = B[coln * K + colk];
                } else {
                    Bs[tx * 4 + j][smem_col] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // Compute
#pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            // Load a 4x1 from As and 4x1 from Bs^T across kk
            half avec[4];
            half bvec[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                avec[i] = As[ty * 4 + i][kk];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                bvec[j] = Bs[tx * 4 + j][kk];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = __half2float(avec[i]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float b = __half2float(bvec[j]);
                    acc[i][j] += a * b;
                }
            }
        }
        __syncthreads();
    }

    // Store C
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = row_base + i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int col = col_base + j;
            if (col < N) {
                C[row * N + col] = __float2half(acc[i][j]);
            }
        }
    }
}

// -------- Kernel 3: Add bias, split QKV and reorder to [B,H,S,d] (d=E/H) ----------
__global__ void bias_split_reorder_qkv(
    const half* __restrict__ qkv_mat, // [M, 3E]
    const half* __restrict__ bias,    // [3E]
    half* __restrict__ Q,             // [B,H,S,d]
    half* __restrict__ K,             // [B,H,S,d]
    half* __restrict__ V,             // [B,H,S,d]
    int B, int Hh, int S, int E
) {
    int M = B * S;
    int d = E / Hh;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * E;
    if (idx >= total) return;
    int m = idx / E;  // 0..M-1
    int e = idx % E;  // 0..E-1
    int b = m / S;
    int s = m % S;
    int h = e / d;
    int t = e % d;

    int qkv_row = m;
    int q_col = e;
    int k_col = E + e;
    int v_col = 2 * E + e;

    half qv = __hadd(qkv_mat[qkv_row * (3 * E) + q_col], bias[q_col]);
    half kv = __hadd(qkv_mat[qkv_row * (3 * E) + k_col], bias[k_col]);
    half vv = __hadd(qkv_mat[qkv_row * (3 * E) + v_col], bias[v_col]);

    int base = ((b * Hh + h) * S + s) * d + t;
    Q[base] = qv;
    K[base] = kv;
    V[base] = vv;
}

// -------- Kernel 4: Scaled Dot-Product Attention (streaming softmax), one warp per query (b,h,s) ----------
__global__ void attention_sdpa_streaming(
    const half* __restrict__ Q, // [B,H,S,d]
    const half* __restrict__ K, // [B,H,S,d]
    const half* __restrict__ V, // [B,H,S,d]
    half* __restrict__ O,       // [B,H,S,d]
    int B, int Hh, int S, int d,
    float scale // = 1/sqrt(d)
) {
    // Each warp computes one query (b,h,sq)
    int warps_per_block = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int q_idx = blockIdx.x * warps_per_block + warp_id;
    int total_queries = B * Hh * S;
    if (q_idx >= total_queries) return;

    // Decode indices
    int tmp = q_idx;
    int b = tmp / (Hh * S);
    tmp = tmp % (Hh * S);
    int h = tmp / S;
    int sq = tmp % S;

    const half* q_ptr = Q + ((b * Hh + h) * S + sq) * d;
    // Load q component per lane
    float ql = 0.0f;
    if (lane < d) {
        ql = __half2float(q_ptr[lane]) * scale;
    }

    float m_i = -INFINITY; // running max
    float l_i = 0.0f;      // running sum exp
    // Accumulator vector across lanes: acc[lane]
    float acc = 0.0f;

    // Sweep over keys
    for (int sk = 0; sk < S; ++sk) {
        const half* k_ptr = K + ((b * Hh + h) * S + sk) * d;
        const half* v_ptr = V + ((b * Hh + h) * S + sk) * d;

        // Compute dot(q, k_sk) in a warp
        float dot = 0.0f;
        if (lane < d) {
            float kval = __half2float(k_ptr[lane]);
            dot = ql * kval; // q already scaled
        }
        dot = warp_reduce_sum(dot); // sum over lanes
        // Broadcast dot to all lanes
        float s_ij = __shfl_sync(0xffffffffu, dot, 0);

        // Update streaming softmax stats
        float m_new = fmaxf(m_i, s_ij);
        float alpha = expf(m_i - m_new);
        float p = expf(s_ij - m_new);
        l_i = alpha * l_i + p;

        // Update accumulator acc (per-lane multiply with v)
        float vcomp = 0.0f;
        if (lane < d) {
            vcomp = __half2float(v_ptr[lane]);
        }
        acc = alpha * acc + p * vcomp;

        m_i = m_new;
    }

    // Normalize
    float out = acc / fmaxf(l_i, 1e-9f);
    // Write result
    half* o_ptr = O + ((b * Hh + h) * S + sq) * d;
    if (lane < d) {
        o_ptr[lane] = __float2half(out);
    }
}

// -------- Kernel 5: Concatenate heads O[B,H,S,d] -> Omat[M=S*B, E=H*d] ----------
__global__ void concat_heads(
    const half* __restrict__ O, // [B,H,S,d]
    half* __restrict__ Omat,    // [M, E]
    int B, int Hh, int S, int d
) {
    int E = Hh * d;
    int M = B * S;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * E;
    if (idx >= total) return;

    int m = idx / E;      // 0..M-1
    int e = idx % E;      // 0..E-1
    int b = m / S;
    int s = m % S;
    int h = e / d;
    int t = e % d;
    int src = ((b * Hh + h) * S + s) * d + t;
    Omat[m * E + e] = O[src];
}

// -------- Kernel 6: Add bias per row for out_proj (C[M,E] += bias[E]) ----------
__global__ void add_bias_rowwise(
    half* __restrict__ C,          // [M,N]
    const half* __restrict__ bias, // [N]
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int n = idx % N;
    C[idx] = __hadd(C[idx], bias[n]);
}

// -------- Kernel 7: Residual + LayerNorm over last dim E (gamma,beta) ----------
// x_res = y + x_in; y_norm = LN(x_res)
__global__ void residual_layernorm(
    const half* __restrict__ y,        // [M,E]
    const half* __restrict__ x_in,     // [M,E]
    const half* __restrict__ gamma,    // [E]
    const half* __restrict__ beta,     // [E]
    half* __restrict__ out,            // [M,E]
    int M, int E, float eps
) {
    int m = blockIdx.x;
    if (m >= M) return;
    // Each block handles one row
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sqsum = sdata + blockDim.x;

    float sum = 0.0f;
    float sqsum = 0.0f;

    // Compute mean and variance
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float val = __half2float(y[m * E + e]) + __half2float(x_in[m * E + e]);
        sum += val;
        sqsum += val * val;
    }
    s_sum[threadIdx.x] = sum;
    s_sqsum[threadIdx.x] = sqsum;
    __syncthreads();

    // Reduce within block
    int tid = threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sqsum[tid] += s_sqsum[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / E;
    float var = s_sqsum[0] / E - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply gamma/beta
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float val = __half2float(y[m * E + e]) + __half2float(x_in[m * E + e]);
        float g = __half2float(gamma[e]);
        float bt = __half2float(beta[e]);
        float norm = (val - mean) * inv_std;
        out[m * E + e] = __float2half(norm * g + bt);
    }
}

// -------- Kernel 8: Unpack mat [M,E] -> output [B,C=E,H,W] (NCHW) ----------
__global__ void unpack_mat_to_output(
    const half* __restrict__ mat, // [M,E]
    half* __restrict__ out,       // [B,E,H,W]
    int B, int E, int H, int W
) {
    int S = H * W;
    int M = B * S;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * E;
    if (idx >= total) return;
    int m = idx / E;  // 0..M-1
    int e = idx % E;
    int b = m / S;
    int s = m % S;
    int y = s / W;
    int xw = s % W;
    int dst_off = ((b * E + e) * H + y) * W + xw;
    out[dst_off] = mat[m * E + e];
}

// -------- Host launcher implementing the full pipeline ----------
void launch_gpu_implementation(
    void* output,            // [B, C, H, W] fp16
    void* input,             // [B, C, H, W] fp16
    void* in_proj_weight,    // [3*E, E] fp16
    void* in_proj_bias,      // [3*E] fp16
    void* out_proj_weight,   // [E, E] fp16
    void* out_proj_bias,     // [E] fp16
    void* ln_weight,         // [E] fp16
    void* ln_bias,           // [E] fp16
    int64_t batch_size,
    int64_t channels,        // == embed_dim
    int64_t height,
    int64_t width,
    int64_t embed_dim,
    int64_t num_heads
) {
    // Aliases and sizes
    const int B = static_cast<int>(batch_size);
    const int E = static_cast<int>(embed_dim);
    const int Hh = static_cast<int>(num_heads);
    const int H_img = static_cast<int>(height);
    const int W_img = static_cast<int>(width);
    const int S = H_img * W_img;
    const int M = B * S;
    const int d = E / Hh;

    const half* x = static_cast<const half*>(input);
    half* y_out = static_cast<half*>(output);
    const half* w_qkv = static_cast<const half*>(in_proj_weight); // [3E, E]
    const half* b_qkv = static_cast<const half*>(in_proj_bias);   // [3E]
    const half* w_out = static_cast<const half*>(out_proj_weight);// [E, E]
    const half* b_out = static_cast<const half*>(out_proj_bias);  // [E]
    const half* ln_w = static_cast<const half*>(ln_weight);       // [E]
    const half* ln_b = static_cast<const half*>(ln_bias);         // [E]

    // Allocate temporaries
    half* d_Xmat = nullptr;      // [M, E]
    half* d_QKV = nullptr;       // [M, 3E]
    half* d_Q = nullptr;         // [B,H,S,d]
    half* d_K = nullptr;         // [B,H,S,d]
    half* d_V = nullptr;         // [B,H,S,d]
    half* d_O = nullptr;         // [B,H,S,d]
    half* d_Omat = nullptr;      // [M, E]
    half* d_proj = nullptr;      // [M, E] (after out proj)
    half* d_ln_out = nullptr;    // [M, E]

    size_t bytes_Xmat = static_cast<size_t>(M) * E * sizeof(half);
    size_t bytes_QKV  = static_cast<size_t>(M) * 3 * E * sizeof(half);
    size_t bytes_BHsD = static_cast<size_t>(B) * Hh * S * d * sizeof(half);
    size_t bytes_M_E  = static_cast<size_t>(M) * E * sizeof(half);

    CUDA_CHECK(cudaMalloc(&d_Xmat, bytes_Xmat));
    CUDA_CHECK(cudaMalloc(&d_QKV,  bytes_QKV));
    CUDA_CHECK(cudaMalloc(&d_Q,    bytes_BHsD));
    CUDA_CHECK(cudaMalloc(&d_K,    bytes_BHsD));
    CUDA_CHECK(cudaMalloc(&d_V,    bytes_BHsD));
    CUDA_CHECK(cudaMalloc(&d_O,    bytes_BHsD));
    CUDA_CHECK(cudaMalloc(&d_Omat, bytes_M_E));
    CUDA_CHECK(cudaMalloc(&d_proj, bytes_M_E));
    CUDA_CHECK(cudaMalloc(&d_ln_out, bytes_M_E));

    // 1) Pack input to [M,E]
    {
        int total = M * E;
        int block = 256;
        int grid = (total + block - 1) / block;
        pack_input_to_mat<<<grid, block>>>(x, d_Xmat, B, E, H_img, W_img);
    }

    // 2) QKV projection: [M,E] * ( [3E,E] )^T -> [M,3E]
    {
        // Launch GEMM: A = d_Xmat [M,E], B = w_qkv [3E,E] (N,K), output C = d_QKV [M,3E]
        const int Mdim = M, Ndim = 3 * E, Kdim = E;
        dim3 block(16, 16);
        // Using TM=64, TN=64, TK=32
        const int TM = 64, TN = 64, TK = 32;
        dim3 grid((Ndim + TN - 1) / TN, (Mdim + TM - 1) / TM);
        gemm_a_row_b_rowT_fp16fp32acc<TM, TN, TK><<<grid, block>>>(
            d_Xmat, w_qkv, d_QKV, Mdim, Ndim, Kdim
        );
        // Add bias and split + reorder to [B,H,S,d]
        int total = M * E;
        int tpb = 256;
        int grd = (total + tpb - 1) / tpb;
        bias_split_reorder_qkv<<<grd, tpb>>>(
            d_QKV, b_qkv, d_Q, d_K, d_V, B, Hh, S, E
        );
    }

    // 3) Attention: O = softmax(Q K^T / sqrt(d)) V, computed streaming per query
    {
        float scale = 1.0f / sqrtf(static_cast<float>(d));
        int total_queries = B * Hh * S;
        int warps_per_block = 8; // 8 warps = 256 threads per block
        int threads = warps_per_block * 32;
        int blocks = (total_queries + warps_per_block - 1) / warps_per_block;
        attention_sdpa_streaming<<<blocks, threads>>>(d_Q, d_K, d_V, d_O, B, Hh, S, d, scale);
    }

    // 4) Concat heads to [M,E]
    {
        int total = M * E;
        int block = 256;
        int grid = (total + block - 1) / block;
        concat_heads<<<grid, block>>>(d_O, d_Omat, B, Hh, S, d);
    }

    // 5) Out projection: [M,E] * ( [E,E] )^T -> [M,E], then add bias
    {
        const int Mdim = M, Ndim = E, Kdim = E;
        dim3 block(16, 16);
        const int TM = 64, TN = 64, TK = 32;
        dim3 grid((Ndim + TN - 1) / TN, (Mdim + TM - 1) / TM);
        gemm_a_row_b_rowT_fp16fp32acc<TM, TN, TK><<<grid, block>>>(
            d_Omat, w_out, d_proj, Mdim, Ndim, Kdim
        );
        int total = M * E;
        int tpb = 256;
        int grd = (total + tpb - 1) / tpb;
        add_bias_rowwise<<<grd, tpb>>>(d_proj, b_out, M, E);
    }

    // 6) Residual + LayerNorm over last dim E
    {
        // y_res = d_proj + d_Xmat; then LayerNorm with ln_w, ln_b
        int threads = 128; // at least E (128)
        size_t shmem = threads * sizeof(float) * 2; // for sums
        residual_layernorm<<<M, threads, shmem>>>(d_proj, d_Xmat, ln_w, ln_b, d_ln_out, M, E, 1e-5f);
    }

    // 7) Unpack to output [B,E,H,W]
    {
        int total = M * E;
        int block = 256;
        int grid = (total + block - 1) / block;
        unpack_mat_to_output<<<grid, block>>>(d_ln_out, y_out, B, E, H_img, W_img);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporaries
    cudaFree(d_Xmat);
    cudaFree(d_QKV);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_Omat);
    cudaFree(d_proj);
    cudaFree(d_ln_out);
}
