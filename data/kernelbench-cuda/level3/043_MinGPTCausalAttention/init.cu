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
// cuda_model.cu / cuda_model.cuh
// Implements a fast CUDA kernel for the provided masked multi-head self-attention forward pass.
// Uses WMMA (Tensor Cores) for GEMMs (fp16 inputs, fp32 accumulation) and a warp-level
// online softmax kernel for attention computation without materializing the T x T attention matrix.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>

using namespace nvcuda;

// Error checking macros
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                 \
        }                                                                    \
    } while (0)

// Utility: ceil division for int64
inline int64_t ceil_div_int64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// Weight transpose: from row-major [rows, cols] to col-major [cols, rows] for a contiguous row range
// Specifically, for W_row shape [R_total, K], we create B_col shape [K, N] where N = rows_to_copy,
// with rows copied from W_row[row_start : row_start + N].
// Output layout: B_col[k + n*K] = W_row[(row_start + n)*K + k]
__global__ void transpose_rows_to_col_major_kernel(
    const half* __restrict__ W_row,
    half* __restrict__ B_col,
    int row_start,   // starting row in W_row
    int N,           // number of rows to copy (also equals output N dimension)
    int K            // number of columns in W_row (also equals output K dimension)
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; // output column index in B's N
    int k = blockIdx.y * blockDim.y + threadIdx.y; // output row index in B's K
    if (n < N && k < K) {
        B_col[k + n * K] = W_row[(row_start + n) * K + k];
    }
}

// Pack [M, C] row-major into [R, T, hs] contiguous (R=B*nh)
// Input: in[m, c], where m in [0, M), c in [0, C)
// Output: out[r, t, d], contiguous, where r = b*nh + h, t in [0, T), d in [0, hs)
__global__ void pack_MxC_to_RTHS_kernel(
    const half* __restrict__ in,
    half* __restrict__ out,
    int64_t B, int64_t T, int64_t C, int64_t nh
) {
    int64_t M = B * T;
    int64_t hs = C / nh;
    int64_t total = M * C;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t m = idx / C;
    int64_t c = idx % C;

    int64_t b = m / T;
    int64_t t = m % T;
    int64_t h = c / hs;
    int64_t d = c % hs;
    int64_t r = b * nh + h;

    int64_t out_idx = ((r * T) + t) * hs + d;
    out[out_idx] = in[idx];
}

// Pack [R, T, hs] contiguous back to [M, C] row-major
__global__ void pack_RTHS_to_MxC_kernel(
    const half* __restrict__ in,
    half* __restrict__ out,
    int64_t B, int64_t T, int64_t C, int64_t nh
) {
    int64_t hs = C / nh;
    int64_t R = B * nh;
    int64_t total = R * T * hs;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t r = idx / (T * hs);
    int64_t rem = idx % (T * hs);
    int64_t t = rem / hs;
    int64_t d = rem % hs;

    int64_t b = r / nh;
    int64_t h = r % nh;

    int64_t c = h * hs + d;
    int64_t m = b * T + t;

    out[m * C + c] = in[idx];
}

// WMMA-based GEMM: C = A * B (+ bias), A row-major [M,K], B col-major [K,N], C row-major [M,N].
// Inputs A,B in half, accumulates in float, outputs in half.
// Bias (optional) is length N, added per column.
template<int WARPS_M, int WARPS_N>
__global__ void wmma_gemm_bias_kernel(
    const half* __restrict__ A,   // [M, K], row-major
    const half* __restrict__ B,   // [K, N], col-major
    const half* __restrict__ bias,// [N] or nullptr
    half* __restrict__ C,         // [M, N], row-major
    int M, int N, int K
) {
    // Tile sizes per block
    constexpr int WM = WARPS_M * 16; // rows per block
    constexpr int WN = WARPS_N * 16; // cols per block

    int block_row = blockIdx.y * WM;
    int block_col = blockIdx.x * WN;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    int warp_row = warpId / WARPS_N; // which warp tile along rows
    int warp_col = warpId % WARPS_N; // which warp tile along cols

    // Each warp computes a 16x16 tile
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K
    for (int kb = 0; kb < K; kb += 16) {
        const half* A_tile = A + (block_row + warp_row * 16) * K + kb;
        const half* B_tile = B + kb + (block_col + warp_col * 16) * K;

        wmma::load_matrix_sync(a_frag, A_tile, K);
        wmma::load_matrix_sync(b_frag, B_tile, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Shared memory for per-warp stores (float accumulation)
    extern __shared__ float sdata[];
    float* warp_smem = sdata + warpId * 16 * 16;

    // Store to shared memory
    wmma::store_matrix_sync(warp_smem, c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    // Each thread writes a portion of the 16x16 tile to global memory with bias add and cast to half
    for (int idx = laneId; idx < 16 * 16; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        int row = block_row + warp_row * 16 + i;
        int col = block_col + warp_col * 16 + j;
        if (row < M && col < N) {
            float val = warp_smem[i * 16 + j];
            if (bias != nullptr) {
                val += __half2float(bias[col]);
            }
            C[row * N + col] = __float2half_rn(val);
        }
    }
}

// Online masked softmax attention kernel per (r, t) row using one warp.
// Input: Q, K, V shapes [R, T, hs] contiguous; output Y [R, T, hs] contiguous.
// Computes for each (r,t): y_t = softmax(q_t K^T)_masked * V, with causal mask j<=t.
// Uses online softmax to avoid storing attention matrices. Accumulation in float, output half.
__global__ void attn_online_softmax_kernel(
    const half* __restrict__ Q, // [R, T, hs]
    const half* __restrict__ K, // [R, T, hs]
    const half* __restrict__ V, // [R, T, hs]
    half* __restrict__ Y,       // [R, T, hs]
    int R, int T, int hs,
    float scale
) {
    // Warp-based mapping
    int warpsPerBlock = blockDim.x / 32;
    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int row_id = blockIdx.x * warpsPerBlock + warpId; // 0..R*T-1
    int total_rows = R * T;
    if (row_id >= total_rows) return;

    int r = row_id / T;
    int t = row_id % T;

    const half* q_ptr = Q + ( (r * T + t) * hs );
    const half* k_ptr = K + ( r * T * hs );
    const half* v_ptr = V + ( r * T * hs );
    half* y_ptr = Y + ( (r * T + t) * hs );

    // Preload q into registers per lane segments
    const int max_segs = 32; // supports hs up to 32*32=1024 dims per head (more than enough here)
    float q_reg[max_segs];
    int segs = (hs + 31) / 32;
    if (segs > max_segs) segs = max_segs; // safety clamp

    for (int s = 0; s < segs; ++s) {
        int d = lane + 32 * s;
        q_reg[s] = (d < hs) ? __half2float(q_ptr[d]) : 0.0f;
    }

    // Online softmax statistics
    float m = -INFINITY;
    float s_acc = 0.0f;

    // Output accumulator vector per lane segments
    float o_buf[max_segs];
    for (int s = 0; s < segs; ++s) o_buf[s] = 0.0f;

    // Iterate over keys up to t (causal mask)
    for (int j = 0; j <= t; ++j) {
        const half* kj = k_ptr + j * hs;
        const half* vj = v_ptr + j * hs;

        // Compute dot(q_t, k_j) across hs using warp reduction
        float dot_partial = 0.0f;
        for (int s = 0; s < segs; ++s) {
            int d = lane + 32 * s;
            if (d < hs) {
                float k_val = __half2float(kj[d]);
                dot_partial += q_reg[s] * k_val;
            }
        }

        // Warp reduce sum
        unsigned mask = 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot_partial += __shfl_down_sync(mask, dot_partial, offset);
        }
        float dot_full = __shfl_sync(mask, dot_partial, 0);
        float alpha = dot_full * scale;

        // Online softmax update
        float new_m = fmaxf(m, alpha);
        float exp_m_m = (m == -INFINITY) ? 0.0f : expf(m - new_m);
        float exp_a_m = expf(alpha - new_m);
        float new_s = s_acc * exp_m_m + exp_a_m;

        // Update output accumulator vector
        for (int s = 0; s < segs; ++s) {
            int d = lane + 32 * s;
            if (d < hs) {
                float v_val = __half2float(vj[d]);
                float o_old = o_buf[s] * exp_m_m;
                o_buf[s] = o_old + exp_a_m * v_val;
            }
        }

        m = new_m;
        s_acc = new_s;
    }

    // Final normalize and store
    float inv_s = 1.0f / s_acc;
    for (int s = 0; s < segs; ++s) {
        int d = lane + 32 * s;
        if (d < hs) {
            float y_val = o_buf[s] * inv_s;
            y_ptr[d] = __float2half_rn(y_val);
        }
    }
}

// Host launcher implementing the full pipeline
extern "C" void launch_gpu_implementation(
    void* output,
    const void* input,
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t n_head,
    const void* c_attn_weight,
    const void* c_attn_bias,
    const void* c_proj_weight,
    const void* c_proj_bias,
    const void* attn_bias,    // causal mask buffer, not used (we apply causal by index <= t)
    int64_t max_seqlen,
    float attn_pdrop,
    float resid_pdrop
) {
    // Constants and derived dims
    const int64_t M = B * T;            // rows for GEMM
    const int64_t hs = C / n_head;      // head size
    const int64_t R = B * n_head;       // number of (batch, head) groups

    // Cast raw pointers
    const half* x = static_cast<const half*>(input);
    half* out = static_cast<half*>(output);
    const half* w_attn = static_cast<const half*>(c_attn_weight); // [3C, C] row-major
    const half* b_attn = static_cast<const half*>(c_attn_bias);   // [3C]
    const half* w_proj = static_cast<const half*>(c_proj_weight); // [C, C] row-major
    const half* b_proj = static_cast<const half*>(c_proj_bias);   // [C]

    // Allocate temporary/transformed weights (col-major B matrices for WMMA)
    half *Wq_col = nullptr, *Wk_col = nullptr, *Wv_col = nullptr, *Wp_col = nullptr;
    size_t CC_elems = static_cast<size_t>(C) * static_cast<size_t>(C);
    CUDA_CHECK(cudaMalloc(&Wq_col, CC_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&Wk_col, CC_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&Wv_col, CC_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&Wp_col, CC_elems * sizeof(half)));

    // Transpose weights from row-major to col-major for WMMA
    dim3 tBlock(32, 8);
    dim3 tGrid_q( (C + tBlock.x - 1) / tBlock.x, (C + tBlock.y - 1) / tBlock.y );
    transpose_rows_to_col_major_kernel<<<tGrid_q, tBlock>>>(w_attn, Wq_col, 0, (int)C, (int)C);
    transpose_rows_to_col_major_kernel<<<tGrid_q, tBlock>>>(w_attn, Wk_col, (int)C, (int)C, (int)C);
    transpose_rows_to_col_major_kernel<<<tGrid_q, tBlock>>>(w_attn, Wv_col, (int)(2*C), (int)C, (int)C);
    transpose_rows_to_col_major_kernel<<<tGrid_q, tBlock>>>(w_proj, Wp_col, 0, (int)C, (int)C);

    // Temporary buffers
    half* gemm_temp = nullptr; // [M, C] row-major
    CUDA_CHECK(cudaMalloc(&gemm_temp, M * C * sizeof(half)));

    // Q,K,V buffers [R, T, hs] contiguous
    half *Q = nullptr, *K = nullptr, *V = nullptr;
    size_t RTHS_elems = static_cast<size_t>(R) * static_cast<size_t>(T) * static_cast<size_t>(hs);
    CUDA_CHECK(cudaMalloc(&Q, RTHS_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&K, RTHS_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&V, RTHS_elems * sizeof(half)));

    // WMMA kernel configuration
    constexpr int WARPS_M = 4; // 4*16=64 rows per block
    constexpr int WARPS_N = 2; // 2*16=32 cols per block
    constexpr int THREADS = (WARPS_M * WARPS_N) * 32; // 256 threads
    dim3 gemmBlock(THREADS);
    dim3 gemmGrid_q( (C + (WARPS_N*16) - 1) / (WARPS_N*16), (M + (WARPS_M*16) - 1) / (WARPS_M*16) );
    size_t smem_wmma = (WARPS_M * WARPS_N) * 16 * 16 * sizeof(float); // per-warp 16x16 floats

    // 1) Compute Q = x * Wq^T + bq, where x: [M, C], Wq_col: [C, C] col-major, output gemm_temp: [M, C]
    const half* bq = b_attn;               // first C
    const half* bk = b_attn + C;           // next C
    const half* bv = b_attn + 2 * C;       // last C

    wmma_gemm_bias_kernel<WARPS_M, WARPS_N><<<gemmGrid_q, gemmBlock, smem_wmma>>>(x, Wq_col, bq, gemm_temp, (int)M, (int)C, (int)C);
    // Pack to [R,T,hs]
    {
        int64_t total = M * C;
        int threads = 256;
        int blocks = (int)ceil_div_int64(total, threads);
        pack_MxC_to_RTHS_kernel<<<blocks, threads>>>(gemm_temp, Q, B, T, C, n_head);
    }

    // 2) Compute K = x * Wk^T + bk
    wmma_gemm_bias_kernel<WARPS_M, WARPS_N><<<gemmGrid_q, gemmBlock, smem_wmma>>>(x, Wk_col, bk, gemm_temp, (int)M, (int)C, (int)C);
    {
        int64_t total = M * C;
        int threads = 256;
        int blocks = (int)ceil_div_int64(total, threads);
        pack_MxC_to_RTHS_kernel<<<blocks, threads>>>(gemm_temp, K, B, T, C, n_head);
    }

    // 3) Compute V = x * Wv^T + bv
    wmma_gemm_bias_kernel<WARPS_M, WARPS_N><<<gemmGrid_q, gemmBlock, smem_wmma>>>(x, Wv_col, bv, gemm_temp, (int)M, (int)C, (int)C);
    {
        int64_t total = M * C;
        int threads = 256;
        int blocks = (int)ceil_div_int64(total, threads);
        pack_MxC_to_RTHS_kernel<<<blocks, threads>>>(gemm_temp, V, B, T, C, n_head);
    }

    // 4) Attention: Y = softmax(Q @ K^T / sqrt(hs)) @ V (causal mask j<=t), per (r, t)
    half* Y = nullptr; // [R, T, hs]
    CUDA_CHECK(cudaMalloc(&Y, RTHS_elems * sizeof(half)));

    // Online softmax kernel launch
    {
        int warpsPerBlock = 8; // 8 warps => 256 threads
        int threads = warpsPerBlock * 32;
        int total_rows = static_cast<int>(R * T);
        int blocks = (total_rows + warpsPerBlock - 1) / warpsPerBlock;
        float scale = 1.0f / std::sqrt(static_cast<float>(hs));
        attn_online_softmax_kernel<<<blocks, threads>>>(Q, K, V, Y, (int)R, (int)T, (int)hs, scale);
    }

    // 5) Convert Y [R,T,hs] -> Y_mat [M,C] for final projection
    {
        int64_t total = R * T * hs;
        int threads = 256;
        int blocks = (int)ceil_div_int64(total, threads);
        pack_RTHS_to_MxC_kernel<<<blocks, threads>>>(Y, gemm_temp, B, T, C, n_head);
    }

    // 6) Final projection: out = Y_mat * Wp^T + b_proj  (Y_mat: [M,C], Wp_col: [C,C])
    wmma_gemm_bias_kernel<WARPS_M, WARPS_N><<<gemmGrid_q, gemmBlock, smem_wmma>>>(gemm_temp, Wp_col, b_proj, out, (int)M, (int)C, (int)C);

    // Synchronize to ensure kernel completion before output is checked.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporaries
    CUDA_CHECK(cudaFree(Wq_col));
    CUDA_CHECK(cudaFree(Wk_col));
    CUDA_CHECK(cudaFree(Wv_col));
    CUDA_CHECK(cudaFree(Wp_col));
    CUDA_CHECK(cudaFree(gemm_temp));
    CUDA_CHECK(cudaFree(Q));
    CUDA_CHECK(cudaFree(K));
    CUDA_CHECK(cudaFree(V));
    CUDA_CHECK(cudaFree(Y));
}
