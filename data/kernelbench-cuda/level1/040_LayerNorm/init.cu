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
/*
 * Fast LayerNorm kernel for fp16 tensors.
 * This kernel normalizes the last 3 dimensions (features, dim1, dim2) for each batch, 
 * using fp32 accumulation for mean/variance for numerical stability.
 * Output:
 *   y = (x - mean) / sqrt(var + eps) * weight + bias
 * where:
 *   - x, y, weight, bias: fp16 tensors
 *   - mean/var: computed in fp32
 *   - weight, bias: shape [features, dim1, dim2]
 *
 * Kernel assumes input/output/weight/bias are all contiguous and stored in row-major order:
 *   [batch_size][features][dim1][dim2]
 *
 * Launch interface:
 *   launch_gpu_implementation(
 *       void* output,
 *       void* input,
 *       void* weight,
 *       void* bias,
 *       int64_t batch_size,
 *       int64_t features,
 *       int64_t dim1,
 *       int64_t dim2
 *   );
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math_constants.h>

#define WARP_SIZE 32
#define EPS 1e-5f

// Utility: warp-level reduction for float (mean/variance accumulation)
// Only works for <= 1024 elements per normalized group (which is true for this use case)
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Utility: block-level reduction for float using shared memory
template <unsigned blockSize>
__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE]; // one per warp
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    __syncthreads();

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// The main LayerNorm kernel
// Each block handles one normalized group (features*dim1*dim2), i.e. one [features, dim1, dim2] per batch
// Each thread processes multiple elements
__global__ void layernorm_fp16_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ y,
    int64_t batch_size,
    int64_t features,
    int64_t dim1,
    int64_t dim2,
    float eps
) {
    extern __shared__ float sdata[]; // used for block reduction

    // Compute the offset for this batch
    int norm_size = features * dim1 * dim2;
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const int64_t base_offset = batch_idx * norm_size;

    // Step 1: compute mean (in fp32) for this group
    float thread_sum = 0.0f;
    for (int i = tid; i < norm_size; i += block_size) {
        thread_sum += __half2float(x[base_offset + i]);
    }
    float mean = block_reduce_sum<1024>(thread_sum); // block reduction
    if (threadIdx.x == 0) sdata[0] = mean;
    __syncthreads();
    mean = sdata[0] / norm_size;

    // Step 2: compute variance (in fp32)
    float thread_var = 0.0f;
    for (int i = tid; i < norm_size; i += block_size) {
        float val = __half2float(x[base_offset + i]) - mean;
        thread_var += val * val;
    }
    float var = block_reduce_sum<1024>(thread_var);
    if (threadIdx.x == 0) sdata[0] = var;
    __syncthreads();
    var = sdata[0] / norm_size;
    float inv_std = rsqrtf(var + eps);

    // Step 3: normalize, scale, bias, and write output
    for (int i = tid; i < norm_size; i += block_size) {
        float inp = __half2float(x[base_offset + i]);
        float w = __half2float(weight[i]);
        float b = __half2float(bias[i]);
        float normed = (inp - mean) * inv_std;
        float out = normed * w + b;
        y[base_offset + i] = __float2half_rn(out);
    }
}

// Host launch function
void launch_gpu_implementation(
    void* output,
    void* input,
    void* weight,
    void* bias,
    int64_t batch_size,
    int64_t features,
    int64_t dim1,
    int64_t dim2
) {
    // All pointers are device pointers to half (fp16)
    half* x = static_cast<half*>(input);
    half* y = static_cast<half*>(output);
    half* w = static_cast<half*>(weight);
    half* b = static_cast<half*>(bias);

    int norm_size = features * dim1 * dim2;

    // Use 256 threads per block or up to norm_size, whichever is smaller
    int block_size = norm_size < 256 ? (norm_size < 128 ? 64 : 128) : 256;
    int grid_size = batch_size;

    // Shared memory for block reduction (one float per warp)
    size_t smem = WARP_SIZE * sizeof(float);

    layernorm_fp16_kernel<<<grid_size, block_size, smem>>>(
        x, w, b, y,
        batch_size, features, dim1, dim2, EPS
    );
    cudaDeviceSynchronize();
}
