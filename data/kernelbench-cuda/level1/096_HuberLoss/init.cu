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
#include <cstdint>
#include <algorithm>
#include <cassert>

// --- CUDA Smooth L1 (Huber) Loss kernel for fp16 tensors ---
// Computes mean smooth_l1 loss over all elements in (batch_size, input_shape[0])
// Input:  predictions [batch_size, input_shape] (fp16)
//         targets     [batch_size, input_shape] (fp16)
// Output: output      [1] (fp16, mean loss scalar)
//
// All reductions are performed in fp32 for numerical stability.
// Final output is cast to fp16.
//
// PyTorch reference loss for each element:
//   x = prediction - target
//   loss = 0.5 * x^2        if |x| < 1
//        = |x| - 0.5        otherwise
//   (mean reduction)
//
// Kernel is optimized for large batch and input (e.g. 128x4096).

// CUDA: warp reduction utility for fp32
__inline__ __device__ float warpReduceSum(float val) {
#if __CUDA_ARCH__ >= 300
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
#endif
    return val;
}

// CUDA: block reduction utility for fp32
template <unsigned int blockSize>
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Up to 1024 threads, 32 warps
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); // Each warp reduces to single value

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Only first warp needs to finish reduction
    float sum = 0.0f;
    if (threadIdx.x < blockDim.x / warpSize)
        sum = shared[lane];
    if (wid == 0)
        sum = warpReduceSum(sum);

    return sum;
}

__global__ void smooth_l1_loss_mean_fp16_kernel(
    const half* __restrict__ predictions,
    const half* __restrict__ targets,
    float* __restrict__ partial_sums,
    int64_t total_elements
) {
    // Each thread computes local sum in fp32
    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        float pred = __half2float(predictions[i]);
        float targ = __half2float(targets[i]);
        float diff = pred - targ;
        float abs_diff = fabsf(diff);

        float loss = (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
        local_sum += loss;
    }

    // Block-wide reduction in fp32
    float block_sum = blockReduceSum<256>(local_sum);

    // Write per-block sum
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = block_sum;
    }
}

// Final reduction kernel: sum partial_sums to scalar, then divide by total_elements, cast to fp16
__global__ void smooth_l1_loss_final_reduce_fp16(
    const float* __restrict__ partial_sums,
    int num_blocks,
    int64_t total_elements,
    half* __restrict__ output
) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    // Block reduction to single value
    sum = blockReduceSum<256>(sum);
    if (threadIdx.x == 0) {
        float mean_loss = sum / (float)total_elements;
        output[0] = __float2half_rn(mean_loss);
    }
}

// --- Host code for launching the CUDA implementation ---
// All pointers are device pointers. All data is fp16.

void launch_gpu_implementation(
    void* output,          // [1] fp16, device pointer
    void* predictions,     // [batch_size, input_shape] fp16, device pointer
    void* targets,         // [batch_size, input_shape] fp16, device pointer
    int64_t batch_size,
    int64_t input0,        // input_shape[0]
    int64_t input1         // not used, always 1
) {
    using namespace std;

    const int64_t N = batch_size * input0; // flat size

    // --- Kernel launch configuration ---
    const int threads_per_block = 256;
    // Use enough blocks to saturate the GPU, but not wasteful
    int num_blocks = static_cast<int>((N + threads_per_block - 1) / threads_per_block);
    // Limit max blocks, e.g. 4096 for large batch
    num_blocks = std::min(num_blocks, 4096);

    // Allocate buffer for per-block partial sums (fp32)
    float* d_partial_sums = nullptr;
    cudaMalloc(&d_partial_sums, sizeof(float) * num_blocks);

    // --- Launch first kernel: compute per-block smooth_l1 sums ---
    smooth_l1_loss_mean_fp16_kernel<<<num_blocks, threads_per_block>>>(
        static_cast<const half*>(predictions),
        static_cast<const half*>(targets),
        d_partial_sums,
        N
    );

    // --- Launch second kernel: final reduction and mean, output fp16 scalar ---
    // Only one block, 256 threads
    smooth_l1_loss_final_reduce_fp16<<<1, threads_per_block>>>(
        d_partial_sums, num_blocks, N,
        static_cast<half*>(output)
    );

    cudaFree(d_partial_sums);

    // Ensure completion
    cudaDeviceSynchronize();
}

// --- END OF FILE ---
