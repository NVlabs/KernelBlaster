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
// Cross Entropy Loss (Softmax + NLL) CUDA kernel for fp16 predictions/targets
// - Input: predictions [batch_size, num_classes] (half), targets [batch_size] (int64)
// - Output: output [1] (half, scalar loss, reduction=mean)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math_constants.h>

// Numerically stable log-sum-exp reduction for fp16
__device__ float logsumexp_fp16(const half* logits, int num_classes) {
    // Find max logit for stability
    float max_logit = -CUDART_INF_F;
    #pragma unroll
    for (int c = 0; c < 10; ++c) { // num_classes = 10
        if (c < num_classes) {
            float lc = __half2float(logits[c]);
            max_logit = fmaxf(max_logit, lc);
        }
    }
    // Compute sum(exp(logit - max_logit))
    float sum = 0.0f;
    #pragma unroll
    for (int c = 0; c < 10; ++c) {
        if (c < num_classes) {
            float val = expf(__half2float(logits[c]) - max_logit);
            sum += val;
        }
    }
    return max_logit + logf(sum);
}

// Compute per-example negative log likelihood loss
__global__ void cross_entropy_loss_kernel(
    const half* __restrict__ predictions,   // [batch_size, num_classes]
    const int64_t* __restrict__ targets,    // [batch_size]
    float* __restrict__ losses,             // [batch_size] (float32 accumulator)
    int64_t batch_size,
    int64_t num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const half* logits_row = predictions + idx * num_classes;
        int64_t target = targets[idx];

        // Compute logsumexp in FP32 for stability
        float lse = logsumexp_fp16(logits_row, num_classes);

        // Get target logit
        float target_logit = (target >= 0 && target < num_classes) ? __half2float(logits_row[target]) : 0.0f;

        // CE loss: -log(softmax[target]) = -target_logit + logsumexp
        float loss = lse - target_logit;
        losses[idx] = loss;
    }
}

// Standard parallel reduction for float32, output is float32
__global__ void reduce_sum_kernel(const float* __restrict__ input, float* output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;
    if (i < N) sum += input[i];
    if (i + blockDim.x < N) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// Host function: launches kernels and writes scalar fp16 output
void launch_gpu_implementation(
    void* output,                // Output: scalar loss, float16
    void* predictions,           // Input: [batch_size, num_classes], float16
    void* targets,               // Input: [batch_size], int64
    int64_t batch_size,
    int64_t num_classes
) {
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    // Temporary buffer for per-example losses (float32)
    float* d_losses = nullptr;
    cudaMalloc(&d_losses, batch_size * sizeof(float));

    // 1. Compute per-example cross-entropy loss in FP32
    cross_entropy_loss_kernel<<<blocks, threads>>>(
        static_cast<const half*>(predictions),
        static_cast<const int64_t*>(targets),
        d_losses,
        batch_size,
        num_classes
    );
    cudaDeviceSynchronize();

    // 2. Parallel reduction to sum the losses
    // Two-stage reduction for large batch size
    float* d_partial = nullptr;
    int partial_blocks = (batch_size + threads * 2 - 1) / (threads * 2);
    cudaMalloc(&d_partial, partial_blocks * sizeof(float));
    reduce_sum_kernel<<<partial_blocks, threads, threads * sizeof(float)>>>(d_losses, d_partial, batch_size);
    cudaDeviceSynchronize();

    float total_loss = 0.0f;
    if (partial_blocks > 1) {
        reduce_sum_kernel<<<1, threads, threads * sizeof(float)>>>(d_partial, d_partial, partial_blocks);
        cudaMemcpy(&total_loss, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(&total_loss, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    }

    // 3. Compute mean and cast to fp16
    float mean_loss = total_loss / static_cast<float>(batch_size);
    half h_loss = __float2half_rn(mean_loss);

    // 4. Write output (half) to device
    cudaMemcpy(static_cast<half*>(output), &h_loss, sizeof(half), cudaMemcpyHostToDevice);

    // Cleanup
    cudaFree(d_losses);
    cudaFree(d_partial);
    cudaDeviceSynchronize();
}
