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
Implements a fast CUDA kernel for the following PyTorch model:
    class Model(nn.Module):
        def forward(self, x): return torch.tanh(x)
with input/output tensors in fp16.

Tested via:
    launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim);
where both input and output are contiguous fp16 arrays of shape [batch_size, dim].

Requirements:
- Use half-precision (fp16) for I/O.
- Accumulate in fp32 for best accuracy.
- Maximize memory throughput (coalesced reads/writes).
- Use block/grid launch for full GPU utilization.
- No main(), all code self-contained.
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>
#include <cassert>

// Fast tanh implementation using __half2 if possible for vectorization.
// Fallback to scalar for odd tail elements.

__device__ __forceinline__ float fast_tanhf(float x) {
    // Numerically stable tanh, fast approximation.
    // For |x| > 5, tanh(x) ~ sign(x)
    if (x > 5.0f) return 1.0f;
    if (x < -5.0f) return -1.0f;
    float e2x = __expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

__device__ __forceinline__ __half fast_tanh_half(__half x) {
    // Convert to float, apply tanh, convert back
    float xf = __half2float(x);
    float tf = fast_tanhf(xf);
    return __float2half_rn(tf);
}

__global__ void kernel_tanh_fp16(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t N
) {
    // Vectorize using __half2 for best throughput.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = N / 2;

    // Process 2 elements at a time using __half2
    for (int i = idx; i < N2; i += gridDim.x * blockDim.x) {
        // Load 2 fp16 values as __half2
        __half2 x2 = reinterpret_cast<const __half2*>(input)[i];
        float2 xf2 = __half22float2(x2);

        float2 tf2;
        tf2.x = fast_tanhf(xf2.x);
        tf2.y = fast_tanhf(xf2.y);

        __half2 t2 = __floats2half2_rn(tf2.x, tf2.y);

        reinterpret_cast<__half2*>(output)[i] = t2;
    }

    // Handle odd tail element if N is odd
    if ((N & 1) && (idx == 0)) {
        output[N - 1] = fast_tanh_half(input[N - 1]);
    }
}

// Host launcher
void launch_gpu_implementation(void* output, void* input, int64_t batch_size, int64_t dim) {
    // input/output: device pointers, shape [batch_size, dim], fp16
    // Launch 2D grid for maximum occupancy
    int64_t N = batch_size * dim;
    const int block_size = 256;
    int grid_size = (N / 2 + block_size - 1) / block_size;
    // Clamp grid size to a reasonable max (e.g., 65535 for 1D grid)
    grid_size = (grid_size > 65535) ? 65535 : grid_size;

    kernel_tanh_fp16<<<grid_size, block_size>>>(
        static_cast<const half*>(input),
        static_cast<half*>(output),
        N
    );
    cudaDeviceSynchronize();
}

