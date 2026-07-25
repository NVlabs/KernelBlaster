#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

__global__ void sum_dim1_half2_kernel(
    half2* __restrict__ output,
    const half2* __restrict__ input,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2_pairs
) {
    const int64_t output_pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_pairs = batch_size * dim2_pairs;
    if (output_pair >= total_pairs) {
        return;
    }
    const int64_t batch = output_pair / dim2_pairs;
    const int64_t column_pair = output_pair - batch * dim2_pairs;
    const int64_t base = batch * dim1 * dim2_pairs + column_pair;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int64_t row = 0; row < dim1; ++row) {
        const float2 value = __half22float2(input[base + row * dim2_pairs]);
        sum0 += value.x;
        sum1 += value.y;
    }
    output[output_pair] = __floats2half2_rn(sum0, sum1);
}

__global__ void sum_dim1_scalar_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2
) {
    const int64_t out = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_size * dim2;
    if (out >= total) {
        return;
    }
    const int64_t batch = out / dim2;
    const int64_t column = out - batch * dim2;
    const int64_t base = batch * dim1 * dim2 + column;
    float sum = 0.0f;
    for (int64_t row = 0; row < dim1; ++row) {
        sum += __half2float(input[base + row * dim2]);
    }
    output[out] = __float2half_rn(sum);
}

void launch_gpu_implementation(
    void* output,
    const void* input,
    int64_t batch_size,
    int64_t dim1,
    int64_t dim2,
    int64_t reduce_dim
) {
    assert(reduce_dim == 1);
    constexpr int threads = 32;
    if ((dim2 & 1) == 0) {
        const int64_t pairs = batch_size * (dim2 / 2);
        sum_dim1_half2_kernel<<<static_cast<int>((pairs + threads - 1) / threads), threads>>>(
            static_cast<half2*>(output),
            static_cast<const half2*>(input),
            batch_size,
            dim1,
            dim2 / 2
        );
    } else {
        const int64_t total = batch_size * dim2;
        sum_dim1_scalar_kernel<<<static_cast<int>((total + threads - 1) / threads), threads>>>(
            static_cast<half*>(output),
            static_cast<const half*>(input),
            batch_size,
            dim1,
            dim2
        );
    }
    cudaDeviceSynchronize();
}
