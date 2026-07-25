#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float block_max(float value, float* warps) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) {
        warps[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x >> 5) ? warps[lane] : -FLT_MAX;
    return warp == 0 ? warp_max(value) : value;
}

__device__ float block_sum(float value, float* warps) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warps[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x >> 5) ? warps[lane] : 0.0f;
    return warp == 0 ? warp_sum(value) : value;
}

__global__ void cached_softmax_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    int rows,
    int columns
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ half cache[];
    __shared__ float warp_values[32];
    __shared__ float row_max;
    __shared__ float inverse_sum;
    const half* row_input = input + static_cast<int64_t>(row) * columns;
    half* row_output = output + static_cast<int64_t>(row) * columns;

    float local_max = -FLT_MAX;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const half value = row_input[column];
        cache[column] = value;
        local_max = fmaxf(local_max, __half2float(value));
    }
    const float maximum = block_max(local_max, warp_values);
    if (threadIdx.x == 0) {
        row_max = maximum;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        local_sum += __expf(__half2float(cache[column]) - row_max);
    }
    const float total = block_sum(local_sum, warp_values);
    if (threadIdx.x == 0) {
        inverse_sum = 1.0f / total;
    }
    __syncthreads();

    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        row_output[column] = __float2half_rn(
            __expf(__half2float(cache[column]) - row_max) * inverse_sum
        );
    }
}

void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t batch_size,
    int64_t dim
) {
    constexpr int threads = 512;
    cached_softmax_kernel<<<
        static_cast<int>(batch_size), threads, static_cast<size_t>(dim) * sizeof(half)
    >>>(
        static_cast<half*>(output),
        static_cast<const half*>(input),
        static_cast<int>(batch_size),
        static_cast<int>(dim)
    );
    cudaDeviceSynchronize();
}
