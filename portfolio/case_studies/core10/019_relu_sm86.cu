#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

union Half8Pack {
    uint4 bits;
    half2 values[4];
};

__global__ void relu_pack8_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    int64_t packs
) {
    const int64_t pack = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pack >= packs) {
        return;
    }
    Half8Pack data;
    data.bits = reinterpret_cast<const uint4*>(input)[pack];
    const half2 zero = __float2half2_rn(0.0f);
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        data.values[index] = __hmax2(data.values[index], zero);
    }
    reinterpret_cast<uint4*>(output)[pack] = data.bits;
}

__global__ void relu_tail_kernel(
    half* output,
    const half* input,
    int64_t begin,
    int64_t total
) {
    const int64_t index = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) {
        output[index] = __hmax(input[index], __float2half(0.0f));
    }
}

void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t batch_size,
    int64_t dim
) {
    constexpr int threads = 256;
    const int64_t total = batch_size * dim;
    const int64_t packs = total / 8;
    if (packs) {
        relu_pack8_kernel<<<static_cast<int>((packs + threads - 1) / threads), threads>>>(
            static_cast<half*>(output),
            static_cast<const half*>(input),
            packs
        );
    }
    const int64_t begin = packs * 8;
    if (begin < total) {
        relu_tail_kernel<<<1, 32>>>(
            static_cast<half*>(output),
            static_cast<const half*>(input),
            begin,
            total
        );
    }
    cudaDeviceSynchronize();
}
