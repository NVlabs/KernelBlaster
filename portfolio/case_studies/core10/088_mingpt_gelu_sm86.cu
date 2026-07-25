#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

union Half8Pack {
    uint4 bits;
    half2 values[4];
};

__device__ __forceinline__ float mingpt_gelu(float value) {
    const float cubic = value * value * value;
    const float argument = 0.7978845608028654f * (value + 0.044715f * cubic);
    float activation;
    asm("tanh.approx.f32 %0, %1;" : "=f"(activation) : "f"(argument));
    return 0.5f * value * (
        1.0f + activation
    );
}

__global__ void mingpt_gelu_pack8_kernel(
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
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 value = __half22float2(data.values[pair]);
        data.values[pair] = __floats2half2_rn(
            mingpt_gelu(value.x), mingpt_gelu(value.y)
        );
    }
    reinterpret_cast<uint4*>(output)[pack] = data.bits;
}

__global__ void mingpt_gelu_tail_kernel(
    half* output,
    const half* input,
    int64_t begin,
    int64_t total
) {
    const int64_t index = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) {
        output[index] = __float2half_rn(mingpt_gelu(__half2float(input[index])));
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
        mingpt_gelu_pack8_kernel<<<static_cast<int>((packs + threads - 1) / threads), threads>>>(
            static_cast<half*>(output),
            static_cast<const half*>(input),
            packs
        );
    }
    const int64_t begin = packs * 8;
    if (begin < total) {
        mingpt_gelu_tail_kernel<<<1, 32>>>(
            static_cast<half*>(output),
            static_cast<const half*>(input),
            begin,
            total
        );
    }
    cudaDeviceSynchronize();
}
