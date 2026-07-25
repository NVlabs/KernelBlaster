#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

__global__ void rmsnorm_half2_rsqrt(
    half2* __restrict__ output,
    const half2* __restrict__ input,
    int64_t batch_size,
    int64_t channels,
    int64_t spatial_pairs,
    float eps
) {
    const int64_t linear_pair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_pairs = batch_size * spatial_pairs;
    if (linear_pair >= total_pairs) {
        return;
    }
    const int64_t batch = linear_pair / spatial_pairs;
    const int64_t pair = linear_pair - batch * spatial_pairs;
    const int64_t base = batch * channels * spatial_pairs + pair;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int64_t channel = 0; channel < channels; ++channel) {
        const float2 value = __half22float2(input[base + channel * spatial_pairs]);
        sum0 = fmaf(value.x, value.x, sum0);
        sum1 = fmaf(value.y, value.y, sum1);
    }
    const float inv0 = rsqrtf(sum0 / static_cast<float>(channels) + eps);
    const float inv1 = rsqrtf(sum1 / static_cast<float>(channels) + eps);
    for (int64_t channel = 0; channel < channels; ++channel) {
        const int64_t index = base + channel * spatial_pairs;
        const float2 value = __half22float2(input[index]);
        output[index] = __floats2half2_rn(value.x * inv0, value.y * inv1);
    }
}

__global__ void rmsnorm_scalar_odd_rsqrt(
    half* __restrict__ output,
    const half* __restrict__ input,
    int64_t total,
    int64_t channels,
    int64_t spatial_size,
    float eps
) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int64_t batch = linear / spatial_size;
    const int64_t spatial = linear - batch * spatial_size;
    const int64_t base = batch * channels * spatial_size + spatial;
    float sum = 0.0f;
    for (int64_t channel = 0; channel < channels; ++channel) {
        const float value = __half2float(input[base + channel * spatial_size]);
        sum = fmaf(value, value, sum);
    }
    const float inverse = rsqrtf(sum / static_cast<float>(channels) + eps);
    for (int64_t channel = 0; channel < channels; ++channel) {
        const int64_t index = base + channel * spatial_size;
        output[index] = __float2half_rn(__half2float(input[index]) * inverse);
    }
}

void launch_gpu_implementation(
    void* output,
    void* input,
    int64_t batch_size,
    int64_t num_features,
    int64_t dim1,
    int64_t dim2,
    float eps
) {
    constexpr int threads = 256;
    const int64_t spatial_size = dim1 * dim2;
    if ((spatial_size & 1) == 0) {
        const int64_t spatial_pairs = spatial_size / 2;
        const int64_t total_pairs = batch_size * spatial_pairs;
        rmsnorm_half2_rsqrt<<<
            static_cast<int>((total_pairs + threads - 1) / threads), threads
        >>>(
            static_cast<half2*>(output),
            static_cast<const half2*>(input),
            batch_size,
            num_features,
            spatial_pairs,
            eps
        );
    } else {
        const int64_t total = batch_size * spatial_size;
        rmsnorm_scalar_odd_rsqrt<<<
            static_cast<int>((total + threads - 1) / threads), threads
        >>>(
            static_cast<half*>(output),
            static_cast<const half*>(input),
            total,
            num_features,
            spatial_size,
            eps
        );
    }
    cudaDeviceSynchronize();
}
