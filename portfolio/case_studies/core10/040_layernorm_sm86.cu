#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <unordered_map>

struct Statistics {
    float mean;
    float inverse_std;
};

namespace {

constexpr int kMaxBatchSize = 16;
constexpr int kTilesPerBatch = 256;
constexpr size_t kPartialsBytes =
    kMaxBatchSize * kTilesPerBatch * sizeof(float2);
constexpr size_t kWorkspaceBytes =
    kPartialsBytes + kMaxBatchSize * sizeof(Statistics);
static_assert(kWorkspaceBytes == 32896, "LayerNorm workspace contract changed.");

[[noreturn]] void fail_cuda(
    const char* operation,
    cudaError_t status
) noexcept {
    std::fprintf(
        stderr,
        "KERNELBLASTER_CUDA_ERROR operation=%s status=%d message=%s\n",
        operation,
        static_cast<int>(status),
        cudaGetErrorString(status)
    );
    std::fflush(stderr);
    std::abort();
}

void require_cuda(const char* operation, cudaError_t status) noexcept {
    if (status != cudaSuccess) {
        fail_cuda(operation, status);
    }
}

[[noreturn]] void fail_cuda_resource(
    const char* operation,
    cudaError_t status
) noexcept {
    std::fprintf(
        stderr,
        "KERNELBLASTER_RESOURCE_BLOCKED kind=cuda operation=%s status=%d message=%s\n",
        operation,
        static_cast<int>(status),
        cudaGetErrorString(status)
    );
    std::fflush(stderr);
    std::abort();
}

void require_cuda_resource(const char* operation, cudaError_t status) noexcept {
    if (status != cudaSuccess) {
        fail_cuda_resource(operation, status);
    }
}

[[noreturn]] void fail_contract(const char* message) noexcept {
    std::fprintf(stderr, "KERNELBLASTER_CONTRACT_ERROR message=%s\n", message);
    std::fflush(stderr);
    std::abort();
}

void require_contract(bool condition, const char* message) noexcept {
    if (!condition) {
        fail_contract(message);
    }
}

void report_cuda_cleanup(const char* operation, cudaError_t status) noexcept {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr,
            "KERNELBLASTER_RESOURCE_CLEANUP_BLOCKED kind=cuda operation=%s status=%d\n",
            operation,
            static_cast<int>(status)
        );
        std::fflush(stderr);
    }
}

class LayerNormContext {
public:
    explicit LayerNormContext(int device) : device_(device), workspace_(nullptr) {
        require_cuda_resource("cudaSetDevice", cudaSetDevice(device_));
        require_cuda_resource(
            "cudaMalloc(layernorm_workspace)",
            cudaMalloc(&workspace_, kWorkspaceBytes)
        );
    }

    ~LayerNormContext() noexcept {
        int previous_device = -1;
        const cudaError_t queried = cudaGetDevice(&previous_device);
        const bool restore = queried == cudaSuccess;
        report_cuda_cleanup("cudaGetDevice", queried);
        const cudaError_t selected = cudaSetDevice(device_);
        report_cuda_cleanup("cudaSetDevice", selected);
        if (selected == cudaSuccess && workspace_ != nullptr) {
            report_cuda_cleanup(
                "cudaFree(layernorm_workspace)", cudaFree(workspace_)
            );
        }
        if (restore && previous_device != device_) {
            report_cuda_cleanup(
                "cudaSetDevice(restore)", cudaSetDevice(previous_device)
            );
        }
    }

    LayerNormContext(const LayerNormContext&) = delete;
    LayerNormContext& operator=(const LayerNormContext&) = delete;

    float2* partials() const { return static_cast<float2*>(workspace_); }
    Statistics* statistics() const {
        return reinterpret_cast<Statistics*>(
            static_cast<unsigned char*>(workspace_) + kPartialsBytes
        );
    }

private:
    int device_;
    void* workspace_;
};

LayerNormContext& thread_device_context() {
    int device = -1;
    require_cuda_resource("cudaGetDevice", cudaGetDevice(&device));
    static thread_local std::unordered_map<int, std::unique_ptr<LayerNormContext>>
        contexts;
    auto found = contexts.find(device);
    if (found == contexts.end()) {
        found = contexts.emplace(
            device, std::make_unique<LayerNormContext>(device)
        ).first;
    }
    return *found->second;
}

}  // namespace

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float block_sum(float value, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x >> 5) ? shared[lane] : 0.0f;
    return warp == 0 ? warp_sum(value) : value;
}

__global__ void layernorm_partials_kernel(
    float2* __restrict__ partials,
    const half* __restrict__ input,
    int64_t norm_size,
    int tiles_per_batch
) {
    const int batch = blockIdx.x / tiles_per_batch;
    const int tile = blockIdx.x - batch * tiles_per_batch;
    const int64_t base = static_cast<int64_t>(batch) * norm_size;
    const int64_t stride = static_cast<int64_t>(tiles_per_batch) * blockDim.x;
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int64_t index = static_cast<int64_t>(tile) * blockDim.x + threadIdx.x;
         index < norm_size;
         index += stride) {
        const float value = __half2float(input[base + index]);
        sum += value;
        square_sum = fmaf(value, value, square_sum);
    }
    __shared__ float shared_sum[8];
    __shared__ float shared_square[8];
    const float total = block_sum(sum, shared_sum);
    __syncthreads();
    const float total_square = block_sum(square_sum, shared_square);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = make_float2(total, total_square);
    }
}

__global__ void layernorm_statistics_kernel(
    Statistics* __restrict__ statistics,
    const float2* __restrict__ partials,
    int64_t norm_size,
    int tiles_per_batch
) {
    const int batch = blockIdx.x;
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int tile = threadIdx.x; tile < tiles_per_batch; tile += blockDim.x) {
        const float2 value = partials[batch * tiles_per_batch + tile];
        sum += value.x;
        square_sum += value.y;
    }
    __shared__ float shared_sum[8];
    __shared__ float shared_square[8];
    const float total = block_sum(sum, shared_sum);
    __syncthreads();
    const float total_square = block_sum(square_sum, shared_square);
    if (threadIdx.x == 0) {
        const float mean = total / static_cast<float>(norm_size);
        const float variance = fmaxf(
            0.0f, total_square / static_cast<float>(norm_size) - mean * mean
        );
        statistics[batch] = {mean, rsqrtf(variance + 1e-5f)};
    }
}

__global__ void layernorm_apply_kernel(
    half2* __restrict__ output,
    const half2* __restrict__ input,
    const half2* __restrict__ weight,
    const half2* __restrict__ bias,
    const Statistics* __restrict__ statistics,
    int64_t total_pairs,
    int64_t norm_pairs
) {
    for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < total_pairs;
         pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int batch = static_cast<int>(pair / norm_pairs);
        const int64_t feature_pair = pair - static_cast<int64_t>(batch) * norm_pairs;
        const float2 value = __half22float2(input[pair]);
        const float2 scale = __half22float2(weight[feature_pair]);
        const float2 offset = __half22float2(bias[feature_pair]);
        const Statistics stats = statistics[batch];
        output[pair] = __floats2half2_rn(
            (value.x - stats.mean) * stats.inverse_std * scale.x + offset.x,
            (value.y - stats.mean) * stats.inverse_std * scale.y + offset.y
        );
    }
}

__global__ void layernorm_apply_scalar_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const Statistics* __restrict__ statistics,
    int64_t total_elements,
    int64_t norm_size
) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_elements;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int batch = static_cast<int>(index / norm_size);
        const int64_t feature = index - static_cast<int64_t>(batch) * norm_size;
        const Statistics stats = statistics[batch];
        const float value = __half2float(input[index]);
        output[index] = __float2half_rn(
            (value - stats.mean) * stats.inverse_std * __half2float(weight[feature])
            + __half2float(bias[feature])
        );
    }
}

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
    constexpr int threads = 256;
    require_contract(
        batch_size > 0 && batch_size <= kMaxBatchSize,
        "batch_size must be in [1, 16]"
    );
    require_contract(
        features > 0 && dim1 > 0 && dim2 > 0,
        "features, dim1, and dim2 must be positive"
    );
    LayerNormContext& context = thread_device_context();
    float2* const partials = context.partials();
    Statistics* const statistics = context.statistics();

    const int64_t norm_size = features * dim1 * dim2;
    layernorm_partials_kernel<<<static_cast<int>(batch_size) * kTilesPerBatch, threads>>>(
        partials,
        static_cast<const half*>(input),
        norm_size,
        kTilesPerBatch
    );
    layernorm_statistics_kernel<<<static_cast<int>(batch_size), threads>>>(
        statistics,
        partials,
        norm_size,
        kTilesPerBatch
    );
    if ((norm_size & 1) == 0) {
        const int64_t norm_pairs = norm_size / 2;
        const int64_t total_pairs = batch_size * norm_pairs;
        const int blocks = static_cast<int>(
            std::min<int64_t>(4096, (total_pairs + threads - 1) / threads)
        );
        layernorm_apply_kernel<<<blocks, threads>>>(
            static_cast<half2*>(output),
            static_cast<const half2*>(input),
            static_cast<const half2*>(weight),
            static_cast<const half2*>(bias),
            statistics,
            total_pairs,
            norm_pairs
        );
    } else {
        const int64_t total_elements = batch_size * norm_size;
        const int blocks = static_cast<int>(
            std::min<int64_t>(4096, (total_elements + threads - 1) / threads)
        );
        layernorm_apply_scalar_kernel<<<blocks, threads>>>(
            static_cast<half*>(output),
            static_cast<const half*>(input),
            static_cast<const half*>(weight),
            static_cast<const half*>(bias),
            statistics,
            total_elements,
            norm_size
        );
    }
    require_cuda("cudaDeviceSynchronize", cudaDeviceSynchronize());
}
