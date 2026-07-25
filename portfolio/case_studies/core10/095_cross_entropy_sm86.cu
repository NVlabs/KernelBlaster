#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace {

constexpr int kMaxBatchSize = 4096;
constexpr int kMaxPartialCount = (kMaxBatchSize + 7) / 8;
constexpr size_t kWorkspaceBytes = kMaxPartialCount * sizeof(float);
static_assert(kWorkspaceBytes == 2048, "Cross-entropy workspace contract changed.");

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

class CrossEntropyContext {
public:
    explicit CrossEntropyContext(int device) : device_(device), partials_(nullptr) {
        require_cuda_resource("cudaSetDevice", cudaSetDevice(device_));
        require_cuda_resource(
            "cudaMalloc(cross_entropy_workspace)",
            cudaMalloc(&partials_, kWorkspaceBytes)
        );
    }

    ~CrossEntropyContext() noexcept {
        int previous_device = -1;
        const cudaError_t queried = cudaGetDevice(&previous_device);
        const bool restore = queried == cudaSuccess;
        report_cuda_cleanup("cudaGetDevice", queried);
        const cudaError_t selected = cudaSetDevice(device_);
        report_cuda_cleanup("cudaSetDevice", selected);
        if (selected == cudaSuccess && partials_ != nullptr) {
            report_cuda_cleanup(
                "cudaFree(cross_entropy_workspace)", cudaFree(partials_)
            );
        }
        if (restore && previous_device != device_) {
            report_cuda_cleanup(
                "cudaSetDevice(restore)", cudaSetDevice(previous_device)
            );
        }
    }

    CrossEntropyContext(const CrossEntropyContext&) = delete;
    CrossEntropyContext& operator=(const CrossEntropyContext&) = delete;

    float* partials() const { return partials_; }

private:
    int device_;
    float* partials_;
};

CrossEntropyContext& thread_device_context() {
    int device = -1;
    require_cuda_resource("cudaGetDevice", cudaGetDevice(&device));
    static thread_local std::unordered_map<
        int, std::unique_ptr<CrossEntropyContext>
    > contexts;
    auto found = contexts.find(device);
    if (found == contexts.end()) {
        found = contexts.emplace(
            device, std::make_unique<CrossEntropyContext>(device)
        ).first;
    }
    return *found->second;
}

}  // namespace

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

__global__ void cross_entropy_partials_kernel(
    float* __restrict__ partials,
    const half* __restrict__ logits,
    const int64_t* __restrict__ targets,
    int batch_size,
    int classes
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int sample = blockIdx.x * 8 + warp;
    float loss = 0.0f;
    if (sample < batch_size) {
        float value = lane < classes
            ? __half2float(logits[static_cast<int64_t>(sample) * classes + lane])
            : -FLT_MAX;
        float maximum = warp_max(value);
        maximum = __shfl_sync(0xffffffff, maximum, 0);
        float exponential = lane < classes ? __expf(value - maximum) : 0.0f;
        float sum = warp_sum(exponential);
        sum = __shfl_sync(0xffffffff, sum, 0);
        const int target = static_cast<int>(targets[sample]);
        const float target_logit = __shfl_sync(0xffffffff, value, target);
        if (lane == 0) {
            loss = maximum + logf(sum) - target_logit;
        }
    }
    __shared__ float warp_losses[8];
    if (lane == 0) {
        warp_losses[warp] = loss;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
#pragma unroll
        for (int index = 0; index < 8; ++index) {
            total += warp_losses[index];
        }
        partials[blockIdx.x] = total;
    }
}

__global__ void cross_entropy_finalize_kernel(
    half* output,
    const float* partials,
    int partial_count,
    int batch_size
) {
    float sum = 0.0f;
    for (int index = threadIdx.x; index < partial_count; index += blockDim.x) {
        sum += partials[index];
    }
    sum = warp_sum(sum);
    __shared__ float warp_totals[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < 8 ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            output[0] = __float2half_rn(total / static_cast<float>(batch_size));
        }
    }
}

void launch_gpu_implementation(
    void* output,
    void* predictions,
    void* targets,
    int64_t batch_size,
    int64_t num_classes
) {
    require_contract(
        batch_size > 0 && batch_size <= kMaxBatchSize,
        "batch_size must be in [1, 4096]"
    );
    require_contract(
        num_classes > 0 && num_classes <= 10,
        "num_classes must be in [1, 10]"
    );
    constexpr int threads = 256;
    const int blocks = static_cast<int>((batch_size + 7) / 8);
    float* const partials = thread_device_context().partials();
    cross_entropy_partials_kernel<<<blocks, threads>>>(
        partials,
        static_cast<const half*>(predictions),
        static_cast<const int64_t*>(targets),
        static_cast<int>(batch_size),
        static_cast<int>(num_classes)
    );
    cross_entropy_finalize_kernel<<<1, threads>>>(
        static_cast<half*>(output),
        partials,
        blocks,
        static_cast<int>(batch_size)
    );
    require_cuda("cudaDeviceSynchronize", cudaDeviceSynchronize());
}
