#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace {

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

[[noreturn]] void fail_cublas(
    const char* operation,
    cublasStatus_t status
) noexcept {
    std::fprintf(
        stderr,
        "KERNELBLASTER_CUBLAS_ERROR operation=%s status=%d\n",
        operation,
        static_cast<int>(status)
    );
    std::fflush(stderr);
    std::abort();
}

void require_cublas(const char* operation, cublasStatus_t status) noexcept {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail_cublas(operation, status);
    }
}

[[noreturn]] void fail_cublas_resource(
    const char* operation,
    cublasStatus_t status
) noexcept {
    std::fprintf(
        stderr,
        "KERNELBLASTER_RESOURCE_BLOCKED kind=cublas operation=%s status=%d\n",
        operation,
        static_cast<int>(status)
    );
    std::fflush(stderr);
    std::abort();
}

void require_cublas_resource(
    const char* operation,
    cublasStatus_t status
) noexcept {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail_cublas_resource(operation, status);
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

void report_cublas_cleanup(
    const char* operation,
    cublasStatus_t status
) noexcept {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(
            stderr,
            "KERNELBLASTER_RESOURCE_CLEANUP_BLOCKED kind=cublas operation=%s status=%d\n",
            operation,
            static_cast<int>(status)
        );
        std::fflush(stderr);
    }
}

class CublasContext {
public:
    explicit CublasContext(int device) : device_(device), handle_(nullptr) {
        require_cuda_resource("cudaSetDevice", cudaSetDevice(device_));
        require_cublas_resource("cublasCreate", cublasCreate(&handle_));
        require_cublas_resource(
            "cublasSetMathMode",
            cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH)
        );
    }

    ~CublasContext() noexcept {
        int previous_device = -1;
        const cudaError_t queried = cudaGetDevice(&previous_device);
        const bool restore = queried == cudaSuccess;
        report_cuda_cleanup("cudaGetDevice", queried);
        const cudaError_t selected = cudaSetDevice(device_);
        report_cuda_cleanup("cudaSetDevice", selected);
        if (selected == cudaSuccess && handle_ != nullptr) {
            report_cublas_cleanup("cublasDestroy", cublasDestroy(handle_));
        }
        if (restore && previous_device != device_) {
            report_cuda_cleanup(
                "cudaSetDevice(restore)", cudaSetDevice(previous_device)
            );
        }
    }

    CublasContext(const CublasContext&) = delete;
    CublasContext& operator=(const CublasContext&) = delete;

    cublasHandle_t handle() const { return handle_; }

private:
    int device_;
    cublasHandle_t handle_;
};

CublasContext& thread_device_context() {
    int device = -1;
    require_cuda_resource("cudaGetDevice", cudaGetDevice(&device));
    static thread_local std::unordered_map<int, std::unique_ptr<CublasContext>>
        contexts;
    auto found = contexts.find(device);
    if (found == contexts.end()) {
        found = contexts.emplace(device, std::make_unique<CublasContext>(device)).first;
    }
    return *found->second;
}

}  // namespace

void launch_gpu_implementation(
    void* output,
    void* input_A,
    void* input_B,
    int64_t M,
    int64_t N,
    int64_t K
) {
    CublasContext& context = thread_device_context();
    const cublasHandle_t handle = context.handle();
    require_cublas_resource(
        "cublasSetStream", cublasSetStream(handle, cudaStreamLegacy)
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;
    require_cublas(
        "cublasGemmEx",
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            input_B,
            CUDA_R_16F,
            static_cast<int>(N),
            input_A,
            CUDA_R_16F,
            static_cast<int>(K),
            &beta,
            output,
            CUDA_R_16F,
            static_cast<int>(N),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );
    require_cuda("cudaDeviceSynchronize", cudaDeviceSynchronize());
}
