#include <cuda_runtime.h>

#include <cmath>
#include <vector>

__global__ void vector_add_kernel(
    const float* left,
    const float* right,
    float* output,
    int count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}

bool run_vector_add(double* elapsed_microseconds) {
    constexpr int count = 1 << 16;
    const size_t bytes = count * sizeof(float);
    std::vector<float> left(count, 1.25F);
    std::vector<float> right(count, 2.5F);
    std::vector<float> output(count, 0.0F);
    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaMalloc(&device_left, bytes) != cudaSuccess ||
        cudaMalloc(&device_right, bytes) != cudaSuccess ||
        cudaMalloc(&device_output, bytes) != cudaSuccess ||
        cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess) {
        return false;
    }
    bool success = true;
    success = success && cudaMemcpy(device_left, left.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    success = success && cudaMemcpy(device_right, right.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (success) {
        cudaEventRecord(start);
        vector_add_kernel<<<(count + 255) / 256, 256>>>(
            device_left, device_right, device_output, count
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0F;
        success = cudaEventElapsedTime(&milliseconds, start, stop) == cudaSuccess;
        *elapsed_microseconds = static_cast<double>(milliseconds) * 1000.0;
        success = success && cudaMemcpy(
            output.data(), device_output, bytes, cudaMemcpyDeviceToHost
        ) == cudaSuccess;
    }
    for (float value : output) {
        if (std::fabs(value - 3.75F) > 1.0e-6F) {
            success = false;
            break;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_left);
    cudaFree(device_right);
    cudaFree(device_output);
    return success;
}
