#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void matvec_row_block_kernel(
    half* __restrict__ output,
    const half* __restrict__ matrix,
    const half* __restrict__ vector,
    int rows,
    int columns
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float local = 0.0f;
    if ((columns & 1) == 0) {
        const half2* matrix2 = reinterpret_cast<const half2*>(matrix + row * columns);
        const half2* vector2 = reinterpret_cast<const half2*>(vector);
        const int pairs = columns / 2;
        for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
            const float2 a = __half22float2(matrix2[pair]);
            const float2 b = __half22float2(vector2[pair]);
            local = fmaf(a.x, b.x, local);
            local = fmaf(a.y, b.y, local);
        }
    } else {
        // An odd row stride makes every second row unsuitable for half2 loads.
        // Keep a scalar path for odd and boundary K without changing the
        // published even-K fast path.
        for (int column = threadIdx.x; column < columns; column += blockDim.x) {
            local = fmaf(
                __half2float(matrix[row * columns + column]),
                __half2float(vector[column]),
                local
            );
        }
    }

    local = warp_sum(local);
    __shared__ float warp_totals[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        float total = lane < 8 ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            output[row] = __float2half_rn(total);
        }
    }
}

void launch_gpu_implementation(
    void* output,
    void* input_A,
    void* input_B,
    int M,
    int K
) {
    matvec_row_block_kernel<<<M, 256>>>(
        static_cast<half*>(output),
        static_cast<const half*>(input_A),
        static_cast<const half*>(input_B),
        M,
        K
    );
    cudaDeviceSynchronize();
}
