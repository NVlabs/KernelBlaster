// SPDX-FileCopyrightText: Copyright (c) 2026 KernelBlaster contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "correctness_metrics.h"
#include "cuda_model.cuh"

void launch_gpu_implementation(void*, void*, void*, int64_t, int64_t);

namespace kc = kernelblaster::correctness;

namespace {

constexpr int64_t kBatch = 17;
constexpr int64_t kClasses = 7;
constexpr int kReuseCalls = 5;
constexpr int kWarmupThreadWaves = 4;
constexpr int kMeasuredThreadWaves = 32;
constexpr size_t kLeakAllowanceBytes = 64 * 1024;
constexpr double kAtol = 1e-2;
constexpr double kRtol = 1e-2;

[[noreturn]] void fail_driver_cuda(
    const char* operation,
    cudaError_t status
) noexcept {
    std::fprintf(
        stderr,
        "KERNELBLASTER_DRIVER_CUDA_ERROR operation=%s status=%d message=%s\n",
        operation,
        static_cast<int>(status),
        cudaGetErrorString(status)
    );
    std::fflush(stderr);
    std::abort();
}

void require_driver_cuda(const char* operation, cudaError_t status) noexcept {
    if (status != cudaSuccess) {
        fail_driver_cuda(operation, status);
    }
}

class StartGate {
public:
    explicit StartGate(int participants) : participants_(participants) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++arrived_;
        condition_.notify_all();
        condition_.wait(lock, [this] { return released_; });
    }

    void release_when_ready() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(
            lock,
            [this] { return arrived_ == participants_; }
        );
        released_ = true;
        lock.unlock();
        condition_.notify_all();
    }

private:
    const int participants_;
    int arrived_ = 0;
    bool released_ = false;
    std::mutex mutex_;
    std::condition_variable condition_;
};

class RepeatGate {
public:
    explicit RepeatGate(int participants) : participants_(participants) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const int generation = generation_;
        ++arrived_;
        if (arrived_ == participants_) {
            arrived_ = 0;
            ++generation_;
            lock.unlock();
            condition_.notify_all();
            return;
        }
        condition_.wait(
            lock,
            [this, generation] { return generation_ != generation; }
        );
    }

private:
    const int participants_;
    int arrived_ = 0;
    int generation_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
};

struct Workload {
    torch::Tensor predictions;
    torch::Tensor targets;
    torch::Tensor reference;
    torch::Tensor output;
};

struct PairResult {
    bool first_deterministic = true;
    bool second_deterministic = true;
};

struct MemorySnapshot {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
};

struct LeakAudit {
    bool bounded = true;
    bool first_deterministic = true;
    bool second_deterministic = true;
};

bool buffers_are_independent(const Workload& first, const Workload& second) {
    return first.predictions.data_ptr() != second.predictions.data_ptr()
        && first.targets.data_ptr() != second.targets.data_ptr()
        && first.output.data_ptr() != second.output.data_ptr();
}

Workload make_workload(int64_t seed, int64_t salt) {
    torch::manual_seed(seed + salt);
    auto half_options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCUDA, 0);
    auto target_options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(torch::kCUDA, 0);
    auto predictions = torch::randn({kBatch, kClasses}, half_options);
    auto targets = torch::randint(
        0, kClasses, {kBatch}, target_options
    );
    auto reference = torch::nn::functional::cross_entropy(
        predictions.to(torch::kFloat32), targets
    );
    return {
        predictions,
        targets,
        reference,
        torch::empty({1}, half_options),
    };
}

std::vector<unsigned char> output_bytes(const torch::Tensor& output) {
    std::vector<unsigned char> bytes(
        static_cast<size_t>(output.numel() * output.element_size())
    );
    require_driver_cuda(
        "cudaMemcpy(output_to_host)",
        cudaMemcpy(
            bytes.data(),
            output.data_ptr(),
            bytes.size(),
            cudaMemcpyDeviceToHost
        )
    );
    return bytes;
}

bool run_calls(
    Workload& workload,
    int calls,
    RepeatGate* repeat_gate = nullptr
) {
    std::vector<unsigned char> first;
    bool deterministic = true;
    for (int call = 0; call < calls; ++call) {
        if (repeat_gate != nullptr) {
            repeat_gate->arrive_and_wait();
        }
        launch_gpu_implementation(
            workload.output.data_ptr(),
            workload.predictions.data_ptr(),
            workload.targets.data_ptr(),
            kBatch,
            kClasses
        );
        require_driver_cuda(
            "cudaStreamSynchronize(cudaStreamLegacy)",
            cudaStreamSynchronize(cudaStreamLegacy)
        );
        auto current = output_bytes(workload.output);
        if (call == 0) {
            first = std::move(current);
        } else {
            deterministic = deterministic && current == first;
        }
    }
    return deterministic;
}

PairResult run_thread_pair(
    Workload& first,
    Workload& second,
    int calls
) {
    StartGate gate(2);
    RepeatGate repeat_gate(2);
    PairResult result;
    std::thread first_thread([&] {
        gate.arrive_and_wait();
        require_driver_cuda("cudaSetDevice(0)", cudaSetDevice(0));
        result.first_deterministic = run_calls(first, calls, &repeat_gate);
    });
    std::thread second_thread([&] {
        gate.arrive_and_wait();
        require_driver_cuda("cudaSetDevice(0)", cudaSetDevice(0));
        result.second_deterministic = run_calls(second, calls, &repeat_gate);
    });
    gate.release_when_ready();
    first_thread.join();
    second_thread.join();
    require_driver_cuda(
        "cudaStreamSynchronize(cudaStreamLegacy)",
        cudaStreamSynchronize(cudaStreamLegacy)
    );
    require_driver_cuda(
        "cudaDeviceSynchronize(thread_pair)", cudaDeviceSynchronize()
    );
    return result;
}

MemorySnapshot memory_snapshot() {
    MemorySnapshot snapshot;
    require_driver_cuda(
        "cudaMemGetInfo",
        cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes)
    );
    return snapshot;
}

size_t memory_decrease(
    const MemorySnapshot& before,
    const MemorySnapshot& after
) {
    if (before.total_bytes != after.total_bytes) {
        return before.total_bytes;
    }
    return before.free_bytes > after.free_bytes
        ? before.free_bytes - after.free_bytes
        : 0;
}

LeakAudit audit_thread_exit_release(Workload& first, Workload& second) {
    LeakAudit audit;
    for (int wave = 0; wave < kWarmupThreadWaves; ++wave) {
        const PairResult result = run_thread_pair(first, second, 1);
        audit.first_deterministic =
            audit.first_deterministic && result.first_deterministic;
        audit.second_deterministic =
            audit.second_deterministic && result.second_deterministic;
    }
    const auto expected_first = output_bytes(first.output);
    const auto expected_second = output_bytes(second.output);
    const MemorySnapshot after_warmup = memory_snapshot();

    for (int wave = 0; wave < kMeasuredThreadWaves; ++wave) {
        run_thread_pair(first, second, 1);
        audit.first_deterministic = audit.first_deterministic
            && output_bytes(first.output) == expected_first;
        audit.second_deterministic = audit.second_deterministic
            && output_bytes(second.output) == expected_second;
    }
    const MemorySnapshot after_first_group = memory_snapshot();

    for (int wave = 0; wave < kMeasuredThreadWaves; ++wave) {
        run_thread_pair(first, second, 1);
        audit.first_deterministic = audit.first_deterministic
            && output_bytes(first.output) == expected_first;
        audit.second_deterministic = audit.second_deterministic
            && output_bytes(second.output) == expected_second;
    }
    const MemorySnapshot after_second_group = memory_snapshot();
    const size_t steady_state_decrease = memory_decrease(
        after_first_group, after_second_group
    );
    audit.bounded = after_warmup.total_bytes == after_first_group.total_bytes
        && after_first_group.total_bytes == after_second_group.total_bytes
        && steady_state_decrease <= kLeakAllowanceBytes;
    if (!audit.bounded) {
        std::cerr
            << "KERNELBLASTER_LIFECYCLE_LEAK task=095 steady_state_decrease="
            << steady_state_decrease
            << " allowance=" << kLeakAllowanceBytes << std::endl;
    }
    return audit;
}

bool append_case(
    const char* case_id,
    int64_t seed,
    const Workload& workload,
    bool deterministic,
    kc::Metrics& aggregate,
    std::vector<std::string>& case_results
) {
    const kc::Metrics metrics = kc::summarize(
        workload.reference, workload.output, kAtol, kRtol
    );
    kc::merge_envelope(aggregate, metrics);
    case_results.push_back(kc::case_json(
        case_id,
        seed,
        "{\"B\":17,\"classes\":7}",
        metrics,
        deterministic
    ));
    return deterministic && metrics.nonfinite_count == 0
        && metrics.mismatch_count == 0 && metrics.normalized_max <= 1.0;
}

}  // namespace

int main() {
    require_driver_cuda("cudaSetDevice(0)", cudaSetDevice(0));
    const std::vector<int64_t> seeds = {0, 42, 20260721};
    kc::Metrics aggregate;
    std::vector<std::string> case_results;
    bool passed = true;
    bool leak_bounded = true;

    for (const int64_t seed : seeds) {
        Workload reuse = make_workload(seed, 1000);
        const bool reuse_deterministic = run_calls(reuse, kReuseCalls);
        passed = append_case(
            "reuse-5-calls",
            seed,
            reuse,
            reuse_deterministic,
            aggregate,
            case_results
        ) && passed;

        Workload parallel_first = make_workload(seed, 2000);
        Workload parallel_second = make_workload(seed, 3000);
        const bool parallel_buffers_independent = buffers_are_independent(
            parallel_first, parallel_second
        );
        const PairResult parallel = run_thread_pair(
            parallel_first, parallel_second, kReuseCalls
        );
        passed = append_case(
            "parallel-host-thread-a",
            seed,
            parallel_first,
            parallel_buffers_independent && parallel.first_deterministic,
            aggregate,
            case_results
        ) && passed;
        passed = append_case(
            "parallel-host-thread-b",
            seed,
            parallel_second,
            parallel_buffers_independent && parallel.second_deterministic,
            aggregate,
            case_results
        ) && passed;

        Workload lifecycle_first = make_workload(seed, 4000);
        Workload lifecycle_second = make_workload(seed, 5000);
        const bool lifecycle_buffers_independent = buffers_are_independent(
            lifecycle_first, lifecycle_second
        );
        bool lifecycle_first_deterministic = true;
        bool lifecycle_second_deterministic = true;
        if (seed == 0) {
            const LeakAudit audit = audit_thread_exit_release(
                lifecycle_first, lifecycle_second
            );
            leak_bounded = audit.bounded;
            lifecycle_first_deterministic = audit.first_deterministic;
            lifecycle_second_deterministic = audit.second_deterministic;
        } else {
            const PairResult lifecycle = run_thread_pair(
                lifecycle_first, lifecycle_second, 1
            );
            lifecycle_first_deterministic = lifecycle.first_deterministic;
            lifecycle_second_deterministic = lifecycle.second_deterministic;
        }
        passed = append_case(
            "thread-exit-waves-a",
            seed,
            lifecycle_first,
            leak_bounded && lifecycle_buffers_independent
                && lifecycle_first_deterministic,
            aggregate,
            case_results
        ) && passed;
        passed = append_case(
            "thread-exit-waves-b",
            seed,
            lifecycle_second,
            leak_bounded && lifecycle_buffers_independent
                && lifecycle_second_deterministic,
            aggregate,
            case_results
        ) && passed;
    }

    const bool finite = aggregate.nonfinite_count == 0;
    std::cout << "KERNELBLASTER_CORRECTNESS_JSON "
              << kc::result_json(aggregate, finite, passed, case_results)
              << std::endl;
    std::cout << (passed ? "passed" : "failed") << std::endl;
    return passed ? 0 : 1;
}
