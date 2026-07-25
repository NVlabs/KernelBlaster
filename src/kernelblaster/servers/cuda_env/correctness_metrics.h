/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace kernelblaster::correctness {

// 汇总一个或多个正确性案例的误差包络。所有绝对误差字段沿用输入张量的
// 数值尺度；normalized_* 为无量纲误差，1.0 对应 atol/rtol 容差边界。
struct Metrics {
    // 实际参与统计的元素数，以及用于估算分位数的降采样元素数。
    int64_t count = 0;
    int64_t quantile_sample_count = 0;
    // mismatch_count 统计 normalized > 1 的元素；nonfinite_count 统计候选中的 NaN/Inf。
    int64_t mismatch_count = 0;
    int64_t nonfinite_count = 0;
    // 全量绝对误差的均值、均方根和若干尾部分位数。
    double abs_mean = 0.0;
    double abs_rmse = 0.0;
    double abs_p50 = 0.0;
    double abs_p90 = 0.0;
    double abs_p99 = 0.0;
    double abs_p999 = 0.0;
    double abs_max = 0.0;
    // 归一化误差分位数，用于直接判断给定 atol/rtol 下的容差余量。
    double normalized_p50 = 0.0;
    double normalized_p90 = 0.0;
    double normalized_p99 = 0.0;
    double normalized_p999 = 0.0;
    double normalized_max = 0.0;
};

// 合并不同案例时，均值和 RMSE 按元素数量加权；分位数和最大值取各案例的
// 最大值，形成保守包络。后者不是把所有样本重新拼接后计算的全局分位数。
inline void merge_envelope(Metrics& aggregate, const Metrics& value) {
    const int64_t combined_count = aggregate.count + value.count;
    if (combined_count == 0) return;
    aggregate.abs_mean = (
        aggregate.abs_mean * aggregate.count + value.abs_mean * value.count
    ) / static_cast<double>(combined_count);
    aggregate.abs_rmse = std::sqrt((
        aggregate.abs_rmse * aggregate.abs_rmse * aggregate.count
        + value.abs_rmse * value.abs_rmse * value.count
    ) / static_cast<double>(combined_count));
    aggregate.count = combined_count;
    aggregate.quantile_sample_count += value.quantile_sample_count;
    aggregate.mismatch_count += value.mismatch_count;
    aggregate.nonfinite_count += value.nonfinite_count;
    aggregate.abs_p50 = std::max(aggregate.abs_p50, value.abs_p50);
    aggregate.abs_p90 = std::max(aggregate.abs_p90, value.abs_p90);
    aggregate.abs_p99 = std::max(aggregate.abs_p99, value.abs_p99);
    aggregate.abs_p999 = std::max(aggregate.abs_p999, value.abs_p999);
    aggregate.abs_max = std::max(aggregate.abs_max, value.abs_max);
    aggregate.normalized_p50 = std::max(
        aggregate.normalized_p50, value.normalized_p50
    );
    aggregate.normalized_p90 = std::max(
        aggregate.normalized_p90, value.normalized_p90
    );
    aggregate.normalized_p99 = std::max(
        aggregate.normalized_p99, value.normalized_p99
    );
    aggregate.normalized_p999 = std::max(
        aggregate.normalized_p999, value.normalized_p999
    );
    aggregate.normalized_max = std::max(
        aggregate.normalized_max, value.normalized_max
    );
}

// 从已升序排列且非空的样本中取得向上取整的 nearest-rank 分位数。
// q 应位于 [0, 1]；调用方负责保证张量已经排序且至少含一个元素。
inline double sample_quantile(const torch::Tensor& sorted, double q) {
    const int64_t index = static_cast<int64_t>(
        std::ceil(q * static_cast<double>(sorted.numel() - 1))
    );
    return sorted[index].item<double>();
}

// 为限制排序开销，最多保留 2^20 个等步长样本。采样完全确定，但得到的是
// 近似分位数；最大值等精确包络指标仍由完整张量单独计算。
inline torch::Tensor sorted_quantile_sample(const torch::Tensor& values) {
    constexpr int64_t max_samples = 1 << 20;
    auto flat = values.flatten();
    if (flat.numel() > max_samples) {
        const int64_t stride = (flat.numel() + max_samples - 1) / max_samples;
        flat = flat.slice(0, 0, flat.numel(), stride);
    }
    return std::get<0>(flat.sort());
}

// 比较参考输出与候选输出。计算统一提升到 FP32，并采用
//     normalized = |candidate - reference| / (atol + rtol * |reference|)
// 作为逐元素无量纲误差；normalized > 1 表示超出调用方给定的混合容差。
// 输入必须形状兼容且非空，否则张量广播、mean/max 或分位数读取会失败。
inline Metrics summarize(
    const torch::Tensor& reference,
    const torch::Tensor& candidate,
    double atol,
    double rtol
) {
    auto reference_fp32 = reference.to(torch::kFloat32);
    auto candidate_fp32 = candidate.to(torch::kFloat32);
    auto absolute = (candidate_fp32 - reference_fp32).abs().flatten();
    auto normalized = (
        absolute / (atol + rtol * reference_fp32.abs().flatten())
    );
    auto absolute_sample = sorted_quantile_sample(absolute);
    auto normalized_sample = sorted_quantile_sample(normalized);
    Metrics metrics;
    metrics.count = absolute.numel();
    metrics.quantile_sample_count = absolute_sample.numel();
    metrics.nonfinite_count = torch::logical_not(
        torch::isfinite(candidate_fp32)
    ).sum().item<int64_t>();
    metrics.mismatch_count = (normalized > 1.0).sum().item<int64_t>();
    metrics.abs_mean = absolute.mean().item<double>();
    metrics.abs_rmse = absolute.square().mean().sqrt().item<double>();
    metrics.abs_p50 = sample_quantile(absolute_sample, 0.50);
    metrics.abs_p90 = sample_quantile(absolute_sample, 0.90);
    metrics.abs_p99 = sample_quantile(absolute_sample, 0.99);
    metrics.abs_p999 = sample_quantile(absolute_sample, 0.999);
    metrics.abs_max = absolute.max().item<double>();
    metrics.normalized_p50 = sample_quantile(normalized_sample, 0.50);
    metrics.normalized_p90 = sample_quantile(normalized_sample, 0.90);
    metrics.normalized_p99 = sample_quantile(normalized_sample, 0.99);
    metrics.normalized_p999 = sample_quantile(normalized_sample, 0.999);
    metrics.normalized_max = normalized.max().item<double>();
    return metrics;
}

// 输出不带最外层花括号的稳定 JSON 字段片段，供单案例与聚合结果复用。
// 数值使用十位有效精度；字段名同时记录分位数采样策略和样本上限。
inline std::string fields_json(const Metrics& metrics) {
    std::ostringstream output;
    output << std::setprecision(10)
           << "\"count\":" << metrics.count
           << ",\"quantile_sample_count\":" << metrics.quantile_sample_count
           << ",\"quantile_sampling\":\"deterministic_stride\""
           << ",\"quantile_max_samples\":1048576"
           << ",\"mismatch_count\":" << metrics.mismatch_count
           << ",\"nonfinite_count\":" << metrics.nonfinite_count
           << ",\"abs_mean\":" << metrics.abs_mean
           << ",\"abs_rmse\":" << metrics.abs_rmse
           << ",\"abs_p50\":" << metrics.abs_p50
           << ",\"abs_p90\":" << metrics.abs_p90
           << ",\"abs_p99\":" << metrics.abs_p99
           << ",\"abs_p999\":" << metrics.abs_p999
           << ",\"abs_max\":" << metrics.abs_max
           << ",\"normalized_p50\":" << metrics.normalized_p50
           << ",\"normalized_p90\":" << metrics.normalized_p90
           << ",\"normalized_p99\":" << metrics.normalized_p99
           << ",\"normalized_p999\":" << metrics.normalized_p999
           << ",\"normalized_max\":" << metrics.normalized_max;
    return output.str();
}

// 序列化单个测试案例。case_id 与 shape_json 由受控调用方生成，必须已经满足
// JSON 字符串/对象语法要求；本函数不会再次转义这些片段。
inline std::string case_json(
    const std::string& case_id,
    int64_t seed,
    const std::string& shape_json,
    const Metrics& metrics,
    bool deterministic
) {
    std::ostringstream output;
    output << "{\"case_id\":\"" << case_id << "\",\"seed\":" << seed
           << ",\"shape\":" << shape_json
           << ",\"deterministic\":" << (deterministic ? "true" : "false")
           << "," << fields_json(metrics) << "}";
    return output.str();
}

// 序列化任务级聚合结果。aggregate 中的分位数采用“各案例分位数最大值包络”
// 语义，cases 则保留每个案例的完整 JSON，便于定位被聚合值掩盖的失败来源。
inline std::string result_json(
    const Metrics& aggregate,
    bool finite,
    bool deterministic,
    const std::vector<std::string>& cases
) {
    std::ostringstream output;
    output << "{\"max_abs_error\":" << std::setprecision(10)
           << aggregate.abs_max
           << ",\"p99_abs_error\":" << aggregate.abs_p99
           << ",\"finite\":" << (finite ? "true" : "false")
           << ",\"deterministic\":" << (deterministic ? "true" : "false")
           << ",\"aggregate_quantile_semantics\":"
              "\"max_per_case_quantile_envelope\""
           << "," << fields_json(aggregate) << ",\"cases\":[";
    for (size_t index = 0; index < cases.size(); ++index) {
        if (index != 0) output << ',';
        output << cases[index];
    }
    output << "]}";
    return output.str();
}

}  // namespace kernelblaster::correctness
