
"""聚合 Benchmark 结果，执行案例门控、基线选择并渲染性能图表。"""

from __future__ import annotations

from collections import Counter
import html
import math
from pathlib import Path
from typing import Any, Iterable


FAILURE_CATEGORIES = {
    "llm_request_failed": "provider_api",
    "llm_fanout_failed": "provider_api",
    "llm_smoke_failed": "provider_api",
    "cuda_compile_failed": "cuda_compile",
    "cuda_correctness_failed": "numerical_correctness",
    "cuda_profile_failed": "ncu_profile",
    "runtime_initialization_failed": "runtime_environment",
    "code_parse_failed": "prompt_code_parse",
}


def geometric_mean(values: Iterable[float]) -> float | None:
    """
    处理 `geometric_mean` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
    values: 调用方提供的 `values` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
    ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    if any(value <= 0 for value in numbers):
        raise ValueError("Geometric mean requires positive values.")
    return math.exp(sum(math.log(value) for value in numbers) / len(numbers))


def classify_event(event: dict[str, Any]) -> str | None:
    """
    分类 `classify_event` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
    event: 调用方提供的 `event` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    event_type = str(event.get("event_type", ""))
    if event_type in FAILURE_CATEGORIES:
        return FAILURE_CATEGORIES[event_type]
    if event.get("status") == "error":
        return "other"
    return None


def failure_counts(events: Iterable[dict[str, Any]]) -> Counter[str]:
    """
    处理 `failure_counts` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
    events: 调用方提供的 `events` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    counts: Counter[str] = Counter()
    for event in events:
        category = classify_event(event)
        if category:
            counts[category] += 1
    return counts


def choose_best_benchmarks(
    summaries: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    选择 `choose_best_benchmarks` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
    summaries: 调用方提供的 `summaries` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    selected: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        comparison = summary.get("comparison")
        task_id = str(summary.get("task_id", ""))
        if not comparison or not task_id:
            continue
        if comparison.get("comparison_kind") != "candidate":
            continue
        if comparison.get("formal_valid") is not True:
            continue
        comparison_scope = comparison.get("comparison_scope")
        if comparison_scope is None:
            baseline_name = (
                summary.get("_manifest", {})
                .get("variants", {})
                .get("baseline", {})
                .get("source_name")
            )
            comparison_scope = (
                "upstream_baseline" if baseline_name == "init.cu" else None
            )
        if comparison_scope != "upstream_baseline":
            continue
        speedup = float(comparison.get("speedup", 0) or 0)
        current = selected.get(task_id)
        if current is None or speedup > float(
            current["comparison"].get("speedup", 0) or 0
        ):
            selected[task_id] = summary
    return selected


def deep_case_gate(
    comparison: dict[str, Any] | None,
    *,
    minimum_speedup: float = 1.05,
) -> dict[str, Any]:
    """
    处理 `deep_case_gate` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
    comparison: 调用方提供的 `comparison` 参数。
    minimum_speedup: 调用方提供的 `minimum_speedup` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    speedup = float((comparison or {}).get("speedup", 0) or 0)
    checks = {
        "candidate_comparison": bool(
            comparison and comparison.get("comparison_kind") == "candidate"
        ),
        "formal_valid": bool(comparison and comparison.get("formal_valid") is True),
        "minimum_speedup": speedup >= minimum_speedup,
        "all_sessions_not_slower": bool(
            comparison and comparison.get("all_sessions_not_slower") is True
        ),
    }
    return {
        "passed": all(checks.values()),
        "minimum_speedup": minimum_speedup,
        "observed_speedup": speedup if comparison else None,
        "checks": checks,
    }


def build_task_rows(
    suite_tasks: Iterable[dict[str, Any]],
    best_benchmarks: dict[str, dict[str, Any]],
    events: Iterable[dict[str, Any]],
    baseline_benchmarks: dict[str, dict[str, Any]] | None = None,
    baseline_attempts: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    构建 `build_task_rows` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
    suite_tasks: 调用方提供的 `suite_tasks` 参数。
    best_benchmarks: 调用方提供的 `best_benchmarks` 参数。
    events: 调用方提供的 `events` 参数。
    baseline_benchmarks: 调用方提供的 `baseline_benchmarks` 参数。
    baseline_attempts: 调用方提供的 `baseline_attempts` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    baseline_benchmarks = baseline_benchmarks or {}
    baseline_attempts = baseline_attempts or {}
    categories_by_task: dict[str, Counter[str]] = {}
    for event in events:
        task_id = str(event.get("task_id") or "")
        category = classify_event(event)
        if task_id and category:
            categories_by_task.setdefault(task_id, Counter())[category] += 1

    rows: list[dict[str, Any]] = []
    for task in suite_tasks:
        task_id = str(task["id"])
        benchmark = best_benchmarks.get(task_id)
        comparison = benchmark.get("comparison") if benchmark else None
        speedup = float(comparison["speedup"]) if comparison else None
        failures = categories_by_task.get(task_id, Counter())
        if comparison:
            status = "verified_improved" if speedup and speedup > 1.0 else "correct_not_improved"
        elif failures:
            status = "failed"
        elif task_id in baseline_benchmarks:
            status = "baseline_only"
        elif task_id in baseline_attempts:
            attempt = baseline_attempts[task_id]
            if attempt.get("status") == "completed" and attempt.get("stable") is True:
                status = "baseline_only"
            elif attempt.get("stable") is False:
                status = "baseline_unstable"
            else:
                status = "baseline_failed"
        else:
            status = "pending"
        baseline_only = baseline_benchmarks.get(task_id, {})
        baseline_only_summary = baseline_only.get("summaries", {}).get(
            "baseline", {}
        )
        baseline_attempt = baseline_attempts.get(task_id, {})
        rows.append(
            {
                "task_id": task_id,
                "name": task.get("name", ""),
                "category": task.get("category", ""),
                "status": status,
                "speedup": speedup,
                "candidate": comparison.get("candidate") if comparison else None,
                "provenance": benchmark.get("_candidate_provenance")
                if benchmark
                else None,
                "baseline_median_us": comparison.get("baseline_median_us")
                if comparison
                else baseline_only_summary.get("session_medians_summary", {}).get(
                    "median_us"
                )
                or baseline_attempt.get("baseline_median_us"),
                "candidate_median_us": comparison.get("candidate_median_us")
                if comparison
                else None,
                "all_sessions_not_slower": comparison.get(
                    "all_sessions_not_slower"
                )
                if comparison
                else None,
                "failure_categories": dict(failures),
            }
        )
    return rows


def choose_baseline_benchmarks(
    summaries: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    选择 `choose_baseline_benchmarks` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
    summaries: 调用方提供的 `summaries` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    selected: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        task_id = str(summary.get("task_id", ""))
        baseline = summary.get("summaries", {}).get("baseline")
        if not task_id or not baseline or summary.get("stable") is not True:
            continue
        current = selected.get(task_id)
        current_count = (
            current.get("summaries", {})
            .get("baseline", {})
            .get("all_samples", {})
            .get("count", 0)
            if current
            else 0
        )
        sample_count = baseline.get("all_samples", {}).get("count", 0)
        if current is None or sample_count > current_count:
            selected[task_id] = summary
    return selected


def render_speedup_svg(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """
    渲染 `render_speedup_svg` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
    rows: 调用方提供的 `rows` 参数。
    path: 待读取、写入或校验的文件系统路径。
    """
    data = list(rows)
    width = 900
    row_height = 34
    margin_left = 170
    margin_right = 50
    height = 80 + row_height * len(data)
    scale_width = width - margin_left - margin_right
    maximum = max([float(row.get("speedup") or 1.0) for row in data] + [1.0])
    maximum = max(1.1, maximum * 1.05)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="sans-serif" font-size="20">Core 10 CUDA Events Speedup (higher is better)</text>',
    ]
    baseline_x = margin_left + scale_width / maximum
    lines.append(
        f'<line x1="{baseline_x:.1f}" y1="45" x2="{baseline_x:.1f}" y2="{height - 20}" stroke="#555" stroke-dasharray="4 4"/>'
    )
    for index, row in enumerate(data):
        y = 58 + index * row_height
        speedup = float(row.get("speedup") or 1.0)
        bar_width = scale_width * speedup / maximum
        color = "#2c7fb8" if row.get("speedup") is not None else "#bdbdbd"
        label = html.escape(f"{row['task_id']} {row['name']}")
        lines.extend(
            [
                f'<text x="10" y="{y + 17}" font-family="sans-serif" font-size="12">{label}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="22" fill="{color}"/>',
                f'<text x="{margin_left + bar_width + 5:.1f}" y="{y + 16}" font-family="sans-serif" font-size="12">{speedup:.3f}x</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
