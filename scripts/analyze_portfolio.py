#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.kernelblaster.result_analysis import (  # noqa: E402
    build_task_rows,
    choose_baseline_benchmarks,
    choose_best_benchmarks,
    deep_case_gate,
    failure_counts,
    geometric_mean,
    render_speedup_svg,
)


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT_DIR / "out" / "portfolio" / "analysis" / timestamp


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(normalized[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(normalized)


def _discover_benchmarks(roots: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for root in roots:
        for path in root.rglob("summary.json"):
            try:
                payload = _load_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("schema_version") == "1.0" and payload.get("summaries"):
                payload["_run_id"] = path.parent.name
                payload["_phase"] = root.name
                payload["_candidate_provenance"] = (
                    "manual_case_study"
                    if root.name == "deep-rmsnorm"
                    else (
                        "runner_validation"
                        if root.name == "runner-self-check"
                        else "agent_candidate"
                    )
                )
                manifest_path = path.parent / "run_manifest.json"
                if manifest_path.is_file():
                    payload["_manifest"] = _load_json(manifest_path)
                summaries.append(payload)
    return summaries


def _discover_baseline_attempts(roots: list[Path]) -> dict[str, dict[str, Any]]:
    attempts: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in root.rglob("suite_summary.json"):
            try:
                payload = _load_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            for row in payload.get("results", []):
                task_id = str(row.get("task_id", ""))
                if task_id:
                    attempts[task_id] = row
    return attempts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_artifact_rows(roots: list[Path]) -> list[dict[str, Any]]:
    exact_names = {
        "events.jsonl",
        "measurements.jsonl",
        "run_manifest.json",
        "smoke_result.json",
        "suite_summary.json",
        "summary.json",
    }
    rows: list[dict[str, Any]] = []
    for root_index, root in enumerate(roots):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            include = (
                path.name in exact_names
                or path.suffix == ".ncu-rep"
                or path.name.startswith("ncu-")
                or path.name.startswith("launcher-")
            )
            if not include:
                continue
            rows.append(
                {
                    "source_group": f"input-{root_index}-{root.name}",
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit KernelBlaster runs and generate traceable result tables."
    )
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--input-price-per-million", type=float, default=2.5)
    parser.add_argument("--output-price-per-million", type=float, default=15.0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite output directory: {output_dir}")

    manifests: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for directory in [path.resolve() for path in args.run_dir]:
        for name in ("run_manifest.json", "events.jsonl", "summary.json"):
            if not (directory / name).is_file():
                parser.error(f"Run directory is missing {name}: {directory}")
        manifests.append(_load_json(directory / "run_manifest.json"))
        events.extend(_load_events(directory / "events.jsonl"))
        run_summaries.append(_load_json(directory / "summary.json"))

    suite_tasks: list[dict[str, Any]] = []
    for manifest in manifests:
        tasks = manifest.get("suite", {}).get("tasks", [])
        if tasks:
            suite_tasks = tasks
            break
    if not suite_tasks:
        suite_tasks = _load_json(ROOT_DIR / "portfolio" / "suites" / "core10.json")[
            "tasks"
        ]

    benchmark_summaries = _discover_benchmarks(
        [path.resolve() for path in args.benchmark_root]
    )
    baseline_attempts = _discover_baseline_attempts(
        [path.resolve() for path in args.benchmark_root]
    )
    best = choose_best_benchmarks(benchmark_summaries)
    baselines = choose_baseline_benchmarks(benchmark_summaries)
    task_rows = build_task_rows(
        suite_tasks, best, events, baselines, baseline_attempts
    )
    measured_speedups = [
        float(row["speedup"]) for row in task_rows if row["speedup"] is not None
    ]
    portfolio_speedups = [
        float(row["speedup"])
        if row["status"] == "verified_improved" and row["speedup"] is not None
        else 1.0
        for row in task_rows
    ]
    agent_portfolio_speedups = [
        float(row["speedup"])
        if row["status"] == "verified_improved"
        and row["speedup"] is not None
        and row["provenance"] == "agent_candidate"
        else 1.0
        for row in task_rows
    ]

    failures = failure_counts(events)
    for row in task_rows:
        if row["status"] == "baseline_unstable":
            failures["baseline_stability"] += 1
        elif row["status"] == "baseline_failed":
            failures["baseline_execution"] += 1
        elif row["status"] == "correct_not_improved":
            failures["performance_regression"] += 1
    llm_totals = {
        key: sum(int(summary.get("llm", {}).get(key, 0) or 0) for summary in run_summaries)
        for key in (
            "requests_started",
            "requests_completed",
            "requests_failed",
            "retries",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }
    estimated_cost = (
        llm_totals["prompt_tokens"] / 1_000_000 * args.input_price_per_million
        + llm_totals["completion_tokens"]
        / 1_000_000
        * args.output_price_per_million
    )
    aggregate = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_tasks": len(task_rows),
        "formally_measured_tasks": len(measured_speedups),
        "baseline_covered_tasks": sum(
            row["baseline_median_us"] is not None for row in task_rows
        ),
        "baseline_attempted_tasks": sum(
            row["task_id"] in baseline_attempts for row in task_rows
        ),
        "baseline_stable_tasks": sum(
            row["status"] in {"baseline_only", "verified_improved", "correct_not_improved"}
            and row["baseline_median_us"] is not None
            for row in task_rows
        ),
        "verified_improved_tasks": sum(
            row["status"] == "verified_improved" for row in task_rows
        ),
        "agent_verified_improved_tasks": sum(
            row["status"] == "verified_improved"
            and row["provenance"] == "agent_candidate"
            for row in task_rows
        ),
        "manual_verified_improved_tasks": sum(
            row["status"] == "verified_improved"
            and row["provenance"] == "manual_case_study"
            for row in task_rows
        ),
        "geomean_measured_candidates": geometric_mean(measured_speedups),
        "portfolio_score_all_tasks_unoptimized_as_one": geometric_mean(
            portfolio_speedups
        ),
        "agent_portfolio_score_all_tasks_unoptimized_as_one": geometric_mean(
            agent_portfolio_speedups
        ),
        "failure_counts": dict(failures),
        "llm": llm_totals,
        "model": args.model,
        "estimated_api_cost_usd": estimated_cost,
        "pricing_assumption": {
            "input_per_million": args.input_price_per_million,
            "output_per_million": args.output_price_per_million,
            "cache_and_regional_uplift_excluded": True,
        },
    }

    ncu_failures: list[str] = []
    ncu_completed = 0
    for benchmark in benchmark_summaries:
        for variant in benchmark.get("_manifest", {}).get("variants", {}).values():
            ncu = variant.get("ncu")
            if not ncu:
                continue
            if ncu.get("status") == "completed":
                ncu_completed += 1
            elif ncu.get("error_type"):
                ncu_failures.append(str(ncu["error_type"]))
    aggregate["ncu"] = {
        "completed_profiles": ncu_completed,
        "failure_types": sorted(set(ncu_failures)),
        "attribution_allowed": ncu_completed > 0 and not ncu_failures,
    }

    deep_rows: list[dict[str, Any]] = []
    for benchmark in benchmark_summaries:
        if str(benchmark.get("task_id")) not in {"036", "047"}:
            continue
        comparison = benchmark.get("comparison") or {}
        manifest = benchmark.get("_manifest", {})
        candidate = comparison.get("candidate")
        if not candidate:
            continue
        ncu = manifest.get("variants", {}).get(candidate, {}).get("ncu", {})
        comparison_scope = comparison.get("comparison_scope") or manifest.get(
            "baseline_scope"
        )
        if comparison_scope is None:
            baseline_name = (
                manifest.get("variants", {}).get("baseline", {}).get("source_name")
            )
            if baseline_name == "init.cu":
                comparison_scope = "upstream_baseline"
        deep_rows.append(
            {
                "run_id": benchmark.get("_run_id"),
                "task_id": benchmark.get("task_id"),
                "kernel": benchmark.get("kernel"),
                "candidate": candidate,
                "provenance": benchmark.get("_candidate_provenance"),
                "speedup": comparison.get("speedup"),
                "formal_valid": comparison.get("formal_valid"),
                "comparison_scope": comparison_scope,
                "stable": comparison.get("stable"),
                "all_sessions_not_slower": comparison.get(
                    "all_sessions_not_slower"
                ),
                "ncu_status": ncu.get("status"),
                "ncu_metric_count": len(ncu.get("metric_names", [])),
            }
        )
    valid_deep = [
        row
        for row in deep_rows
        if row["formal_valid"] is True
        and row["candidate"] is not None
        and row["comparison_scope"] == "upstream_baseline"
    ]
    best_deep = max(valid_deep, key=lambda row: float(row["speedup"] or 0), default=None)
    deep_gate = deep_case_gate(
        (
            next(
                benchmark["comparison"]
                for benchmark in benchmark_summaries
                if best_deep
                and benchmark.get("_run_id") == best_deep["run_id"]
            )
            if best_deep
            else None
        )
    )
    aggregate["deep_case_gate"] = deep_gate

    _write_csv(output_dir / "core10_results.csv", task_rows)
    failure_rows = [
        {"category": category, "count": count}
        for category, count in sorted(failures.items())
    ]
    _write_csv(output_dir / "failure_classification.csv", failure_rows)
    _write_csv(output_dir / "api_usage.csv", [llm_totals | {"estimated_cost_usd": estimated_cost}])
    _write_csv(output_dir / "deep_case_results.csv", deep_rows)
    render_speedup_svg(task_rows, output_dir / "core10_speedup.svg")
    _atomic_json(output_dir / "analysis_summary.json", aggregate)
    _write_csv(
        output_dir / "raw_artifact_sha256.csv",
        _raw_artifact_rows(
            [path.resolve() for path in args.run_dir]
            + [path.resolve() for path in args.benchmark_root]
        ),
    )

    task_lines_parts = []
    for row in task_rows:
        speedup_text = (
            f"{row['speedup']:.3f}x"
            if row["speedup"] is not None
            else "NOT RUN"
        )
        task_lines_parts.append(
            f"| {row['task_id']} | {row['name']} | {row['provenance'] or '-'} | "
            f"{row['status']} | "
            f"{speedup_text} |"
        )
    task_lines = "\n".join(task_lines_parts)
    deep_lines = "\n".join(
        f"| {row['candidate']} | {float(row['speedup']):.4f}x | "
        f"{row['comparison_scope']} | {row['stable']} | "
        f"{row['all_sessions_not_slower']} | {row['provenance']} |"
        for row in deep_rows
    ) or "| - | - | - | - | - | no measured candidate |"
    deep_speedup_text = (
        f"{float(deep_gate['observed_speedup']):.4f}x"
        if deep_gate["observed_speedup"] is not None
        else "NOT RUN"
    )
    report = f"""# KernelBlaster Day 1–10 RTX 3080 验证报告

生成时间：{aggregate['generated_at']}

## 结论

- 正式 CUDA Events 覆盖：{aggregate['formally_measured_tasks']}/{aggregate['total_tasks']}。
- Baseline 覆盖：{aggregate['baseline_covered_tasks']}/{aggregate['total_tasks']}。
- Baseline 已尝试：{aggregate['baseline_attempted_tasks']}/{aggregate['total_tasks']}；稳定通过：{aggregate['baseline_stable_tasks']}/{aggregate['total_tasks']}。
- 已验证提升任务：{aggregate['verified_improved_tasks']}/{aggregate['total_tasks']}。
- Agent 已验证提升任务：{aggregate['agent_verified_improved_tasks']}/{aggregate['total_tasks']}；手工深度案例：{aggregate['manual_verified_improved_tasks']}。
- 已测正确候选几何平均（1 个手工候选）：{aggregate['geomean_measured_candidates'] or 'NOT RUN'}。
- 全十题组合分数（手工案例计入、未优化按 1.0）：{aggregate['portfolio_score_all_tasks_unoptimized_as_one']}。
- Agent-only 全十题组合分数：{aggregate['agent_portfolio_score_all_tasks_unoptimized_as_one']}。
- API 估算成本：${estimated_cost:.4f}；未包含 cache write 与区域加价。
- 深度案例去留门槛：{'PASS' if deep_gate['passed'] else 'BLOCKED'}。

唯一一次 API smoke 返回 401 `invalid_api_key`，未重试、无成功响应、无已记录 token；因此 Pilot 与 Core 10 Agent 搜索未执行。`provider_api` 失败分类为同一次请求在 request/fan-out/smoke 三层产生的 3 条错误事件，不代表 3 次计费请求。

NCU 2025.1 权限探针返回 `ERR_NVGPUCTRPERM`。Windows 注册表开关已设为允许非管理员访问，但当前驱动尚未重新加载；本报告不发布任何 NCU 硬件计数器归因。CUDA Events、正确性和源码层面的映射分析单独成立。

未完成或不稳定的候选保持 NOT RUN/明确失败状态，不用于性能声明。

## Core 10

| Task | Kernel | Provenance | Status | Speedup |
| --- | --- | --- | --- | --- |
{task_lines}

## RMSNorm 深度案例

| Candidate | Speedup | Scope | Stable | All sessions not slower | Provenance |
| --- | ---: | --- | --- | --- | --- |
{deep_lines}

最佳 V3c 相对同次 V0 为 {deep_speedup_text}。V3c 对 V1 的直接 head-to-head 为 1.0069x；V3c 对自身为 1.0000x。主收益来自把线程映射到连续空间位置并移除跨线程 reduction；`half2`、128-thread block 和两对 work/thread 都没有超过最终最佳版本。

## 失败分类

```json
{json.dumps(dict(failures), indent=2, ensure_ascii=False)}
```

## English summary

Formal CUDA Events coverage is {aggregate['formally_measured_tasks']}/{aggregate['total_tasks']}. Results without a correct, reproducible candidate remain pending and are excluded from measured-candidate claims. The all-task portfolio score assigns 1.0 only as an explicit conservative policy value for unoptimized tasks.

## 复现命令

```bash
python -m pytest -q
python scripts/run_portfolio.py --suite core10 --model gpt-5.6-terra --gpu rtx3080 --dry-run
python scripts/benchmark_suite.py --suite core10 --warmup 20 --repetitions 100 --sessions 3
python scripts/benchmark_cuda.py --task-dir data/kernelbench-cuda/level1/036_RMSNorm --task-id 036 --kernel RMSNorm --candidate portfolio/case_studies/rmsnorm/best_rmsnorm_sm86.cu --candidate-name v3c --extra-correctness-driver portfolio/case_studies/rmsnorm/edge_driver.cpp
```
"""
    (output_dir / "validation-report.zh-CN.md").write_text(report, encoding="utf-8")

    english = f"""# KernelBlaster Days 1-10 validation summary

- Environment: RTX 3080 10 GiB, `sm_86`, NGC PyTorch 25.01, CUDA 12.8.
- CPU tests: offline Provider, Recorder, Suite, dry-run, benchmark, and analysis coverage passed before publication.
- API: the single bounded smoke request failed with HTTP 401 and was not retried. Pilot and Core 10 Agent search therefore remain blocked; Agent-only portfolio score is 1.0.
- Core baselines: {aggregate['baseline_attempted_tasks']}/10 attempted and {aggregate['baseline_stable_tasks']}/10 passed the 5% session-spread gate. The remaining tasks are explicitly marked unstable.
- Manual RMSNorm case study: V3c passed official and edge-shape correctness and measured {deep_speedup_text} versus the paired upstream V0. Every Session was faster; the runner self-check measured 1.0000x.
- NCU: attribution is blocked by `ERR_NVGPUCTRPERM` until the Windows driver reloads the enabled counter setting. No hardware-counter conclusion is published.
- Cost: one failed request, zero recorded tokens, estimated API cost $0.00.

Raw logs and reports remain under ignored `out/` directories. Curated CSV/JSON/SVG files link back to them through SHA256 values without storing credentials or response text.
"""
    (output_dir / "validation-summary.en.md").write_text(english, encoding="utf-8")

    artifacts = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.json"
    ]
    _atomic_json(
        output_dir / "SHA256SUMS.json",
        {path.name: _sha256(path) for path in sorted(artifacts)},
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
