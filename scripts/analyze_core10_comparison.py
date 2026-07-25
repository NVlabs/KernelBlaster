#!/usr/bin/env python3
"""Join formal CUDA candidate and same-GPU PyTorch Core 10 results."""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
TASK_IDS = ("004", "007", "019", "023", "026", "036", "040", "047", "088", "095")
MIN_MATERIAL_SPEEDUP = 1.01
MIN_CONFIRMATION_SESSIONS = 5
BOOTSTRAP_RESAMPLES = 10_000


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("Geometric mean requires positive values.")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def paired_bootstrap_interval(
    session_speedups: list[float],
    *,
    seed: int = 20260719,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float] | None:
    if len(session_speedups) < 2:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [rng.choice(session_speedups) for _ in session_speedups]
        estimates.append(statistics.median(sample))
    estimates.sort()
    lower_index = int(0.025 * (len(estimates) - 1))
    upper_index = int(0.975 * (len(estimates) - 1))
    return estimates[lower_index], estimates[upper_index]


def _eager_method(task_id: str) -> str:
    return "pytorch_driver_formula" if task_id == "088" else "pytorch_eager"


def augment_candidate_details(
    candidate_payload: dict[str, Any], base_dir: Path
) -> None:
    """Join p10/p90/session medians from each append-only task summary."""
    for row in candidate_payload.get("results", []):
        relative = row.get("summary")
        if not relative:
            continue
        path = (base_dir / relative).resolve()
        detail = _load(path)
        manifest_path = path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Task {row['task_id']} is missing run_manifest.json.")
        manifest = _load(manifest_path)
        candidate_variant = manifest.get("variants", {}).get(row["candidate"])
        baseline_variant = manifest.get("variants", {}).get("baseline")
        if not candidate_variant or not baseline_variant:
            raise ValueError(f"Task {row['task_id']} manifest is missing measured variants.")
        source_path = (
            ROOT_DIR / "portfolio" / "case_studies" / "core10" / row["source"]
        ).resolve()
        measured_source_hash = str(candidate_variant["source_sha256"])
        if not source_path.is_file() or _sha256(source_path) != measured_source_hash:
            raise ValueError(
                f"Task {row['task_id']} candidate source does not match the measured hash."
            )
        row.update(
            {
                "measurement_git_commit": manifest.get("git_commit"),
                "profiling_mode": manifest.get("profiling_mode"),
                "correctness_error_regression": manifest.get(
                    "correctness_error_regression", {}
                ).get("status"),
                "candidate_source_sha256": measured_source_hash,
                "candidate_normalized_sha256": candidate_variant.get(
                    "normalized_sha256"
                ),
                "baseline_source_sha256": baseline_variant.get("source_sha256"),
                "driver_sha256": manifest.get("driver_sha256"),
                "extra_correctness_drivers": manifest.get(
                    "extra_correctness_drivers", []
                ),
                "raw_summary_sha256": _sha256(path),
                "raw_manifest_sha256": _sha256(manifest_path),
            }
        )
        variants = detail.get("summaries", {})
        for prefix, variant in (("baseline", "baseline"), ("candidate", row["candidate"])):
            selected = variants.get(variant)
            if not selected:
                continue
            row[f"{prefix}_p10_us"] = selected["all_samples"]["p10_us"]
            row[f"{prefix}_p90_us"] = selected["all_samples"]["p90_us"]
            row[f"{prefix}_session_medians_us"] = selected["session_medians"]
            row[f"{prefix}_session_spread_percent"] = selected[
                "session_spread_percent"
            ]


def build_comparison_rows(
    candidate_payload: dict[str, Any], pytorch_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    candidate_rows = {
        str(row["task_id"]): row for row in candidate_payload.get("results", [])
    }
    pytorch_rows: dict[str, list[dict[str, Any]]] = {}
    for row in pytorch_payload.get("results", []):
        pytorch_rows.setdefault(str(row["task_id"]), []).append(row)
    missing_candidates = sorted(set(TASK_IDS) - set(candidate_rows))
    missing_pytorch = sorted(set(TASK_IDS) - set(pytorch_rows))
    if missing_candidates or missing_pytorch:
        raise ValueError(
            f"Incomplete Core 10 inputs; candidate={missing_candidates}, pytorch={missing_pytorch}"
        )

    rows: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        candidate = candidate_rows[task_id]
        torch_methods = pytorch_rows[task_id]
        eager_name = _eager_method(task_id)
        eager = next((row for row in torch_methods if row["method"] == eager_name), None)
        if eager is None:
            raise ValueError(f"Task {task_id} is missing {eager_name}.")
        correct_methods = [row for row in torch_methods if row.get("correct")]
        if not correct_methods:
            raise ValueError(f"Task {task_id} has no correct PyTorch method.")
        stable_methods = [row for row in correct_methods if row.get("stable")]
        best = (
            min(stable_methods, key=lambda row: float(row["median_us"]))
            if stable_methods
            else None
        )

        baseline_us = float(candidate["baseline_median_us"])
        candidate_us = float(candidate["candidate_median_us"])
        attempted_speedup = float(candidate["speedup"])
        session_speedups = [
            float(value) for value in (candidate.get("session_speedups") or [])
        ]
        bootstrap_interval = paired_bootstrap_interval(session_speedups)
        bootstrap_lower = bootstrap_interval[0] if bootstrap_interval else None
        confirmation_ready = len(session_speedups) >= MIN_CONFIRMATION_SESSIONS
        verified_improvement = bool(
            candidate.get("correct")
            and candidate.get("stable")
            and candidate.get("performance_claim_allowed")
            and candidate.get("all_sessions_not_slower")
            and attempted_speedup >= MIN_MATERIAL_SPEEDUP
            and confirmation_ready
            and bootstrap_lower is not None
            and bootstrap_lower > 1.0
        )
        if verified_improvement:
            candidate_outcome = "improved"
            candidate_exclusion_reason = None
        elif not candidate.get("correct"):
            candidate_outcome = "failed"
            candidate_exclusion_reason = "correctness_failed"
        elif not candidate.get("stable"):
            candidate_outcome = "inconclusive"
            candidate_exclusion_reason = "session_spread_exceeded"
        elif not confirmation_ready or bootstrap_lower is None:
            candidate_outcome = "inconclusive"
            candidate_exclusion_reason = "insufficient_confirmation"
        else:
            candidate_outcome = "no_improvement"
            candidate_exclusion_reason = (
                "paired_session_slower"
                if not candidate.get("all_sessions_not_slower")
                else "formal_performance_gate_failed"
            )
        selected_us = candidate_us if verified_improvement else baseline_us
        selected_variant = candidate["candidate"] if verified_improvement else "upstream_baseline"
        eager_us = float(eager["median_us"])
        best_us = float(best["median_us"]) if best is not None else None
        rows.append(
            {
                "task_id": task_id,
                "kernel": candidate["kernel"],
                "candidate": candidate["candidate"],
                "source": candidate["source"],
                "measurement_git_commit": candidate.get("measurement_git_commit"),
                "profiling_mode": candidate.get("profiling_mode"),
                "correctness_error_regression": candidate.get(
                    "correctness_error_regression"
                ),
                "candidate_source_sha256": candidate.get("candidate_source_sha256"),
                "candidate_normalized_sha256": candidate.get(
                    "candidate_normalized_sha256"
                ),
                "baseline_source_sha256": candidate.get("baseline_source_sha256"),
                "driver_sha256": candidate.get("driver_sha256"),
                "extra_correctness_drivers": candidate.get(
                    "extra_correctness_drivers", []
                ),
                "raw_summary_sha256": candidate.get("raw_summary_sha256"),
                "raw_manifest_sha256": candidate.get("raw_manifest_sha256"),
                "correct": bool(candidate.get("correct")),
                "stable": bool(candidate.get("stable")),
                "all_sessions_not_slower": bool(candidate.get("all_sessions_not_slower")),
                "baseline_median_us": baseline_us,
                "baseline_p10_us": candidate.get("baseline_p10_us"),
                "baseline_p90_us": candidate.get("baseline_p90_us"),
                "baseline_session_medians_us": candidate.get(
                    "baseline_session_medians_us"
                ),
                "baseline_session_spread_percent": candidate.get(
                    "baseline_session_spread_percent"
                ),
                "candidate_median_us": candidate_us,
                "candidate_p10_us": candidate.get("candidate_p10_us"),
                "candidate_p90_us": candidate.get("candidate_p90_us"),
                "candidate_session_medians_us": candidate.get(
                    "candidate_session_medians_us"
                ),
                "candidate_session_spread_percent": candidate.get(
                    "candidate_session_spread_percent"
                ),
                "attempted_speedup": attempted_speedup,
                "session_speedups": session_speedups,
                "confirmation_sessions": len(session_speedups),
                "confirmation_ready": confirmation_ready,
                "candidate_outcome": candidate_outcome,
                "candidate_exclusion_reason": candidate_exclusion_reason,
                "bootstrap_95_lower": bootstrap_lower,
                "bootstrap_95_upper": (
                    bootstrap_interval[1] if bootstrap_interval else None
                ),
                "verified_improvement": verified_improvement,
                "selected_variant": selected_variant,
                "selected_median_us": selected_us,
                "portfolio_speedup": baseline_us / selected_us,
                "pytorch_eager_method": eager_name,
                "pytorch_eager_median_us": eager_us,
                "pytorch_eager_stable": bool(eager.get("stable")),
                "pytorch_eager_p10_us": eager.get("p10_us"),
                "pytorch_eager_p90_us": eager.get("p90_us"),
                "candidate_vs_pytorch_eager": eager_us / candidate_us,
                "selected_vs_pytorch_eager": eager_us / selected_us,
                "pytorch_comparison_status": (
                    "comparable" if best is not None else "inconclusive"
                ),
                "pytorch_best_method": best["method"] if best is not None else None,
                "pytorch_best_allocation_mode": (
                    best["allocation_mode"] if best is not None else None
                ),
                "pytorch_best_median_us": best_us,
                "pytorch_best_stable": bool(best and best.get("stable")),
                "pytorch_best_p10_us": best.get("p10_us") if best else None,
                "pytorch_best_p90_us": best.get("p90_us") if best else None,
                "candidate_vs_pytorch_best": (
                    best_us / candidate_us if best_us is not None else None
                ),
                "selected_vs_pytorch_best": (
                    best_us / selected_us if best_us is not None else None
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [float(row["attempted_speedup"]) for row in rows]
    selected = [float(row["portfolio_speedup"]) for row in rows]
    library = [
        float(row["selected_vs_pytorch_best"])
        for row in rows
        if row["selected_vs_pytorch_best"] is not None
    ]
    attempted_library = [
        float(row["candidate_vs_pytorch_best"])
        for row in rows
        if row["candidate_vs_pytorch_best"] is not None
    ]
    return {
        "denominator": len(rows),
        "material_speedup_threshold": MIN_MATERIAL_SPEEDUP,
        "minimum_confirmation_sessions": MIN_CONFIRMATION_SESSIONS,
        "verified_improved_tasks": sum(row["verified_improvement"] for row in rows),
        "no_improvement_tasks": sum(
            row["candidate_outcome"] == "no_improvement" for row in rows
        ),
        "inconclusive_tasks": sum(
            row["candidate_outcome"] == "inconclusive" for row in rows
        ),
        "correct_tasks": sum(row["correct"] for row in rows),
        "stable_tasks": sum(row["stable"] for row in rows),
        "attempted_candidate_geomean_speedup": _geomean(attempted),
        "all10_selected_portfolio_geomean_speedup": _geomean(selected),
        "pytorch_comparable_tasks": len(library),
        "selected_vs_pytorch_best_geomean": _geomean(library) if library else None,
        "attempted_candidate_vs_pytorch_best_geomean": (
            _geomean(attempted_library) if attempted_library else None
        ),
        "custom_faster_than_pytorch_best_tasks": sum(value > 1.0 for value in library),
        "pytorch_best_faster_tasks": sum(value < 1.0 for value in library),
        "attempted_candidate_faster_than_pytorch_best_tasks": sum(
            value > 1.0 for value in attempted_library
        ),
        "pytorch_best_stable_tasks": sum(row["pytorch_best_stable"] for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    serializable = [
        {
            key: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (list, dict))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(serializable[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(serializable)


def render_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width, left, plot_width = 1120, 250, 650
    top, row_height = 82, 48
    height = top + row_height * len(rows) + 70
    low, high = -2.0, 8.0

    def x_for(ratio: float) -> float:
        value = min(high, max(low, math.log2(max(ratio, 2**low))))
        return left + (value - low) / (high - low) * plot_width

    one_x = x_for(1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.label{font-size:13px}.small{font-size:11px;fill:#526079}</style>',
        '<text x="24" y="32" class="title">Core 10 — upstream, selected custom, and same-RTX-3080 PyTorch</text>',
        '<text x="24" y="55" class="small">Log2 ratio; right of 1.0 means the custom result is faster. Blue: upstream/candidate attempt. Green/orange: selected/PyTorch best.</text>',
    ]
    for tick in (-2, 0, 2, 4, 6, 8):
        x = left + (tick - low) / (high - low) * plot_width
        parts.append(f'<line x1="{x:.1f}" y1="66" x2="{x:.1f}" y2="{height - 35}" stroke="#d7dce5" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" class="small">{2**tick:g}×</text>')
    parts.append(f'<line x1="{one_x:.1f}" y1="66" x2="{one_x:.1f}" y2="{height - 35}" stroke="#172033" stroke-width="2"/>')
    for index, row in enumerate(rows):
        y = top + index * row_height
        attempted = float(row["attempted_speedup"])
        library_raw = row["selected_vs_pytorch_best"]
        library = float(library_raw) if library_raw is not None else None
        attempted_x = x_for(attempted)
        library_x = x_for(library) if library is not None else one_x
        parts.append(f'<text x="24" y="{y + 18}" class="label">{escape(row["task_id"] + " " + row["kernel"])}</text>')
        parts.append(f'<line x1="{min(one_x, attempted_x):.1f}" y1="{y + 10}" x2="{max(one_x, attempted_x):.1f}" y2="{y + 10}" stroke="#3578d4" stroke-width="8"/>')
        color = "#9ca3af" if library is None else ("#248a5b" if library >= 1.0 else "#d07428")
        parts.append(f'<line x1="{min(one_x, library_x):.1f}" y1="{y + 28}" x2="{max(one_x, library_x):.1f}" y2="{y + 28}" stroke="{color}" stroke-width="8"/>')
        parts.append(f'<text x="920" y="{y + 14}" class="small">candidate {attempted:.3f}×</text>')
        library_label = f"{library:.3f}×" if library is not None else "inconclusive"
        parts.append(f'<text x="920" y="{y + 32}" class="small">vs PyTorch {library_label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_report(payload: dict[str, Any], *, chinese: bool) -> str:
    summary = payload["summary"]
    rows = payload["results"]

    def ratio(value: float | None) -> str:
        return "—" if value is None else f"{float(value):.3f}×"

    if chinese:
        title = "# RTX 3080 Core 10 schema-v2 完整确认"
        language = "**简体中文** | [English](core10-rtx3080-confirmation.en.md)"
        intro = (
            "本报告是手工候选的完整 Core 10 确认，不是 Agent 自动搜索结果。"
            "正式提升要求正确、五个独立进程 Session、双方 spread 不超过 5%、"
            "中位加速至少 1.01×，且配对 bootstrap 95% 下界大于 1.0。"
        )
        header = (
            "| 任务 | 候选结论 | 候选加速 | Bootstrap 95% 下界 | "
            "基线/候选 spread | 稳定 PyTorch 基线 | 严格选中结果 / PyTorch |"
        )
        divider = "| --- | --- | ---: | ---: | ---: | --- | ---: |"
        labels = {
            "improved": "正式提升",
            "no_improvement": "无提升",
            "inconclusive": "无法定论",
            "failed": "失败",
        }
        summary_text = (
            f"严格结果为 {summary['verified_improved_tasks']} 项提升、"
            f"{summary['no_improvement_tasks']} 项无提升、"
            f"{summary['inconclusive_tasks']} 项无法定论。"
            f"相对上游的严格 Core 10 几何平均为 "
            f"{ratio(summary['all10_selected_portfolio_geomean_speedup'])}。"
            f"PyTorch 有 {summary['pytorch_comparable_tasks']}/10 题存在正确且稳定的方法；"
            f"仅在这些可比题上，严格结果相对最快稳定 PyTorch 的几何平均为 "
            f"{ratio(summary['selected_vs_pytorch_best_geomean'])}。"
        )
        caveat = (
            "026 没有稳定 PyTorch 方法，因此不进入 PyTorch 几何平均。"
            "019、023、026、047、095 在自动重测后仍未满足 CUDA Session 稳定性门槛；"
            "其诊断 speedup 不作为正式声明。NCU 计数器仍不可用，本地模式为 `events_only`。"
        )
    else:
        title = "# RTX 3080 Core 10 schema-v2 full confirmation"
        language = "**English** | [简体中文](core10-rtx3080-confirmation.zh-CN.md)"
        intro = (
            "This is a full Core 10 confirmation of manual candidates, not an "
            "Agent-generated search result. A formal improvement requires correctness, "
            "five independent process Sessions, at most 5% spread on both sides, at "
            "least 1.01× median speedup, and a paired-bootstrap 95% lower bound above 1.0."
        )
        header = (
            "| Task | Candidate outcome | Candidate speedup | Bootstrap 95% lower | "
            "Baseline/candidate spread | Stable PyTorch baseline | Strict selected / PyTorch |"
        )
        divider = "| --- | --- | ---: | ---: | ---: | --- | ---: |"
        labels = {
            "improved": "improved",
            "no_improvement": "no improvement",
            "inconclusive": "inconclusive",
            "failed": "failed",
        }
        summary_text = (
            f"The strict result contains {summary['verified_improved_tasks']} improvements, "
            f"{summary['no_improvement_tasks']} no-improvement result, and "
            f"{summary['inconclusive_tasks']} inconclusive results. The strict Core 10 "
            f"geometric mean versus upstream is "
            f"{ratio(summary['all10_selected_portfolio_geomean_speedup'])}. "
            f"A correct and stable PyTorch method exists for "
            f"{summary['pytorch_comparable_tasks']}/10 tasks; on only those comparable "
            f"tasks, the strict geometric mean versus the fastest stable PyTorch method "
            f"is {ratio(summary['selected_vs_pytorch_best_geomean'])}."
        )
        caveat = (
            "Task 026 has no stable PyTorch method and is excluded from the PyTorch "
            "geometric mean. Tasks 019, 023, 026, 047, and 095 still fail the CUDA "
            "Session-stability gate after automatic retesting, so their diagnostic "
            "speedups are not formal claims. NCU counters remain unavailable and the "
            "local profiling mode is `events_only`."
        )

    table_rows = []
    for row in rows:
        pytorch = row["pytorch_best_method"] or "—"
        table_rows.append(
            "| "
            + " | ".join(
                (
                    str(row["task_id"]),
                    labels[row["candidate_outcome"]],
                    ratio(row["attempted_speedup"]),
                    ratio(row["bootstrap_95_lower"]),
                    (
                        f"{float(row['baseline_session_spread_percent']):.3f}% / "
                        f"{float(row['candidate_session_spread_percent']):.3f}%"
                    ),
                    pytorch,
                    ratio(row["selected_vs_pytorch_best"]),
                )
            )
            + " |"
        )
    return "\n\n".join(
        (
            title,
            language,
            intro,
            "\n".join((header, divider, *table_rows)),
            summary_text,
            caveat,
            "Canonical machine-readable evidence: "
            "[JSON](core10_rtx3080_comparison.json).",
        )
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--pytorch-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate_payload = _load(args.candidate_summary.resolve())
    pytorch_payload = _load(args.pytorch_summary.resolve())
    augment_candidate_details(candidate_payload, args.candidate_summary.resolve().parent)
    rows = build_comparison_rows(candidate_payload, pytorch_payload)
    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite output directory: {output_dir}")
    comparison_csv = output_dir / "core10_rtx3080_comparison.csv"
    comparison_json = output_dir / "core10_rtx3080_comparison.json"
    figure = output_dir / "core10_rtx3080_comparison.svg"
    _write_csv(comparison_csv, rows)
    comparison_payload = {
        "schema_version": "2.0",
        "hardware": "NVIDIA GeForce RTX 3080 (sm_86)",
        "profiling_mode": "events_only",
        "candidate_protocol": candidate_payload.get("protocol"),
        "pytorch_protocol": pytorch_payload.get("protocol"),
        "summary": summarize(rows),
        "results": rows,
    }
    _atomic_json(comparison_json, comparison_payload)
    render_svg(rows, figure)
    report_en = output_dir / "core10-rtx3080-confirmation.en.md"
    report_zh = output_dir / "core10-rtx3080-confirmation.zh-CN.md"
    report_en.write_text(
        render_report(comparison_payload, chinese=False), encoding="utf-8"
    )
    report_zh.write_text(
        render_report(comparison_payload, chinese=True), encoding="utf-8"
    )
    files = [comparison_csv, comparison_json, figure, report_en, report_zh]
    _atomic_json(
        output_dir / "core10_rtx3080_SHA256SUMS.json",
        {path.name: _sha256(path) for path in files},
    )
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
