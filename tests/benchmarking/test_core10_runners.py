from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PYTORCH = _load("benchmark_pytorch", "benchmark_pytorch.py")
CANDIDATES = _load("benchmark_candidates", "benchmark_candidates.py")
ANALYZE = _load("analyze_core10_comparison", "analyze_core10_comparison.py")


def _manifest(tasks):
    return {
        "schema_version": "2.0",
        "runtime_contract": {
            "device": "cuda",
            "gpu_architectures": ["sm_86"],
            "input_dtype": "fp16",
            "accumulation_dtype": "fp32",
            "layout": "contiguous_row_major",
            "stream_mode": "legacy_default",
            "max_streams": 1,
            "graph_capture": False,
            "directions": ["inference_forward"],
            "backward": False,
            "fallback": "none",
            "production_ready": False,
        },
        "tasks": tasks,
    }


def _candidate(task_id, source, **extra):
    candidate = {
        "id": task_id,
        "name": "candidate",
        "source": source,
        "supported_cases": [
            {
                "case_id": "canonical",
                "class": "canonical",
                "shape": {"M": 1, "K": 1},
                "correctness": True,
                "performance": True,
            }
        ],
    }
    if task_id in {"004", "007", "036", "040", "095"}:
        candidate.update(
            {
                "numerics_profile": "test",
                "resource_policy": {
                    "kind": "none",
                    "ownership": "none",
                    "workspace_bytes": 0,
                },
                "reentrant_under_contract": True,
                "requires_prewarm": False,
            }
        )
    candidate.update(extra)
    return candidate


def test_pytorch_matrix_covers_every_core10_task():
    expected = {"004", "007", "019", "023", "026", "036", "040", "047", "088", "095"}
    assert set(PYTORCH.CORE10_TASKS) == expected
    assert set(PYTORCH.PYTORCH_METHODS) == expected
    assert all(PYTORCH.PYTORCH_METHODS[task_id] for task_id in expected)
    assert "pytorch_fused_gelu_tanh" in PYTORCH.PYTORCH_METHODS["088"]
    assert "pytorch_preallocated_out" in PYTORCH.PYTORCH_METHODS["019"]


def test_candidate_manifest_resolves_relative_sources(tmp_path):
    source = tmp_path / "candidate.cu"
    driver = tmp_path / "edge.cpp"
    source.write_text("// candidate\n", encoding="utf-8")
    driver.write_text("// driver\n", encoding="utf-8")
    manifest = tmp_path / "candidates.json"
    manifest.write_text(
        json.dumps(
            _manifest(
                [
                    _candidate(
                        "004",
                        source.name,
                        extra_correctness_drivers=[driver.name],
                    )
                ]
            )
        ),
        encoding="utf-8",
    )
    loaded = CANDIDATES.load_candidates(manifest)
    assert loaded["004"]["source_path"] == source.resolve()
    assert loaded["004"]["extra_driver_paths"] == [driver.resolve()]


def test_candidate_manifest_rejects_duplicate_ids(tmp_path):
    source = tmp_path / "candidate.cu"
    source.write_text("// candidate\n", encoding="utf-8")
    manifest = tmp_path / "candidates.json"
    manifest.write_text(
        json.dumps(
            _manifest(
                [
                    _candidate("004", source.name, name="a"),
                    _candidate("004", source.name, name="b"),
                ]
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        CANDIDATES.load_candidates(manifest)


def test_candidate_runner_forwards_explicit_gpu_and_sm():
    assert CANDIDATES._benchmark_target_arguments(
        "NVIDIA A100-SXM4-80GB",
        "sm_80",
    ) == [
        "--gpu",
        "NVIDIA A100-SXM4-80GB",
        "--sm",
        "sm_80",
    ]


def test_comparison_selects_only_verified_improvements(tmp_path):
    candidate_results = []
    pytorch_results = []
    for task_id in ANALYZE.TASK_IDS:
        improved = task_id != "088"
        candidate_results.append(
            {
                "task_id": task_id,
                "kernel": f"kernel-{task_id}",
                "candidate": f"candidate-{task_id}",
                "source": f"{task_id}.cu",
                "correct": True,
                "stable": True,
                "performance_claim_allowed": True,
                "all_sessions_not_slower": improved,
                "baseline_median_us": 20.0,
                "candidate_median_us": 10.0 if improved else 21.0,
                "speedup": 2.0 if improved else 20.0 / 21.0,
                "session_speedups": [2.0] * 5 if improved else [0.95] * 5,
            }
        )
        eager_method = (
            "pytorch_driver_formula" if task_id == "088" else "pytorch_eager"
        )
        pytorch_results.append(
            {
                "task_id": task_id,
                "method": eager_method,
                "allocation_mode": "framework_allocated",
                "median_us": 15.0,
                "correct": True,
                "stable": True,
            }
        )
        if task_id == "088":
            pytorch_results.append(
                {
                    "task_id": task_id,
                    "method": "pytorch_fused_gelu_tanh",
                    "allocation_mode": "framework_allocated_equivalent",
                    "median_us": 8.0,
                    "correct": True,
                    "stable": True,
                }
            )
    rows = ANALYZE.build_comparison_rows(
        {"results": candidate_results}, {"results": pytorch_results}
    )
    assert len(rows) == 10
    failed = next(row for row in rows if row["task_id"] == "088")
    assert failed["selected_variant"] == "upstream_baseline"
    assert failed["portfolio_speedup"] == 1.0
    assert failed["candidate_outcome"] == "no_improvement"
    assert failed["candidate_exclusion_reason"] == "paired_session_slower"
    assert failed["pytorch_best_method"] == "pytorch_fused_gelu_tanh"
    assert ANALYZE.summarize(rows)["verified_improved_tasks"] == 9

    candidate_results[0]["session_speedups"] = [2.0] * 3
    incomplete_rows = ANALYZE.build_comparison_rows(
        {"results": candidate_results}, {"results": pytorch_results}
    )
    incomplete = next(row for row in incomplete_rows if row["task_id"] == "004")
    assert incomplete["candidate_outcome"] == "inconclusive"
    assert incomplete["candidate_exclusion_reason"] == "insufficient_confirmation"

    csv_path = tmp_path / "comparison.csv"
    ANALYZE._write_csv(csv_path, rows)
    assert b"\r\n" not in csv_path.read_bytes()


def test_comparison_excludes_unstable_pytorch_method():
    candidate_results = []
    pytorch_results = []
    for task_id in ANALYZE.TASK_IDS:
        candidate_results.append(
            {
                "task_id": task_id,
                "kernel": f"kernel-{task_id}",
                "candidate": f"candidate-{task_id}",
                "source": f"{task_id}.cu",
                "correct": True,
                "stable": True,
                "performance_claim_allowed": True,
                "all_sessions_not_slower": True,
                "baseline_median_us": 20.0,
                "candidate_median_us": 10.0,
                "speedup": 2.0,
                "session_speedups": [2.0] * 5,
            }
        )
        eager_name = (
            "pytorch_driver_formula" if task_id == "088" else "pytorch_eager"
        )
        pytorch_results.extend(
            [
                {
                    "task_id": task_id,
                    "method": eager_name,
                    "allocation_mode": "framework_allocated",
                    "median_us": 15.0,
                    "correct": True,
                    "stable": True,
                },
                {
                    "task_id": task_id,
                    "method": "unstable_but_fast",
                    "allocation_mode": "framework_allocated",
                    "median_us": 1.0,
                    "correct": True,
                    "stable": False,
                },
            ]
        )
    rows = ANALYZE.build_comparison_rows(
        {"results": candidate_results}, {"results": pytorch_results}
    )
    assert all(row["pytorch_best_method"] != "unstable_but_fast" for row in rows)


def test_core10_manifest_declares_hardened_edge_contracts():
    payload = json.loads(
        (ROOT / "portfolio" / "case_studies" / "core10" / "candidates.json").read_text(
            encoding="utf-8"
        )
    )
    tasks = {task["id"]: task for task in payload["tasks"]}
    for task_id in ("004", "007", "036", "040", "095"):
        assert tasks[task_id]["extra_correctness_drivers"]
        assert tasks[task_id]["reentrant_under_contract"] is True
        assert tasks[task_id]["supported_cases"][0]["case_id"] == "canonical"
        assert tasks[task_id]["supported_cases"][0]["performance"] is True
        assert all(
            not case["performance"]
            for case in tasks[task_id]["supported_cases"][1:]
        )
        for relative in tasks[task_id]["extra_correctness_drivers"]:
            assert (ROOT / "portfolio" / "case_studies" / "core10" / relative).resolve().is_file()
    assert tasks["040"]["resource_policy"]["workspace_bytes"] == 32896
    assert tasks["095"]["resource_policy"]["workspace_bytes"] == 2048
    assert tasks["095"]["target_dtype"] == "int64"


def test_bilingual_confirmation_report_labels_outcomes_and_links_json():
    payload = {
        "summary": {
            "verified_improved_tasks": 1,
            "no_improvement_tasks": 0,
            "inconclusive_tasks": 0,
            "all10_selected_portfolio_geomean_speedup": 2.0,
            "pytorch_comparable_tasks": 1,
            "selected_vs_pytorch_best_geomean": 1.1,
        },
        "results": [
            {
                "task_id": "004",
                "candidate_outcome": "improved",
                "attempted_speedup": 2.0,
                "bootstrap_95_lower": 1.8,
                "baseline_session_spread_percent": 1.0,
                "candidate_session_spread_percent": 0.5,
                "pytorch_best_method": "pytorch_eager",
                "selected_vs_pytorch_best": 1.1,
            }
        ],
    }
    english = ANALYZE.render_report(payload, chinese=False)
    chinese = ANALYZE.render_report(payload, chinese=True)
    assert "Candidate outcome" in english
    assert "正式提升" in chinese
    assert "core10_rtx3080_comparison.json" in english
    assert "core10-rtx3080-confirmation.zh-CN.md" in english


def test_gpu_ci_keeps_correctness_and_ncu_probes_out_of_formal_timing():
    workflow = (ROOT / ".github" / "workflows" / "gpu-validation.yml").read_text(
        encoding="utf-8"
    )
    assert workflow.count("--correctness-only") == 2
