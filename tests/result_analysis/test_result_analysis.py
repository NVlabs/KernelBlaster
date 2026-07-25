from __future__ import annotations

import pytest

from src.kernelblaster.result_analysis import (
    build_task_rows,
    choose_baseline_benchmarks,
    choose_best_benchmarks,
    deep_case_gate,
    failure_counts,
    geometric_mean,
)


def test_geometric_mean_and_invalid_values():
    assert geometric_mean([1.0, 4.0]) == pytest.approx(2.0)
    assert geometric_mean([]) is None
    with pytest.raises(ValueError):
        geometric_mean([1.0, 0.0])


def test_best_benchmark_is_selected_per_task():
    selected = choose_best_benchmarks(
        [
            {
                "task_id": "036",
                "comparison": {
                    "speedup": 1.05,
                    "comparison_kind": "candidate",
                    "formal_valid": True,
                    "comparison_scope": "upstream_baseline",
                },
            },
            {
                "task_id": "036",
                "comparison": {
                    "speedup": 1.11,
                    "comparison_kind": "candidate",
                    "formal_valid": True,
                    "comparison_scope": "upstream_baseline",
                },
            },
            {
                "task_id": "023",
                "comparison": {
                    "speedup": 0.98,
                    "comparison_kind": "candidate",
                    "formal_valid": True,
                    "comparison_scope": "upstream_baseline",
                },
            },
            {
                "task_id": "023",
                "comparison": {
                    "speedup": 1.5,
                    "comparison_kind": "candidate",
                    "formal_valid": False,
                    "comparison_scope": "upstream_baseline",
                },
            },
            {
                "task_id": "004",
                "comparison": {
                    "speedup": 1.0,
                    "comparison_kind": "self_check",
                    "formal_valid": True,
                    "comparison_scope": "upstream_baseline",
                },
            },
            {
                "task_id": "026",
                "comparison": {
                    "speedup": 1.2,
                    "comparison_kind": "candidate",
                    "formal_valid": True,
                    "comparison_scope": "variant_head_to_head",
                },
            },
        ]
    )
    assert selected["036"]["comparison"]["speedup"] == 1.11
    assert selected["023"]["comparison"]["speedup"] == 0.98
    assert "004" not in selected
    assert "026" not in selected


def test_failure_taxonomy_and_task_rows():
    events = [
        {"event_type": "llm_request_failed", "status": "error", "task_id": "004"},
        {"event_type": "cuda_compile_failed", "status": "error", "task_id": "004"},
        {"event_type": "unexpected", "status": "error", "task_id": "007"},
    ]
    counts = failure_counts(events)
    assert counts["provider_api"] == 1
    assert counts["cuda_compile"] == 1
    assert counts["other"] == 1

    rows = build_task_rows(
        [
            {"id": "004", "name": "MatVec", "category": "memory"},
            {"id": "007", "name": "MatMul", "category": "matmul"},
            {"id": "036", "name": "RMSNorm", "category": "normalization"},
        ],
        {
            "036": {
                "comparison": {
                    "speedup": 1.1,
                    "candidate": "v2",
                    "baseline_median_us": 10,
                    "candidate_median_us": 9,
                    "all_sessions_not_slower": True,
                    "comparison_kind": "candidate",
                    "formal_valid": True,
                    "comparison_scope": "upstream_baseline",
                }
            }
        },
        events,
    )
    assert rows[0]["status"] == "failed"
    assert rows[1]["status"] == "failed"
    assert rows[2]["status"] == "verified_improved"


def test_deep_case_gate_requires_threshold_and_every_session():
    comparison = {
        "comparison_kind": "candidate",
        "formal_valid": True,
        "speedup": 1.06,
        "all_sessions_not_slower": True,
    }
    assert deep_case_gate(comparison)["passed"] is True
    assert deep_case_gate(comparison | {"all_sessions_not_slower": False})[
        "passed"
    ] is False
    assert deep_case_gate(comparison | {"speedup": 1.049})["passed"] is False


def test_baseline_coverage_prefers_formal_run_with_more_samples():
    selected = choose_baseline_benchmarks(
        [
            {
                "task_id": "004",
                "stable": True,
                "summaries": {"baseline": {"all_samples": {"count": 3}}},
            },
            {
                "task_id": "004",
                "stable": True,
                "summaries": {"baseline": {"all_samples": {"count": 300}}},
            },
            {
                "task_id": "007",
                "stable": False,
                "summaries": {"baseline": {"all_samples": {"count": 300}}},
            },
        ]
    )
    assert selected["004"]["summaries"]["baseline"]["all_samples"]["count"] == 300
    assert "007" not in selected


def test_task_rows_preserve_unstable_and_failed_baseline_states():
    rows = build_task_rows(
        [
            {"id": "019", "name": "ReLU", "category": "elementwise"},
            {"id": "026", "name": "GELU", "category": "elementwise"},
        ],
        {},
        [],
        {},
        {
            "019": {
                "status": "failed",
                "stable": False,
                "baseline_median_us": 9.5,
            },
            "026": {"status": "failed", "stable": None},
        },
    )
    assert rows[0]["status"] == "baseline_unstable"
    assert rows[0]["baseline_median_us"] == 9.5
    assert rows[1]["status"] == "baseline_failed"

    recovered = build_task_rows(
        [{"id": "026", "name": "GELU", "category": "elementwise"}],
        {},
        [],
        {
            "026": {
                "summaries": {
                    "baseline": {
                        "session_medians_summary": {"median_us": 10.0}
                    }
                }
            }
        },
        {"026": {"status": "failed", "stable": None}},
    )
    assert recovered[0]["status"] == "baseline_only"
    assert recovered[0]["baseline_median_us"] == 10.0
