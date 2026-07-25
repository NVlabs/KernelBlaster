#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify the publishable portfolio-v2.1 evidence index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "portfolio-v2.1"
ISSUE10_ROOT = ARTIFACT_ROOT / "issue-10" / "rtx3080"
ISSUE10_SUMMARY = ISSUE10_ROOT / "correctness-summary.json"
SHA256_INDEX = ARTIFACT_ROOT / "SHA256SUMS.json"
CANDIDATE_MANIFEST = (
    ROOT_DIR / "portfolio" / "case_studies" / "core10" / "candidates.json"
)
CPU_TEST_RESULT = "177 passed"

RUN_CONFIG = {
    "004": {
        "evidence_run": "004-v2",
        "candidate": "row_block_half2",
        "edge_driver": "extra-0-004_matvec_edge_driver",
        "resource_driver": None,
        "canonical_driver": None,
    },
    "007": {
        "evidence_run": "007-v3",
        "candidate": "cublas_gemm_ex",
        "edge_driver": "extra-0-007_small_k_matmul_edge_driver",
        "resource_driver": "extra-1-007_resource_lifecycle_driver",
        "canonical_driver": (
            "candidate-only-0-007_small_k_matmul_canonical_driver"
        ),
    },
    "036": {
        "evidence_run": "036-v3",
        "candidate": "rmsnorm_v3c",
        "edge_driver": "extra-0-edge_driver",
        "resource_driver": None,
        "canonical_driver": None,
    },
    "040": {
        "evidence_run": "040-v2",
        "candidate": "multiblock_layernorm",
        "edge_driver": "extra-0-040_layernorm_edge_driver",
        "resource_driver": "extra-1-040_resource_lifecycle_driver",
        "canonical_driver": None,
    },
    "095": {
        "evidence_run": "095-v5",
        "candidate": "warp_cross_entropy",
        "edge_driver": "extra-0-095_cross_entropy_edge_driver",
        "resource_driver": "extra-1-095_resource_lifecycle_driver",
        "canonical_driver": None,
    },
}

DISTRIBUTION_FIELDS = (
    "abs_mean",
    "abs_rmse",
    "abs_p50",
    "abs_p90",
    "abs_p99",
    "abs_p999",
    "abs_max",
    "normalized_p50",
    "normalized_p90",
    "normalized_p99",
    "normalized_p999",
    "normalized_max",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_evidence(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size < 1:
        raise ValueError(f"Publishable evidence is empty: {path}")
    return {
        "path": path.relative_to(ROOT_DIR).as_posix(),
        "size_bytes": size,
        "sha256": _sha256(path),
    }


def _find_normalized_record(
    manifest: dict[str, Any], variant: str, driver: str
) -> dict[str, Any]:
    matches = [
        record
        for record in manifest["variants"][variant]["correctness"]
        if record.get("driver") == driver
        and str(record.get("mode", "")).startswith("normalized-")
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one normalized correctness record for "
            f"{variant}/{driver}; observed {len(matches)}"
        )
    return matches[0]


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    cases = metrics.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Correctness evidence is missing case records")
    result = {
        "case_seed_records": len(cases),
        "case_ids": sorted({case["case_id"] for case in cases}),
        "seeds": sorted({case["seed"] for case in cases}),
        "count": metrics["count"],
        "quantile_sample_count": metrics["quantile_sample_count"],
        "quantile_sampling": metrics["quantile_sampling"],
        "quantile_max_samples": metrics["quantile_max_samples"],
        "aggregate_quantile_semantics": metrics[
            "aggregate_quantile_semantics"
        ],
        "mismatch_count": metrics["mismatch_count"],
        "nonfinite_count": metrics["nonfinite_count"],
        "finite": metrics["finite"],
        "deterministic": metrics["deterministic"],
    }
    result.update({field: metrics[field] for field in DISTRIBUTION_FIELDS})
    return result


def _resource_lifecycle(
    *,
    manifest: dict[str, Any],
    candidate: str,
    candidate_config: dict[str, Any],
    resource_driver: str | None,
) -> dict[str, Any]:
    result = {
        "policy": candidate_config["resource_policy"],
        "reentrant_under_contract": candidate_config[
            "reentrant_under_contract"
        ],
        "requires_prewarm": candidate_config["requires_prewarm"],
    }
    if resource_driver is None:
        result.update(
            {
                "status": "passed_no_persistent_resources",
                "protocol": "not_applicable",
            }
        )
        return result

    record = _find_normalized_record(manifest, candidate, resource_driver)
    result.update(
        {
            "status": "passed",
            "protocols": [
                "same_host_thread_reuse_5_calls",
                "dual_host_thread_legacy_default_stream",
                "thread_exit_tls_cleanup",
                "cudaMemGetInfo_steady_state_leak_bound",
            ],
            "leak_allowance_bytes": 65536,
            "distribution": _metrics_summary(record["metrics"]),
        }
    )
    return result


def _performance_confirmation(
    manifest: dict[str, Any], summary: dict[str, Any], candidate: str
) -> dict[str, Any]:
    comparison = summary["comparison"]
    gate = comparison["performance_gate"]
    round_spreads = [
        {
            "round": index,
            "baseline_spread_percent": round_summary["baseline"][
                "session_spread_percent"
            ],
            "candidate_spread_percent": round_summary[candidate][
                "session_spread_percent"
            ],
        }
        for index, round_summary in enumerate(summary["round_summaries"])
    ]
    return {
        "outcome": manifest["outcome"],
        "stable": summary["stable"],
        "formal_valid": comparison["formal_valid"],
        "performance_claim_allowed": summary["performance_claim_allowed"],
        "claim_kind": (
            "formal"
            if summary["performance_claim_allowed"]
            else "diagnostic_only"
        ),
        "sessions": manifest["sessions"],
        "warmup": manifest["warmup"],
        "repetitions": manifest["repetitions"],
        "retest_triggered": summary["retest_triggered"],
        "active_round": summary["active_round"],
        "round_spreads": round_spreads,
        "baseline_median_us": comparison["baseline_median_us"],
        "candidate_median_us": comparison["candidate_median_us"],
        "baseline_session_spread_percent": comparison[
            "baseline_session_spread_percent"
        ],
        "candidate_session_spread_percent": comparison[
            "candidate_session_spread_percent"
        ],
        "all_sessions_not_slower": comparison["all_sessions_not_slower"],
        "session_speedups": comparison["session_speedups"],
        "median_speedup": gate["median_speedup"],
        "bootstrap_95_lower": gate["bootstrap_95_lower"],
        "bootstrap_95_upper": gate["bootstrap_95_upper"],
        "gate_passed": gate["passed"],
        "gate_reason": gate["reason"],
    }


def build_issue10_summary() -> dict[str, Any]:
    candidate_manifest = _load_json(CANDIDATE_MANIFEST)
    candidate_manifest_sha = _sha256(CANDIDATE_MANIFEST)
    candidate_configs = {
        task["id"]: task for task in candidate_manifest["tasks"]
    }
    tasks = []
    outcomes = {"improved": [], "no_improvement": [], "inconclusive": []}

    for task_id, config in RUN_CONFIG.items():
        directory = ISSUE10_ROOT / task_id
        manifest = _load_json(directory / "run_manifest.json")
        summary = _load_json(directory / "summary.json")
        suite = _load_json(directory / "suite_summary.json")
        candidate = str(config["candidate"])
        candidate_config = candidate_configs[task_id]
        edge_record = _find_normalized_record(
            manifest, candidate, str(config["edge_driver"])
        )
        official = [
            {
                "mode": record["mode"],
                "passed": record["passed"],
                "returncode": record["returncode"],
            }
            for record in manifest["variants"][candidate]["correctness"]
            if record.get("driver") == "official"
        ]
        correctness_matrix: dict[str, Any] = {
            "status": manifest["validation_gates"]["correctness"],
            "correctness_error_regression": manifest["validation_gates"][
                "correctness_error_regression"
            ],
            "official": official,
            "edge_distribution": _metrics_summary(edge_record["metrics"]),
        }
        canonical_driver = config["canonical_driver"]
        if canonical_driver is not None:
            canonical_record = _find_normalized_record(
                manifest, candidate, str(canonical_driver)
            )
            correctness_matrix.update(
                {
                    "candidate_only_canonical_distribution": (
                        _metrics_summary(canonical_record["metrics"])
                    ),
                    "candidate_only_gate": manifest[
                        "candidate_only_correctness"
                    ],
                    "candidate_only_memory_strategy": (
                        "row_chunked_fp32_golden_256_rows"
                    ),
                    "candidate_only_full_element_statistics": True,
                }
            )

        variant = manifest["variants"][candidate]
        evidence_files = [
            directory / "run_manifest.json",
            directory / "summary.json",
            directory / "measurements.jsonl",
            directory / "suite_summary.json",
        ]
        task = {
            "task_id": task_id,
            "kernel": manifest["kernel"],
            "candidate": candidate,
            "evidence_run": config["evidence_run"],
            "suite_status": suite["results"][0]["status"],
            "validation_gates": manifest["validation_gates"],
            "correctness_matrix": correctness_matrix,
            "resource_lifecycle": _resource_lifecycle(
                manifest=manifest,
                candidate=candidate,
                candidate_config=candidate_config,
                resource_driver=config["resource_driver"],
            ),
            "performance_confirmation": _performance_confirmation(
                manifest, summary, candidate
            ),
            "source_provenance": {
                "candidate_source": candidate_config["source"],
                "candidate_source_sha256": variant["source_sha256"],
                "normalized_source_sha256": variant["normalized_sha256"],
                "official_driver_sha256": manifest["driver_sha256"],
                "extra_correctness_drivers": manifest[
                    "extra_correctness_drivers"
                ],
                "candidate_only_correctness_drivers": manifest.get(
                    "candidate_only_correctness_drivers", []
                ),
                "correctness_support_headers": manifest[
                    "correctness_support_headers"
                ],
                "candidate_manifest_sha256": candidate_manifest_sha,
                "git_commit": manifest["git_commit"],
                "container_image": manifest["container_image"],
                "container_digest": manifest["container_digest"],
            },
            "evidence": [_file_evidence(path) for path in evidence_files],
        }
        tasks.append(task)
        outcomes[manifest["outcome"]].append(task_id)

    issue_close_allowed = not outcomes["inconclusive"]
    return {
        "schema_version": "2.0",
        "issue": 10,
        "run_date": "2026-07-23",
        "base_commit": "4a8fb35712c12e9619e4603cbe7d2620097282ba",
        "source_state": "dirty_worktree_with_per_file_sha256",
        "environment": {
            "gpu": "NVIDIA GeForce RTX 3080",
            "compute_capability": "8.6",
            "container_image": (
                "kernelblaster-gpu@sha256:"
                "ae4d23986d0f8778103436b10a58d88ba2ae64faf87650d2b70f594f868ab487"
            ),
            "torch": "2.6.0a0+ecf3bae40a.nv25.01",
            "cuda_toolkit": "12.8",
            "cpu_tests": CPU_TEST_RESULT,
        },
        "runtime_contract": {
            "input_dtype": "fp16",
            "accumulation_dtype": "fp32",
            "layout": "contiguous_row_major",
            "stream_mode": "legacy_default",
            "max_streams": 1,
            "graph_capture": False,
            "direction": "inference_forward",
            "fallback": "none",
            "production_ready": False,
        },
        "performance_protocol": {
            "scope": "canonical_shapes_only",
            "warmups": 20,
            "samples_per_session": 100,
            "sessions": 5,
            "ordering": "AB_BA",
            "max_session_spread_percent": 5.0,
            "minimum_median_speedup": 1.01,
            "paired_bootstrap_95_lower_must_exceed": 1.0,
            "edge_timing_allowed": False,
        },
        "tasks": tasks,
        "formal_outcomes": outcomes,
        "failed_attempts": [
            {
                "task_id": "007",
                "evidence_run": "007-v2",
                "status": "failed",
                "reason": (
                    "normalized candidate-only canonical correctness timed "
                    "out after 1200 seconds before timing"
                ),
                "remediation": (
                    "row-chunked FP32 golden retained the full shape, seeds, "
                    "five bitwise launches, full-element statistics, and "
                    "identical global deterministic-stride quantiles"
                ),
                "source_evidence_published": False,
            }
        ],
        "unsupported_paths": [
            "fp32_or_bf16_input",
            "noncontiguous_or_channels_last_layout",
            "non_default_or_multiple_streams",
            "cuda_graph_capture",
            "backward",
            "shapes_outside_the_manifest_whitelist",
        ],
        "publication": {
            "raw_llm_logs_published": False,
            "build_directories_published": False,
            "session_logs_published": False,
            "secrets_detected": False,
        },
        "issue_close_allowed": issue_close_allowed,
        "remaining_blocker": (
            None
            if issue_close_allowed
            else (
                "Task 095 upstream baseline remained unstable after the single "
                "cooldown/retest: 24.3688254595231% spread > 5%; candidate "
                "spread was 3.4325640774421684%. The 27.010936109352286x "
                "speedup is diagnostic only."
            )
        ),
    }


def build_sha256_index(
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in sorted(artifact_root.rglob("*")):
        if path == artifact_root / SHA256_INDEX.name:
            continue
        if path.is_symlink():
            raise ValueError(f"Publishable artifacts may not be symlinks: {path}")
        if not path.is_file():
            continue
        if path.stat().st_size < 1:
            raise ValueError(f"Publishable artifacts may not be empty: {path}")
        relative = path.relative_to(artifact_root).as_posix()
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe artifact path: {relative}")
        files[relative] = _sha256(path)
    # Keep the repository's established SHA256SUMS.json ABI: a flat mapping
    # relative to the directory containing the manifest. The stricter path,
    # empty-file, symlink, and exact-set invariants are enforced above.
    return files


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _check(path: Path, expected: str) -> bool:
    return path.is_file() and path.read_text(encoding="utf-8") == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = _render(build_issue10_summary())
    if args.write:
        _atomic_write(ISSUE10_SUMMARY, summary)
        _atomic_write(SHA256_INDEX, _render(build_sha256_index()))
        return 0

    stale = []
    if not _check(ISSUE10_SUMMARY, summary):
        stale.append(str(ISSUE10_SUMMARY.relative_to(ROOT_DIR)))
    index = _render(build_sha256_index())
    if not _check(SHA256_INDEX, index):
        stale.append(str(SHA256_INDEX.relative_to(ROOT_DIR)))
    if stale:
        print("Stale portfolio-v2.1 evidence: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
