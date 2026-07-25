#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.kernelblaster.benchmarking import (  # noqa: E402
    BENCHMARK_MARKER,
    BENCHMARK_SCHEMA_VERSION,
    comparison_validity,
    instrument_profiler_driver,
    instrument_driver,
    latency_summary,
    normalize_cuda_source,
    session_spread_percent,
    sha256_text,
    write_compilation_units,
)
from src.kernelblaster.profiling import evaluate_performance_gate  # noqa: E402


NCU_SECTIONS = (
    "SpeedOfLight",
    "LaunchStats",
    "Occupancy",
)
NCU_SECTION_NAMES = {
    "SpeedOfLight": "GPU Speed Of Light Throughput",
    "LaunchStats": "Launch Statistics",
    "Occupancy": "Occupancy",
}
NCU_REQUIRED_METRIC_GROUPS = {
    "GPU Speed Of Light Throughput": (
        ("Compute (SM) Throughput",),
        ("Memory Throughput", "DRAM Throughput"),
    ),
    "Launch Statistics": (
        ("Block Size",),
        ("Grid Size",),
        ("Registers Per Thread",),
    ),
    "Occupancy": (
        ("Theoretical Occupancy",),
        ("Achieved Occupancy",),
    ),
}
CORRECTNESS_MARKER = "KERNELBLASTER_CORRECTNESS_JSON "
CORRECTNESS_ERROR_METRICS = ("p99_abs_error", "max_abs_error")
CORRECTNESS_ERROR_RELATIVE_ALLOWANCE = 1.10
CORRECTNESS_ERROR_ABSOLUTE_ALLOWANCE = 1e-4
CORRECTNESS_DISTRIBUTION_METRICS = (
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
CORRECTNESS_SEEDS = {0, 42, 20260721}
AGGREGATE_QUANTILE_SEMANTICS = "max_per_case_quantile_envelope"
RESOURCE_BLOCKED_MARKER = "KERNELBLASTER_RESOURCE_"


class BlockedResourceError(RuntimeError):
    """A CUDA/cuBLAS context or owned resource could not be used safely."""


def _raise_if_resource_blocked(
    completed: subprocess.CompletedProcess[str],
    *,
    log_path: Path,
) -> None:
    combined = completed.stdout + completed.stderr
    if RESOURCE_BLOCKED_MARKER in combined:
        marker_line = next(
            (
                line
                for line in combined.splitlines()
                if RESOURCE_BLOCKED_MARKER in line
            ),
            RESOURCE_BLOCKED_MARKER,
        )
        raise BlockedResourceError(
            f"CUDA resource lifecycle blocked: {marker_line}; see {log_path.name}."
        )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_output_dir() -> Path:
    return ROOT_DIR / "out" / "portfolio" / "benchmark" / _timestamp()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_correctness_distribution(metrics: dict[str, Any]) -> None:
    cases = metrics.get("cases")
    if cases is None:
        return
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("Correctness distribution must contain case records.")
    aggregate_count = metrics.get("count")
    if not isinstance(aggregate_count, int) or aggregate_count < 1:
        raise RuntimeError("Correctness distribution has an invalid count.")
    aggregate_sample_count = metrics.get("quantile_sample_count")
    if (
        not isinstance(aggregate_sample_count, int)
        or aggregate_sample_count < 1
        or aggregate_sample_count > aggregate_count
        or metrics.get("quantile_sampling") != "deterministic_stride"
        or metrics.get("quantile_max_samples") != 1048576
        or metrics.get("aggregate_quantile_semantics")
        != AGGREGATE_QUANTILE_SEMANTICS
    ):
        raise RuntimeError("Correctness distribution has invalid quantile sampling.")
    for name in CORRECTNESS_DISTRIBUTION_METRICS:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not 0 <= float(value) < float(
            "inf"
        ):
            raise RuntimeError(
                f"Correctness distribution has invalid {name}: {value!r}."
            )
    for count_name in ("mismatch_count", "nonfinite_count"):
        value = metrics.get(count_name)
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"Correctness distribution has invalid {count_name}: {value!r}."
            )
    if metrics["mismatch_count"] != 0 or metrics["nonfinite_count"] != 0:
        raise RuntimeError("Correctness distribution contains mismatches or NaN/Inf.")
    if float(metrics["normalized_max"]) > 1.0:
        raise RuntimeError("Correctness normalized error exceeded the declared tolerance.")

    observed_seeds: set[int] = set()
    case_ids: set[str] = set()
    observed_case_seed_pairs: set[tuple[str, int]] = set()
    count_sum = 0
    sample_count_sum = 0
    for case in cases:
        if not isinstance(case, dict):
            raise RuntimeError("Correctness case record must be an object.")
        case_id = case.get("case_id")
        seed = case.get("seed")
        shape = case.get("shape")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError("Correctness case record is missing case_id.")
        if not isinstance(seed, int) or seed not in CORRECTNESS_SEEDS:
            raise RuntimeError(f"Correctness case has an unexpected seed: {seed!r}.")
        if not isinstance(shape, dict) or not shape:
            raise RuntimeError("Correctness case record has an invalid shape.")
        if case.get("deterministic") is not True:
            raise RuntimeError("Correctness case record is not deterministic.")
        if case.get("mismatch_count") != 0 or case.get("nonfinite_count") != 0:
            raise RuntimeError("Correctness case contains mismatches or NaN/Inf.")
        if float(case.get("normalized_max", float("inf"))) > 1.0:
            raise RuntimeError("Correctness case exceeded the declared tolerance.")
        case_count = case.get("count")
        if not isinstance(case_count, int) or case_count < 1:
            raise RuntimeError("Correctness case has an invalid count.")
        count_sum += case_count
        case_sample_count = case.get("quantile_sample_count")
        if (
            not isinstance(case_sample_count, int)
            or case_sample_count < 1
            or case_sample_count > case_count
            or case.get("quantile_sampling") != "deterministic_stride"
            or case.get("quantile_max_samples") != 1048576
        ):
            raise RuntimeError("Correctness case has invalid quantile sampling.")
        sample_count_sum += case_sample_count
        case_seed_pair = (case_id, seed)
        if case_seed_pair in observed_case_seed_pairs:
            raise RuntimeError(
                "Correctness cases contain a duplicate case_id/seed pair."
            )
        observed_case_seed_pairs.add(case_seed_pair)
        observed_seeds.add(seed)
        case_ids.add(case_id)
    if observed_seeds != CORRECTNESS_SEEDS:
        raise RuntimeError("Correctness cases do not cover the required seed set.")
    expected_case_seed_pairs = {
        (case_id, seed) for case_id in case_ids for seed in CORRECTNESS_SEEDS
    }
    if observed_case_seed_pairs != expected_case_seed_pairs:
        raise RuntimeError("Correctness cases do not cover every case/seed pair exactly.")
    if count_sum != aggregate_count:
        raise RuntimeError("Correctness aggregate count does not match its case records.")
    if sample_count_sum != aggregate_sample_count:
        raise RuntimeError(
            "Correctness aggregate sample count does not match its case records."
        )


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    log_path: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log_path.write_text(
        "COMMAND\n"
        + json.dumps(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path.name}."
        )
    return completed


def _compile(
    source_dir: Path,
    *,
    sm: str,
    timeout: float,
) -> Path:
    template = ROOT_DIR / "src" / "kernelblaster" / "servers" / "cuda_env" / "CMakeLists.txt"
    shutil.copy2(template, source_dir / "CMakeLists.txt")
    build_dir = source_dir / "build"
    try:
        import torch
        from torch.utils import cmake_prefix_path
    except ImportError as error:
        raise RuntimeError(
            "benchmark_cuda.py must run in the KernelBlaster container with PyTorch."
        ) from error

    _run_command(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DGPU_ARCH_VERSION={sm.removeprefix('sm_')}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        cwd=source_dir,
        timeout=timeout,
        log_path=source_dir / "cmake-configure.log",
    )
    _run_command(
        ["cmake", "--build", str(build_dir), "--parallel"],
        cwd=source_dir,
        timeout=timeout,
        log_path=source_dir / "cmake-build.log",
    )
    executable = build_dir / "main"
    if not executable.is_file():
        raise RuntimeError("CMake completed without producing build/main.")
    return executable


def _compile_program(
    root: Path,
    *,
    label: str,
    mode: str,
    driver: str,
    source: str,
    sm: str,
    timeout: float,
) -> Path:
    directory = root / "build" / label / mode
    write_compilation_units(directory, driver, source)
    return _compile(directory, sm=sm, timeout=timeout)


def _run_correctness(
    executable: Path,
    *,
    output_dir: Path,
    label: str,
    mode: str,
    timeout: float,
    require_distribution: bool = False,
) -> dict[str, Any]:
    log_path = output_dir / f"correctness-{label}-{mode}.log"
    completed = _run_command(
        [str(executable)],
        cwd=executable.parent,
        timeout=timeout,
        log_path=log_path,
        check=False,
    )
    _raise_if_resource_blocked(completed, log_path=log_path)
    marker_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(CORRECTNESS_MARKER)
    ]
    if len(marker_lines) > 1:
        raise RuntimeError(
            "Correctness output contained more than one result marker."
        )
    if require_distribution and not marker_lines:
        raise RuntimeError(
            "Candidate-only correctness Driver did not emit a distribution marker."
        )
    metrics: dict[str, Any] | None = None
    if marker_lines:
        try:
            metrics = json.loads(marker_lines[0][len(CORRECTNESS_MARKER) :])
        except (json.JSONDecodeError, TypeError) as error:
            raise RuntimeError("Correctness result marker is not valid JSON.") from error
        for key in CORRECTNESS_ERROR_METRICS:
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not 0 <= float(value) < float(
                "inf"
            ):
                raise RuntimeError(
                    f"Correctness result marker has invalid {key}: {value!r}."
                )
        if metrics.get("finite") is not True or metrics.get("deterministic") is not True:
            raise RuntimeError(
                "Correctness metrics reported NaN/Inf or non-deterministic output."
            )
        if require_distribution and metrics.get("cases") is None:
            raise RuntimeError(
                "Candidate-only correctness Driver did not emit case distributions."
            )
        _validate_correctness_distribution(metrics)

    passed = completed.returncode == 0 and "passed" in completed.stdout.lower()
    result = {
        "mode": mode,
        "returncode": completed.returncode,
        "passed": passed,
    }
    if metrics is not None:
        result["metrics"] = metrics
    return result


def _validate_correctness_error_regression(
    variants: dict[str, Any],
    *,
    candidate_label: str | None,
    required_drivers: set[str] | None = None,
) -> dict[str, Any]:
    """Compare normalized extra-driver errors against the upstream baseline."""
    if candidate_label is None:
        return {"status": "not_applicable", "comparisons": []}

    def indexed_metrics(label: str) -> dict[str, dict[str, float]]:
        indexed: dict[str, dict[str, float]] = {}
        for result in variants[label]["correctness"]:
            driver = str(result["driver"])
            if (
                driver == "official"
                or not str(result["mode"]).startswith("normalized-")
                or result.get("driver_scope") == "candidate_only"
            ):
                continue
            metrics = result.get("metrics")
            if metrics is None:
                continue
            indexed[driver] = {
                key: float(metrics[key]) for key in CORRECTNESS_ERROR_METRICS
            }
        return indexed

    baseline = indexed_metrics("baseline")
    candidate = indexed_metrics(candidate_label)
    if required_drivers is not None and set(baseline) != required_drivers:
        missing = sorted(required_drivers - set(baseline))
        raise RuntimeError(
            "Extra correctness drivers did not emit required metrics: "
            + ", ".join(missing)
        )
    if set(baseline) != set(candidate):
        raise RuntimeError(
            "Candidate and baseline did not emit matching extra-driver correctness metrics."
        )

    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    for driver in sorted(baseline):
        metric_results: dict[str, Any] = {}
        for metric in CORRECTNESS_ERROR_METRICS:
            baseline_value = baseline[driver][metric]
            candidate_value = candidate[driver][metric]
            allowed = (
                baseline_value * CORRECTNESS_ERROR_RELATIVE_ALLOWANCE
                + CORRECTNESS_ERROR_ABSOLUTE_ALLOWANCE
            )
            passed = candidate_value <= allowed
            metric_results[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "allowed": allowed,
                "passed": passed,
            }
            if not passed:
                failures.append(
                    f"{driver}:{metric} candidate={candidate_value:.8g} "
                    f"allowed={allowed:.8g}"
                )
        comparisons.append(
            {"driver": driver, "metrics": metric_results, "passed": not any(
                not item["passed"] for item in metric_results.values()
            )}
        )

    status = "passed" if not failures else "failed"
    gate = {
        "status": status,
        "relative_allowance": CORRECTNESS_ERROR_RELATIVE_ALLOWANCE,
        "absolute_allowance": CORRECTNESS_ERROR_ABSOLUTE_ALLOWANCE,
        "comparisons": comparisons,
    }
    if failures:
        raise RuntimeError(
            "Candidate correctness error regressed beyond the 10% gate: "
            + "; ".join(failures)
        )
    return gate


def _validate_candidate_only_correctness(
    variants: dict[str, Any],
    *,
    candidate_label: str | None,
    required_drivers: set[str],
) -> dict[str, Any]:
    """Require both source forms of each strict candidate-only Driver."""

    if not required_drivers:
        return {"status": "not_applicable", "drivers": [], "executions": 0}
    if candidate_label is None or candidate_label not in variants:
        raise RuntimeError("Candidate-only correctness Drivers require a candidate.")

    results = [
        result
        for result in variants[candidate_label]["correctness"]
        if result.get("driver_scope") == "candidate_only"
    ]
    observed_drivers = {str(result.get("driver")) for result in results}
    if observed_drivers != required_drivers:
        missing = sorted(required_drivers - observed_drivers)
        unexpected = sorted(observed_drivers - required_drivers)
        raise RuntimeError(
            "Candidate-only correctness Driver coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )

    expected_modes = {
        (driver, f"{source_mode}-{driver}")
        for driver in required_drivers
        for source_mode in ("original", "normalized")
    }
    observed_modes = {
        (str(result.get("driver")), str(result.get("mode"))) for result in results
    }
    if observed_modes != expected_modes or len(results) != len(expected_modes):
        raise RuntimeError(
            "Candidate-only correctness Drivers did not cover original and normalized sources."
        )
    if any(
        result.get("passed") is not True
        or not isinstance(result.get("metrics", {}).get("cases"), list)
        for result in results
    ):
        raise RuntimeError(
            "Candidate-only correctness Drivers did not pass strict distribution validation."
        )
    return {
        "status": "passed",
        "drivers": sorted(required_drivers),
        "executions": len(results),
    }


def _gpu_telemetry() -> dict[str, Any]:
    fields = (
        "name,uuid,driver_version,temperature.gpu,power.draw,"
        "clocks.sm,clocks.mem,utilization.gpu,memory.total"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error_type": type(error).__name__}
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    names = fields.split(",")
    return {"available": True, **dict(zip(names, values, strict=False))}


def _parse_benchmark_output(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith(BENCHMARK_MARKER)]
    if len(lines) != 1:
        raise RuntimeError("Benchmark output did not contain exactly one result marker.")
    return json.loads(lines[0][len(BENCHMARK_MARKER) :])


def _parse_ncu_details(csv_text: str) -> dict[str, Any]:
    """Parse NCU's long-form details export and validate requested coverage."""
    rows = list(csv.reader(csv_text.splitlines()))
    section_column: int | None = None
    metric_column: int | None = None
    value_column: int | None = None
    header_index: int | None = None
    for index, row in enumerate(rows):
        normalized = [cell.strip().casefold() for cell in row]
        if all(
            column in normalized
            for column in ("section name", "metric name", "metric value")
        ):
            header_index = index
            section_column = normalized.index("section name")
            metric_column = normalized.index("metric name")
            value_column = normalized.index("metric value")
            break
    if (
        header_index is None
        or section_column is None
        or metric_column is None
        or value_column is None
    ):
        return {
            "parse_valid": False,
            "observed_section_names": [],
            "metrics_by_section": {},
            "empty_metrics_by_section": {},
            "metric_names": [],
            "metric_count": 0,
            "missing_section_ids": list(NCU_SECTIONS),
            "missing_metrics_by_section": {},
        }

    metrics_by_section: dict[str, list[str]] = {}
    empty_metric_candidates: dict[str, list[str]] = {}
    for row in rows[header_index + 1 :]:
        if len(row) <= max(section_column, metric_column):
            continue
        section = row[section_column].strip()
        metric = row[metric_column].strip()
        value = row[value_column].strip() if len(row) > value_column else ""
        if not section:
            continue
        section_metrics = metrics_by_section.setdefault(section, [])
        if metric and value and metric not in section_metrics:
            section_metrics.append(metric)
        elif metric and not value:
            empty_metrics = empty_metric_candidates.setdefault(section, [])
            if metric not in empty_metrics:
                empty_metrics.append(metric)

    empty_metrics_by_section = {
        section: [
            metric
            for metric in metrics
            if metric not in metrics_by_section.get(section, [])
        ]
        for section, metrics in empty_metric_candidates.items()
    }
    empty_metrics_by_section = {
        section: metrics
        for section, metrics in empty_metrics_by_section.items()
        if metrics
    }

    observed_section_names = list(metrics_by_section)
    missing_section_ids = [
        section_id
        for section_id in NCU_SECTIONS
        if NCU_SECTION_NAMES[section_id] not in metrics_by_section
    ]
    missing_metrics_by_section: dict[str, list[list[str]]] = {}
    for section_id in NCU_SECTIONS:
        section_name = NCU_SECTION_NAMES[section_id]
        observed_metrics = set(metrics_by_section.get(section_name, []))
        missing_groups = [
            list(group)
            for group in NCU_REQUIRED_METRIC_GROUPS[section_name]
            if observed_metrics.isdisjoint(group)
        ]
        if missing_groups:
            missing_metrics_by_section[section_name] = missing_groups

    metric_names = list(
        dict.fromkeys(
            metric
            for metrics in metrics_by_section.values()
            for metric in metrics
        )
    )
    return {
        "parse_valid": True,
        "observed_section_names": observed_section_names,
        "metrics_by_section": metrics_by_section,
        "empty_metrics_by_section": empty_metrics_by_section,
        "metric_names": metric_names,
        "metric_count": sum(len(metrics) for metrics in metrics_by_section.values()),
        "missing_section_ids": missing_section_ids,
        "missing_metrics_by_section": missing_metrics_by_section,
    }


def _file_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.name, "size_bytes": 0, "sha256": None}
    content = path.read_bytes()
    return {
        "path": path.name,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest() if content else None,
    }


def _ncu_failure(
    error_type: str,
    *,
    artifacts: dict[str, dict[str, Any]],
    returncode: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "failed",
        "error_type": error_type,
        "requested_section_ids": list(NCU_SECTIONS),
        "artifacts": artifacts,
        "artifact_sha256": {
            metadata["path"]: metadata["sha256"]
            for metadata in artifacts.values()
            if metadata.get("sha256")
        },
    }
    if returncode is not None:
        result["returncode"] = returncode
    if details is not None:
        result.update(details)
    return result


def _run_ncu(
    executable: Path,
    *,
    output_dir: Path,
    label: str,
    timeout: float,
) -> dict[str, Any]:
    report_base = output_dir / f"ncu-{label}"
    report_path = report_base.with_suffix(".ncu-rep")
    details_path = output_dir / f"ncu-{label}-details.csv"
    raw_path = output_dir / f"ncu-{label}-raw.csv"
    metrics_path = output_dir / f"ncu-{label}-metrics.json"

    def artifacts() -> dict[str, dict[str, Any]]:
        return {
            "report": _file_evidence(report_path),
            "details_csv": _file_evidence(details_path),
            "raw_csv": _file_evidence(raw_path),
        }

    if not executable.is_file():
        return _ncu_failure("NCUTargetMissing", artifacts=artifacts())

    command = ["ncu"]
    for section in NCU_SECTIONS:
        command.extend(["--section", section])
    command.extend(
        [
            "--profile-from-start",
            "off",
            "--export",
            str(report_base),
            "--force-overwrite",
            str(executable),
        ]
    )
    try:
        completed = _run_command(
            command,
            cwd=executable.parent,
            timeout=timeout,
            log_path=output_dir / f"ncu-{label}.log",
            check=False,
        )
    except FileNotFoundError:
        return _ncu_failure("NCUExecutableUnavailable", artifacts=artifacts())
    except subprocess.TimeoutExpired:
        return _ncu_failure("NCUCommandTimeout", artifacts=artifacts())
    except (OSError, subprocess.SubprocessError) as error:
        return _ncu_failure(type(error).__name__, artifacts=artifacts())
    if completed.returncode != 0:
        error_name = (
            "ERR_NVGPUCTRPERM"
            if "ERR_NVGPUCTRPERM" in completed.stderr + completed.stdout
            else "NCUCommandFailed"
        )
        return _ncu_failure(
            error_name,
            artifacts=artifacts(),
            returncode=completed.returncode,
        )
    report_evidence = _file_evidence(report_path)
    if report_evidence["size_bytes"] == 0:
        return _ncu_failure("NCUReportMissingOrEmpty", artifacts=artifacts())

    export_results: dict[str, subprocess.CompletedProcess[str]] = {}
    for page, destination in (("details", details_path), ("raw", raw_path)):
        try:
            exported = _run_command(
                ["ncu", "--import", str(report_path), "--csv", "--page", page],
                cwd=executable.parent,
                timeout=timeout,
                log_path=output_dir / f"ncu-{label}-export-{page}.log",
                check=False,
            )
        except FileNotFoundError:
            return _ncu_failure("NCUExecutableUnavailable", artifacts=artifacts())
        except subprocess.TimeoutExpired:
            return _ncu_failure(
                f"NCU{page.title()}ExportTimeout", artifacts=artifacts()
            )
        except (OSError, subprocess.SubprocessError) as error:
            return _ncu_failure(type(error).__name__, artifacts=artifacts())
        destination.write_text(exported.stdout, encoding="utf-8")
        export_results[page] = exported

    failed_exports = [
        page for page, exported in export_results.items() if exported.returncode != 0
    ]
    evidence = artifacts()
    if failed_exports:
        return _ncu_failure(
            "NCUCSVExportFailed",
            artifacts=evidence,
            details={"failed_pages": failed_exports},
        )
    empty_artifacts = [
        name for name, metadata in evidence.items() if metadata["size_bytes"] == 0
    ]
    if empty_artifacts:
        return _ncu_failure(
            "NCUArtifactMissingOrEmpty",
            artifacts=evidence,
            details={"empty_artifacts": empty_artifacts},
        )

    coverage = _parse_ncu_details(export_results["details"].stdout)
    metrics_payload = {
        "requested_section_ids": list(NCU_SECTIONS),
        **coverage,
    }
    _atomic_json(
        metrics_path,
        metrics_payload,
    )
    evidence = artifacts()
    evidence["metrics_json"] = _file_evidence(metrics_path)
    if not coverage["parse_valid"]:
        return _ncu_failure(
            "NCUDetailsParseFailed",
            artifacts=evidence,
            details={**coverage, "metrics_json": metrics_path.name},
        )
    if coverage["missing_section_ids"]:
        return _ncu_failure(
            "NCUSectionCoverageIncomplete",
            artifacts=evidence,
            details={**coverage, "metrics_json": metrics_path.name},
        )
    if coverage["missing_metrics_by_section"]:
        return _ncu_failure(
            "NCUMetricCoverageIncomplete",
            artifacts=evidence,
            details={**coverage, "metrics_json": metrics_path.name},
        )
    return {
        "status": "completed",
        "requested_section_ids": list(NCU_SECTIONS),
        **coverage,
        "report": report_path.name,
        "details_csv": details_path.name,
        "raw_csv": raw_path.name,
        "metrics_json": metrics_path.name,
        "artifacts": evidence,
        "artifact_sha256": {
            metadata["path"]: metadata["sha256"]
            for metadata in evidence.values()
            if metadata.get("sha256")
        },
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def _version(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "error_type": type(error).__name__}
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip()[:10_000],
    }


def _summarize_records(
    records: list[dict[str, Any]], labels: list[str]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for label in labels:
        selected = [record for record in records if record["variant"] == label]
        all_samples = [
            sample for record in selected for sample in record["samples_us"]
        ]
        session_medians = [record["latency"]["median_us"] for record in selected]
        summaries[label] = {
            "all_samples": latency_summary(all_samples),
            "session_medians": session_medians,
            "session_medians_summary": latency_summary(session_medians),
            "session_spread_percent": session_spread_percent(session_medians),
        }
    return summaries


def _validate_session_protocol(
    phase: str, sessions: int, *, correctness_only: bool = False
) -> None:
    if correctness_only:
        return
    minimum = 3 if phase == "discovery" else 5
    if sessions < minimum:
        raise ValueError(
            f"{phase.capitalize()} requires at least {minimum} independent process sessions."
        )


def _classify_benchmark_result(
    *, phase: str, stable: bool, comparison: dict[str, Any] | None
) -> dict[str, Any]:
    """Classify evidence without turning diagnostic timing into a formal claim."""
    formally_comparable = comparison is None or bool(comparison.get("formal_valid"))
    execution_valid = stable and formally_comparable
    if not execution_valid:
        return {
            "outcome": "inconclusive",
            "execution_valid": False,
            "performance_claim_allowed": False,
        }
    if comparison is None or phase == "discovery":
        return {
            "outcome": "completed",
            "execution_valid": True,
            "performance_claim_allowed": False,
        }

    performance_claim_allowed = bool(
        comparison.get("all_sessions_not_slower")
        and comparison.get("performance_gate", {}).get("passed")
    )
    return {
        "outcome": "improved" if performance_claim_allowed else "no_improvement",
        "execution_valid": True,
        "performance_claim_allowed": performance_claim_allowed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness-first CUDA Events benchmark for KernelBench-CUDA."
    )
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument(
        "--extra-correctness-driver",
        type=Path,
        action="append",
        default=[],
        help="Additional Driver with the same launcher signature; may be repeated.",
    )
    parser.add_argument(
        "--candidate-only-correctness-driver",
        type=Path,
        action="append",
        default=[],
        help=(
            "Strict distribution Driver compiled only with the candidate; "
            "may be repeated."
        ),
    )
    parser.add_argument("--gpu", default="RTX 3080")
    parser.add_argument("--sm", default="sm_86")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--shape", default="driver-defined")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--inner-loops",
        type=int,
        default=0,
        help="Zero enables calibration to approximately one millisecond per sample.",
    )
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument(
        "--phase",
        choices=("discovery", "confirmation"),
        default="discovery",
    )
    parser.add_argument("--max-session-spread-percent", type=float, default=5.0)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=60.0,
        help="Cooldown before the single automatic retest of unstable sessions.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Compile and run all correctness gates without CUDA Events timing.",
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()

    task_dir = args.task_dir.resolve()
    driver_path = task_dir / "driver.cpp"
    upstream_baseline_path = (task_dir / "init.cu").resolve()
    baseline_path = (args.baseline or upstream_baseline_path).resolve()
    candidate_path = args.candidate.resolve() if args.candidate else None
    for path in (driver_path, baseline_path):
        if not path.is_file():
            parser.error(f"Required input does not exist: {path}")
    if candidate_path is not None and not candidate_path.is_file():
        parser.error(f"Candidate does not exist: {candidate_path}")
    extra_driver_paths = [path.resolve() for path in args.extra_correctness_driver]
    for path in extra_driver_paths:
        if not path.is_file():
            parser.error(f"Extra correctness Driver does not exist: {path}")
    candidate_only_driver_paths = [
        path.resolve() for path in args.candidate_only_correctness_driver
    ]
    for path in candidate_only_driver_paths:
        if not path.is_file():
            parser.error(f"Candidate-only correctness Driver does not exist: {path}")
    if candidate_only_driver_paths and candidate_path is None:
        parser.error("Candidate-only correctness Drivers require --candidate.")
    if min(args.warmup, args.repetitions, args.sessions) < 1:
        parser.error("Warmup, repetitions, and sessions must be positive.")
    try:
        _validate_session_protocol(
            args.phase,
            args.sessions,
            correctness_only=args.correctness_only,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.max_session_spread_percent <= 0 or args.cooldown_seconds < 0:
        parser.error("Session spread must be positive and cooldown non-negative.")

    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite output directory: {output_dir}")

    driver = driver_path.read_text(encoding="utf-8")
    instrumented_driver = instrument_driver(
        driver,
        seed=args.seed,
        warmup=args.warmup,
        repetitions=args.repetitions,
        inner_loops=args.inner_loops,
    )
    profiler_driver = instrument_profiler_driver(driver)
    variant_paths = {"baseline": baseline_path}
    if candidate_path is not None:
        variant_paths[args.candidate_name] = candidate_path

    manifest: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "kernel": args.kernel,
        "git_commit": _git_commit(),
        "gpu": args.gpu,
        "sm": args.sm,
        "dtype": args.dtype,
        "shape": args.shape,
        "seed": args.seed,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "sessions": args.sessions,
        "phase": args.phase,
        "max_session_spread_percent": args.max_session_spread_percent,
        "cooldown_seconds": args.cooldown_seconds,
        "container_image": os.getenv("KERNELBLASTER_CONTAINER_IMAGE"),
        "container_digest": os.getenv("KERNELBLASTER_CONTAINER_DIGEST"),
        "driver_sha256": sha256_text(driver),
        "extra_correctness_drivers": [
            {"name": path.name, "sha256": sha256_text(path.read_text(encoding="utf-8"))}
            for path in extra_driver_paths
        ],
        "candidate_only_correctness_drivers": [
            {
                "name": path.name,
                "scope": "candidate_only",
                "sha256": sha256_text(path.read_text(encoding="utf-8")),
            }
            for path in candidate_only_driver_paths
        ],
        "correctness_support_headers": [],
        "environment": {
            "nvidia_smi": _version(["nvidia-smi"]),
            "nvcc": _version(["nvcc", "--version"]),
            "ncu": _version(["ncu", "--version"]),
        },
        "variants": {},
        "baseline_scope": (
            "upstream_baseline"
            if baseline_path == upstream_baseline_path
            else "variant_head_to_head"
        ),
    }
    if any(
        '#include "correctness_metrics.h"' in path.read_text(encoding="utf-8")
        for path in [*extra_driver_paths, *candidate_only_driver_paths]
    ):
        support_header = (
            ROOT_DIR
            / "src"
            / "kernelblaster"
            / "servers"
            / "cuda_env"
            / "correctness_metrics.h"
        )
        manifest["correctness_support_headers"].append(
            {
                "path": str(support_header.relative_to(ROOT_DIR)),
                "sha256": sha256_text(support_header.read_text(encoding="utf-8")),
            }
        )

    candidate_label = args.candidate_name if candidate_path is not None else None
    normalized_sources: dict[str, str] = {}
    for label, source_path in variant_paths.items():
        original = source_path.read_text(encoding="utf-8")
        normalized, transformations = normalize_cuda_source(original)
        normalized_sources[label] = normalized
        correctness_results: list[dict[str, Any]] = []
        normalized_executable: Path | None = None
        correctness_drivers = [("official", driver, "shared", False)] + [
            (
                f"extra-{index}-{path.stem}",
                path.read_text(encoding="utf-8"),
                "shared",
                False,
            )
            for index, path in enumerate(extra_driver_paths)
        ]
        if label == candidate_label:
            correctness_drivers.extend(
                (
                    f"candidate-only-{index}-{path.stem}",
                    path.read_text(encoding="utf-8"),
                    "candidate_only",
                    True,
                )
                for index, path in enumerate(candidate_only_driver_paths)
            )
        for (
            driver_label,
            correctness_driver,
            driver_scope,
            require_distribution,
        ) in correctness_drivers:
            for source_mode, correctness_source in (
                ("original", original),
                ("normalized", normalized),
            ):
                executable = _compile_program(
                    output_dir,
                    label=label,
                    mode=f"{source_mode}-correctness-{driver_label}",
                    driver=correctness_driver,
                    source=correctness_source,
                    sm=args.sm,
                    timeout=args.timeout_seconds,
                )
                result = _run_correctness(
                    executable,
                    output_dir=output_dir,
                    label=label,
                    mode=f"{source_mode}-{driver_label}",
                    timeout=args.timeout_seconds,
                    require_distribution=require_distribution,
                )
                result["driver"] = driver_label
                result["driver_scope"] = driver_scope
                correctness_results.append(result)
                if source_mode == "normalized" and driver_label == "official":
                    normalized_executable = executable
        if not all(result["passed"] for result in correctness_results):
            raise RuntimeError(f"Correctness failed for {label}; timing was not started.")
        if normalized_executable is None:
            raise RuntimeError("Official normalized correctness executable is missing.")

        manifest["variants"][label] = {
            "source_name": source_path.name,
            "source_sha256": sha256_text(original),
            "normalized_sha256": sha256_text(normalized),
            "transformations": transformations,
            "correctness": correctness_results,
        }
    manifest["correctness_error_regression"] = _validate_correctness_error_regression(
        manifest["variants"],
        candidate_label=candidate_label,
        required_drivers={
            f"extra-{index}-{path.stem}"
            for index, path in enumerate(extra_driver_paths)
        },
    )
    manifest["candidate_only_correctness"] = _validate_candidate_only_correctness(
        manifest["variants"],
        candidate_label=candidate_label,
        required_drivers={
            f"candidate-only-{index}-{path.stem}"
            for index, path in enumerate(candidate_only_driver_paths)
        },
    )
    _atomic_json(output_dir / "run_manifest.json", manifest)

    # No benchmark executable, NCU target, or Events sample is created until
    # every shared and candidate-only correctness gate has passed.
    executables: dict[str, Path] = {}
    for label, normalized in normalized_sources.items():
        executables[label] = _compile_program(
            output_dir,
            label=label,
            mode="benchmark",
            driver=instrumented_driver,
            source=normalized,
            sm=args.sm,
            timeout=args.timeout_seconds,
        )
        if args.ncu:
            profiler_executable = _compile_program(
                output_dir,
                label=label,
                mode="ncu-profile",
                driver=profiler_driver,
                source=normalized,
                sm=args.sm,
                timeout=args.timeout_seconds,
            )
            manifest["variants"][label]["ncu"] = _run_ncu(
                profiler_executable,
                output_dir=output_dir,
                label=label,
                timeout=args.timeout_seconds,
            )
    _atomic_json(output_dir / "run_manifest.json", manifest)

    if args.correctness_only:
        ncu_attribution_valid = None
        if args.ncu:
            ncu_attribution_valid = all(
                variant.get("ncu", {}).get("status") == "completed"
                and bool(variant.get("ncu", {}).get("metric_names"))
                for variant in manifest["variants"].values()
            )
        blocked = args.ncu and not ncu_attribution_valid
        manifest["profiling_mode"] = (
            "ncu" if args.ncu and ncu_attribution_valid else "events_only"
        )
        manifest["outcome"] = "blocked" if blocked else "completed"
        manifest["failure_classification"] = (
            "ncu_attribution_unavailable" if blocked else None
        )
        manifest["budget"] = {"api_requests": 0, "tokens": 0}
        manifest["validation_gates"] = {
            "correctness": "passed",
            "correctness_error_regression": manifest[
                "correctness_error_regression"
            ]["status"],
            "candidate_only_correctness": manifest[
                "candidate_only_correctness"
            ]["status"],
            "events_stability": None,
            "performance": None,
            "ncu_attribution": ncu_attribution_valid,
        }
        summary = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "run_id": output_dir.name,
            "task_id": args.task_id,
            "kernel": args.kernel,
            "phase": "correctness_only",
            "stable": None,
            "comparison": None,
            "ncu_attribution_valid": ncu_attribution_valid,
            "performance_claim_allowed": False,
            "validation_gates": manifest["validation_gates"],
        }
        _atomic_json(output_dir / "summary.json", summary)
        _atomic_json(output_dir / "run_manifest.json", manifest)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 4 if blocked else 0

    records: list[dict[str, Any]] = []
    jsonl_path = output_dir / "measurements.jsonl"
    labels = list(executables)
    round_summaries: list[dict[str, Any]] = []

    def run_round(round_index: int) -> dict[str, Any]:
        round_records: list[dict[str, Any]] = []
        for session in range(args.sessions):
            order = labels if session % 2 == 0 else list(reversed(labels))
            for position, label in enumerate(order):
                before = _gpu_telemetry()
                log_path = (
                    output_dir
                    / f"round-{round_index}-session-{session}-{position}-{label}.log"
                )
                completed = _run_command(
                    [str(executables[label])],
                    cwd=executables[label].parent,
                    timeout=args.timeout_seconds,
                    log_path=log_path,
                    check=False,
                )
                _raise_if_resource_blocked(completed, log_path=log_path)
                if (
                    completed.returncode != 0
                    or "passed" not in completed.stdout.lower()
                ):
                    raise RuntimeError(
                        f"Benchmark execution failed correctness for {label}."
                    )
                parsed = _parse_benchmark_output(completed.stdout)
                samples = [float(value) for value in parsed["samples_us"]]
                after = _gpu_telemetry()
                record = {
                    "schema_version": BENCHMARK_SCHEMA_VERSION,
                    "run_id": output_dir.name,
                    "task_id": args.task_id,
                    "kernel": args.kernel,
                    "variant": label,
                    "round": round_index,
                    "session": session,
                    "order": "AB" if order == labels else "BA",
                    "order_position": position,
                    "seed": args.seed,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "inner_loops": int(parsed["inner_loops"]),
                    "correct": True,
                    "samples_us": samples,
                    "latency": latency_summary(samples),
                    "telemetry_before": before,
                    "telemetry_after": after,
                    "source_sha256": manifest["variants"][label]["source_sha256"],
                    "normalized_sha256": manifest["variants"][label][
                        "normalized_sha256"
                    ],
                    "driver_sha256": manifest["driver_sha256"],
                    "git_commit": manifest["git_commit"],
                    "container_digest": manifest["container_digest"],
                    "gpu": args.gpu,
                    "sm": args.sm,
                    "dtype": args.dtype,
                    "shape": args.shape,
                }
                records.append(record)
                round_records.append(record)
                with jsonl_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        return _summarize_records(round_records, labels)

    summaries = run_round(0)
    round_summaries.append(summaries)
    unstable = any(
        summary["session_spread_percent"] > args.max_session_spread_percent
        for summary in summaries.values()
    )
    if unstable:
        if args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
        summaries = run_round(1)
        round_summaries.append(summaries)

    comparison = None
    if len(labels) == 2:
        baseline_median = summaries["baseline"]["session_medians_summary"][
            "median_us"
        ]
        candidate_label = next(label for label in labels if label != "baseline")
        candidate_median = summaries[candidate_label]["session_medians_summary"][
            "median_us"
        ]
        session_speedups = [
            baseline / candidate
            for baseline, candidate in zip(
                summaries["baseline"]["session_medians"],
                summaries[candidate_label]["session_medians"],
                strict=True,
            )
        ]
        comparison = {
            "candidate": candidate_label,
            "baseline_median_us": baseline_median,
            "candidate_median_us": candidate_median,
            "speedup": baseline_median / candidate_median,
            "session_speedups": session_speedups,
            "all_sessions_not_slower": all(value >= 1.0 for value in session_speedups),
            "comparison_scope": manifest["baseline_scope"],
        }
        comparison.update(
            comparison_validity(
                baseline_source_sha256=manifest["variants"]["baseline"][
                    "source_sha256"
                ],
                candidate_source_sha256=manifest["variants"][candidate_label][
                    "source_sha256"
                ],
                baseline_session_medians=summaries["baseline"]["session_medians"],
                candidate_session_medians=summaries[candidate_label][
                    "session_medians"
                ],
                speedup=comparison["speedup"],
                max_session_spread_percent=args.max_session_spread_percent,
            )
        )
        if args.phase == "confirmation":
            performance_gate = evaluate_performance_gate(
                summaries["baseline"]["session_medians"],
                summaries[candidate_label]["session_medians"],
            ).to_dict()
        else:
            performance_gate = {
                "passed": False,
                "reason": "Discovery phase does not produce a formal performance claim.",
                "session_speedups": session_speedups,
            }
        comparison["performance_gate"] = performance_gate
        with (output_dir / "comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            csv_comparison = {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in comparison.items()
            }
            writer = csv.DictWriter(stream, fieldnames=list(csv_comparison))
            writer.writeheader()
            writer.writerow(csv_comparison)

    stable = all(
        summary["session_spread_percent"] <= args.max_session_spread_percent
        for summary in summaries.values()
    )
    ncu_attribution_valid = None
    if args.ncu:
        ncu_attribution_valid = all(
            variant.get("ncu", {}).get("status") == "completed"
            and bool(variant.get("ncu", {}).get("metric_names"))
            for variant in manifest["variants"].values()
        )
    classification = _classify_benchmark_result(
        phase=args.phase,
        stable=stable,
        comparison=comparison,
    )
    execution_valid = classification["execution_valid"]
    performance_claim_allowed = classification["performance_claim_allowed"]

    summary = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": output_dir.name,
        "task_id": args.task_id,
        "kernel": args.kernel,
        "summaries": summaries,
        "round_summaries": round_summaries,
        "retest_triggered": len(round_summaries) == 2,
        "active_round": len(round_summaries) - 1,
        "stable": stable,
        "comparison": comparison,
        "ncu_attribution_valid": ncu_attribution_valid,
        "performance_claim_allowed": performance_claim_allowed,
        "phase": args.phase,
    }
    manifest["profiling_mode"] = (
        "ncu" if args.ncu and ncu_attribution_valid else "events_only"
    )
    manifest["outcome"] = classification["outcome"]
    manifest["failure_classification"] = (
        None if execution_valid else "unstable_or_invalid_comparison"
    )
    manifest["budget"] = {"api_requests": 0, "tokens": 0}
    manifest["validation_gates"] = {
        "correctness": "passed",
        "correctness_error_regression": manifest[
            "correctness_error_regression"
        ]["status"],
        "candidate_only_correctness": manifest[
            "candidate_only_correctness"
        ]["status"],
        "events_stability": "passed" if stable else "failed",
        "performance": (
            comparison.get("performance_gate", {}).get("passed")
            if comparison is not None
            else None
        ),
        "ncu_attribution": ncu_attribution_valid,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.ncu and not ncu_attribution_valid:
        return 4
    if classification["outcome"] == "inconclusive":
        return 3
    if classification["outcome"] == "no_improvement":
        return 3
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BlockedResourceError as error:
        print(f"KERNELBLASTER_BLOCKED {error}", file=sys.stderr)
        exit_code = 4
    raise SystemExit(exit_code)
