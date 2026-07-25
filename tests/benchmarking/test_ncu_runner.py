# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_cuda_ncu", ROOT / "scripts" / "benchmark_cuda.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


DETAILS_CSV = """==PROF== Connected
"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"
"0","kernel","GPU Speed Of Light Throughput","Compute (SM) Throughput","%","42"
"0","kernel","GPU Speed Of Light Throughput","Memory Throughput","%","73"
"0","kernel","Launch Statistics","Block Size","","256"
"0","kernel","Launch Statistics","Grid Size","","1024"
"0","kernel","Launch Statistics","Registers Per Thread","register/thread","32"
"0","kernel","Occupancy","Theoretical Occupancy","%","100"
"0","kernel","Occupancy","Achieved Occupancy","%","75"
"0","kernel","Occupancy","","",""
"""


def _blank_all_metric_values(csv_text: str) -> str:
    for value in ("42", "73", "256", "1024", "32", "100", "75"):
        csv_text = csv_text.replace(f',"{value}"\n', ',""\n')
    return csv_text


def test_ncu_details_parser_requires_all_sections_and_key_metrics():
    parsed = MODULE._parse_ncu_details(DETAILS_CSV)

    assert parsed["parse_valid"] is True
    assert parsed["observed_section_names"] == [
        "GPU Speed Of Light Throughput",
        "Launch Statistics",
        "Occupancy",
    ]
    assert parsed["metric_count"] == 7
    assert parsed["missing_section_ids"] == []
    assert parsed["missing_metrics_by_section"] == {}
    assert parsed["metrics_by_section"]["Occupancy"] == [
        "Theoretical Occupancy",
        "Achieved Occupancy",
    ]


def test_ncu_details_parser_reports_missing_section_and_metric_coverage():
    without_occupancy = "\n".join(
        line for line in DETAILS_CSV.splitlines() if '"Occupancy"' not in line
    )
    missing_section = MODULE._parse_ncu_details(without_occupancy)
    assert missing_section["missing_section_ids"] == ["Occupancy"]
    assert missing_section["missing_metrics_by_section"]["Occupancy"] == [
        ["Theoretical Occupancy"],
        ["Achieved Occupancy"],
    ]

    without_achieved = DETAILS_CSV.replace(
        '"0","kernel","Occupancy","Achieved Occupancy","%","75"\n', ""
    )
    missing_metric = MODULE._parse_ncu_details(without_achieved)
    assert missing_metric["missing_section_ids"] == []
    assert missing_metric["missing_metrics_by_section"] == {
        "Occupancy": [["Achieved Occupancy"]]
    }


def test_ncu_details_parser_requires_metric_value_column():
    without_value_column = DETAILS_CSV.replace(',"Metric Value"', "")
    parsed = MODULE._parse_ncu_details(without_value_column)

    assert parsed["parse_valid"] is False
    assert parsed["metric_count"] == 0


def test_ncu_details_parser_treats_blank_required_values_as_missing():
    blank_achieved = DETAILS_CSV.replace(
        '"Occupancy","Achieved Occupancy","%","75"',
        '"Occupancy","Achieved Occupancy","%",""',
    )
    parsed = MODULE._parse_ncu_details(blank_achieved)

    assert parsed["parse_valid"] is True
    assert parsed["empty_metrics_by_section"] == {
        "Occupancy": ["Achieved Occupancy"]
    }
    assert parsed["missing_metrics_by_section"] == {
        "Occupancy": [["Achieved Occupancy"]]
    }


def test_ncu_details_parser_rejects_all_blank_metric_values():
    parsed = MODULE._parse_ncu_details(_blank_all_metric_values(DETAILS_CSV))

    assert parsed["parse_valid"] is True
    assert parsed["metric_count"] == 0
    assert set(parsed["empty_metrics_by_section"]) == {
        "GPU Speed Of Light Throughput",
        "Launch Statistics",
        "Occupancy",
    }
    assert set(parsed["missing_metrics_by_section"]) == {
        "GPU Speed Of Light Throughput",
        "Launch Statistics",
        "Occupancy",
    }


def test_ncu_details_parser_rejects_wide_raw_csv():
    parsed = MODULE._parse_ncu_details(
        '"ID","Kernel Name","sm__throughput.avg","launch__block_size"\n'
        '"0","kernel","42","256"\n'
    )
    assert parsed["parse_valid"] is False
    assert parsed["metric_count"] == 0


def test_run_ncu_exports_details_and_raw_with_hashes(monkeypatch, tmp_path):
    executable = tmp_path / "main"
    executable.write_text("executable", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if "--export" in command:
            report_base = Path(command[command.index("--export") + 1])
            report_base.with_suffix(".ncu-rep").write_bytes(b"ncu-report")
            return subprocess.CompletedProcess(command, 0, "", "")
        page = command[command.index("--page") + 1]
        stdout = DETAILS_CSV if page == "details" else '"ID","sm__throughput"\n"0","42"\n'
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(MODULE, "_run_command", fake_run)
    result = MODULE._run_ncu(
        executable,
        output_dir=tmp_path,
        label="candidate",
        timeout=10,
    )

    assert result["status"] == "completed"
    assert result["requested_section_ids"] == [
        "SpeedOfLight",
        "LaunchStats",
        "Occupancy",
    ]
    assert result["metric_count"] == 7
    assert result["missing_section_ids"] == []
    assert result["missing_metrics_by_section"] == {}
    assert set(result["artifacts"]) == {
        "report",
        "details_csv",
        "raw_csv",
        "metrics_json",
    }
    assert all(metadata["size_bytes"] > 0 for metadata in result["artifacts"].values())
    assert len(result["artifact_sha256"]) == 4
    assert all(len(digest) == 64 for digest in result["artifact_sha256"].values())
    assert [call[call.index("--page") + 1] for call in calls[1:]] == [
        "details",
        "raw",
    ]
    for section in MODULE.NCU_SECTIONS:
        assert ["--section", section] in [
            calls[0][index : index + 2] for index in range(len(calls[0]) - 1)
        ]

    metrics = json.loads((tmp_path / result["metrics_json"]).read_text(encoding="utf-8"))
    assert metrics["requested_section_ids"] == result["requested_section_ids"]
    assert metrics["metrics_by_section"] == result["metrics_by_section"]


def test_run_ncu_fails_when_report_is_missing(monkeypatch, tmp_path):
    executable = tmp_path / "main"
    executable.write_text("executable", encoding="utf-8")
    monkeypatch.setattr(
        MODULE,
        "_run_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    result = MODULE._run_ncu(
        executable,
        output_dir=tmp_path,
        label="baseline",
        timeout=10,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "NCUReportMissingOrEmpty"


def test_run_ncu_fails_when_metric_coverage_is_incomplete(monkeypatch, tmp_path):
    executable = tmp_path / "main"
    executable.write_text("executable", encoding="utf-8")
    incomplete = DETAILS_CSV.replace(
        '"0","kernel","Occupancy","Achieved Occupancy","%","75"\n', ""
    )

    def fake_run(command, **_kwargs):
        if "--export" in command:
            report_base = Path(command[command.index("--export") + 1])
            report_base.with_suffix(".ncu-rep").write_bytes(b"ncu-report")
            return subprocess.CompletedProcess(command, 0, "", "")
        page = command[command.index("--page") + 1]
        stdout = incomplete if page == "details" else "raw\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(MODULE, "_run_command", fake_run)
    result = MODULE._run_ncu(
        executable,
        output_dir=tmp_path,
        label="baseline",
        timeout=10,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "NCUMetricCoverageIncomplete"
    assert result["missing_metrics_by_section"] == {
        "Occupancy": [["Achieved Occupancy"]]
    }


def test_run_ncu_fails_when_required_metric_value_is_blank(monkeypatch, tmp_path):
    executable = tmp_path / "main"
    executable.write_text("executable", encoding="utf-8")
    blank_achieved = DETAILS_CSV.replace(
        '"Occupancy","Achieved Occupancy","%","75"',
        '"Occupancy","Achieved Occupancy","%",""',
    )

    def fake_run(command, **_kwargs):
        if "--export" in command:
            report_base = Path(command[command.index("--export") + 1])
            report_base.with_suffix(".ncu-rep").write_bytes(b"ncu-report")
            return subprocess.CompletedProcess(command, 0, "", "")
        page = command[command.index("--page") + 1]
        stdout = blank_achieved if page == "details" else "raw\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(MODULE, "_run_command", fake_run)
    result = MODULE._run_ncu(
        executable,
        output_dir=tmp_path,
        label="baseline",
        timeout=10,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "NCUMetricCoverageIncomplete"
    assert result["empty_metrics_by_section"] == {
        "Occupancy": ["Achieved Occupancy"]
    }


def test_run_ncu_fails_when_all_required_metric_values_are_blank(
    monkeypatch, tmp_path
):
    executable = tmp_path / "main"
    executable.write_text("executable", encoding="utf-8")
    blank_details = _blank_all_metric_values(DETAILS_CSV)

    def fake_run(command, **_kwargs):
        if "--export" in command:
            report_base = Path(command[command.index("--export") + 1])
            report_base.with_suffix(".ncu-rep").write_bytes(b"ncu-report")
            return subprocess.CompletedProcess(command, 0, "", "")
        page = command[command.index("--page") + 1]
        stdout = blank_details if page == "details" else "raw\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(MODULE, "_run_command", fake_run)
    result = MODULE._run_ncu(
        executable,
        output_dir=tmp_path,
        label="baseline",
        timeout=10,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "NCUMetricCoverageIncomplete"
    assert result["metric_count"] == 0


def test_run_ncu_fails_when_raw_export_command_fails(monkeypatch, tmp_path):
    executable = tmp_path / "main"
    executable.write_text("executable", encoding="utf-8")

    def fake_run(command, **_kwargs):
        if "--export" in command:
            report_base = Path(command[command.index("--export") + 1])
            report_base.with_suffix(".ncu-rep").write_bytes(b"ncu-report")
            return subprocess.CompletedProcess(command, 0, "", "")
        page = command[command.index("--page") + 1]
        if page == "raw":
            return subprocess.CompletedProcess(command, 7, "partial raw", "export failed")
        return subprocess.CompletedProcess(command, 0, DETAILS_CSV, "")

    monkeypatch.setattr(MODULE, "_run_command", fake_run)
    result = MODULE._run_ncu(
        executable,
        output_dir=tmp_path,
        label="baseline",
        timeout=10,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "NCUCSVExportFailed"
    assert result["failed_pages"] == ["raw"]


def test_run_ncu_fails_when_raw_export_is_empty(monkeypatch, tmp_path):
    executable = tmp_path / "main"
    executable.write_text("executable", encoding="utf-8")

    def fake_run(command, **_kwargs):
        if "--export" in command:
            report_base = Path(command[command.index("--export") + 1])
            report_base.with_suffix(".ncu-rep").write_bytes(b"ncu-report")
            return subprocess.CompletedProcess(command, 0, "", "")
        page = command[command.index("--page") + 1]
        return subprocess.CompletedProcess(
            command, 0, DETAILS_CSV if page == "details" else "", ""
        )

    monkeypatch.setattr(MODULE, "_run_command", fake_run)
    result = MODULE._run_ncu(
        executable,
        output_dir=tmp_path,
        label="baseline",
        timeout=10,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "NCUArtifactMissingOrEmpty"
    assert result["empty_artifacts"] == ["raw_csv"]


def test_run_ncu_preserves_counter_permission_failure(monkeypatch, tmp_path):
    executable = tmp_path / "main"
    executable.write_text("executable", encoding="utf-8")
    monkeypatch.setattr(
        MODULE,
        "_run_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", "ERR_NVGPUCTRPERM"
        ),
    )

    result = MODULE._run_ncu(
        executable,
        output_dir=tmp_path,
        label="baseline",
        timeout=10,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "ERR_NVGPUCTRPERM"
    assert result["returncode"] == 1
