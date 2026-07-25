# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_portfolio_v2_1_evidence",
    ROOT / "scripts" / "build_portfolio_v2_1_evidence.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_issue10_summary_preserves_formal_outcome_boundaries():
    payload = MODULE.build_issue10_summary()
    tasks = {task["task_id"]: task for task in payload["tasks"]}

    assert payload["formal_outcomes"] == {
        "improved": ["004", "007", "036", "040", "095"],
        "no_improvement": [],
        "inconclusive": [],
    }
    assert payload["issue_close_allowed"] is True
    assert payload["remaining_blocker"] is None
    assert tasks["007"]["correctness_matrix"][
        "candidate_only_full_element_statistics"
    ] is True
    assert tasks["095"]["evidence_run"] == "095-v5"
    assert tasks["095"]["performance_confirmation"]["formal_valid"] is True
    assert tasks["095"]["performance_confirmation"]["claim_kind"] == "formal"


def test_sha256_index_exactly_covers_nonempty_regular_files(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "nested" / "b.bin").write_bytes(b"evidence")
    (tmp_path / "SHA256SUMS.json").write_text("ignored\n", encoding="utf-8")

    payload = MODULE.build_sha256_index(tmp_path)

    assert list(payload) == ["a.json", "nested/b.bin"]
    assert all(len(digest) == 64 for digest in payload.values())


def test_sha256_index_rejects_empty_or_symlinked_artifacts(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.touch()
    with pytest.raises(ValueError, match="may not be empty"):
        MODULE.build_sha256_index(tmp_path)

    empty.unlink()
    target = tmp_path / "target.txt"
    target.write_text("evidence\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="may not be symlinks"):
        MODULE.build_sha256_index(tmp_path)
