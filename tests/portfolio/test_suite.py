from __future__ import annotations

import json

import pytest

from src.kernelblaster.portfolio import load_suite


def _task(repo_root, number=1, task_id="001", path="data/task"):
    task_dir = repo_root / path
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "init.cu").write_text("// init\n")
    (task_dir / "driver.cpp").write_text("// driver\n")
    return {
        "number": number,
        "id": task_id,
        "name": "Task",
        "path": path,
        "category": "unit",
    }


def _write_suite(repo_root, tasks):
    path = repo_root / "suite.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "unit",
                "subset": "level1",
                "precision": "fp16",
                "defaults": {"rollouts": 1, "steps": 1},
                "tasks": tasks,
            }
        )
    )
    return path


def test_valid_suite_loads(tmp_path):
    suite = load_suite(str(_write_suite(tmp_path, [_task(tmp_path)])), tmp_path)
    assert suite.problem_numbers == "1"


def test_duplicate_number_is_rejected(tmp_path):
    first = _task(tmp_path, number=1, task_id="001", path="data/one")
    second = _task(tmp_path, number=1, task_id="002", path="data/two")
    with pytest.raises(ValueError, match="numbers must be unique"):
        load_suite(str(_write_suite(tmp_path, [first, second])), tmp_path)


def test_duplicate_id_is_rejected(tmp_path):
    first = _task(tmp_path, number=1, task_id="001", path="data/one")
    second = _task(tmp_path, number=2, task_id="001", path="data/two")
    with pytest.raises(ValueError, match="IDs must be unique"):
        load_suite(str(_write_suite(tmp_path, [first, second])), tmp_path)


def test_path_escape_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside-kernelblaster-test"
    outside.mkdir(exist_ok=True)
    (outside / "init.cu").write_text("// init\n")
    (outside / "driver.cpp").write_text("// driver\n")
    escaped = {
        "number": 1,
        "id": "001",
        "name": "Escaped",
        "path": "../outside-kernelblaster-test",
        "category": "unit",
    }
    with pytest.raises(ValueError, match="escapes the repository"):
        load_suite(str(_write_suite(tmp_path, [escaped])), tmp_path)


def test_missing_artifact_is_rejected(tmp_path):
    task = _task(tmp_path)
    (tmp_path / task["path"] / "driver.cpp").unlink()
    with pytest.raises(ValueError, match="missing driver.cpp"):
        load_suite(str(_write_suite(tmp_path, [task])), tmp_path)


def test_artifact_symlink_escape_is_rejected(tmp_path):
    task = _task(tmp_path)
    outside = tmp_path.parent / "outside-kernelblaster-artifact.cu"
    outside.write_text("// outside\n")
    init_path = tmp_path / task["path"] / "init.cu"
    init_path.unlink()
    try:
        init_path.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symbolic links are not available: {error}")

    with pytest.raises(ValueError, match="artifact escapes the repository"):
        load_suite(str(_write_suite(tmp_path, [task])), tmp_path)
