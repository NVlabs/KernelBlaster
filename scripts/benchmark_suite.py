#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.kernelblaster.portfolio import load_suite  # noqa: E402


def _default_output_dir(suite_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT_DIR / "out" / "portfolio" / "baseline" / suite_name / timestamp


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _command(
    *,
    task_dir: Path,
    task_id: str,
    kernel: str,
    output_dir: Path,
    warmup: int,
    repetitions: int,
    sessions: int,
    cooldown_seconds: float,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "benchmark_cuda.py"),
        "--task-dir",
        str(task_dir),
        "--task-id",
        task_id,
        "--kernel",
        kernel,
        "--warmup",
        str(warmup),
        "--repetitions",
        str(repetitions),
        "--sessions",
        str(sessions),
        "--cooldown-seconds",
        str(cooldown_seconds),
        "--output-dir",
        str(output_dir),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run serial CUDA Events baselines for every task in a suite."
    )
    parser.add_argument("--suite", default="core10")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    suite = load_suite(args.suite, ROOT_DIR)
    selected_ids = set(args.task_id)
    tasks = [
        task for task in suite.tasks if not selected_ids or task.task_id in selected_ids
    ]
    unknown = selected_ids - {task.task_id for task in suite.tasks}
    if unknown:
        parser.error(f"Task IDs are not in suite {suite.name}: {sorted(unknown)}")
    if not tasks:
        parser.error("No tasks selected.")

    output_dir = (args.output_dir or _default_output_dir(suite.name)).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite output directory: {output_dir}")

    results: list[dict[str, Any]] = []
    for task in tasks:
        task_output = output_dir / task.task_id
        command = _command(
            task_dir=(ROOT_DIR / task.path).resolve(),
            task_id=task.task_id,
            kernel=task.name,
            output_dir=task_output,
            warmup=args.warmup,
            repetitions=args.repetitions,
            sessions=args.sessions,
            cooldown_seconds=args.cooldown_seconds,
        )
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
        (output_dir / f"launcher-{task.task_id}.log").write_text(
            "COMMAND\n"
            + json.dumps(command)
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\n\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )
        summary_path = task_output / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else None
        )
        results.append(
            {
                "task_id": task.task_id,
                "kernel": task.name,
                "returncode": completed.returncode,
                "status": "completed" if completed.returncode == 0 else "failed",
                "stable": summary.get("stable") if summary else None,
                "baseline_median_us": (
                    summary.get("summaries", {})
                    .get("baseline", {})
                    .get("session_medians_summary", {})
                    .get("median_us")
                    if summary
                    else None
                ),
            }
        )
        _atomic_json(
            output_dir / "suite_summary.json",
            {
                "schema_version": "1.0",
                "suite": suite.name,
                "results": results,
                "complete": False,
            },
        )

    payload = {
        "schema_version": "1.0",
        "suite": suite.name,
        "results": results,
        "complete": True,
        "completed_tasks": sum(row["status"] == "completed" for row in results),
        "failed_tasks": sum(row["status"] == "failed" for row in results),
    }
    _atomic_json(output_dir / "suite_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload["failed_tasks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
