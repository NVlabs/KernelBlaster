#!/usr/bin/env python3
"""Run correctness-first CUDA Events comparisons for Core 10 candidates."""
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

from src.kernelblaster.portfolio import (  # noqa: E402
    CAPABILITY_MARKER,
    CapabilityResult,
    describe_capabilities,
    load_capability_manifest,
    load_suite,
    parse_shape,
    task_map,
    validate_candidate_request,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_capability_manifest(path)
    candidates: dict[str, dict[str, Any]] = {}
    for item in manifest.get("tasks", []):
        task_id = str(item.get("id", ""))
        if not task_id or task_id in candidates:
            raise ValueError(f"Missing or duplicate candidate task ID: {task_id!r}")
        source = (path.parent / item["source"]).resolve()
        if not source.is_file():
            raise ValueError(f"Candidate source does not exist: {source}")
        extra_drivers = [
            (path.parent / driver).resolve()
            for driver in item.get("extra_correctness_drivers", [])
        ]
        if not all(driver.is_file() for driver in extra_drivers):
            raise ValueError(f"Candidate {task_id} has a missing correctness Driver.")
        candidate_only_drivers = [
            (path.parent / driver).resolve()
            for driver in item.get("candidate_only_correctness_drivers", [])
        ]
        if not all(driver.is_file() for driver in candidate_only_drivers):
            raise ValueError(
                f"Candidate {task_id} has a missing candidate-only correctness Driver."
            )
        candidates[task_id] = {
            **item,
            "source_path": source,
            "extra_driver_paths": extra_drivers,
            "candidate_only_driver_paths": candidate_only_drivers,
        }
    return candidates


def _emit_capability(payload: dict[str, Any]) -> None:
    print(CAPABILITY_MARKER + json.dumps(payload, sort_keys=True))


def _request_for_task(
    args: argparse.Namespace,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        # A portability replay validates the source contract but compiles for
        # the explicitly recorded target architecture. It never expands the
        # advertised production capability.
        "arch": args.portability_replay_from or args.sm,
        "compile_arch": args.sm,
        "dtype": args.dtype,
        "target_dtype": args.target_dtype,
        "layout": args.layout,
        "stream_mode": args.stream_mode,
        "stream_count": args.stream_count,
        "graph_capture": args.graph_capture,
        "backward": args.backward,
        "shape": parse_shape(args.shape, candidate),
        "portability_replay": args.portability_replay_from is not None,
    }


def _invalid_cli_request(
    args: argparse.Namespace, selected_ids: list[str]
) -> dict[str, Any] | None:
    """Validate task-independent request syntax before resolving task IDs."""

    string_fields = {
        "gpu": args.gpu,
        "sm": args.sm,
        "dtype": args.dtype,
        "target_dtype": args.target_dtype,
        "layout": args.layout,
        "stream_mode": args.stream_mode,
        "shape": args.shape,
    }
    invalid = (
        not selected_ids
        or any(not task_id.strip() for task_id in selected_ids)
        or any(not isinstance(value, str) or not value.strip() for value in string_fields.values())
        or args.stream_count < 1
        or args.warmup < 1
        or args.repetitions < 1
        or args.sessions < 1
        or args.cooldown_seconds < 0
        or args.max_session_spread_percent <= 0
        or (
            args.portability_replay_from is not None
            and args.sm != "sm_80"
        )
    )
    if not invalid:
        return None
    return {
        "task_ids": selected_ids,
        **string_fields,
        "stream_count": args.stream_count,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "sessions": args.sessions,
        "cooldown_seconds": args.cooldown_seconds,
        "max_session_spread_percent": args.max_session_spread_percent,
        "portability_replay_from": args.portability_replay_from,
    }


def _benchmark_target_arguments(gpu: str, sm: str) -> list[str]:
    """Return the explicit target arguments forwarded to benchmark_cuda.py."""
    return ["--gpu", gpu, "--sm", sm]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Core 10 candidates against upstream init.cu."
    )
    parser.add_argument("--suite", default="core10")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT_DIR
        / "portfolio"
        / "case_studies"
        / "core10"
        / "candidates.json",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--describe-capabilities",
        action="store_true",
        help="Print the machine-readable contract without compiling or using CUDA.",
    )
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 3080")
    parser.add_argument("--sm", default="sm_86")
    parser.add_argument(
        "--portability-replay-from",
        choices=("sm_86",),
        default=None,
        help=(
            "Validate the sm_86 source contract while compiling for --sm; "
            "used only for explicitly labelled cross-GPU replay."
        ),
    )
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--target-dtype", default="int64")
    parser.add_argument("--layout", default="contiguous_row_major")
    parser.add_argument("--stream-mode", default="legacy_default")
    parser.add_argument("--stream-count", type=int, default=1)
    parser.add_argument("--graph-capture", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--shape",
        default="canonical",
        help="Case id, JSON shape, or comma-separated name=value dimensions.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--phase",
        choices=("discovery", "confirmation"),
        default="confirmation",
        help="Discovery uses 3 sessions; formal confirmation uses 5.",
    )
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--max-session-spread-percent", type=float, default=5.0)
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run compile/correctness gates without CUDA Events timing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "out" / "portfolio" / "candidates" / _timestamp(),
    )
    args = parser.parse_args()
    if args.sessions is None:
        args.sessions = 3 if args.phase == "discovery" else 5

    manifest_path = args.manifest.resolve()
    capability_manifest = load_capability_manifest(manifest_path)
    candidates = load_candidates(manifest_path)
    if args.describe_capabilities:
        description = describe_capabilities(
            capability_manifest, args.task_id if args.task_id else None
        )
        payload = {
            "supported": not description["unknown_tasks"],
            "reason_code": (
                "unknown_task" if description["unknown_tasks"] else None
            ),
            **description,
        }
        _emit_capability(payload)
        return 2 if description["unknown_tasks"] else 0

    suite = load_suite(args.suite, ROOT_DIR)
    selected_ids = args.task_id or [
        task.task_id
        for task in suite.tasks
        if candidates.get(task.task_id, {}).get("capability_status") == "hardened"
    ]
    invalid_request = _invalid_cli_request(args, selected_ids)
    if invalid_request is not None:
        result = CapabilityResult(
            False,
            selected_ids[0] if selected_ids else None,
            invalid_request,
            "invalid_request",
        )
        _emit_capability(result.to_dict())
        return result.exit_code

    suite_ids = {task.task_id for task in suite.tasks}
    unknown = sorted(set(selected_ids) - suite_ids)
    missing = sorted(set(selected_ids) - set(candidates))
    if unknown:
        result = CapabilityResult(
            False,
            unknown[0],
            {"task_ids": selected_ids, "suite": suite.name},
            "unknown_task",
        )
        _emit_capability(result.to_dict())
        return result.exit_code
    if missing:
        result = CapabilityResult(
            False,
            missing[0],
            {"task_ids": selected_ids, "manifest": manifest_path.name},
            "unknown_task",
        )
        _emit_capability(result.to_dict())
        return result.exit_code

    capability_results: dict[str, dict[str, Any]] = {}
    requested_cases: dict[str, dict[str, Any]] = {}
    for task_id in selected_ids:
        request = _request_for_task(args, candidates[task_id])
        result = validate_candidate_request(
            capability_manifest, task_id, request
        )
        _emit_capability(result.to_dict())
        if not result.supported:
            return result.exit_code
        capability_results[task_id] = result.to_dict()
        requested_cases[task_id] = next(
            case
            for case in candidates[task_id]["supported_cases"]
            if case["shape"] == request["shape"]
        )

    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite output directory: {output_dir}")

    task_map = {task.task_id: task for task in suite.tasks}
    rows: list[dict[str, Any]] = []
    for task_id in selected_ids:
        task = task_map[task_id]
        candidate = candidates[task_id]
        task_output = output_dir / task_id
        command = [
            sys.executable,
            str(SCRIPT_DIR / "benchmark_cuda.py"),
            "--task-dir",
            str((ROOT_DIR / task.path).resolve()),
            "--task-id",
            task_id,
            "--kernel",
            task.name,
            "--candidate",
            str(candidate["source_path"]),
            "--candidate-name",
            candidate["name"],
            *_benchmark_target_arguments(args.gpu, args.sm),
            "--dtype",
            args.dtype,
            "--shape",
            json.dumps(requested_cases[task_id]["shape"], sort_keys=True),
            "--warmup",
            str(args.warmup),
            "--repetitions",
            str(args.repetitions),
            "--sessions",
            str(args.sessions),
            "--phase",
            args.phase,
            "--cooldown-seconds",
            str(args.cooldown_seconds),
            "--max-session-spread-percent",
            str(args.max_session_spread_percent),
            "--output-dir",
            str(task_output),
        ]
        for extra_driver in candidate["extra_driver_paths"]:
            command.extend(["--extra-correctness-driver", str(extra_driver)])
        for candidate_only_driver in candidate["candidate_only_driver_paths"]:
            command.extend(
                [
                    "--candidate-only-correctness-driver",
                    str(candidate_only_driver),
                ]
            )
        if args.ncu:
            command.append("--ncu")
        if (
            args.correctness_only
            or not requested_cases[task_id]["performance"]
            or args.portability_replay_from is not None
        ):
            command.append("--correctness-only")
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
        log_path = output_dir / f"launcher-{task_id}.log"
        log_path.write_text(
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
        comparison = summary.get("comparison") if summary else None
        if completed.returncode == 0:
            status = "completed"
        elif completed.returncode == 3:
            status = (
                "no_improvement"
                if summary
                and summary.get("stable")
                and comparison
                and comparison.get("formal_valid")
                else "inconclusive"
            )
        elif completed.returncode == 4:
            status = "blocked"
        else:
            status = "failed"
        rows.append(
            {
                "task_id": task_id,
                "kernel": task.name,
                "candidate": candidate["name"],
                "source": candidate["source"],
                "capability": capability_results[task_id],
                "requested_case": requested_cases[task_id],
                "portability_replay": (
                    {
                        "source_contract_arch": args.portability_replay_from,
                        "compile_arch": args.sm,
                        "label": "sm86_candidate_portability_replay_on_sm80",
                    }
                    if args.portability_replay_from is not None
                    else None
                ),
                "returncode": completed.returncode,
                "status": status,
                # benchmark_cuda emits summary only after every original and
                # normalized correctness executable has passed.
                "correct": bool(
                    summary
                    and (
                        comparison
                        or summary.get("validation_gates", {}).get("correctness")
                        == "passed"
                    )
                ),
                "stable": summary.get("stable") if summary else None,
                "performance_claim_allowed": summary.get(
                    "performance_claim_allowed"
                )
                if summary
                else False,
                "baseline_median_us": comparison.get("baseline_median_us")
                if comparison
                else None,
                "candidate_median_us": comparison.get("candidate_median_us")
                if comparison
                else None,
                "speedup": comparison.get("speedup") if comparison else None,
                "session_speedups": comparison.get("session_speedups")
                if comparison
                else None,
                "all_sessions_not_slower": comparison.get(
                    "all_sessions_not_slower"
                )
                if comparison
                else None,
                "summary": str(summary_path.relative_to(output_dir))
                if summary
                else None,
                "log": log_path.name,
            }
        )
        _atomic_json(
            output_dir / "suite_summary.json",
            {
                "schema_version": "2.0",
                "suite": suite.name,
                "complete": False,
                "results": rows,
            },
        )

    payload = {
        "schema_version": "2.0",
        "suite": suite.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.relative_to(ROOT_DIR)),
        "protocol": {
            "phase": args.phase,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "sessions": args.sessions,
            "cooldown_seconds": args.cooldown_seconds,
            "max_session_spread_percent": args.max_session_spread_percent,
            "ncu": args.ncu,
            "correctness_only": args.correctness_only,
            "gpu": args.gpu,
            "sm": args.sm,
            "dtype": args.dtype,
            "target_dtype": args.target_dtype,
            "layout": args.layout,
            "stream_mode": args.stream_mode,
            "stream_count": args.stream_count,
            "graph_capture": args.graph_capture,
            "backward": args.backward,
            "shape": args.shape,
            "portability_replay_from": args.portability_replay_from,
        },
        "results": rows,
        "complete": len(rows) == len(selected_ids),
        "completed_tasks": sum(row["status"] == "completed" for row in rows),
        "inconclusive_tasks": sum(row["status"] == "inconclusive" for row in rows),
        "no_improvement_tasks": sum(row["status"] == "no_improvement" for row in rows),
        "blocked_tasks": sum(row["status"] == "blocked" for row in rows),
        "failed_tasks": sum(row["status"] == "failed" for row in rows),
    }
    _atomic_json(output_dir / "suite_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["failed_tasks"]:
        return 2
    if payload["blocked_tasks"]:
        return 4
    if payload["inconclusive_tasks"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
