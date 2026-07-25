#!/usr/bin/env python3
"""CUDA Events benchmark for the PyTorch references of KernelBlaster Core 10."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.kernelblaster.benchmarking import (  # noqa: E402
    BENCHMARK_SCHEMA_VERSION,
    latency_summary,
    session_spread_percent,
)


WORKER_MARKER = "KERNELBLASTER_PYTORCH_JSON "
CORE10_TASKS: dict[str, dict[str, str]] = {
    "004": {"kernel": "Matrix-vector multiplication", "shape": "A[256,131072] @ B[131072,1]"},
    "007": {"kernel": "Small-K matrix multiplication", "shape": "A[16384,32] @ B[32,16384]"},
    "019": {"kernel": "ReLU", "shape": "[16,16384]"},
    "023": {"kernel": "Softmax", "shape": "[16,16384], dim=1"},
    "026": {"kernel": "GELU", "shape": "[16,16384], approximate=none"},
    "036": {"kernel": "RMSNorm", "shape": "[16,64,256,256], reduce dim=1"},
    "040": {"kernel": "LayerNorm", "shape": "[16,64,256,256], normalized_shape=[64,256,256]"},
    "047": {"kernel": "Sum reduction", "shape": "[16,256,256], dim=1, keepdim=true"},
    "088": {"kernel": "MinGPT GELU", "shape": "[2000,2000]"},
    "095": {"kernel": "Cross entropy loss", "shape": "logits[4096,10], targets[4096]"},
}
PYTORCH_METHODS: dict[str, list[str]] = {
    "004": ["pytorch_eager", "pytorch_preallocated_out"],
    "007": ["pytorch_eager", "pytorch_preallocated_out"],
    "019": ["pytorch_eager", "pytorch_preallocated_out"],
    "023": ["pytorch_eager", "pytorch_preallocated_out"],
    "026": ["pytorch_eager"],
    "036": ["pytorch_eager"],
    "040": ["pytorch_eager"],
    "047": ["pytorch_eager", "pytorch_preallocated_out"],
    "088": ["pytorch_driver_formula", "pytorch_fused_gelu_tanh"],
    "095": ["pytorch_eager"],
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _telemetry() -> dict[str, Any]:
    fields = "name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error_type": type(error).__name__}
    values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    return {"available": True, **dict(zip(fields.split(","), values, strict=False))}


def _mingpt_formula(torch: Any, value: Any) -> Any:
    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * value * (
        1.0 + torch.tanh(coefficient * (value + 0.044715 * value.pow(3)))
    )


def _task_methods(task_id: str, torch: Any) -> dict[str, dict[str, Any]]:
    """Allocate fixed Driver-equivalent inputs and return timed callables."""
    device = torch.device("cuda")
    dtype = torch.float16
    if task_id == "004":
        a = torch.randn((256, 131072), device=device, dtype=dtype)
        b = torch.randn((131072, 1), device=device, dtype=dtype)
        output = torch.empty((256, 1), device=device, dtype=dtype)
        reference = lambda: torch.matmul(a, b)
        run = lambda: torch.ops.aten.mm.out(a, b, out=output)
        methods = {
            "pytorch_eager": (reference, reference, "framework_allocated"),
            "pytorch_preallocated_out": (run, reference, "preallocated_out"),
        }
    elif task_id == "007":
        a = torch.randn((16384, 32), device=device, dtype=dtype)
        b = torch.randn((32, 16384), device=device, dtype=dtype)
        output = torch.empty((16384, 16384), device=device, dtype=dtype)
        reference = lambda: torch.matmul(a, b)
        run = lambda: torch.ops.aten.mm.out(a, b, out=output)
        methods = {
            "pytorch_eager": (reference, reference, "framework_allocated"),
            "pytorch_preallocated_out": (run, reference, "preallocated_out"),
        }
    elif task_id == "019":
        value = torch.randn((16, 16384), device=device, dtype=dtype)
        output = torch.empty_like(value)
        reference = lambda: torch.relu(value)
        run = lambda: torch.ops.aten.relu.out(value, out=output)
        methods = {
            "pytorch_eager": (reference, reference, "framework_allocated"),
            "pytorch_preallocated_out": (run, reference, "preallocated_out"),
        }
    elif task_id == "023":
        value = torch.randn((16, 16384), device=device, dtype=dtype)
        output = torch.empty_like(value)
        reference = lambda: torch.softmax(value, dim=1)
        run = lambda: torch.ops.aten._softmax.out(
            value, 1, False, out=output
        )
        methods = {
            "pytorch_eager": (reference, reference, "framework_allocated"),
            "pytorch_preallocated_out": (run, reference, "preallocated_out"),
        }
    elif task_id == "026":
        value = torch.randn((16, 16384), device=device, dtype=dtype)
        reference = lambda: torch.nn.functional.gelu(value)
        methods = {"pytorch_eager": (reference, reference, "framework_allocated")}
    elif task_id == "036":
        value = torch.randn((16, 64, 256, 256), device=device, dtype=dtype)

        def reference() -> Any:
            return value / (value.square().mean(dim=1, keepdim=True) + 1.0e-5).sqrt()

        methods = {"pytorch_eager": (reference, reference, "framework_allocated_multiop")}
    elif task_id == "040":
        value = torch.randn((16, 64, 256, 256), device=device, dtype=dtype)
        normalized_shape = (64, 256, 256)
        weight = torch.ones(normalized_shape, device=device, dtype=dtype)
        bias = torch.zeros(normalized_shape, device=device, dtype=dtype)
        reference = lambda: torch.nn.functional.layer_norm(
            value, normalized_shape, weight, bias
        )
        methods = {"pytorch_eager": (reference, reference, "framework_allocated")}
    elif task_id == "047":
        value = torch.randn((16, 256, 256), device=device, dtype=dtype)
        output = torch.empty((16, 1, 256), device=device, dtype=dtype)
        reference = lambda: torch.sum(value, dim=1, keepdim=True)
        run = lambda: torch.ops.aten.sum.IntList_out(
            value, [1], True, dtype=None, out=output
        )
        methods = {
            "pytorch_eager": (reference, reference, "framework_allocated"),
            "pytorch_preallocated_out": (run, reference, "preallocated_out"),
        }
    elif task_id == "088":
        value = torch.randn((2000, 2000), device=device, dtype=dtype)
        reference = lambda: _mingpt_formula(torch, value)
        methods = {
            "pytorch_driver_formula": (reference, reference, "framework_allocated_multiop"),
            "pytorch_fused_gelu_tanh": (
                lambda: torch.nn.functional.gelu(value, approximate="tanh"),
                reference,
                "framework_allocated_equivalent",
            ),
        }
    elif task_id == "095":
        predictions = torch.randn((4096, 10), device=device, dtype=dtype)
        targets = torch.randint(0, 10, (4096,), device=device, dtype=torch.int64)
        reference = lambda: torch.nn.functional.cross_entropy(predictions, targets)
        methods = {"pytorch_eager": (reference, reference, "framework_allocated")}
    else:
        raise ValueError(f"Unknown Core 10 task: {task_id}")
    return {
        name: {"run": run, "reference": reference, "allocation_mode": allocation}
        for name, (run, reference, allocation) in methods.items()
    }


def _worker(args: argparse.Namespace) -> int:
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    method = _task_methods(args.worker_task, torch)[args.worker_method]
    run: Callable[[], Any] = method["run"]
    expected = method["reference"]()
    actual = run()
    torch.cuda.synchronize()
    correct = bool(torch.allclose(actual, expected, rtol=1.0e-1, atol=1.0e-1))
    del actual, expected
    if not correct:
        raise RuntimeError("PyTorch method did not match the task reference.")

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    inner_loops = args.inner_loops
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    if inner_loops == 0:
        calibration_loops = 10
        start.record()
        for _ in range(calibration_loops):
            run()
        stop.record()
        stop.synchronize()
        single_us = max(
            0.001, start.elapsed_time(stop) * 1000.0 / calibration_loops
        )
        inner_loops = min(10_000, max(1, math.ceil(1000.0 / single_us)))

    samples_us: list[float] = []
    before = _telemetry()
    for _ in range(args.repetitions):
        start.record()
        for _ in range(inner_loops):
            run()
        stop.record()
        stop.synchronize()
        samples_us.append(start.elapsed_time(stop) * 1000.0 / inner_loops)
    after = _telemetry()
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "task_id": args.worker_task,
        "kernel": CORE10_TASKS[args.worker_task]["kernel"],
        "shape": CORE10_TASKS[args.worker_task]["shape"],
        "method": args.worker_method,
        "allocation_mode": method["allocation_mode"],
        "session": args.worker_session,
        "seed": args.seed,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "inner_loops": inner_loops,
        "correct": correct,
        "samples_us": samples_us,
        "latency": latency_summary(samples_us),
        "telemetry_before": before,
        "telemetry_after": after,
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
    }
    print(WORKER_MARKER + json.dumps(payload, sort_keys=True))
    return 0


def _parse_worker(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith(WORKER_MARKER)]
    if len(lines) != 1:
        raise RuntimeError("PyTorch worker did not emit exactly one result marker.")
    return json.loads(lines[0][len(WORKER_MARKER) :])


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _worker_command(
    args: argparse.Namespace, task_id: str, method: str, session: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-task",
        task_id,
        "--worker-method",
        method,
        "--worker-session",
        str(session),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
        "--inner-loops",
        str(args.inner_loops),
        "--seed",
        str(args.seed + session),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Core 10 PyTorch references on one CUDA GPU."
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--inner-loops", type=int, default=0)
    parser.add_argument(
        "--phase",
        choices=("discovery", "confirmation"),
        default="confirmation",
        help="Discovery uses 3 sessions; formal confirmation uses 5.",
    )
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-session-spread-percent", type=float, default=5.0)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "out" / "portfolio" / "pytorch" / _timestamp(),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-task", help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-session", type=int, default=0, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.sessions is None:
        args.sessions = 3 if args.phase == "discovery" else 5
    if args.worker:
        return _worker(args)
    if min(args.warmup, args.repetitions, args.sessions) < 1 or args.inner_loops < 0:
        parser.error("Warmup/repetitions/sessions must be positive; inner-loops may be zero.")

    selected = args.task_id or list(CORE10_TASKS)
    unknown = sorted(set(selected) - set(CORE10_TASKS))
    if unknown:
        parser.error(f"Unknown Core 10 task IDs: {unknown}")
    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite output directory: {output_dir}")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    jsonl_path = output_dir / "measurements.jsonl"
    commit = _git_commit()
    for task_id in selected:
        methods = PYTORCH_METHODS[task_id]
        if args.reference_only:
            methods = methods[:1]
        for session in range(args.sessions):
            order = methods if session % 2 == 0 else list(reversed(methods))
            for method in order:
                command = _worker_command(args, task_id, method, session)
                completed = subprocess.run(
                    command, cwd=ROOT_DIR, check=False, capture_output=True, text=True
                )
                log = output_dir / f"worker-{task_id}-{session}-{method}.log"
                log.write_text(
                    "COMMAND\n"
                    + json.dumps(command)
                    + "\n\nSTDOUT\n"
                    + completed.stdout
                    + "\n\nSTDERR\n"
                    + completed.stderr,
                    encoding="utf-8",
                )
                if completed.returncode != 0:
                    failures.append(
                        {
                            "task_id": task_id,
                            "method": method,
                            "session": session,
                            "returncode": completed.returncode,
                            "log": log.name,
                        }
                    )
                    continue
                record = _parse_worker(completed.stdout)
                record.update(
                    {
                        "order": "AB" if order == methods else "BA",
                        "git_commit": commit,
                        "container_image": os.getenv("KERNELBLASTER_CONTAINER_IMAGE"),
                        "container_digest": os.getenv("KERNELBLASTER_CONTAINER_DIGEST"),
                    }
                )
                records.append(record)
                with jsonl_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")

    rows: list[dict[str, Any]] = []
    for task_id in selected:
        methods = sorted(
            {record["method"] for record in records if record["task_id"] == task_id}
        )
        for method in methods:
            chosen = [
                record
                for record in records
                if record["task_id"] == task_id and record["method"] == method
            ]
            medians = [record["latency"]["median_us"] for record in chosen]
            all_samples = [
                sample for record in chosen for sample in record["samples_us"]
            ]
            spread = session_spread_percent(medians) if medians else None
            rows.append(
                {
                    "task_id": task_id,
                    "kernel": CORE10_TASKS[task_id]["kernel"],
                    "shape": CORE10_TASKS[task_id]["shape"],
                    "method": method,
                    "allocation_mode": chosen[0]["allocation_mode"] if chosen else None,
                    "sessions": len(chosen),
                    "session_medians_us": medians,
                    "median_us": latency_summary(medians)["median_us"] if medians else None,
                    "p10_us": latency_summary(all_samples)["p10_us"] if all_samples else None,
                    "p90_us": latency_summary(all_samples)["p90_us"] if all_samples else None,
                    "session_spread_percent": spread,
                    "stable": spread is not None
                    and spread <= args.max_session_spread_percent,
                    "correct": bool(chosen)
                    and all(record["correct"] for record in chosen),
                }
            )

    flat_rows = [
        {**row, "session_medians_us": json.dumps(row["session_medians_us"])}
        for row in rows
    ]
    if flat_rows:
        with (output_dir / "pytorch_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    expected_methods = sum(
        1 if args.reference_only else len(PYTORCH_METHODS[task_id])
        for task_id in selected
    )
    complete = (
        not failures
        and len(rows) == expected_methods
        and all(row["sessions"] == args.sessions and row["correct"] for row in rows)
    )
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "protocol": {
            "phase": args.phase,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "sessions": args.sessions,
            "inner_loops": args.inner_loops,
            "max_session_spread_percent": args.max_session_spread_percent,
        },
        "results": rows,
        "failures": failures,
        "complete": complete,
    }
    _atomic_json(output_dir / "pytorch_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
