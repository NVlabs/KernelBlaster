#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.kernelblaster.config import config  # noqa: E402
from src.kernelblaster.observability import RunRecorder  # noqa: E402
from src.kernelblaster.portfolio import load_suite  # noqa: E402
from src.kernelblaster.storage import StateStore, state_storage_requested  # noqa: E402


GPU_TARGETS = (
    "rtx3080",
    "a100",
    "a6000",
    "l40",
    "l40s",
    "l40g",
    "rtx4090",
    "rtx5000",
    "rtx6000",
    "h100",
    "h200",
    "b200",
)


def _default_output_dir(suite_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT_DIR / "out" / "portfolio" / suite_name / timestamp


def _public_provider_config() -> dict:
    return {
        "provider": config.LLM_PROVIDER,
        "base_url": config.LLM_BASE_URL_PUBLIC,
        "timeout_seconds": config.LLM_REQUEST_TIMEOUT_SECONDS,
        "max_concurrency": config.LLM_MAX_CONCURRENCY,
        "max_retries": config.LLM_MAX_RETRIES,
        "max_requests": config.LLM_MAX_REQUESTS,
        "max_total_tokens": config.LLM_MAX_TOTAL_TOKENS,
        "max_completion_tokens": config.LLM_MAX_COMPLETION_TOKENS,
        "reasoning_effort": config.LLM_REASONING_EFFORT or None,
        "stream": config.STREAM.lower() in ("true", "1", "yes", "y", "on"),
        "fanout_mode": "client",
        "api_key_configured": bool(config.API_KEY),
        "log_content": config.LLM_LOG_CONTENT,
    }


def _ensure_new_artifact_dir(output_dir: Path) -> None:
    artifact_names = ("run_manifest.json", "events.jsonl", "summary.json")
    existing = [name for name in artifact_names if (output_dir / name).exists()]
    if existing:
        raise ValueError(
            f"Output directory already contains run artifacts: {', '.join(existing)}"
        )


def _build_command(args, suite, output_dir: Path, rollouts: int, steps: int) -> list[str]:
    experiment_name = f"portfolio-{suite.name}-{output_dir.name}"
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_RL.py"),
        "--experiment-name",
        experiment_name,
        "--dataset",
        "kernelbench-cuda",
        "--precision",
        suite.precision,
        "--cuda",
        "--cuda-perf",
        "--use-rl",
        "--rl-iterations",
        str(rollouts),
        "--rl-rollout-steps",
        str(steps),
        "--rl-buffer-size",
        "100",
        "--rl-update-frequency",
        "3",
        "--concurrency",
        "1",
        "--problem-numbers",
        suite.problem_numbers,
        "--subset",
        suite.subset,
        "--gpu",
        args.gpu,
        "--model",
        args.model,
        "--run-record-dir",
        str(output_dir),
        "--portfolio-suite",
        str(suite.source_path),
    ]
    for option, value in (
        ("--state-dir", args.state_dir),
        ("--sqlite-path", args.sqlite_path),
        ("--cas-dir", args.cas_dir),
    ):
        if value is not None:
            command.extend((option, str(value)))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a reproducible KernelBlaster portfolio suite."
    )
    parser.add_argument(
        "--suite",
        default="core10",
        help="Suite alias from portfolio/suites or a JSON suite path.",
    )
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--gpu", choices=GPU_TARGETS, default="l40s")
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=None)
    parser.add_argument("--cas-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and write artifacts without API or CUDA calls.",
    )
    args = parser.parse_args()

    try:
        suite = load_suite(args.suite, ROOT_DIR)
        rollouts = args.rollouts if args.rollouts is not None else suite.rollouts
        steps = args.steps if args.steps is not None else suite.steps
        if rollouts < 1 or steps < 1:
            raise ValueError("--rollouts and --steps must both be positive.")
        if not args.model.strip():
            raise ValueError("--model cannot be empty.")

        output_dir = (args.output_dir or _default_output_dir(suite.name)).resolve()
        _ensure_new_artifact_dir(output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    resolved_suite = dict(suite.raw)
    try:
        suite_source = str(suite.source_path.relative_to(ROOT_DIR))
    except ValueError:
        suite_source = str(suite.source_path)
    resolved_suite["source"] = suite_source
    resolved_suite["resolved"] = {"rollouts": rollouts, "steps": steps}

    state_store = None
    if (
        args.state_dir is not None
        or args.sqlite_path is not None
        or args.cas_dir is not None
        or state_storage_requested()
    ):
        try:
            state_store = StateStore(
                state_dir=args.state_dir,
                sqlite_path=args.sqlite_path,
                cas_dir=args.cas_dir,
            )
        except (OSError, PermissionError, ValueError) as error:
            parser.error(str(error))

    if args.dry_run:
        recorder = RunRecorder(
            output_dir,
            model=args.model,
            provider_config=_public_provider_config(),
            suite=resolved_suite,
            gpu_target=args.gpu,
            dry_run=True,
            repo_root=ROOT_DIR,
            state_store=state_store,
        )
        recorder.record_event(
            "portfolio_dry_run_resolved",
            data={
                "suite": suite.name,
                "task_count": len(suite.tasks),
                "rollouts": rollouts,
                "steps": steps,
                "network_calls": 0,
                "cuda_calls": 0,
            },
        )
        recorder.close("dry_run")
        print(f"Dry-run artifacts written to {output_dir}")
        return 0

    if not config.API_KEY:
        parser.error(
            "Execution requires KERNELBLASTER_LLM_API_KEY or OPENAI_API_KEY. "
            "Use --dry-run to resolve the suite without credentials."
        )

    command = _build_command(args, suite, output_dir, rollouts, steps)
    environment = os.environ.copy()
    environment["MODEL"] = args.model
    completed = subprocess.run(command, cwd=ROOT_DIR, env=environment, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
