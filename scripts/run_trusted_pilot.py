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


ROOT_DIR = Path(__file__).resolve().parents[1]

NCU_PERMISSION_ERROR = "ERR_NVGPUCTRPERM"
OPENAI_BASE_URL = "https://api.openai.com/v1"
PILOT_MODEL = "gpt-5.6-sol"
LLM_SECRET_ENV_VARS = ("KERNELBLASTER_LLM_API_KEY", "OPENAI_API_KEY")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
    timeout: float,
):
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        completed = subprocess.CompletedProcess(
            command,
            124,
            stdout=str(error.stdout or ""),
            stderr=f"TimeoutExpired: exceeded {timeout} seconds",
        )
    except OSError as error:
        completed = subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=f"{type(error).__name__}: {error}",
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n"
        + json.dumps(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    return completed


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _environment_without_llm_secrets() -> dict[str, str]:
    environment = os.environ.copy()
    for name in LLM_SECRET_ENV_VARS:
        environment.pop(name, None)
    return environment


def _classify_ncu_probe(
    completed: subprocess.CompletedProcess[str],
) -> tuple[str, str | None]:
    """Classify the NCU gate without hiding setup or execution failures."""
    if completed.returncode == 0:
        return "available", "ncu"
    if completed.returncode not in {124, 127} and NCU_PERMISSION_ERROR in (
        completed.stdout + completed.stderr
    ):
        return "events_only", "events_only"
    return "failed", None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the gated RMSNorm Agent Pilot in the required startup order."
    )
    parser.add_argument(
        "--model",
        default=PILOT_MODEL,
        choices=(PILOT_MODEL,),
        help="Fixed trusted-pilot model; alternate model IDs are rejected.",
    )
    parser.add_argument("--gpu", default="rtx3080")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "out" / "trusted-pilot" / _timestamp(),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    stages: list[dict[str, Any]] = []
    profiler_environment = _environment_without_llm_secrets()

    # 1. Environment and dependency check.
    runtime = _run(
        [sys.executable, "scripts/check_runtime_versions.py", "--require-gpu"],
        log_path=output_dir / "01-runtime.log",
        env=profiler_environment,
        timeout=120,
    )
    stages.append({"stage": "runtime", "returncode": runtime.returncode})
    if runtime.returncode:
        _atomic_json(output_dir / "preflight.json", {"schema_version": "2.0", "stages": stages})
        return 2

    # 2-4. CUDA compilation, official/edge correctness and three-session Events stability.
    events_dir = output_dir / "events-discovery"
    events = _run(
        [
            sys.executable,
            "scripts/benchmark_candidates.py",
            "--task-id",
            "036",
            "--phase",
            "discovery",
            "--cooldown-seconds",
            "0",
            "--output-dir",
            str(events_dir),
        ],
        log_path=output_dir / "02-04-events.log",
        env=profiler_environment,
        timeout=3600,
    )
    stages.append({"stage": "compile_correctness_events", "returncode": events.returncode})
    if events.returncode:
        _atomic_json(output_dir / "preflight.json", {"schema_version": "2.0", "stages": stages})
        return 2

    # 5. NCU gate; only ERR_NVGPUCTRPERM may explicitly select events_only.
    executable = (
        events_dir
        / "036"
        / "build"
        / "rmsnorm_v3c"
        / "benchmark"
        / "build"
        / "main"
    )
    ncu = _run(
        ["ncu", "--section", "SpeedOfLight", str(executable)],
        log_path=output_dir / "05-ncu-probe.log",
        env=profiler_environment,
        timeout=600,
    )
    ncu_status, profiling_mode = _classify_ncu_probe(ncu)
    stages.append(
        {
            "stage": "ncu_permission_probe",
            "returncode": ncu.returncode,
            "status": ncu_status,
            "profiling_mode": profiling_mode,
        }
    )
    if profiling_mode is None:
        _atomic_json(
            output_dir / "preflight.json",
            {
                "schema_version": "2.0",
                "profiling_mode": None,
                "stages": stages,
            },
        )
        return 2

    # 6. Exactly one bounded authentication smoke request.
    smoke = _run(
        [
            sys.executable,
            "scripts/smoke_llm.py",
            "--model",
            args.model,
            "--base-url",
            OPENAI_BASE_URL,
            "--max-completion-tokens",
            "64",
            "--max-total-tokens",
            "10000",
            "--reasoning-effort",
            "none",
            "--output-dir",
            str(output_dir / "api-smoke"),
        ],
        log_path=output_dir / "06-api-smoke.log",
        timeout=300,
    )
    stages.append({"stage": "api_smoke", "returncode": smoke.returncode})
    if smoke.returncode:
        _atomic_json(
            output_dir / "preflight.json",
            {
                "schema_version": "2.0",
                "profiling_mode": profiling_mode,
                "stages": stages,
            },
        )
        return 2

    # 7. RMSNorm Pilot only: 2 rollouts x 2 steps, bounded to 32 requests/250k tokens.
    environment = os.environ.copy()
    environment.update(
        {
            "LLM_MAX_REQUESTS": "32",
            "LLM_MAX_TOTAL_TOKENS": "250000",
            "LLM_MAX_CONCURRENCY": "2",
            "LLM_MAX_RETRIES": "2",
            "LLM_REASONING_EFFORT": "low",
            "KERNELBLASTER_LLM_PROVIDER": "openai_compatible",
            "KERNELBLASTER_LLM_BASE_URL": OPENAI_BASE_URL,
            "MODEL": args.model,
        }
    )
    pilot = _run(
        [
            sys.executable,
            "scripts/run_RL.py",
            "--experiment-name",
            "trusted-rmsnorm-pilot",
            "--dataset",
            "kernelbench-cuda",
            "--precision",
            "fp16",
            "--cuda",
            "--cuda-perf",
            "--use-rl",
            "--rl-iterations",
            "2",
            "--rl-rollout-steps",
            "2",
            "--rl-buffer-size",
            "16",
            "--rl-update-frequency",
            "2",
            "--concurrency",
            "1",
            "--problem-numbers",
            "36",
            "--portfolio-suite",
            "portfolio/suites/rmsnorm.json",
            "--subset",
            "level1",
            "--gpu",
            args.gpu,
            "--model",
            args.model,
            "--run-record-dir",
            str(output_dir / "pilot-record"),
        ],
        log_path=output_dir / "07-pilot.log",
        env=environment,
        timeout=7200,
    )
    stages.append({"stage": "agent_pilot", "returncode": pilot.returncode})
    _atomic_json(
        output_dir / "preflight.json",
        {
            "schema_version": "2.0",
            "profiling_mode": profiling_mode,
            "stages": stages,
        },
    )
    return pilot.returncode


if __name__ == "__main__":
    raise SystemExit(main())
