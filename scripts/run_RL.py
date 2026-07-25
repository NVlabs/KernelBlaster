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
import argparse
import asyncio
import os
import sys
import signal
import glob
import shutil
from loguru import logger
from pathlib import Path
import json

# Add parent directory to path so we can import from root
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.kernelblaster.config import config, GPUType
from src.kernelblaster.llm import get_llm_provider
from src.kernelblaster.observability import (
    RunRecorder,
    event_context,
    record_event,
    set_run_recorder,
)
from src.kernelblaster.outcomes import RunStatus
from src.kernelblaster.resources import *
from src.kernelblaster.storage import StateStore, state_storage_requested
from src.kernelblaster.workflow import run_workflow
from src.kernelblaster.agents.database import LLMInterface, OptimizationDatabase

from data import get_dataset
from utils.arguments import *

COMPILE_SERVER = None
GPU_SERVER = None
CLEANUP_IN_PROGRESS = False
SIGNAL_COUNT = 0
COMPREHENSIVE_ANALYSIS_CACHE = None
RUN_RECORDER = None
TRUSTED_RMSNORM_SUITE = (
    ROOT_DIR / "portfolio" / "suites" / "rmsnorm.json"
).resolve()
TRUSTED_RMSNORM_TASK_IDS = ("036",)
TRUSTED_RMSNORM_ROLLOUTS = 2
TRUSTED_RMSNORM_STEPS = 2


def resolve_target_gpu(gpu: str | None) -> GPUType:
    """Resolve the GPU whose server URL should be configured for this run."""
    return GPUType(gpu) if gpu is not None else GPUType.current()


def _normalize_task_id(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Invalid portfolio task ID: {value!r}")
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f"Invalid portfolio task ID: {value!r}")
    task_number = int(text)
    if task_number < 1:
        raise ValueError(f"Invalid portfolio task ID: {value!r}")
    return f"{task_number:03d}"


def _parse_problem_ids(problem_numbers: str | None) -> list[str]:
    if not problem_numbers:
        raise ValueError(
            "--portfolio-suite requires explicit --problem-numbers for validation."
        )
    task_ids: list[str] = []
    for raw_part in problem_numbers.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("--problem-numbers contains an empty entry.")
        if "-" in part and not part.startswith("-"):
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending problem range: {part!r}")
            task_ids.extend(_normalize_task_id(number) for number in range(start, end + 1))
        else:
            task_ids.append(_normalize_task_id(part))
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("--problem-numbers contains duplicate tasks.")
    return task_ids


def resolve_portfolio_suite(
    path: Path,
    *,
    problem_numbers: str | None,
    rollouts: int,
    steps: int,
    trusted_pilot: bool = False,
) -> dict:
    """Validate a suite against the requested task set and resolved pilot shape."""
    resolved_path = path.resolve()
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load portfolio suite {resolved_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Portfolio suite must be a JSON object.")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Portfolio suite must contain a non-empty tasks list.")

    task_ids: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Every portfolio suite task must be an object.")
        raw_id = task.get("id", task.get("number"))
        task_id = _normalize_task_id(raw_id)
        if "number" in task and _normalize_task_id(task["number"]) != task_id:
            raise ValueError(f"Portfolio task id/number disagree for {task!r}.")
        task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Portfolio suite contains duplicate task IDs.")

    requested_task_ids = _parse_problem_ids(problem_numbers)
    if set(task_ids) != set(requested_task_ids):
        raise ValueError(
            "Portfolio suite task IDs do not match --problem-numbers: "
            f"suite={task_ids}, requested={requested_task_ids}."
        )

    if trusted_pilot and (
        resolved_path != TRUSTED_RMSNORM_SUITE
        or tuple(task_ids) != TRUSTED_RMSNORM_TASK_IDS
        or rollouts != TRUSTED_RMSNORM_ROLLOUTS
        or steps != TRUSTED_RMSNORM_STEPS
    ):
        raise ValueError(
            "Trusted RMSNorm pilot must use portfolio/suites/rmsnorm.json and "
            "resolve to task 036 with 2 rollouts x 2 steps."
        )

    try:
        suite_source = str(resolved_path.relative_to(ROOT_DIR.resolve()))
    except ValueError:
        suite_source = str(resolved_path)
    payload["source"] = suite_source
    payload["resolved"] = {
        "task_ids": task_ids,
        "rollouts": rollouts,
        "steps": steps,
    }
    return payload


def load_comprehensive_analysis_results():
    """
    Load and cache comprehensive analysis results from all JSON files.
    
    Returns:
        dict: Dictionary with op_name as key and list of matching entries as value
    """
    global COMPREHENSIVE_ANALYSIS_CACHE
    
    if COMPREHENSIVE_ANALYSIS_CACHE is not None:
        return COMPREHENSIVE_ANALYSIS_CACHE
    
    logger.info("Loading comprehensive analysis results...")
    
    # Find all JSON files in the comprehensive_analysis_results directory
    analysis_dir = ROOT_DIR / "comprehensive_analysis_results"
    json_files = glob.glob(str(analysis_dir / "detailed_analysis_chunk_*.json"))
    
    # Dictionary to store results indexed by op_name
    analysis_data = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for entry in data:
                op_name = entry.get("metadata", {}).get("op_name", "")
                if op_name:
                    if op_name not in analysis_data:
                        analysis_data[op_name] = []
                    analysis_data[op_name].append(entry)
                    
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
    
    COMPREHENSIVE_ANALYSIS_CACHE = analysis_data
    logger.info(f"Loaded analysis results for {len(analysis_data)} operations")
    
    return analysis_data


def find_matching_optimization_data(task_id, level_id, op_name):
    """
    Find matching optimization data based on task_id, level_id, and op_name.
    
    Args:
        task_id: Task ID to match
        level_id: Level ID to match  
        op_name: Operation name to match
        
    Returns:
        dict: Best matching entry or None if no match found
    """
    analysis_data = load_comprehensive_analysis_results()
    
    if op_name not in analysis_data:
        return None
    
    # Find entries that match the op_name
    matching_entries = analysis_data[op_name]
    
    # Filter by level_id and task_id if provided
    best_match = None
    best_score = -1
    
    for entry in matching_entries:
        metadata = entry.get("metadata", {})
        entry_level_id = metadata.get("level_id")
        entry_task_id = metadata.get("task_id")
        
        # Calculate match score
        score = 0
        if entry_level_id == level_id:
            score += 10
        if entry_task_id == task_id:
            score += 10
        
        # Prefer entries with higher quality scores
        quality_score = metadata.get("quality_score", 0)
        score += quality_score
        
        if score > best_score:
            best_score = score
            best_match = entry
    
    return best_match


def enhance_user_message_with_optimization_data(user_message, task_id, level_id, op_name):
    """
    Enhance the user message with optimization data from comprehensive analysis results.
    
    Args:
        user_message: Original user message
        task_id: Task ID
        level_id: Level ID
        op_name: Operation name
        
    Returns:
        str: Enhanced user message with optimization data
    """
    optimization_data = find_matching_optimization_data(task_id, level_id, op_name)
    
    if not optimization_data:
        logger.debug(f"No optimization data found for op_name: {op_name}, level: {level_id}, task: {task_id}")
        return user_message
    
    # Extract optimization information
    file_path = optimization_data.get("implementation_info", {}).get("file_path", "")
    optimizations_detected = optimization_data.get("cuda_analysis", {}).get("optimizations_detected", [])
    memory_patterns = optimization_data.get("cuda_analysis", {}).get("memory_patterns", [])
    thread_patterns = optimization_data.get("cuda_analysis", {}).get("thread_patterns", [])
    quality_score = optimization_data.get("metadata", {}).get("quality_score", 0)
    complexity_score = optimization_data.get("cuda_analysis", {}).get("complexity_score", 0)
    lines_of_code = optimization_data.get("cuda_analysis", {}).get("lines_of_code", 0)
    
    # Try to read the reference file content
    reference_code_content = ""
    if file_path and Path(file_path).exists():
        try:
            with open(file_path, 'r') as f:
                reference_code_content = f.read()[:2000]  # Limit to first 2000 chars
                if len(reference_code_content) == 2000:
                    reference_code_content += "\n... (truncated for brevity)"
        except Exception as e:
            logger.warning(f"Could not read reference file {file_path}: {e}")
    
    # Build optimization context
    optimization_context = f"""

## Optimization Context from Analysis Results

Based on previous analysis of similar kernels for operation '{op_name}':

**Reference Implementation:** {file_path}
**Quality Score:** {quality_score:.4f}
**Complexity Score:** {complexity_score:.1f}
**Lines of Code:** {lines_of_code}

**Detected Optimizations:**
{chr(10).join(f"- {opt}" for opt in optimizations_detected) if optimizations_detected else "- None detected"}

**Memory Patterns:**
{chr(10).join(f"- {pattern}" for pattern in memory_patterns) if memory_patterns else "- Standard memory access patterns"}

**Thread Patterns:**
{chr(10).join(f"- {pattern}" for pattern in thread_patterns) if thread_patterns else "- Standard thread organization"}

**Optimization Recommendations:**
Consider implementing similar optimization techniques in your solution, particularly:
{chr(10).join(f"- {opt}" for opt in optimizations_detected[:3]) if optimizations_detected else "- Focus on memory coalescing and thread utilization"}

"""
    
    # Add reference code if available
    if reference_code_content:
        optimization_context += f"""
**Reference Code Sample:**
```cuda
{reference_code_content}
```

"""
    
    # Add optimization context to user message
    enhanced_message = user_message + optimization_context
    
    logger.info(f"Enhanced user message with optimization data for {op_name} (quality: {quality_score:.4f})")
    
    return enhanced_message


def cleanup_servers():
    """Clean up servers on exit."""
    global COMPILE_SERVER, GPU_SERVER, CLEANUP_IN_PROGRESS
    
    if CLEANUP_IN_PROGRESS:
        return
    
    CLEANUP_IN_PROGRESS = True
    
    try:
        if COMPILE_SERVER is not None:
            logger.info("Cleaning up compiler server...")
            # Add timeout to prevent hanging
            import threading
            cleanup_thread = threading.Thread(target=COMPILE_SERVER.cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            cleanup_thread.join(timeout=5.0)  # 5 second timeout
            if cleanup_thread.is_alive():
                logger.warning("Compiler server cleanup timed out")
        
        if GPU_SERVER is not None:
            logger.info("Cleaning up GPU server...")
            # Add timeout to prevent hanging
            cleanup_thread = threading.Thread(target=GPU_SERVER.cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            cleanup_thread.join(timeout=5.0)  # 5 second timeout
            if cleanup_thread.is_alive():
                logger.warning("GPU server cleanup timed out")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        CLEANUP_IN_PROGRESS = False


def signal_handler(signum, frame):
    """Handle termination signals."""
    global SIGNAL_COUNT
    SIGNAL_COUNT += 1
    
    if SIGNAL_COUNT == 1:
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_servers()
        logger.info("Cleanup complete, exiting...")
        sys.exit(0)
    elif SIGNAL_COUNT == 2:
        logger.warning("Received second signal, forcing exit...")
        sys.exit(1)
    else:
        logger.error("Received multiple signals, forcing immediate exit...")
        os._exit(1)


async def process_problem(
    entry,
    folder,
    semaphore,
    workflow_config,
    timeout_minutes,
) -> tuple[dict[str, Path], RunStatus]:
    problem_id = entry["id"]
    user_message = entry.get("user_message", "")
    reference_code = None

    # Extract task information for optimization data lookup
    task_id = entry.get("task_id")
    level_id = entry.get("level_id")
    op_name = entry.get("op_name")
    
    # If not directly available, try to extract from entry structure
    if task_id is None and "problem_num" in entry:
        task_id = entry["problem_num"]
    
    if level_id is None and "level" in entry:
        level_str = entry["level"]
        if level_str and "level" in level_str:
            level_id = int(level_str.replace("level", ""))
    
    if op_name is None and task_id is not None:
        # Try to construct op_name from problem information
        # Extract operation name from problem_name or id
        problem_name = entry.get("problem_name", "")
        if problem_name:
            # Remove numeric prefix and underscores to get operation name
            parts = problem_name.split("_")
            if len(parts) > 1:
                operation_name = "_".join(parts[1:])  # Skip the numeric prefix
                op_name = f"{task_id}_{operation_name}"
        
        # Fallback: use just the task_id if we can't determine operation name
        if op_name is None:
            op_name = str(task_id)
    
    # Enhance user message with optimization data if available
    if op_name and task_id is not None and level_id is not None:
        user_message = enhance_user_message_with_optimization_data(
            user_message, task_id, level_id, op_name
        )

    job_logger = logger.bind(problem_id=problem_id)
    async with semaphore:
        job_logger_id = job_logger.add(
            folder / "run.log",
            level=config.LOG_LEVEL,
            backtrace=True,
            diagnose=True,
            format=config.CUSTOM_LOGGER_FORMAT,
            filter=lambda record: record["extra"].get("problem_id") == problem_id,
        )
        with event_context(task_id=task_id or problem_id, stage="workflow"):
            result = await run_workflow(
                problem_id,
                user_message,
                reference_code,
                folder,
                workflow_config,
                job_logger=job_logger,
                timeout_seconds=timeout_minutes * 60,
                shared_database=workflow_config.shared_optimization_database,
            )
        if result.success:
            logger.info(
                f"Successfully generated codes for {problem_id}:\n{json.dumps(result.generated_codes, indent=2)}"
            )
        else:
            logger.error(
                f"❌ Failed to generate codes for {problem_id}: {result.error}"
            )
        record_event(
            "task_outcome",
            status=(
                "ok"
                if result.outcome.status.value in {"improved", "no_improvement"}
                else "error"
            ),
            task_id=task_id or problem_id,
            stage="terminal",
            data={
                "task_id": task_id or problem_id,
                "outcome": result.outcome.status.value,
                "profiling_mode": result.outcome.profiling_mode,
                "reason": result.outcome.reason,
                "reason_code": result.outcome.reason_code.value,
                "execution_status": result.outcome.execution_status.value,
                "correctness_status": result.outcome.correctness_status.value,
                "timing_status": result.outcome.timing_status.value,
                "diagnostic_status": result.outcome.diagnostic_status.value,
                "measurement": (
                    result.outcome.measurement.to_dict()
                    if result.outcome.measurement is not None else None
                ),
                "metrics": result.outcome.metrics,
            },
        )
        job_logger.remove(job_logger_id)
    return result.generated_codes, result.outcome.status


async def async_main() -> int:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of problems to process in parallel",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing results.",
    )
    parser.add_argument(
        "--compiler-port",
        type=int,
        default=None,
        help="Port for compiler server (default: auto-assign starting from 2001)",
    )
    parser.add_argument(
        "--gpu-port",
        type=int,
        default=None,
        help="Port for GPU server (default: auto-assign starting from 2002)",
    )
    parser.add_argument(
        "--gpu-server-url",
        type=str,
        default=None,
        help="URL of existing GPU server to use (e.g., http://localhost:2002)",
    )
    parser.add_argument(
        "--run-record-dir",
        type=Path,
        default=None,
        help="Write run_manifest.json, events.jsonl, and summary.json here.",
    )
    parser.add_argument(
        "--portfolio-suite",
        type=Path,
        default=None,
        help="Resolved portfolio suite JSON to embed in the run manifest.",
    )
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=None)
    parser.add_argument("--cas-dir", type=Path, default=None)
    args = parser.parse_args()
    validate_common_arguments(parser, args)

    suite_config = {}
    if args.portfolio_suite is not None:
        try:
            suite_config = resolve_portfolio_suite(
                args.portfolio_suite,
                problem_numbers=args.problem_numbers,
                rollouts=args.rl_iterations,
                steps=args.rl_rollout_steps,
                trusted_pilot=args.experiment_name == "trusted-rmsnorm-pilot",
            )
        except ValueError as error:
            parser.error(str(error))

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

    if args.run_record_dir is not None:
        global RUN_RECORDER
        config.MODEL = args.model
        provider = get_llm_provider(type(config))
        RUN_RECORDER = RunRecorder(
            args.run_record_dir,
            model=args.model,
            provider_config=provider.public_config(),
            suite=suite_config,
            gpu_target=args.gpu,
            repo_root=ROOT_DIR,
            state_store=state_store,
        )
        set_run_recorder(RUN_RECORDER)
        record_event(
            "portfolio_run_started",
            data={
                "dataset": args.dataset,
                "subset": args.subset,
                "problem_numbers": args.problem_numbers,
                "rollouts": args.rl_iterations,
                "steps": args.rl_rollout_steps,
            },
        )

    dataset_str = args.dataset

    dataset, dataset_iter = get_dataset(
        args.dataset,
        args.subset,
        args.dataset_split,
        args.precision,
        args.problem_numbers,
        args.start,
        args.end,
        args.single_file_path,
    )
    # Append precision to dataset string when provided (avoid dataset-specific special-casing)
    if getattr(args, "precision", None):
        dataset_str += "/" + args.precision

    # set output directory
    model_name = (
        config.MODEL.replace("llmgateway/", "")
        .replace("eos/", "")
        .replace("chipnemo/", "")
        .replace("azure/", "")
        .replace("/", "-")
        .lower()
    )
    OUT_DIR = (
        ROOT_DIR / "out" / dataset_str / args.experiment_name / model_name
    )

    # configure loggers
    log_file = OUT_DIR / f"run.log"
    logger.configure(
        handlers=[
            dict(
                sink=sys.stderr,
                format=config.CUSTOM_LOGGER_FORMAT,
                level=config.LOG_LEVEL,
                colorize=True,
                backtrace=True,
                diagnose=True,
            ),
            dict(
                sink=log_file,
                format=config.CUSTOM_LOGGER_FORMAT,
                level=config.LOG_LEVEL,
                colorize=False,
                backtrace=True,
                diagnose=True,
            ),
        ],
        extra=dict(agent_name="main", attempt_id=None, task_id=None),
    )
    logger.info(f"Logging to {log_file}")

    # initialize resources
    try:
        global COMPILE_SERVER, GPU_SERVER
        COMPILE_SERVER = CompileServer(logger, OUT_DIR, port=args.compiler_port)
        
        # Use existing GPU server if URL provided, otherwise create new one
        if args.gpu_server_url:
            logger.info(f"Using existing GPU server at {args.gpu_server_url}")
            config.set_gpu_server_url(resolve_target_gpu(args.gpu), args.gpu_server_url)
            GPU_SERVER = None  # No need to manage our own server
        else:
            GPU_SERVER = GPUServer(logger, OUT_DIR, gpu=args.gpu, port=args.gpu_port)
            GPU_SERVER.wait_for_connection()
            if GPU_SERVER.is_managed:
                assert (
                    args.gpu is None or args.gpu == GPUType.current().value
                ), f"GPU type mismatch: {args.gpu} != {GPUType.current().value}. Please supply your own GPU_SERVER_URL_<GPU_TYPE> since --gpu differs from the current GPU type."
                config.set_gpu_server_url(GPUType.current(), GPU_SERVER.url)
        
        COMPILE_SERVER.wait_for_connection()
        if COMPILE_SERVER.is_managed:
            config.set_compile_server_url(COMPILE_SERVER.url)
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        record_event(
            "runtime_initialization_failed",
            status="error",
            data={"error_type": type(e).__name__},
        )
        return 2

    config.print_config(logger)

    # Load comprehensive analysis results for optimization data
    logger.info("Loading comprehensive analysis results...")
    load_comprehensive_analysis_results()

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.concurrency)

    # Create a list to hold all the tasks
    tasks = []

    workflow_config = create_workflow_config(args)
    workflow_config.shared_optimization_database = OptimizationDatabase(
        OUT_DIR / "optimization_database.md",
        None,
        LLMInterface(args.model, logger),
    )

    logger.info(f"Processing {len(dataset)} problems")
    for entry in dataset_iter:
        problem_id = entry["id"]
        folder = (OUT_DIR / problem_id).resolve()
        if folder == OUT_DIR.resolve() or not folder.is_relative_to(OUT_DIR.resolve()):
            logger.error(f"Skipping unsafe task output path for {problem_id!r}: {folder}")
            continue
        if workflow_config.should_skip_folder(folder):
            continue
        elif args.no_resume:
            logger.warning(
                f"Retrying {problem_id} from scratch because --no-resume flag is set."
            )
            shutil.rmtree(folder, ignore_errors=True)
        elif folder.exists():
            logger.debug(f"Resuming {problem_id}")

        # Create a task for this problem
        task = asyncio.create_task(
            process_problem(
                entry,
                folder,
                semaphore,
                workflow_config,
                args.timeout,
            )
        )
        logger.debug(f"Created task for {problem_id}")
        tasks.append(task)

    logger.info(f"Waiting for {len(tasks)} tasks to complete")
    # Wait for all tasks to complete
    exit_code = 0
    if tasks:
        logger.info(
            f"Processing {len(tasks)} problems with concurrency {args.concurrency}"
        )
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        for task_result in task_results:
            if isinstance(task_result, Exception):
                exit_code = 2
                logger.error(
                    f"A task escaped workflow error isolation: "
                    f"{type(task_result).__name__}: {task_result}"
                )
            elif task_result[1] in {
                RunStatus.FAILED,
                RunStatus.TIMEOUT,
                RunStatus.BLOCKED,
            }:
                exit_code = 2
    else:
        logger.info("No problems to process")
    return exit_code


def main():
    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    exit_code = 0
    try:
        exit_code = asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt detected, cleaning up...")
        exit_code = 130
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise e
    finally:
        cleanup_servers()
        if RUN_RECORDER is not None:
            RUN_RECORDER.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
