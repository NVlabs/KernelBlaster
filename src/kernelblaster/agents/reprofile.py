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
"""
Re-Profiling Agent for existing success_rl_optimization.cu files.

This agent discovers, compiles, and profiles existing success_rl_optimization.cu files
to gather fresh NCU profiling data without generating new code.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import asyncio
import re
import csv
import loguru
from dataclasses import dataclass, asdict

from ..config import GPUType
from .utils import (
    compile_and_run_cu_file,
    run_gpu_executable,
    find_kernel_names_ncu,
    get_elapsed_cycles_ncu_log,
    NamedTimer,
    FeedbackError,
    UTILIZATION_METRICS,
    format_ncu_details_as_csv,
    format_ncu_source_as_csv,
    annotate_source,
)


@dataclass
class ProfilingResult:
    """Results from profiling a single file."""
    success_file: str
    test_code_file: str
    cycles: int
    ncu_log: str
    kernel_names: List[str]
    metrics: Dict[str, float]
    output_path: str
    success: bool
    error: Optional[str] = None
    annotated_ncu: Optional[str] = None


class ReProfileAgent:
    """
    Agent to re-profile existing success_rl_optimization.cu files.
    
    This agent does NOT generate new code - it only compiles and profiles
    existing optimized kernels to gather fresh profiling data.
    """
    
    def __init__(
        self,
        base_folder: Path,
        gpu: GPUType,
        logger: loguru.Logger,
        timeout: int = 3600,
        cycles_only: bool = False,
        detailed_profiling: bool = True,
        profile_init: bool = False,
    ):
        self.base_folder = Path(base_folder)
        self.gpu = gpu
        self.logger = logger
        self.timeout = timeout
        self.cycles_only = cycles_only
        self.detailed_profiling = detailed_profiling
        self.profile_init = profile_init
        
    def discover_success_files(
        self, 
        pattern: Optional[str] = None,
        base_directory: Optional[Path] = None,
        problem_numbers: Optional[List[str]] = None,
    ) -> List[Tuple[Path, Optional[Path]]]:
        """
        Find all success files and attempt to find their corresponding test code.
        
        Args:
            pattern: File pattern to search for (default: "init.cu" if profile_init, else "success_rl_optimization.cu")
            base_directory: Directory to search in
            problem_numbers: Optional list of problem numbers to filter by (e.g., ["8", "10", "25"])
        
        Returns:
            List of (success_file_path, test_code_path) tuples.
            test_code_path may be None if not found.
        """
        if pattern is None:
            pattern = "init.cu" if self.profile_init else "success_rl_optimization.cu"
        """
        Find all success files and attempt to find their corresponding test code.
        
        Args:
            pattern: File pattern to search for
            base_directory: Directory to search in
            problem_numbers: Optional list of problem numbers to filter by (e.g., ["8", "10", "25"])
        
        Returns:
            List of (success_file_path, test_code_path) tuples.
            test_code_path may be None if not found.
        """
        if base_directory is None:
            base_directory = self.base_folder
        
        # Normalize problem numbers (remove leading zeros for matching)
        normalized_problem_numbers = None
        if problem_numbers:
            normalized_problem_numbers = [pn.lstrip('0') or '0' for pn in problem_numbers]
            self.logger.info(f"Filtering to problem numbers: {problem_numbers}")
            
        success_files = []
        for success_file in base_directory.rglob(pattern):
            # Filter by problem number if specified
            if problem_numbers is not None:
                problem_num = self._get_problem_number(success_file)
                if problem_num is None:
                    self.logger.debug(f"Could not extract problem number from {success_file}, skipping")
                    continue
                # Normalize the extracted problem number for comparison
                normalized_num = problem_num.lstrip('0') or '0'
                if normalized_num not in normalized_problem_numbers:
                    continue
            
            test_code = self.find_test_code(success_file)
            success_files.append((success_file, test_code))
            if test_code is None:
                self.logger.warning(
                    f"Could not find test code for {success_file}. "
                    f"It will be skipped during profiling."
                )
        
        self.logger.info(f"Discovered {len(success_files)} success files")
        return success_files
    
    def find_test_code(self, success_file: Path) -> Optional[Path]:
        """
        Find test code file for a given success file.
        
        Typical structures:
        - success_file: <base>/<problem>/rl_ncu/success_rl_optimization.cu
          test_code: <base>/<problem>/driver.cpp
        - success_file: <base>/<problem>/init.cu
          test_code: <base>/<problem>/driver.cpp
        - success_file: <base>/<problem>/<subdir>/init.cu
          test_code: <base>/<problem>/driver.cpp
        
        Also checks state.json for test_code_fp field.
        """
        # Strategy 1: Check same directory for driver.cpp
        same_dir = success_file.parent
        driver_cpp = same_dir / "driver.cpp"
        if driver_cpp.exists():
            return driver_cpp
        
        # Strategy 2: Check parent directory for driver.cpp
        parent_dir = success_file.parent.parent
        driver_cpp = parent_dir / "driver.cpp"
        if driver_cpp.exists():
            return driver_cpp
        
        # Strategy 2: Check for state.json in parent directories
        for check_dir in [success_file.parent, parent_dir, parent_dir.parent]:
            state_json = check_dir / "state.json"
            if state_json.exists():
                try:
                    with open(state_json, 'r') as f:
                        state = json.load(f)
                        if "test_code_fp" in state:
                            test_code_path = Path(state["test_code_fp"])
                            if test_code_path.exists():
                                return test_code_path
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    self.logger.debug(f"Failed to read state.json at {state_json}: {e}")
        
        # Strategy 3: Check for test_code.cu in same directory
        test_code_cu = success_file.parent / "test_code.cu"
        if test_code_cu.exists():
            return test_code_cu
        
        # Strategy 4: Check for 0_init_annotated.cu (may contain test harness)
        init_file = success_file.parent / "0_init_annotated.cu"
        if init_file.exists():
            # Check if it has a main function (test harness)
            content = init_file.read_text()
            if "int main(" in content or "void launch_gpu_implementation" in content:
                return init_file
        
        # Strategy 5: Look for files with "test", "main", or "driver" in parent directories
        for check_dir in [success_file.parent, parent_dir]:
            for pattern in ["*test*.cu", "*test*.cpp", "*main*.cu", "*main*.cpp", "*driver*"]:
                matches = list(check_dir.glob(pattern))
                if matches:
                    return matches[0]
        
        return None
    
    def _get_problem_name(self, success_file: Path) -> str:
        """Extract problem name from file path."""
        # Path structure: .../level2/025_Conv2d_Min_Tanh_Tanh/rl_ncu/success_rl_optimization.cu
        # Extract the problem directory name
        parts = success_file.parts
        for i, part in enumerate(parts):
            if part.startswith(('level1', 'level2', 'level3')) and i + 1 < len(parts):
                return parts[i + 1]  # Return the problem name directory
        # Fallback: use parent directory name
        return success_file.parent.parent.name if success_file.parent.parent.name != 'rl_ncu' else success_file.parent.parent.parent.name
    
    def _get_problem_number(self, success_file: Path) -> Optional[str]:
        """Extract problem number from file path."""
        # Problem name format: 025_Conv2d_Min_Tanh_Tanh -> problem number is "025"
        problem_name = self._get_problem_name(success_file)
        # Extract the numeric prefix (e.g., "025" from "025_Conv2d_Min_Tanh_Tanh")
        match = re.match(r"^(\d+)_", problem_name)
        if match:
            return match.group(1)
        return None
    
    async def profile_file(
        self,
        success_file: Path,
        test_code_fp: Path,
        output_dir: Optional[Path] = None
    ) -> ProfilingResult:
        """
        Profile a single success_rl_optimization.cu file.
        
        Args:
            success_file: Path to the success_rl_optimization.cu file
            test_code_fp: Path to the test code file (driver.cpp)
            output_dir: Base directory to save profiling results (default: same as success_file)
                      A subdirectory will be created for each problem
        
        Returns:
            ProfilingResult with profiling data
        """
        # Determine output directory structure
        problem_name = self._get_problem_name(success_file)
        
        if output_dir is None:
            # Create subdirectory in the problem's parent directory
            output_dir = success_file.parent.parent / "reprofile_results" / problem_name
        else:
            # Create subdirectory in the specified output base
            output_dir = Path(output_dir) / problem_name
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for logs
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Profiling {success_file} with test code {test_code_fp}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Store all logs for saving
        compilation_logs = {"stdout": [], "stderr": [], "command": None}
        ncu_logs = {"basic": {"stdout": [], "stderr": [], "commands": []}, "details": {"stdout": [], "stderr": [], "commands": []}, "source": {"stdout": [], "stderr": [], "commands": []}}
        
        # Log compilation command info
        compilation_info = {
            "main_file": str(test_code_fp),
            "cuda_file": str(success_file),
            "gpu": self.gpu.value,
            "sm_version": self.gpu.sm,
            "timeout": self.timeout,
        }
        (logs_dir / "compilation_info.json").write_text(json.dumps(compilation_info, indent=2))
        
        try:
            # Step 1: Compile and run
            timer = NamedTimer()
            
            # Log compilation command (note: actual compilation happens on server, but we log what we're requesting)
            compilation_command_info = {
                "description": "Compilation request to compile server",
                "main_file": str(test_code_fp),
                "cuda_file": str(success_file),
                "gpu": self.gpu.value,
                "sm_version": self.gpu.sm,
                "timeout": self.timeout,
                "persistent_artifacts": True,
                "num_runs": 1,
                "passed_keyword": "passed",
            }
            compilation_logs["command"] = compilation_command_info
            (logs_dir / "compilation_command.json").write_text(json.dumps(compilation_command_info, indent=2))
            
            stdout_list, stderr_list, compiled_path, success = await compile_and_run_cu_file(
                test_code_fp,
                success_file,
                self.gpu,
                timer,
                self.logger,
                persistent_artifacts=True,
                timeout=self.timeout,
                num_runs=1,
                passed_keyword="passed",
            )
            
            # Save compilation logs
            compilation_logs["stdout"] = stdout_list if isinstance(stdout_list, list) else [stdout_list]
            compilation_logs["stderr"] = stderr_list if isinstance(stderr_list, list) else [stderr_list]
            (logs_dir / "compilation_stdout.txt").write_text("\n".join(compilation_logs["stdout"]))
            (logs_dir / "compilation_stderr.txt").write_text("\n".join(compilation_logs["stderr"]))
            
            # Log compiled binary path
            compilation_logs["compiled_binary"] = compiled_path
            (logs_dir / "compilation_info.json").write_text(json.dumps(compilation_logs, indent=2, default=str))
            
            if not success:
                error_msg = f"Compilation/execution failed. stdout: {stdout_list}, stderr: {stderr_list}"
                self.logger.error(f"Failed to compile/run {success_file}: {error_msg}")
                
                # Save error summary
                error_summary = {
                    "problem_name": problem_name,
                    "success_file": str(success_file),
                    "test_code_file": str(test_code_fp),
                    "error": error_msg,
                    "compilation_stdout": compilation_logs["stdout"],
                    "compilation_stderr": compilation_logs["stderr"],
                }
                (output_dir / "error_summary.json").write_text(json.dumps(error_summary, indent=2))
                
                return ProfilingResult(
                    success_file=str(success_file),
                    test_code_file=str(test_code_fp),
                    cycles=0,
                    ncu_log="",
                    kernel_names=[],
                    metrics={},
                    output_path=str(output_dir),
                    success=False,
                    error=error_msg,
                )
            
            # Step 2: Find kernel names
            kernel_names = await find_kernel_names_ncu(
                Path(compiled_path), success_file, self.gpu, self.timeout
            )
            
            if not kernel_names:
                error_msg = "No kernel names found for NCU profiling"
                self.logger.error(f"{error_msg} for {success_file}")
                
                # Save error summary
                error_summary = {
                    "problem_name": problem_name,
                    "success_file": str(success_file),
                    "test_code_file": str(test_code_fp),
                    "error": error_msg,
                    "compilation_stdout": compilation_logs["stdout"],
                    "compilation_stderr": compilation_logs["stderr"],
                }
                (output_dir / "error_summary.json").write_text(json.dumps(error_summary, indent=2))
                
                return ProfilingResult(
                    success_file=str(success_file),
                    test_code_file=str(test_code_fp),
                    cycles=0,
                    ncu_log="",
                    kernel_names=[],
                    metrics={},
                    output_path=str(output_dir),
                    success=False,
                    error=error_msg,
                )
            
            # Step 3: Run NCU profiling
            if self.cycles_only:
                # Cycles-only mode: just get cycles
                # Parallelize kernel profiling within this file
                async def profile_kernel_basic(kernel_name):
                    # Get text output with sections (for cycles extraction per kernel)
                    ncu_text_command = f"NVIDIA_TF32_OVERRIDE=0 ncu --section=SpeedOfLight --section=SpeedOfLight_RooflineChart -k {kernel_name}"
                    ncu_logs["basic"]["commands"].append({
                        "kernel": kernel_name,
                        "command": ncu_text_command,
                        "description": "NCU cycles-only profiling with SpeedOfLight sections (text output)",
                    })
                    
                    ncu_stdout, ncu_stderr = await run_gpu_executable(
                        Path(compiled_path),
                        self.gpu,
                        self.timeout,
                        job_name=f"{success_file} (ncu cycles-only {kernel_name})",
                        prefix_command=ncu_text_command,
                    )
                    
                    # Save NCU logs
                    ncu_logs["basic"]["stdout"].append(ncu_stdout)
                    ncu_logs["basic"]["stderr"].append(ncu_stderr)
                    (logs_dir / f"ncu_basic_{kernel_name}_stdout.txt").write_text(ncu_stdout)
                    (logs_dir / f"ncu_basic_{kernel_name}_stderr.txt").write_text(ncu_stderr)
                    (logs_dir / f"ncu_basic_{kernel_name}_command.txt").write_text(ncu_text_command)
                    
                    return ncu_stdout, ncu_stderr, get_elapsed_cycles_ncu_log(ncu_stdout)
                
                # Profile all kernels in parallel
                kernel_results = await asyncio.gather(*[
                    profile_kernel_basic(kernel_name) for kernel_name in kernel_names
                ])
                
                cycles = 0
                combined_ncu_logs = ""
                for ncu_stdout, ncu_stderr, kernel_cycles in kernel_results:
                    combined_ncu_logs += ncu_stdout + "\n\n"
                    cycles += kernel_cycles
                
                # Generate single global NCU report for all kernels (after all per-kernel text runs)
                global_report_file = f"/tmp/kernelagent_gpu_{problem_name}_all_kernels.ncu-rep"
                ncu_global_export_command = f"NVIDIA_TF32_OVERRIDE=0 ncu --set full --section=SpeedOfLight --section=SpeedOfLight_RooflineChart --export {global_report_file} --force-overwrite"
                ncu_logs["basic"]["commands"].append({
                    "kernel": "all_kernels",
                    "command": ncu_global_export_command,
                    "description": "NCU global export report file (all kernels)",
                    "report_file": global_report_file,
                })
                
                # Run global export (profiles all kernels at once, no -k flag)
                await run_gpu_executable(
                    Path(compiled_path),
                    self.gpu,
                    self.timeout,
                    job_name=f"{success_file} (ncu global export all kernels)",
                    prefix_command=ncu_global_export_command,
                )
                (logs_dir / "ncu_global_export_command.txt").write_text(ncu_global_export_command)
                
                annotated_ncu = None
                metrics = {"elapsed_cycles": cycles}
            else:
                # Full profiling mode
                cycles = 0
                combined_ncu_logs = ""
                source_dfs = []
                details_dfs = []
                metrics = {}
                
                if self.detailed_profiling:
                    details_command = (
                        "ncu --page details --section=SchedulerStats --section=Occupancy "
                        "--section=SpeedOfLight --section=SpeedOfLight_RooflineChart --section=LaunchStats --section=WarpStateStats "
                        "--section=InstructionStats --csv --metrics "
                        + ",".join(UTILIZATION_METRICS)
                    )
                    source_command = (
                        "ncu --page source --print-source=cuda,sass "
                        "--section=InstructionStats --section=SourceCounters "
                        "--import-source yes --csv"
                    )
                
                # Parallelize kernel profiling within this file
                async def profile_kernel_full(kernel_name):
                    """Profile a single kernel with all profiling types."""
                    self.logger.info(f"Profiling kernel {kernel_name}")
                    
                    kernel_results = {
                        "kernel": kernel_name,
                        "basic": None,
                        "details": None,
                        "source": None,
                    }
                    
                    # Basic NCU profiling for cycles and raw log
                    # Run NCU twice: once for text output (for cycles extraction), once for export file
                    # This is because --export suppresses the normal text output
                    safe_kernel_name = kernel_name.replace(" ", "_").replace("/", "_")
                    ncu_report_file = f"/tmp/kernelagent_gpu_{safe_kernel_name}_{problem_name}.ncu-rep"
                    
                    # First run: get text output with sections (for cycles extraction)
                    ncu_basic_command = f"NVIDIA_TF32_OVERRIDE=0 ncu --section=SpeedOfLight --section=SpeedOfLight_RooflineChart -k {kernel_name}"
                    ncu_logs["basic"]["commands"].append({
                        "kernel": kernel_name,
                        "command": ncu_basic_command,
                        "description": "NCU basic profiling (cycles and raw log) with SpeedOfLight sections (text output)",
                    })
                    
                    ncu_stdout, ncu_stderr = await run_gpu_executable(
                        Path(compiled_path),
                        self.gpu,
                        self.timeout,
                        job_name=f"{success_file} (ncu basic {kernel_name})",
                        prefix_command=ncu_basic_command,
                    )
                    
                    # Extract cycles from text output
                    try:
                        kernel_cycles = get_elapsed_cycles_ncu_log(ncu_stdout)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to extract cycles from NCU output for {kernel_name}. "
                            f"stdout length: {len(ncu_stdout)}, stderr length: {len(ncu_stderr)}. "
                            f"Error: {e}. "
                            f"First 500 chars of stdout: {ncu_stdout[:500]}"
                        )
                        raise
                    kernel_results["basic"] = (ncu_stdout, ncu_stderr, kernel_cycles)
                    
                    # Save basic NCU logs (from first run with text output)
                    ncu_logs["basic"]["stdout"].append(ncu_stdout)
                    ncu_logs["basic"]["stderr"].append(ncu_stderr)
                    (logs_dir / f"ncu_basic_{kernel_name}_stdout.txt").write_text(ncu_stdout)
                    (logs_dir / f"ncu_basic_{kernel_name}_stderr.txt").write_text(ncu_stderr)
                    (logs_dir / f"ncu_basic_{kernel_name}_command.txt").write_text(ncu_basic_command)
                    (logs_dir / f"ncu_basic_{kernel_name}_export_command.txt").write_text(ncu_export_command)
                    
                    if self.detailed_profiling:
                        # Detailed profiling with report file
                        # Run NCU twice: once for text output (for CSV parsing), once for export file
                        # This is because --export suppresses the normal text output
                        ncu_details_report_file = f"/tmp/kernelagent_gpu_{safe_kernel_name}_{problem_name}_details.ncu-rep"
                        
                        # First run: get text output with sections (for CSV parsing)
                        ncu_details_command = f"NVIDIA_TF32_OVERRIDE=0 {details_command} -k {kernel_name}"
                        ncu_logs["details"]["commands"].append({
                            "kernel": kernel_name,
                            "command": ncu_details_command,
                            "description": "NCU detailed profiling (metrics and CSV) with SpeedOfLight sections (text output)",
                            "full_command": details_command,
                        })
                        
                        details_stdout, details_stderr = await run_gpu_executable(
                            Path(compiled_path),
                            self.gpu,
                            self.timeout,
                            job_name=f"{success_file} (ncu details {kernel_name})",
                            prefix_command=ncu_details_command,
                        )
                        
                        kernel_results["details"] = (details_stdout, details_stderr)
                        
                        # Save details logs
                        ncu_logs["details"]["stdout"].append(details_stdout)
                        ncu_logs["details"]["stderr"].append(details_stderr)
                        (logs_dir / f"ncu_details_{kernel_name}_stdout.txt").write_text(details_stdout)
                        (logs_dir / f"ncu_details_{kernel_name}_stderr.txt").write_text(details_stderr)
                        (logs_dir / f"ncu_details_{kernel_name}_command.txt").write_text(ncu_details_command)
                        
                        if "No Kernels were profiled" not in details_stdout:
                            ncu_source_command = f"NVIDIA_TF32_OVERRIDE=0 {source_command} -k {kernel_name}"
                            ncu_logs["source"]["commands"].append({
                                "kernel": kernel_name,
                                "command": ncu_source_command,
                                "description": "NCU source profiling (source annotation)",
                                "full_command": source_command,
                            })
                            
                            source_stdout, source_stderr = await run_gpu_executable(
                                Path(compiled_path),
                                self.gpu,
                                self.timeout,
                                job_name=f"{success_file} (ncu source {kernel_name})",
                                prefix_command=ncu_source_command,
                            )
                            
                            kernel_results["source"] = (source_stdout, source_stderr)
                            
                            # Save source logs
                            ncu_logs["source"]["stdout"].append(source_stdout)
                            ncu_logs["source"]["stderr"].append(source_stderr)
                            (logs_dir / f"ncu_source_{kernel_name}_stdout.txt").write_text(source_stdout)
                            (logs_dir / f"ncu_source_{kernel_name}_stderr.txt").write_text(source_stderr)
                            (logs_dir / f"ncu_source_{kernel_name}_command.txt").write_text(ncu_source_command)
                    
                    return kernel_results
                
                # Profile all kernels in parallel
                kernel_results_list = await asyncio.gather(*[
                    profile_kernel_full(kernel_name) for kernel_name in kernel_names
                ])
                
                # Aggregate results
                cycles = 0
                combined_ncu_logs = ""
                for kernel_result in kernel_results_list:
                    if kernel_result["basic"]:
                        ncu_stdout, _, kernel_cycles = kernel_result["basic"]
                        combined_ncu_logs += ncu_stdout + "\n\n"
                        cycles += kernel_cycles
                    
                    if self.detailed_profiling and kernel_result["details"]:
                        details_stdout, _ = kernel_result["details"]
                        if "No Kernels were profiled" not in details_stdout and kernel_result["source"]:
                            source_stdout, _ = kernel_result["source"]
                            try:
                                details_df = format_ncu_details_as_csv(details_stdout)
                                source_df = format_ncu_source_as_csv(source_stdout)
                                source_dfs.append(source_df)
                                details_dfs.append(details_df)
                                
                                # Extract metrics from details
                                for _, row in details_df.iterrows():
                                    metric_name = row["Metric Name"]
                                    try:
                                        # Handle both string and numeric values
                                        metric_value_raw = row["Metric Value"]
                                        if isinstance(metric_value_raw, str):
                                            metric_value = float(metric_value_raw.replace(",", ""))
                                        else:
                                            metric_value = float(metric_value_raw)
                                        metrics[metric_name] = metrics.get(metric_name, 0) + metric_value
                                    except (ValueError, KeyError, TypeError):
                                        pass
                            except ValueError as e:
                                self.logger.warning(
                                    f"Failed to parse CSV from NCU logs for {kernel_result['kernel']}: {e}"
                                )
                
                # Annotate source if we have detailed profiling data
                if self.detailed_profiling and source_dfs and details_dfs:
                    try:
                        annotated_ncu = annotate_source(success_file, source_dfs, details_dfs)
                    except Exception as e:
                        self.logger.warning(f"Failed to annotate source: {e}")
                        annotated_ncu = None
                else:
                    annotated_ncu = None
                
                # Generate single global NCU report for all kernels (after all per-kernel text runs)
                # For detailed profiling, create both basic and details global reports
                global_basic_report_file = f"/tmp/kernelagent_gpu_{problem_name}_all_kernels.ncu-rep"
                ncu_global_basic_export_command = f"NVIDIA_TF32_OVERRIDE=0 ncu --section=SpeedOfLight --section=SpeedOfLight_RooflineChart --export {global_basic_report_file} --force-overwrite"
                ncu_logs["basic"]["commands"].append({
                    "kernel": "all_kernels",
                    "command": ncu_global_basic_export_command,
                    "description": "NCU global export report file (all kernels, basic)",
                    "report_file": global_basic_report_file,
                })
                
                # Run global basic export (profiles all kernels at once, no -k flag)
                await run_gpu_executable(
                    Path(compiled_path),
                    self.gpu,
                    self.timeout,
                    job_name=f"{success_file} (ncu global export all kernels basic)",
                    prefix_command=ncu_global_basic_export_command,
                )
                (logs_dir / "ncu_global_basic_export_command.txt").write_text(ncu_global_basic_export_command)
                
                # For detailed profiling, also create a global details report
                if self.detailed_profiling:
                    global_details_report_file = f"/tmp/kernelagent_gpu_{problem_name}_all_kernels_details.ncu-rep"
                    ncu_global_details_export_command = f"NVIDIA_TF32_OVERRIDE=0 {details_command} --export {global_details_report_file} --force-overwrite"
                    ncu_logs["details"]["commands"].append({
                        "kernel": "all_kernels",
                        "command": ncu_global_details_export_command,
                        "description": "NCU global export report file (all kernels, details)",
                        "full_command": details_command,
                        "report_file": global_details_report_file,
                    })
                    
                    # Run global details export (profiles all kernels at once, no -k flag)
                    await run_gpu_executable(
                        Path(compiled_path),
                        self.gpu,
                        self.timeout,
                        job_name=f"{success_file} (ncu global export all kernels details)",
                        prefix_command=ncu_global_details_export_command,
                    )
                    (logs_dir / "ncu_global_details_export_command.txt").write_text(ncu_global_details_export_command)
                
                metrics["elapsed_cycles"] = cycles
            
            # Step 4: Save results in organized structure
            if self.profile_init:
                result_file = output_dir / "init_profiled.cu"
            else:
                result_file = output_dir / "success_rl_optimization_profiled.cu"
            ncu_log_file = output_dir / "ncu_log.txt"
            metrics_file = output_dir / "metrics.json"
            annotated_file = output_dir / "annotated_source.txt" if annotated_ncu else None
            
            # Initialize list for tracking copied NCU report files
            ncu_report_files_copied = []
            
            # Write profiled file with NCU log appended
            original_content = success_file.read_text()
            profiled_content = original_content + f"\n\n/*\nNCU Profiling Log:\n{combined_ncu_logs}\n*/\n"
            if annotated_ncu:
                profiled_content += f"\n\n/*\nAnnotated Source:\n{annotated_ncu}\n*/\n"
            result_file.write_text(profiled_content)
            
            # Write NCU log separately
            ncu_log_file.write_text(combined_ncu_logs)
            
            # Write annotated source separately if available
            if annotated_ncu:
                annotated_file.write_text(annotated_ncu)
            
            # Write metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Try to copy NCU report files from server temp directory to output directory
            # Note: These files are created on the GPU server at /tmp/kernelagent_gpu_*
            # If server and client are on the same machine, we can copy them here
            import shutil
            
            # Copy global basic report file (contains all kernels)
            global_basic_report_file = f"/tmp/kernelagent_gpu_{problem_name}_all_kernels.ncu-rep"
            server_basic_path = Path(global_basic_report_file)
            if server_basic_path.exists():
                client_basic_path = output_dir / "all_kernels.ncu-rep"
                try:
                    shutil.copy2(server_basic_path, client_basic_path)
                    ncu_report_files_copied.append(str(client_basic_path.relative_to(output_dir)))
                    self.logger.info(f"Copied global NCU report file: {server_basic_path} -> {client_basic_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to copy global NCU report file {server_basic_path}: {e}")
            
            # Copy global details report file if detailed profiling was done
            if self.detailed_profiling:
                global_details_report_file = f"/tmp/kernelagent_gpu_{problem_name}_all_kernels_details.ncu-rep"
                server_details_path = Path(global_details_report_file)
                if server_details_path.exists():
                    client_details_path = output_dir / "all_kernels_details.ncu-rep"
                    try:
                        shutil.copy2(server_details_path, client_details_path)
                        ncu_report_files_copied.append(str(client_details_path.relative_to(output_dir)))
                        self.logger.info(f"Copied global NCU details report file: {server_details_path} -> {client_details_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to copy global NCU details report file {server_details_path}: {e}")
            
            if not ncu_report_files_copied:
                self.logger.warning(
                    f"NCU report files were not found in /tmp/kernelagent_gpu_*. "
                    f"This may be because the GPU server and client are on different machines, "
                    f"or the files were cleaned up. Report files are created on the server side."
                )
            
            # Save all NCU commands to a single JSON file for easy reference
            ncu_commands_summary = {
                "basic": ncu_logs["basic"]["commands"],
                "details": ncu_logs["details"]["commands"] if self.detailed_profiling else [],
                "source": ncu_logs["source"]["commands"] if self.detailed_profiling else [],
            }
            (logs_dir / "ncu_commands.json").write_text(json.dumps(ncu_commands_summary, indent=2))
            
            # Write summary info
            summary = {
                "problem_name": problem_name,
                "success_file": str(success_file),
                "test_code_file": str(test_code_fp),
                "kernel_names": kernel_names,
                "cycles": cycles,
                "metrics": metrics,
                "profiling_mode": "cycles_only" if self.cycles_only else ("detailed" if self.detailed_profiling else "basic"),
                "output_directory": str(output_dir),
                "files": {
                    "profiled_cu": str(result_file.relative_to(output_dir)),
                    "ncu_log": str(ncu_log_file.relative_to(output_dir)),
                    "metrics": str(metrics_file.relative_to(output_dir)),
                    "logs_directory": "logs/",
                    "ncu_reports": ncu_report_files_copied if ncu_report_files_copied else "No .ncu-rep files found (may be on different machine than GPU server)",
                }
            }
            if annotated_ncu:
                summary["files"]["annotated_source"] = str(annotated_file.relative_to(output_dir))
            summary_file = output_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(
                f"Successfully profiled {success_file}: {cycles} cycles. "
                f"Results saved to {output_dir}"
            )
            
            return ProfilingResult(
                success_file=str(success_file),
                test_code_file=str(test_code_fp),
                cycles=cycles,
                ncu_log=combined_ncu_logs,
                kernel_names=kernel_names,
                metrics=metrics,
                output_path=str(output_dir),
                success=True,
                annotated_ncu=annotated_ncu,
            )
            
        except Exception as e:
            error_msg = f"Error profiling {success_file}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Save error summary with exception details
            import traceback
            error_summary = {
                "problem_name": problem_name,
                "success_file": str(success_file),
                "test_code_file": str(test_code_fp),
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "compilation_stdout": compilation_logs.get("stdout", []),
                "compilation_stderr": compilation_logs.get("stderr", []),
            }
            try:
                (output_dir / "error_summary.json").write_text(json.dumps(error_summary, indent=2))
            except Exception:
                pass  # Don't fail if we can't write the error summary
            
            return ProfilingResult(
                success_file=str(success_file),
                test_code_file=str(test_code_fp),
                cycles=0,
                ncu_log="",
                kernel_names=[],
                metrics={},
                output_path=str(output_dir),
                success=False,
                error=error_msg,
            )
    
    async def profile_all(
        self,
        base_directory: Optional[Path] = None,
        output_base: Optional[Path] = None,
        max_workers: int = 1,
        skip_existing: bool = True,
        problem_numbers: Optional[List[str]] = None,
    ) -> List[ProfilingResult]:
        """
        Discover and profile all success_rl_optimization.cu files.
        
        Args:
            base_directory: Root directory to search (default: self.base_folder)
            output_base: Where to save profiling results (default: alongside each file)
            max_workers: Number of parallel profiling jobs (default: 1 for sequential)
            skip_existing: Skip files that already have profiling results
            problem_numbers: Optional list of problem numbers to filter by (e.g., ["8", "10", "25"])
        
        Returns:
            List of profiling results for each file
        """
        if base_directory is None:
            base_directory = self.base_folder
        
        # Discover all success files (with optional problem number filtering)
        success_files = self.discover_success_files(
            base_directory=base_directory,
            problem_numbers=problem_numbers,
        )
        
        # Filter out files without test code
        files_to_profile = [
            (sf, tc) for sf, tc in success_files if tc is not None
        ]
        
        if skip_existing:
            # Filter out files that already have profiling results
            if self.profile_init:
                # Check for init_profiled.cu
                files_to_profile = [
                    (sf, tc) for sf, tc in files_to_profile
                    if not (sf.parent / "init_profiled.cu").exists()
                ]
            else:
                # Check for success_rl_optimization_profiled.cu
                files_to_profile = [
                    (sf, tc) for sf, tc in files_to_profile
                    if not (sf.parent / "success_rl_optimization_profiled.cu").exists()
                ]
        
        self.logger.info(
            f"Found {len(files_to_profile)} files to profile "
            f"(out of {len(success_files)} total success files)"
        )
        
        if not files_to_profile:
            self.logger.info("No files to profile")
            return []
        
        # Profile files
        if max_workers == 1:
            # Sequential processing
            results = []
            for success_file, test_code in files_to_profile:
                result = await self.profile_file(
                    success_file,
                    test_code,
                    output_dir=output_base if output_base else None,
                )
                results.append(result)
        else:
            # Parallel processing with semaphore
            semaphore = asyncio.Semaphore(max_workers)
            
            async def profile_with_semaphore(success_file, test_code):
                async with semaphore:
                    return await self.profile_file(
                        success_file,
                        test_code,
                        output_dir=output_base if output_base else None,
                    )
            
            tasks = [
                profile_with_semaphore(sf, tc) for sf, tc in files_to_profile
            ]
            results = await asyncio.gather(*tasks)
        
        # Save summary
        if output_base:
            output_base = Path(output_base)
            output_base.mkdir(parents=True, exist_ok=True)
            summary_file = output_base / "profiling_results.jsonl"
            csv_file = output_base / "profiling_results.csv"
        else:
            summary_file = base_directory / "profiling_results.jsonl"
            csv_file = base_directory / "profiling_results.csv"
        
        # Write JSONL summary
        with open(summary_file, 'w') as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + "\n")
        
        # Write CSV summary in the same format as baseline files
        successful_results = [r for r in results if r.success and r.cycles > 0]
        
        if successful_results:
            # Extract problem names and cycles
            csv_data = []
            for result in successful_results:
                # Extract problem name from success_file path
                success_file_path = Path(result.success_file)
                problem_name = self._get_problem_name(success_file_path)
                csv_data.append({
                    "problem": problem_name,
                    "avg_Elapsed_Cycles": result.cycles
                })
            
            # Sort by problem name (numeric prefix)
            def get_problem_number_for_sort(entry):
                problem = entry["problem"]
                match = re.match(r"^(\d+)_", problem)
                if match:
                    return int(match.group(1))
                return 999999  # Put non-matching at end
            
            csv_data.sort(key=get_problem_number_for_sort)
            
            # Write CSV
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["problem", "avg_Elapsed_Cycles"])
                writer.writeheader()
                writer.writerows(csv_data)
            
            self.logger.info(
                f"CSV summary saved to {csv_file} with {len(csv_data)} successful profiles"
            )
        else:
            self.logger.warning("No successful profiles to write to CSV")
        
        self.logger.info(
            f"Profiling complete. {sum(1 for r in results if r.success)}/{len(results)} successful. "
            f"Summary saved to {summary_file}"
        )
        
        return results
