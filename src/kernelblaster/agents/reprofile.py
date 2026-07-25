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
现有 success_rl_optimization.cu 文件的重新分析代理。

该代理发现、编译和分析现有的 success_rl_optimization.cu 文件
收集新的 NCU 分析数据而不生成新代码。
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
    """分析单个文件的结果。"""
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
    代理重新分析现有 success_rl_optimization.cu 文件。

    该代理不会生成新代码 - 它只会编译和分析
    现有的优化内核来收集新的分析数据。
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
        """
        初始化 ReProfileAgent 实例，并保存后续流程所需的配置与依赖。

        参数:
        base_folder: 当前 Agent 使用的工作目录。
        gpu: 执行或分析任务使用的 GPU 配置。
        logger: 记录诊断信息和任务进度的日志器。
        timeout: 允许操作等待的最长秒数。
        cycles_only: 调用方提供的 `cycles_only` 参数。
        detailed_profiling: 调用方提供的 `detailed_profiling` 参数。
        profile_init: 调用方提供的 `profile_init` 参数。
        """
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
        查找所有成功文件并尝试找到其对应的测试代码。

        参数：
        模式：要搜索的文件模式（默认值：如果是 profile_init，则为“init.cu”，否则为“success_rl_optimization.cu”）
        base_directory：要搜索的目录
        problem_numbers：可选的问题编号列表（e.g.，[“8”，“10”，“25”]）

        返回：
        (success_file_path, test_code_path) 元组列表。
        如果未找到，test_code_path 可能为 None。

        参数:
        pattern: 调用方提供的 `pattern` 参数。
        base_directory: 调用方提供的 `base_directory` 参数。
        problem_numbers: 调用方提供的 `problem_numbers` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
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
        
        # 标准化问题编号（删除前导零以进行匹配）
        normalized_problem_numbers = None
        if problem_numbers:
            normalized_problem_numbers = [pn.lstrip('0') or '0' for pn in problem_numbers]
            self.logger.info(f"Filtering to problem numbers: {problem_numbers}")
            
        success_files = []
        for success_file in base_directory.rglob(pattern):
            # 按问题编号（如果指定）过滤
            if problem_numbers is not None:
                problem_num = self._get_problem_number(success_file)
                if problem_num is None:
                    self.logger.debug(f"Could not extract problem number from {success_file}, skipping")
                    continue
                # 将提取的问题数归一化以进行比较
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
        查找给定成功文件的测试代码文件。

        典型结构：
        - success_file：<基础>/<问题>/rl_ncu/success_rl_optimization.cu
        test_code：<基础>/<问题>/driver.cpp
        - success_file：<基础>/<问题>/init.cu
        test_code：<基础>/<问题>/driver.cpp
        - success_file：<基础>/<问题>/<子目录>/init.cu
        test_code：<基础>/<问题>/driver.cpp

        还检查 state.json 中的 test_code_fp 字段。

        参数:
        success_file: 调用方提供的 `success_file` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 策略1：检查同一目录中是否有driver.cpp
        same_dir = success_file.parent
        driver_cpp = same_dir / "driver.cpp"
        if driver_cpp.exists():
            return driver_cpp
        
        # 策略2：检查父目录中是否有driver.cpp
        parent_dir = success_file.parent.parent
        driver_cpp = parent_dir / "driver.cpp"
        if driver_cpp.exists():
            return driver_cpp
        
        # 策略2：检查父目录中是否有state.json
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
        
        # 策略3：检查同一目录中是否有test_code.cu
        test_code_cu = success_file.parent / "test_code.cu"
        if test_code_cu.exists():
            return test_code_cu
        
        # 策略 4：检查 0_init_annotated.cu（可能包含测试工具）
        init_file = success_file.parent / "0_init_annotated.cu"
        if init_file.exists():
            # 检查它是否有主要功能（测试工具）
            content = init_file.read_text()
            if "int main(" in content or "void launch_gpu_implementation" in content:
                return init_file
        
        # 策略 5：在父目录中查找包含“test”、“main”或“driver”的文件
        for check_dir in [success_file.parent, parent_dir]:
            for pattern in ["*test*.cu", "*test*.cpp", "*main*.cu", "*main*.cpp", "*driver*"]:
                matches = list(check_dir.glob(pattern))
                if matches:
                    return matches[0]
        
        return None
    
    def _get_problem_name(self, success_file: Path) -> str:
        """
        从文件路径中提取问题名称。

        参数:
        success_file: 调用方提供的 `success_file` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 路径结构：.../level2/025_Conv2d_Min_Tanh_Tanh/rl_ncu/success_rl_optimization.cu
        # 提取问题目录名
        parts = success_file.parts
        for i, part in enumerate(parts):
            if part.startswith(('level1', 'level2', 'level3')) and i + 1 < len(parts):
                return parts[i + 1]  # 返回问题名称目录
        # 后备：使用父目录名称
        return success_file.parent.parent.name if success_file.parent.parent.name != 'rl_ncu' else success_file.parent.parent.parent.name
    
    def _get_problem_number(self, success_file: Path) -> Optional[str]:
        """
        从文件路径中提取问题编号。

        参数:
        success_file: 调用方提供的 `success_file` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 问题名称格式：025_Conv2d_Min_Tanh_Tanh -> 问题编号为“025”
        problem_name = self._get_problem_name(success_file)
        # 提取数字前缀（e.g.，“025_Conv2d_Min_Tanh_Tanh”中的“025”）
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
        对单个 success_rl_optimization.cu 候选重新执行性能分析。

        参数：
        success_file：success_rl_optimization.cu 文件的路径
        test_code_fp：测试代码文件的路径（driver.cpp）
        output_dir：保存分析结果的基目录（默认：与success_file相同）
        将为每个问题创建一个子目录

        返回：
        带有分析数据的 ProfilingResult

        参数:
        success_file: 调用方提供的 `success_file` 参数。
        test_code_fp: 调用方提供的 `test_code_fp` 参数。
        output_dir: 调用方提供的 `output_dir` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 确定输出目录结构
        problem_name = self._get_problem_name(success_file)
        
        if output_dir is None:
            # 在问题的父目录中创建子目录
            output_dir = success_file.parent.parent / "reprofile_results" / problem_name
        else:
            # 在指定的输出库中创建子目录
            output_dir = Path(output_dir) / problem_name
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 为日志创建子目录
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Profiling {success_file} with test code {test_code_fp}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # 存储所有日志以供保存
        compilation_logs = {"stdout": [], "stderr": [], "command": None}
        ncu_logs = {"basic": {"stdout": [], "stderr": [], "commands": []}, "details": {"stdout": [], "stderr": [], "commands": []}, "source": {"stdout": [], "stderr": [], "commands": []}}
        
        # 记录编译命令信息
        compilation_info = {
            "main_file": str(test_code_fp),
            "cuda_file": str(success_file),
            "gpu": self.gpu.value,
            "sm_version": self.gpu.sm,
            "timeout": self.timeout,
        }
        (logs_dir / "compilation_info.json").write_text(json.dumps(compilation_info, indent=2))
        
        try:
            # 第1步：编译并运行
            timer = NamedTimer()
            
            # 记录编译命令（注意：实际编译发生在服务器上，但我们记录我们所请求的内容）
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
            
            # 保存编译日志
            compilation_logs["stdout"] = stdout_list if isinstance(stdout_list, list) else [stdout_list]
            compilation_logs["stderr"] = stderr_list if isinstance(stderr_list, list) else [stderr_list]
            (logs_dir / "compilation_stdout.txt").write_text("\n".join(compilation_logs["stdout"]))
            (logs_dir / "compilation_stderr.txt").write_text("\n".join(compilation_logs["stderr"]))
            
            # 记录编译后的二进制路径
            compilation_logs["compiled_binary"] = compiled_path
            (logs_dir / "compilation_info.json").write_text(json.dumps(compilation_logs, indent=2, default=str))
            
            if not success:
                error_msg = f"Compilation/execution failed. stdout: {stdout_list}, stderr: {stderr_list}"
                self.logger.error(f"Failed to compile/run {success_file}: {error_msg}")
                
                # 保存错误摘要
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
            
            # 第 2 步：查找内核名称
            kernel_names = await find_kernel_names_ncu(
                Path(compiled_path), success_file, self.gpu, self.timeout
            )
            
            if not kernel_names:
                error_msg = "No kernel names found for NCU profiling"
                self.logger.error(f"{error_msg} for {success_file}")
                
                # 保存错误摘要
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
            
            # 第 3 步：运行 NCU 分析
            if self.cycles_only:
                # 仅循环模式：仅获取循环
                # 在此文件中并行化内核分析
                async def profile_kernel_basic(kernel_name):
                    # 获取带有部分的文本输出（用于每个内核的周期提取）
                    """
                    处理 `profile_kernel_basic` 对应的领域操作，并返回调用方所需的标准化结果。

                    参数:
                        kernel_name: 调用方提供的 `kernel_name` 参数。

                    返回:
                        当前操作产生的结果；具体类型由返回注解和调用约定确定。
                    """
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
                    
                    # 保存NCU日志
                    ncu_logs["basic"]["stdout"].append(ncu_stdout)
                    ncu_logs["basic"]["stderr"].append(ncu_stderr)
                    (logs_dir / f"ncu_basic_{kernel_name}_stdout.txt").write_text(ncu_stdout)
                    (logs_dir / f"ncu_basic_{kernel_name}_stderr.txt").write_text(ncu_stderr)
                    (logs_dir / f"ncu_basic_{kernel_name}_command.txt").write_text(ncu_text_command)
                    
                    return ncu_stdout, ncu_stderr, get_elapsed_cycles_ncu_log(ncu_stdout)
                
                # 并行分析所有内核
                kernel_results = await asyncio.gather(*[
                    profile_kernel_basic(kernel_name) for kernel_name in kernel_names
                ])
                
                cycles = 0
                combined_ncu_logs = ""
                for ncu_stdout, ncu_stderr, kernel_cycles in kernel_results:
                    combined_ncu_logs += ncu_stdout + "\n\n"
                    cycles += kernel_cycles
                
                # 为所有内核生成单个全局 NCU 报告（在所有每个内核文本运行之后）
                global_report_file = f"/tmp/kernelagent_gpu_{problem_name}_all_kernels.ncu-rep"
                ncu_global_export_command = f"NVIDIA_TF32_OVERRIDE=0 ncu --set full --section=SpeedOfLight --section=SpeedOfLight_RooflineChart --export {global_report_file} --force-overwrite"
                ncu_logs["basic"]["commands"].append({
                    "kernel": "all_kernels",
                    "command": ncu_global_export_command,
                    "description": "NCU global export report file (all kernels)",
                    "report_file": global_report_file,
                })
                
                # 运行全局导出（一次分析所有内核，无 -k 标志）
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
                # 完整分析模式
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
                
                # 在此文件中并行化内核分析
                async def profile_kernel_full(kernel_name):
                    """
                    使用所有分析类型分析单个内核。

                    参数:
                        kernel_name: 调用方提供的 `kernel_name` 参数。

                    返回:
                        当前操作产生的结果；具体类型由返回注解和调用约定确定。
                    """
                    self.logger.info(f"Profiling kernel {kernel_name}")
                    
                    kernel_results = {
                        "kernel": kernel_name,
                        "basic": None,
                        "details": None,
                        "source": None,
                    }
                    
                    # 循环和原始日志的基本 NCU 分析
                    # 运行 NCU 两次：一次用于文本输出（用于循环提取），一次用于导出文件
                    # 这是因为 --export 抑制了正常的文本输出
                    safe_kernel_name = kernel_name.replace(" ", "_").replace("/", "_")
                    # 第一次运行：获取带有部分的文本输出（用于循环提取）
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
                    
                    # 从文本输出中提取循环
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
                    
                    # 保存基本 NCU 日志（从第一次运行时使用文本输出）
                    ncu_logs["basic"]["stdout"].append(ncu_stdout)
                    ncu_logs["basic"]["stderr"].append(ncu_stderr)
                    (logs_dir / f"ncu_basic_{kernel_name}_stdout.txt").write_text(ncu_stdout)
                    (logs_dir / f"ncu_basic_{kernel_name}_stderr.txt").write_text(ncu_stderr)
                    (logs_dir / f"ncu_basic_{kernel_name}_command.txt").write_text(ncu_basic_command)
                    if self.detailed_profiling:
                        # 带有报告文件的详细分析
                        # 运行 NCU 两次：一次用于文本输出（用于 CSV 解析），一次用于导出文件
                        # 这是因为 --export 抑制了正常的文本输出
                        ncu_details_report_file = f"/tmp/kernelagent_gpu_{safe_kernel_name}_{problem_name}_details.ncu-rep"
                        
                        # 第一次运行：获取带有部分的文本输出（用于 CSV 解析）
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
                        
                        # 保存详细日志
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
                            
                            # 保存源日志
                            ncu_logs["source"]["stdout"].append(source_stdout)
                            ncu_logs["source"]["stderr"].append(source_stderr)
                            (logs_dir / f"ncu_source_{kernel_name}_stdout.txt").write_text(source_stdout)
                            (logs_dir / f"ncu_source_{kernel_name}_stderr.txt").write_text(source_stderr)
                            (logs_dir / f"ncu_source_{kernel_name}_command.txt").write_text(ncu_source_command)
                    
                    return kernel_results
                
                # 并行分析所有内核
                kernel_results_list = await asyncio.gather(*[
                    profile_kernel_full(kernel_name) for kernel_name in kernel_names
                ])
                
                # 汇总结果
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
                                
                                # 从细节中提取指标
                                for _, row in details_df.iterrows():
                                    metric_name = row["Metric Name"]
                                    try:
                                        # 处理字符串和数值
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
                
                # 如果我们有详细的分析数据，请注释来源
                if self.detailed_profiling and source_dfs and details_dfs:
                    try:
                        annotated_ncu = annotate_source(success_file, source_dfs, details_dfs)
                    except Exception as e:
                        self.logger.warning(f"Failed to annotate source: {e}")
                        annotated_ncu = None
                else:
                    annotated_ncu = None
                
                # 为所有内核生成单个全局 NCU 报告（在所有每个内核文本运行之后）
                # 对于详细的分析，创建基本和详细的全局报告
                global_basic_report_file = f"/tmp/kernelagent_gpu_{problem_name}_all_kernels.ncu-rep"
                ncu_global_basic_export_command = f"NVIDIA_TF32_OVERRIDE=0 ncu --section=SpeedOfLight --section=SpeedOfLight_RooflineChart --export {global_basic_report_file} --force-overwrite"
                ncu_logs["basic"]["commands"].append({
                    "kernel": "all_kernels",
                    "command": ncu_global_basic_export_command,
                    "description": "NCU global export report file (all kernels, basic)",
                    "report_file": global_basic_report_file,
                })
                
                # 运行全局基本导出（一次分析所有内核，无 -k 标志）
                await run_gpu_executable(
                    Path(compiled_path),
                    self.gpu,
                    self.timeout,
                    job_name=f"{success_file} (ncu global export all kernels basic)",
                    prefix_command=ncu_global_basic_export_command,
                )
                (logs_dir / "ncu_global_basic_export_command.txt").write_text(ncu_global_basic_export_command)
                
                # 对于详细的分析，还可以创建全局详细信息报告
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
                    
                    # 运行全局详细信息导出（一次分析所有内核，无 -k 标志）
                    await run_gpu_executable(
                        Path(compiled_path),
                        self.gpu,
                        self.timeout,
                        job_name=f"{success_file} (ncu global export all kernels details)",
                        prefix_command=ncu_global_details_export_command,
                    )
                    (logs_dir / "ncu_global_details_export_command.txt").write_text(ncu_global_details_export_command)
                
                metrics["elapsed_cycles"] = cycles
            
            # 第 4 步：将结果保存在有组织的结构中
            if self.profile_init:
                result_file = output_dir / "init_profiled.cu"
            else:
                result_file = output_dir / "success_rl_optimization_profiled.cu"
            ncu_log_file = output_dir / "ncu_log.txt"
            metrics_file = output_dir / "metrics.json"
            annotated_file = output_dir / "annotated_source.txt" if annotated_ncu else None
            
            # 初始化用于跟踪复制的 NCU 报告文件的列表
            ncu_report_files_copied = []
            
            # 写入附带 NCU 日志的性能分析结果。
            original_content = success_file.read_text()
            profiled_content = original_content + f"\n\n/*\nNCU Profiling Log:\n{combined_ncu_logs}\n*/\n"
            if annotated_ncu:
                profiled_content += f"\n\n/*\nAnnotated Source:\n{annotated_ncu}\n*/\n"
            result_file.write_text(profiled_content)
            
            # 单独写入NCU日志
            ncu_log_file.write_text(combined_ncu_logs)
            
            # 如果有的话，单独编写带注释的源代码
            if annotated_ncu:
                annotated_file.write_text(annotated_ncu)
            
            # 写入指标
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # 尝试将 NCU 报告文件从服务器临时目录复制到输出目录
            # 注意：这些文件是在 GPU 服务器上创建的，路径为 /tmp/kernelagent_gpu_*
            # 如果服务器和客户端在同一台机器上，我们可以在这里复制它们
            import shutil
            
            # 复制全局基本报告文件（包含所有内核）
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
            
            # 如果完成了详细分析，则复制全局详细信息报告文件
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
            
            # 将所有 NCU 命令保存到单个 JSON 文件以方便参考
            ncu_commands_summary = {
                "basic": ncu_logs["basic"]["commands"],
                "details": ncu_logs["details"]["commands"] if self.detailed_profiling else [],
                "source": ncu_logs["source"]["commands"] if self.detailed_profiling else [],
            }
            (logs_dir / "ncu_commands.json").write_text(json.dumps(ncu_commands_summary, indent=2))
            
            # 写入摘要信息
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
            
            # 保存错误摘要以及异常详细信息
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
                pass  # 如果我们不能写出错误摘要，也不要失败
            
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
        发现并分析所有 success_rl_optimization.cu 文件。

        参数：
        base_directory：要搜索的根目录（默认：self.base_folder）
        output_base：保存分析结果的位置（默认：每个文件旁边）
        max_workers：并行分析作业的数量（默认值：1 表示顺序）
        skip_existing：跳过已有分析结果的文件
        problem_numbers：可选的问题编号列表（e.g.，[“8”，“10”，“25”]）

        返回：
        每个文件的分析结果列表

        参数:
        base_directory: 调用方提供的 `base_directory` 参数。
        output_base: 调用方提供的 `output_base` 参数。
        max_workers: 调用方提供的 `max_workers` 参数。
        skip_existing: 调用方提供的 `skip_existing` 参数。
        problem_numbers: 调用方提供的 `problem_numbers` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        if base_directory is None:
            base_directory = self.base_folder
        
        # 发现所有成功文件（带有可选的问题编号过滤）
        success_files = self.discover_success_files(
            base_directory=base_directory,
            problem_numbers=problem_numbers,
        )
        
        # 过滤掉没有测试代码的文件
        files_to_profile = [
            (sf, tc) for sf, tc in success_files if tc is not None
        ]
        
        if skip_existing:
            # 过滤掉已经有分析结果的文件
            if self.profile_init:
                # 检查 init_profiled.cu
                files_to_profile = [
                    (sf, tc) for sf, tc in files_to_profile
                    if not (sf.parent / "init_profiled.cu").exists()
                ]
            else:
                # 检查 success_rl_optimization_profiled.cu
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
        
        # 配置文件
        if max_workers == 1:
            # 顺序处理
            results = []
            for success_file, test_code in files_to_profile:
                result = await self.profile_file(
                    success_file,
                    test_code,
                    output_dir=output_base if output_base else None,
                )
                results.append(result)
        else:
            # 使用信号量进行并行处理
            semaphore = asyncio.Semaphore(max_workers)
            
            async def profile_with_semaphore(success_file, test_code):
                """
                处理 `profile_with_semaphore` 对应的领域操作，并返回调用方所需的标准化结果。

                参数:
                    success_file: 调用方提供的 `success_file` 参数。
                    test_code: 调用方提供的 `test_code` 参数。

                返回:
                    当前操作产生的结果；具体类型由返回注解和调用约定确定。
                """
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
        
        # 保存摘要
        if output_base:
            output_base = Path(output_base)
            output_base.mkdir(parents=True, exist_ok=True)
            summary_file = output_base / "profiling_results.jsonl"
            csv_file = output_base / "profiling_results.csv"
        else:
            summary_file = base_directory / "profiling_results.jsonl"
            csv_file = base_directory / "profiling_results.csv"
        
        # 编写 JSONL 摘要
        with open(summary_file, 'w') as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + "\n")
        
        # 以与基线文件相同的格式编写 CSV 摘要
        successful_results = [r for r in results if r.success and r.cycles > 0]
        
        if successful_results:
            # 提取问题名称和周期
            csv_data = []
            for result in successful_results:
                # 从 success_file 路径中提取问题名称
                success_file_path = Path(result.success_file)
                problem_name = self._get_problem_name(success_file_path)
                csv_data.append({
                    "problem": problem_name,
                    "avg_Elapsed_Cycles": result.cycles
                })
            
            # 按问题名称排序（数字前缀）
            def get_problem_number_for_sort(entry):
                """
                获取 `get_problem_number_for_sort` 对应的领域操作，并返回调用方所需的标准化结果。

                参数:
                    entry: 调用方提供的 `entry` 参数。

                返回:
                    当前操作产生的结果；具体类型由返回注解和调用约定确定。
                """
                problem = entry["problem"]
                match = re.match(r"^(\d+)_", problem)
                if match:
                    return int(match.group(1))
                return 999999  # 将不匹配的放在最后
            
            csv_data.sort(key=get_problem_number_for_sort)
            
            # 写入 CSV
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
