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
实现基于强化学习的 CUDA 优化 Agent。
通过性能分析、LLM 策略生成、rollout 和经验回放形成闭环搜索。
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import loguru
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import json
import asyncio
import sys
from ..config import config
from ..observability import event_context
from ..outcomes import RunOutcome, RunStatus
from ..profiling import ProfilerBackend, ProfilerUnavailable, ProfilingMode
from ..measurements import (
    Measurement,
    MeasurementComparisonError,
    MeasurementSource,
    MeasurementUnit,
    format_measurement,
    hardware_fingerprint,
)
from .feedback import FeedbackAgent, Feedback, FeedbackConfig
from .database import OptimizationDatabase, OptimizationEntry, CompositeOptimization
from .rl_agents import (
    ReplayBuffer, Trajectory, TrajectoryStep,
    PolicyEvaluationAgent, PerfGapAnalysisAgent, ParameterUpdateAgent
)
from .utils import (
    FeedbackError,
    compile_and_run_cu_file,
    run_gpu_executable,
    format_ncu_source_as_csv,
    format_ncu_details_as_csv,
    annotate_source,
    UTILIZATION_METRICS,
    find_kernel_names_ncu,
    get_elapsed_cycles_ncu_log,
    NamedTimer,
)
from .database import LLMInterface

import os



def parse_ncu_metrics(ncu_log: str) -> Dict[str, float]:
    """
    从 NCU 日志中解析关键指标以进行状态确定。

    参数:
    ncu_log: 调用方提供的 `ncu_log` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    metrics = {}

    
    # Nsight-Compute 文本表并不总是在值后打印尾随“%”。
    # 相反，列布局为：<指标名称> <指标单位> <指标值>
    # 因此，我们搜索*名称*并获取**该行上的最后一个数字标记**。

    def _build_pattern(keyword: str) -> str:
        """
        返回捕获匹配行上最后一个数字的正则表达式。

        参数:
        keyword: 调用方提供的 `keyword` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # .*？非贪婪直到最终数字（处理可变列/间距）
        return rf"{keyword}.*?([0-9]+(?:\.[0-9]+)?)"

    patterns = {
        'memory_throughput'        : _build_pattern(r"Memory\s+Throughput"),
        'compute_throughput'       : _build_pattern(r"Compute\s*\(SM\)\s*Throughput"),
        'sm_efficiency'            : _build_pattern(r"SM\s+Efficiency"),
        'occupancy'                : _build_pattern(r"Achieved\s+Occupancy"),
        'coalescing_efficiency'    : _build_pattern(r"Global\s+Memory\s+Coalescing"),
        'cache_hit_rate'           : _build_pattern(r"L2\s+Cache\s+Hit\s+Rate"),
        'shared_memory_efficiency' : _build_pattern(r"Shared\s+Memory\s+Efficiency"),
        'tensor_core_usage'        : _build_pattern(r"Tensor\s+Core\s+Usage"),
        'register_usage'           : _build_pattern(r"Registers\s+Per\s+Thread"),
        'shared_memory_usage'      : _build_pattern(r"Shared\s+Memory\s+Usage"),
    }
    
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, ncu_log, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                metrics[metric_name] = float(match.group(1))
            except ValueError:
                metrics[metric_name] = 0.0
        else:
            metrics[metric_name] = 0.0
    
    return metrics


def generate_strategy_guided_prompt(
    optimization_entry: OptimizationEntry | CompositeOptimization,
    annotated_ncu: str,
    ncu_log: str,
    database_content: str = "",
    override_description: str | None = None,
    original_code: str | None = None,
) -> str:
    """
    生成指导 LLM 使用综合优化数据库的提示。

    参数:
    optimization_entry: 调用方提供的 `optimization_entry` 参数。
    annotated_ncu: 调用方提供的 `annotated_ncu` 参数。
    ncu_log: 调用方提供的 `ncu_log` 参数。
    database_content: 调用方提供的 `database_content` 参数。
    override_description: 调用方提供的 `override_description` 参数。
    original_code: 调用方提供的 `original_code` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    
    technique_descriptions = {
        "1.1_coalesced_access": "Focus on ensuring memory accesses are coalesced. Rearrange thread-to-data mapping so consecutive threads access consecutive memory locations.",
        "1.1_occupancy_tuning": "Optimize occupancy by tuning threads per block and shared memory usage. Aim for high occupancy while avoiding resource bottlenecks.",
        "1.1_register_optimization": "Reduce register pressure by minimizing local variables and using shared memory for frequently accessed data.",
        "1.1_shared_memory_optimization": "Optimize shared memory usage by reducing bank conflicts and improving access patterns.",
        "1.1_block_size_tuning": "Experiment with different block sizes to maximize occupancy and resource utilization.",
        "2.1_shared_memory_tiling": "Implement tiling using shared memory to reduce global memory accesses through data reuse.",
        "2.1_tensor_core_utilization": "Modify the code to use tensor cores by ensuring proper data types (half, bfloat16) and matrix sizes.",
        "2.2_thread_data_mapping": "Rearrange how threads map to data elements to improve memory access patterns and reduce conflicts.",
        "2.2_functional_unit_optimization": "Balance the workload across different functional units (ALU, SFU, memory units).",
        "2.2_instruction_mix_optimization": "Optimize the instruction mix to better utilize available compute resources.",
        "2.3_data_layout_optimization": "Reorganize data layout in memory to improve cache utilization and memory bandwidth.",
        "2.3_constant_cache_usage": "Move read-only data to constant memory to leverage the constant cache.",
        "3.1_increase_thread_count": "Launch more threads by increasing grid size or using multiple kernels.",
        "3.1_thread_work_remapping": "Remap thread work assignment to reduce warp divergence.",
        "3.2_work_per_thread_increase": "Increase work per thread through loop unrolling or processing multiple elements per thread.",
        "3.2_data_layout_for_divergence": "Restructure data layout to minimize control flow divergence.",
        "3.3_vector_load_usage": "Use vector loads (float2, float4) to process multiple elements efficiently.",
        "3.4_maximum_occupancy_tuning": "Fine-tune launch parameters to achieve maximum theoretical occupancy.",
        "4.1_shared_memory_caching": "Cache frequently accessed global memory data in shared memory.",
        "4.1_shared_memory_bank_conflict_removal": "Eliminate shared memory bank conflicts by padding or restructuring access patterns.",
        "4.1_register_tiling": "Use register tiling to keep frequently accessed data in registers.",
        "6.1_thread_coarsening": "Assign multiple work items to each thread to amortize parallelization overhead."
    }
    
    # 以不同方式处理复合优化
    if isinstance(optimization_entry, CompositeOptimization):
        # 多种技术的复合优化
        techniques = [t for t in [optimization_entry.technique1, optimization_entry.technique2, optimization_entry.technique3] if t]
        technique_descs = []
        for tech in techniques:
            desc = technique_descriptions.get(tech, f"Apply {tech}")
            technique_descs.append(f"- {tech}: {desc}")
        
        composite_desc = "\n".join(technique_descs)
        order_desc = "\n".join(optimization_entry.order_of_techniques) if optimization_entry.order_of_techniques else "Apply techniques in the order listed above"
        
        params_desc = ""
        if optimization_entry.parameters_to_fine_tune:
            params_list = [f"- {k}: {v}" for k, v in optimization_entry.parameters_to_fine_tune.items()]
            params_desc = f"\n\nPARAMETER TUNING:\n" + "\n".join(params_list)
        
        side_effects_note = ""
        if optimization_entry.side_effects:
            side_effects_note = f"\n\nWARNING - POTENTIAL SIDE EFFECTS:\n{optimization_entry.side_effects}"
        
        # 如果 annotated_ncu 为空，则使用原始代码作为后备
        source_code_display = annotated_ncu if annotated_ncu.strip() else (original_code or "// Source code not available")
        source_code_label = "ANNOTATED SOURCE CODE (with per-line analysis):" if annotated_ncu.strip() else "SOURCE CODE:"
        
        # 仅包含有意义的内容的 NCU 分析日志部分
        # （不仅仅是表示提取失败的“Kernels：...”）
        ncu_section = ""
        ncu_log_stripped = ncu_log.strip()
        if ncu_log_stripped and not (ncu_log_stripped.startswith("Kernels:") and len(ncu_log_stripped.split('\n')) <= 2):
            ncu_section = f"""
RAW NCU PROFILING LOG (Speed Of Light Throughput Summary):
```
{ncu_log[:4000] if len(ncu_log) > 4000 else ncu_log}
```
"""
        
        return f"""You are a CUDA optimization expert with access to comprehensive optimization knowledge.

COMPREHENSIVE GPU OPTIMIZATION DATABASE:
```
{database_content[:6000] if database_content else "Database not available - using fallback descriptions"}
```

COMPOSITE OPTIMIZATION STRATEGY:
{optimization_entry.get_composite_id()}
PREDICTED IMPROVEMENT: {optimization_entry.predicted_improvement}%

TECHNIQUES TO APPLY:
{composite_desc}

APPLICATION ORDER:
{order_desc}{params_desc}{side_effects_note}

CURRENT KERNEL ANALYSIS:

{source_code_label}
```
{source_code_display}
```
{ncu_section}

OPTIMIZATION TASK:
You are an expert CUDA optimization agent, and you are provided an optimization plan. Your task is to apply the optimization plan to the current kernel. You will be provided with the annotated source code, the raw NCU profiling log, and the optimization plan.

CRITICAL REQUIREMENTS:
1. Reference the optimization plan for detailed implementation guidance
2. Apply ALL specified techniques in the given order: {' -> '.join(techniques)}
3. Use the specified parameter values for fine-tuning
4. Generate COMPLETE, COMPILABLE CUDA code
5. Include ALL necessary components:
   - #include statements (cuda_fp16.h, cuda_runtime.h, etc.)
   - #define constants - DEFINE ALL CONSTANTS BEFORE USING THEM
   - Complete __global__ kernel function with proper signature
   - Complete launch_gpu_implementation(void*, void*, void*, int64_t) function
6. Format ALL code in a single ```cpp code block
7. Consider potential side effects mentioned in the database
8. COMPILATION SAFETY: Ensure all constants are properly defined


You are a knowledgeable and efficient CUDA programming assistant, skilled in analyzing NSight Compute logs and optimizing the cuda kernels. Your task is to analyze the provided NSight Compute logs and generate optimized CUDA code based on the analysis. You should focus on finding the largest deficiencies from the NCU log and optimize those attributes first. 

For perf comparisons, please use the "Elapsed Cycles" metric in the GPU Speed of Light Throughput section of the NCU log. The lower the better. Please only write one kernel in the output.

Optimization Tips:
* For better memory bandwidth utilization, please try to use coalescing and coalesced memory access patterns. You can also use vectorized datatypes like float4, int4, uint4, __nv_bfloat162, etc.

APPROACH:
1. Analyze the profiling data to understand current bottlenecks
2. Consult the optimization database for best practices
3. Apply the composite strategy systematically
4. Generate optimized code addressing the identified performance issues"""
    
    else:
        # 单一技术优化
        technique_name = (
            optimization_entry.get_composite_id()
            if isinstance(optimization_entry, CompositeOptimization)
            else getattr(optimization_entry, "technique", str(optimization_entry))
        )
        technique_desc = (
            override_description
            if override_description
            else technique_descriptions.get(
                technique_name,
                f"Apply the {technique_name} optimization technique.",
            )
        )
        pred_impr = getattr(optimization_entry, "predicted_improvement", None)
        category = getattr(optimization_entry, "category", "general")
        pred_impr_str = f"{pred_impr}%" if pred_impr is not None else "N/A"
        
        # 如果 annotated_ncu 为空，则使用原始代码作为后备
        source_code_display = annotated_ncu if annotated_ncu.strip() else (original_code or "// Source code not available")
        source_code_label = "ANNOTATED SOURCE CODE (with per-line analysis):" if annotated_ncu.strip() else "SOURCE CODE:"
        
        # 仅包含有意义的内容的 NCU 分析日志部分
        # （不仅仅是表示提取失败的“Kernels：...”）
        ncu_section = ""
        ncu_log_stripped = ncu_log.strip()
        if ncu_log_stripped and not (ncu_log_stripped.startswith("Kernels:") and len(ncu_log_stripped.split('\n')) <= 2):
            ncu_section = f"""
RAW NCU PROFILING LOG (Speed Of Light Throughput Summary):
```
{ncu_log[:4000] if len(ncu_log) > 4000 else ncu_log}
```
"""
        
        return f"""OPTIMIZATION TASK:
You are an expert CUDA optimization agent, and you are provided an optimization plan. Your task is to apply the optimization plan to the current kernel. You will be provided with the annotated source code, the raw NCU profiling log, and the optimization plan.

OPTIMIZATION STRATEGY: {technique_name}
PREDICTED IMPROVEMENT: {pred_impr_str}
CATEGORY: {category}

STRATEGY DESCRIPTION:
{technique_desc}

CURRENT KERNEL ANALYSIS:

{source_code_label}
```
{source_code_display}
```
{ncu_section}

COMPREHENSIVE GPU OPTIMIZATION DATABASE:
```
{database_content}
```

CRITICAL REQUIREMENTS:
1. Reference the optimization database for detailed implementation guidance
2. Generate COMPLETE, COMPILABLE CUDA code
3. Include ALL necessary components:
   - #include statements (cuda_fp16.h, cuda_runtime.h, etc.)
   - #define constants - DEFINE ALL CONSTANTS BEFORE USING THEM
   - Complete __global__ kernel function with proper signature
   - Complete launch_gpu_implementation function
4. Format ALL code in a single ```cpp code block
5. Focus specifically on the technique described in the database
6. COMPILATION SAFETY: Ensure all constants are properly defined
7. Summarize the optimization technique applied and the reason for the improvement before the code.



APPROACH:
1. Apply requested the optimization technique systematically
2. If applying a new technique not yet attempted in the code, start with the most minimal example, focusing on correctness.
4. Please use the reference code provided in the prompt as helper functions for your optimized kernel.
3. Generate optimized code addressing the identified performance issues"""


@dataclass
class RLNCUFeedback(Feedback):
    """封装 `RLNCUFeedback` 对应的领域状态与操作。"""
    elapsed_cycles: Optional[int] = None
    measurement: Measurement | None = None
    ncu_log: Optional[str] = None
    annotated_ncu: Optional[str] = None
    optimization_technique: Optional[str] = None
    predicted_improvement: Optional[float] = None
    actual_improvement: Optional[float] = None
    state: Optional[str] = None


class RLNCUAgent(FeedbackAgent):
    """以正确性和实测性能为反馈，执行多轮 CUDA Kernel rollout 优化。"""

    @staticmethod
    def next_performance_state(current_state: str | None, new_state: str | None) -> str:
        """
        合并新旧性能状态；Profiler 未给出新分类时保留已知状态。

        参数:
        current_state: 调用方提供的 `current_state` 参数。
        new_state: 调用方提供的 `new_state` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return new_state or current_state or "events_only/unknown"

    @property
    def initial_cycles(self) -> int | None:
        """Deprecated compatibility view; Events measurements never become cycles."""
        if getattr(self, "initial_measurement", None) is None:
            return None
        if self.initial_measurement.unit is not MeasurementUnit.CYCLES:
            return None
        return int(self.initial_measurement.value)

    @property
    def best_cycles(self) -> int | None:
        """Deprecated compatibility view; use ``best_measurement`` internally."""
        if getattr(self, "best_measurement", None) is None:
            return None
        if self.best_measurement.unit is not MeasurementUnit.CYCLES:
            return None
        return int(self.best_measurement.value)
    
    def __init__(
        self,
        fb_config: FeedbackConfig,
        code_to_optimize_fp: Path,
        database_path: Path,
        max_rollout_steps: int = 5,
        replay_buffer_size: int = 1000,
        update_frequency: int = 10,  # 每 N 个轨迹更新数据库
        database: Optional[OptimizationDatabase] = None,
        profiler_backend: Optional[ProfilerBackend] = None,
    ):
        # 初始化基础反馈代理
        """
        初始化 RLNCUAgent 实例，并保存后续流程所需的配置与依赖。

        参数:
        fb_config: 调用方提供的 `fb_config` 参数。
        code_to_optimize_fp: 调用方提供的 `code_to_optimize_fp` 参数。
        database_path: 调用方提供的 `database_path` 参数。
        max_rollout_steps: 调用方提供的 `max_rollout_steps` 参数。
        replay_buffer_size: 调用方提供的 `replay_buffer_size` 参数。
        update_frequency: 调用方提供的 `update_frequency` 参数。
        database: 保存历史状态与优化经验的共享数据库。
        profiler_backend: 调用方提供的 `profiler_backend` 参数。
        """
        super().__init__(fb_config)
        
        self.test_code_fp = fb_config.test_code_fp
        self.test_code = fb_config.test_code_fp.read_text()
        self.code_to_optimize_fp = code_to_optimize_fp
        self.code_to_optimize = code_to_optimize_fp.read_text()
        
        # RL 特定组件 - 使用带有 GPU 优化报告的增强型数据库
        gpu_report_path = Path(__file__).parent.parent.parent.parent.parent / "algo-sol-modeling/algo-space/gpu_optimization_report.md"
        llm_interface = LLMInterface(self.model, self.agent_logger)
        # 使用提供的共享数据库（如果可用）；否则创建一个新的
        if database is not None:
            self.database = database
        else:
            self.database = OptimizationDatabase(database_path, gpu_report_path, llm_interface)
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.max_rollout_steps = max_rollout_steps
        self.update_frequency = update_frequency
        self.profiler_backend = profiler_backend
        
        # RL代理
        self.policy_evaluation_agent = PolicyEvaluationAgent()
        self.perf_gap_analysis_agent = PerfGapAnalysisAgent()
        self.parameter_update_agent = ParameterUpdateAgent()
        
        # 追踪
        self.iteration_count = 0
        self.total_trajectories = 0
        self._next_trajectory_id = 0
        self.best_measurement: Measurement | None = None
        self.initial_measurement: Measurement | None = None
        self.profiling_mode = "ncu"
        
        # 并发助手
        import asyncio as _asyncio
        self._trajectory_lock: _asyncio.Lock = _asyncio.Lock()
        self._policy_lock: _asyncio.Lock = _asyncio.Lock()
        
        # 当前轨迹
        self.current_trajectory = None
        
        # 要运行的 RL 迭代次数（可以通过工作流程设置）
        self.num_rl_iterations = 50  # 默认为 50 次 RL 迭代

    async def _profile_candidate(self, filepath: Path) -> Tuple[str, str, str, Measurement]:
        """
        处理 `profile_candidate` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
        filepath: 目标文件路径。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
        ProfilerUnavailable: 输入、外部调用或状态不满足执行要求时抛出。
        """
        if self.profiler_backend is None:
            annotated, raw_output, stderr, cycles = await self.gather_perf_metrics(filepath)
            return (
                annotated,
                raw_output,
                stderr,
                Measurement(
                    value=int(cycles),
                    unit=MeasurementUnit.CYCLES,
                    source=MeasurementSource.NCU,
                    protocol_id="ncu-default",
                    hardware_fingerprint=hardware_fingerprint(getattr(self, "gpu", None)),
                ),
            )

        result = await self.profiler_backend.profile(filepath)
        self.profiling_mode = result.mode.value
        if not result.available:
            raise ProfilerUnavailable(result.error or "Profiler returned no timing metric")
        measurement = result.measurement
        if measurement is None or measurement.value <= 0:
            raise ProfilerUnavailable("Profiler returned no positive timing metric")
        return (
            result.annotated_source,
            result.raw_output,
            result.stderr,
            measurement,
        )

    async def _classify_profile_state(
        self,
        ncu_log: str,
        metrics: dict,
        code: str,
        measurement: Measurement,
    ) -> str:
        """
        仅当存在硬件反证据时才导出 NCU 状态。

        参数:
        ncu_log: 调用方提供的 `ncu_log` 参数。
        metrics: 性能分析或正确性检查产生的指标集合。
        code: 待处理的源码文本。
        measurement: 调用方提供的 `measurement` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        if self.profiling_mode == ProfilingMode.EVENTS_ONLY.value or not ncu_log.strip():
            return "events_only/unknown"
        return await self.database.get_state_from_ncu_report(
            ncu_log,
            metrics,
            code,
            elapsed_cycles=(
                int(measurement.value)
                if measurement.unit is MeasurementUnit.CYCLES
                else None
            ),
        )

    async def initialize(self):
        """通过收集初始分析数据来初始化代理。"""
        # 将 init cu 文件复制到文件夹
        self.code_to_optimize_fp = self.folder / "init.cu"
        self.code_to_optimize_fp.write_text(self.code_to_optimize)

        self.agent_logger.info(f"Gathering initial NCU log...")
        try:
            annotated_ncu, init_ncu_log, _, measurement = await self._profile_candidate(
                self.code_to_optimize_fp
            )
            self.initial_measurement = measurement
            self.best_measurement = measurement

            # 保留第一个 NCU 日志，以便后续步骤可以执行分析
            self.last_ncu_log = init_ncu_log
            
            # 保存初始状态
            if self.profiling_mode == ProfilingMode.EVENTS_ONLY.value:
                initial_state = "events_only/unknown"
            else:
                init_metrics = parse_ncu_metrics(init_ncu_log)
                initial_state = await self.database.get_state_from_ncu_report(
                    init_ncu_log,
                    init_metrics,
                    self.code_to_optimize,
                    elapsed_cycles=(
                        int(measurement.value)
                        if measurement.unit is MeasurementUnit.CYCLES
                        else None
                    ),
                )
            
            self.agent_logger.info(
                f"Initial state: {initial_state}, measurement: {format_measurement(measurement)}"
            )
            
            # 保存初始文件
            (self.folder / "0_init_annotated.cu").write_text(annotated_ncu)
            
        except FeedbackError as e:
            # 记录失败，但继续进行回退分析，以便代理可以继续。
            self.agent_logger.warning(
                f"Initial profiling failed numeric verification; proceeding with fallback state. Details: {e}"
            )

            # 使用基本后备状态；将周期保持为“无”，这样我们就不会报告虚假值。
            init_metrics = {}
            initial_state_profile = self.database._fallback_state_analysis("", init_metrics)
            initial_state = initial_state_profile.state_name
            # 保持 self.initial_cycles 不变（默认为 None）。保持 best_cycles 原样。
            # 保留下游步骤的占位符 NCU 日志
            self.last_ncu_log = ""

            # 后备：使用原始 init.cu 写入带注释的文件，以便下游步骤可以继续进行
            try:
                init_src = self.code_to_optimize_fp.read_text()
                (self.folder / "0_init_annotated.cu").write_text(init_src)
            except Exception as _e:
                self.agent_logger.warning(f"Failed to write fallback 0_init_annotated.cu: {_e}")

        except Exception as e:
            if "ERR_NVGPUCTRPERM" not in str(e):
                raise
            self.profiling_mode = "events_only"
            self.initial_measurement = None
            self.last_ncu_log = ""
            initial_state = "events_only/unknown"
            self.agent_logger.warning(
                "NCU performance counters are unavailable; the run is marked "
                "events_only and cannot make a profile-guided success claim."
            )
            (self.folder / "0_init_annotated.cu").write_text(
                self.code_to_optimize,
                encoding="utf-8",
            )

    async def run(self) -> RunOutcome:
        """
        重写基本运行方法以实现特定于 RL 的行为。
        **并行**运行多个 RL 迭代并返回最佳结果。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        import asyncio as _asyncio
        
        best_filename = None
        best_measurement: Measurement | None = None
        iteration_errors: list[str] = []

        # 确保初始分析数据在生成任务之前可用一次
        if not hasattr(self, "last_ncu_log"):
            await self.initialize()
        # 计算并共享从初始 NCU 日志导出的初始状态
        if self.last_ncu_log and self.profiling_mode == ProfilingMode.NCU.value:
            initial_state_shared = await self.database.get_state_from_ncu_report(
                self.last_ncu_log,
                parse_ncu_metrics(self.last_ncu_log),
                self.code_to_optimize,
                elapsed_cycles=self.initial_cycles,
            )
        else:
            initial_state_shared = "events_only/unknown"

        async def _run_single_iteration(iteration_idx: int):
            """
            执行一次展开并返回其轨迹的助手。

            参数:
            iteration_idx: 调用方提供的 `iteration_idx` 参数。

            返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
            """
            self.agent_logger.info(
                f"[Async] RL Iteration {iteration_idx + 1}/{self.num_rl_iterations}")
            try:
                # 初始状态导出一次并在迭代之间共享
                initial_state = initial_state_shared

                trajectory = await self.run_rollout(self.code_to_optimize, initial_state)
                return iteration_idx, trajectory
            except Exception as exc:
                iteration_errors.append(str(exc))
                self.agent_logger.error(
                    f"RL iteration {iteration_idx + 1} failed: {exc}")
                return iteration_idx, None

        # 同时启动所有迭代
        tasks = [_asyncio.create_task(_run_single_iteration(i)) for i in range(self.num_rl_iterations)]

        for coro in _asyncio.as_completed(tasks):
            iteration_idx, trajectory = await coro
            if trajectory is None:
                continue

            # 处理轨迹结果
            if trajectory.steps:
                for best_step in trajectory.steps:
                    try:
                        is_best = best_measurement is None or best_step.measurement.is_faster_than(
                            best_measurement
                        )
                    except MeasurementComparisonError as error:
                        self.agent_logger.warning(f"Skipping incomparable measurement: {error}")
                        continue
                    if not is_best:
                        continue
                    best_measurement = best_step.measurement
                    best_filename_path = self.folder / f"rl_iter_{iteration_idx}_best.cu"
                    best_filename_path.write_text(
                        best_step.code + f"\n\n// Measurement: {format_measurement(best_measurement)}\n")
                    best_filename = best_filename_path
                    self.agent_logger.info(
                        f"[Async] New best result from iter {iteration_idx}: "
                        f"{format_measurement(best_measurement)}")

            await self._record_completed_trajectory(trajectory)

        # 所有任务完成后
        # 保留优化数据库 JSON 的编号快照
        try:
            # 确保当前数据库状态持续存在
            self.database._persist_database()
            persist_fp = self.database._persist_json_fp
            snapshots_dir = persist_fp.parent / "snapshots"
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(snapshots_dir.glob("optimization_database_*.json"))
            next_idx = len(existing)
            snapshot_fp = snapshots_dir / f"optimization_database_{next_idx}.json"
            snapshot_fp.write_text(persist_fp.read_text(encoding="utf-8"), encoding="utf-8")
            self.agent_logger.info(f"Saved database snapshot to {snapshot_fp}")
        except Exception as snap_exc:
            self.agent_logger.warning(f"Failed to write database snapshot: {snap_exc}")

        if best_filename is not None and best_measurement is not None:
            # 确保我们有基线周期来判断改进情况。
            try:
                if self.initial_measurement is None:
                    init_fp = getattr(self, "code_to_optimize_fp", None)
                    if not init_fp or not init_fp.exists():
                        self.code_to_optimize_fp = self.folder / "init.cu"
                        self.code_to_optimize_fp.write_text(self.code_to_optimize)
                    _, _, _, baseline_measurement = await self._profile_candidate(self.code_to_optimize_fp)
                    self.initial_measurement = baseline_measurement
            except Exception as e:
                self.agent_logger.warning(
                    f"Failed to obtain baseline cycles before finalizing result: {e}"
                )

            try:
                preliminary_speedup = (
                    best_measurement.speedup_over(self.initial_measurement)
                    if self.initial_measurement is not None
                    else None
                )
            except MeasurementComparisonError as error:
                preliminary_speedup = None
                self.agent_logger.warning(f"Cannot rank final result: {error}")
            performance_gate = {
                "passed": False,
                "reason": "Five-session CUDA Events confirmation was not run.",
            }
            confirmation_error: str | None = None
            confirm_pair = getattr(self.profiler_backend, "confirm_pair", None)
            if confirm_pair is not None and self.initial_measurement is not None:
                try:
                    gate_result = await confirm_pair(
                        self.code_to_optimize_fp,
                        best_filename,
                    )
                    performance_gate = gate_result.to_dict()
                except Exception as error:
                    confirmation_error = f"{type(error).__name__}: {error}"
                    performance_gate = {
                        "passed": False,
                        "reason": "CUDA Events confirmation failed.",
                        "error": confirmation_error,
                    }

            # 搜索阶段的改进只是暂定结果；只有独立的配对性能门控通过后，
            # 候选才能晋升为正式结果产物。
            if performance_gate.get("passed") is True:
                final_filename = self.folder / "success_rl_optimization.cu"
                final_filename.write_text(
                    best_filename.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                return RunOutcome(
                    status=RunStatus.IMPROVED,
                    artifact_path=final_filename,
                    profiling_mode=self.profiling_mode,
                    metrics={
                        "baseline_measurement": self.initial_measurement.to_dict(),
                        "best_measurement": best_measurement.to_dict(),
                        "search_speedup": preliminary_speedup,
                        "performance_gate": performance_gate,
                    },
                )

            failure_file = self.folder / "failure_rl_optimization.cu"
            baseline_str = format_measurement(self.initial_measurement)
            try:
                failure_file.write_text(
                    self.code_to_optimize + f"\n\n// Measurement: {baseline_str}\n"
                )
            except Exception:
                try:
                    init_fp = getattr(self, "code_to_optimize_fp", None)
                    if init_fp and init_fp.exists():
                        failure_file.write_text(
                            init_fp.read_text() + f"\n\n// Measurement: {baseline_str}\n"
                        )
                except Exception:
                    pass
            self.agent_logger.error(
                "RL did not produce an improvement; wrote failure_rl_optimization.cu with baseline (if available)"
            )
            return RunOutcome(
                status=RunStatus.NO_IMPROVEMENT,
                artifact_path=failure_file,
                reason=(
                    "Candidate search completed, but the formal performance gate "
                    "did not confirm an improvement."
                ),
                profiling_mode=self.profiling_mode,
                metrics={
                    "baseline_measurement": (
                        self.initial_measurement.to_dict()
                        if self.initial_measurement is not None else None
                    ),
                    "best_measurement": best_measurement.to_dict(),
                    "search_speedup": preliminary_speedup,
                    "performance_gate": performance_gate,
                },
            )

        # 没有产生候选人的轨迹。仍然编写一个失败文件（如果有的话，带有基线）。
        try:
            if self.initial_measurement is None:
                init_fp = getattr(self, "code_to_optimize_fp", None)
                if not init_fp or not init_fp.exists():
                    self.code_to_optimize_fp = self.folder / "init.cu"
                    self.code_to_optimize_fp.write_text(self.code_to_optimize)
                _, _, _, measurement = await self._profile_candidate(self.code_to_optimize_fp)
                self.initial_measurement = measurement
                self.best_measurement = measurement
        except Exception as e:
            self.agent_logger.warning(f"Failed to obtain baseline cycles for original code: {e}")

        fallback_measurement = format_measurement(self.initial_measurement)
        failure_file = self.folder / "failure_rl_optimization.cu"
        try:
            failure_file.write_text(self.code_to_optimize + f"\n\n// Measurement: {fallback_measurement}\n")
        except Exception:
            try:
                init_fp = getattr(self, "code_to_optimize_fp", None)
                if init_fp and init_fp.exists():
                    failure_file.write_text(
                        init_fp.read_text()
                        + f"\n\n// Measurement: {fallback_measurement}\n"
                    )
            except Exception:
                pass
        self.agent_logger.error(
            "All RL iterations failed; wrote failure_rl_optimization.cu with baseline (if available)"
        )
        permission_blocked = any(
            "ERR_NVGPUCTRPERM" in error for error in iteration_errors
        )
        return RunOutcome(
            status=(RunStatus.BLOCKED if permission_blocked else RunStatus.FAILED),
            artifact_path=failure_file,
            reason=(
                "NCU performance-counter access is blocked."
                if permission_blocked
                else "All RL iterations failed before producing a valid candidate."
            ),
            profiling_mode=self.profiling_mode,
            metrics={"iteration_errors": iteration_errors},
        )

    async def gather_perf_metrics(self, filepath: Path) -> Tuple[str, str, str, int]:
        """
        使用 NCU 分析收集性能指标。

        参数:
        filepath: 目标文件路径。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        # 重用 opt_ncu_annot_fixed5.py 中的现有分析逻辑
        # 使用单次执行运行以避免非确定性内核导致虚假
        # 重复运行时验证失败。
        stdout_list, stderr_list, path, success = await compile_and_run_cu_file(
            self.test_code_fp,
            filepath,
            self.gpu,
            NamedTimer(),
            self.agent_logger,
            persistent_artifacts=True,
            timeout=3600,
            num_runs=1,
            passed_keyword="passed",
        )
        
        if not success:
            FeedbackAgent.raise_numerics_verification_error(stdout_list, stderr_list)

        # 可选：仅循环模式，以避免在代理流程中包含完整的 NCU 日志。
        # 仍然运行 NCU 以获得准确的周期计数，但仅返回周期（而不是完整日志）。
        cycles_only = os.getenv("KERNELAGENT_RL_NCU_CYCLES_ONLY", "0") in (
            "1",
            "true",
            "True",
            "yes",
            "YES",
            "y",
            "on",
            "ON",
        )
        if cycles_only:
            err_text = "\n".join(stderr_list or [])
            cycles = None
            try:
                # 仍然运行 NCU 以从光速部分获得准确的周期计数
                kernel_names = await find_kernel_names_ncu(path, filepath, self.gpu, 3600)
                
                if not kernel_names:
                    raise ValueError("No kernel names found for NCU profiling")
                
                # 运行基本 NCU 分析以获取周期（包括光速部分）
                # 使用第一个内核名称（大多数内核都有一个主内核）
                kernel_name = kernel_names[0]
                ncu_stdout, ncu_stderr = await run_gpu_executable(
                    path, self.gpu, 3600,
                    job_name=f"{filepath} (ncu cycles-only)",
                    prefix_command=f"NVIDIA_TF32_OVERRIDE=0 ncu -k {kernel_name}",
                )
                
                if "No Kernels were profiled" in ncu_stdout:
                    raise ValueError("NCU did not profile any kernels")
                
                # 使用现有实用函数从 NCU 输出解析周期
                cycles = get_elapsed_cycles_ncu_log(ncu_stdout)
                
                err_text += f"\nNCU stderr: {ncu_stderr}"
                
            except Exception as e:
                self.agent_logger.warning(
                    f"KERNELAGENT_RL_NCU_CYCLES_ONLY is set but failed to parse elapsed cycles from NCU output: {e}"
                )
                cycles = None  # 使用None代替0表示解析失败
            # 返回空的 NCU 日志/注释，以便提示保持较小。
            # 如果 Cycles 为 None（解析失败），则使用 0 以保持与 int 返回类型的向后兼容性
            return "", "", err_text, cycles if cycles is not None else 0

        kernel_names = await find_kernel_names_ncu(path, filepath, self.gpu, 3600)
        
        # 调试：记录正在分析的内核名称
        self.agent_logger.info(f"Profiling {len(kernel_names)} kernel(s) from CUDA file: {kernel_names}")

        # 单个 NCU 调用以获取详细信息 CSV 和原始日志
        # 使用 --csv 标志获取 CSV 格式进行解析，但输出仍然包含嵌入 CSV 的全文
        # 构建内核过滤器：如果是单内核，则使用 -k 标志；如果有多个，则分析全部（无 -k 标志）
        if len(kernel_names) == 1:
            # 单内核：使用 -k 标志进行过滤
            kernel_filter = f"-k {kernel_names[0]}"
        else:
            # 多个内核：分析全部（NCU 不支持多个 -k 标志）
            # 我们将在后处理中进行过滤，仅处理 CUDA 文件中的内核
            kernel_filter = ""
            self.agent_logger.debug(f"Multiple kernels detected, profiling all and filtering to: {kernel_names}")
        
        details_command = (
            f"ncu {kernel_filter} --page details --section=SchedulerStats --section=Occupancy --section=SpeedOfLight --section=LaunchStats --section=WarpStateStats --section=InstructionStats --csv --metrics "
            + ",".join(UTILIZATION_METRICS)
        )

        # 在单个 NCU 调用中分析内核
        # 从一次调用中获取详细信息 CSV（从文本解析）和原始日志
        details_stdout, details_stderr = await run_gpu_executable(
            path, self.gpu, 3600,
            job_name=f"{filepath} (details)",
            prefix_command=f"NVIDIA_TF32_OVERRIDE=0 {details_command} ",
        )

        if "No Kernels were profiled" in details_stdout:
            self.agent_logger.warning(f"No kernels were profiled for {filepath}")
            return "", "", details_stderr, 0
        
        # 使用原始日志的详细输出（它包含全面的分析信息）
        combined_ncu_logs = details_stdout
        
        stderr = f"details: {details_stderr}\n"
        
        # 解析详细的 CSV 输出并按内核拆分
        try:
            all_details_df = format_ncu_details_as_csv(details_stdout)
        except ValueError as e:
            raise ValueError(f"Failed to extract CSV from NCU logs: {e}")

        # 按内核名称拆分详细数据帧
        details_dfs = []
        cycles = 0
        
        # 有关详细信息 CSV，按“内核名称”列拆分
        # 仅处理 CUDA 文件中找到的内核（来自 find_kernel_names_ncu）
        if "Kernel Name" in all_details_df.columns:
            # 记录我们发现的内容与我们期望的内容
            all_profiled_kernels = all_details_df["Kernel Name"].str.split("(").str[0].str.strip().unique().tolist()
            self.agent_logger.info(
                f"Found {len(all_profiled_kernels)} kernels in NCU CSV output: {all_profiled_kernels}"
            )
            self.agent_logger.info(
                f"Processing {len(kernel_names)} kernels from CUDA file: {kernel_names}"
            )
            
            # 仅处理 CUDA 文件中找到的内核
            for kernel_name in kernel_names:
                # 过滤此内核的行（处理带或不带参数的内核名称）
                kernel_base_name = kernel_name.split("(")[0].strip()
                name_series = all_details_df["Kernel Name"].astype(str)
                base_series = name_series.str.split("(").str[0].str.strip()

                # 首先尝试精确的基本名称匹配
                kernel_mask = base_series == kernel_base_name

                # 如果没有行，则回退到模糊包含匹配来处理模板，例如
                # “void linear_bias_relu_kernel<1>”与“linear_bias_relu_kernel”
                if not kernel_mask.any():
                    import re as _re

                    pattern = _re.escape(kernel_base_name)
                    kernel_mask = base_series.str.contains(pattern, case=False, regex=True)

                kernel_details_df = all_details_df[kernel_mask].copy()
                
                if len(kernel_details_df) > 0:
                    details_dfs.append(kernel_details_df)
                    
                    # 从该内核的详细信息中获取周期
                    for _, row in kernel_details_df.iterrows():
                        if row["Metric Name"] == "Elapsed Cycles":
                            cycles += int(row["Metric Value"].replace(",", ""))
                    
                    self.agent_logger.debug(
                        f"Extracted {len(kernel_details_df)} metric rows for kernel '{kernel_name}'"
                    )
                else:
                    # 未找到此内核的详细信息 - 如果跳过源分析，则可能会发生这种情况
                    # 或者如果内核没有实际执行，或者内核名称不匹配
                    # 尝试模糊匹配来帮助诊断
                    similar_kernels = [
                        k for k in all_profiled_kernels 
                        if kernel_base_name.lower() in k.lower() or k.lower() in kernel_base_name.lower()
                    ]
                    if similar_kernels:
                        self.agent_logger.warning(
                            f"Expected kernel '{kernel_name}' was not found in NCU details. "
                            f"Similar kernel names found: {similar_kernels}. "
                            f"This may indicate a kernel name mismatch or the kernel was not executed."
                        )
                    else:
                        self.agent_logger.warning(
                            f"Expected kernel '{kernel_name}' was not found in NCU details - "
                            f"may not have been executed. Profiled kernels: {all_profiled_kernels}"
                        )
                    # 添加空数据框以保持对齐
                    details_dfs.append(pd.DataFrame())
        else:
            # 后备：如果没有“内核名称”列，则假定为单个内核
            self.agent_logger.warning("No 'Kernel Name' column in NCU CSV - assuming single kernel")
            details_dfs.append(all_details_df)
            for _, row in all_details_df.iterrows():
                if row["Metric Name"] == "Elapsed Cycles":
                    cycles += int(row["Metric Value"].replace(",", ""))

        # 创建空的源数据帧（不需要源分析 - 我们有原始日志和详细信息）
        # annotate_source 函数需要 source_dfs，但我们将传递空的，因为我们不需要每行注释
        # 确保 source_dfs 与 details_dfs 的数量匹配（现在包括所有分析的内核）
        source_dfs = [pd.DataFrame() for _ in range(len(details_dfs))]

        # 注释源（仅使用详细信息，源注释将是最小/空）
        # 为 CSV 中识别出的所有 Kernel 生成性能分析摘要。
        annotated_ncu = annotate_source(filepath, source_dfs, details_dfs)
        
        # 处理内容的日志摘要
        kernels_with_details = sum(1 for df in details_dfs if not df.empty)
        self.agent_logger.info(
            f"NCU profiling summary: {kernels_with_details}/{len(details_dfs)} kernels have detailed metrics"
        )

        # 仅提取 GPU 光速吞吐量部分以减少令牌使用
        # 与最小代理类似 - 仅包含摘要信息，不包含完整的详细日志
        combined_ncu_logs = self._extract_speed_of_light_section(combined_ncu_logs, kernel_names)
        
        return annotated_ncu, combined_ncu_logs, stderr, cycles
    
    def _extract_speed_of_light_section(self, ncu_output: str, kernel_names: list) -> str:
        """
        从 NCU 日志中仅提取 GPU 光速吞吐量部分。
        返回带有内核名称的简化日志以及每个内核的摘要表。
        这显着减少了令牌的使用，同时保留了基本的性能指标。

        参数:
        ncu_output: 调用方提供的 `ncu_output` 参数。
        kernel_names: 调用方提供的 `kernel_names` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        import re
        
        sections = []
        
        # 按内核标记（如果存在）分割
        kernel_blocks = []
        if "[Kernel:" in ncu_output:
            # 按内核标记拆分（来自我们的手动标记）
            kernel_pattern = r"\[Kernel: ([^\]]+)\]\n(.*?)(?=\[Kernel:|\Z)"
            for match in re.finditer(kernel_pattern, ncu_output, re.DOTALL):
                kernel_name = match.group(1)
                kernel_log = match.group(2)
                kernel_blocks.append((kernel_name, kernel_log))
        else:
            # 无内核标记 - NCU 在每个部分之前输出内核信息
            # 在“部分：GPU 光吞吐量速度”之前查找内核名称模式
            section_pattern = r"Section: GPU Speed Of Light Throughput"
            section_matches = list(re.finditer(section_pattern, ncu_output, re.MULTILINE))
            
            for i, section_match in enumerate(section_matches):
                # 从节头向后查找内核名称
                section_start = section_match.start()
                # 获取本节之前的 50 行以查找内核名称
                lines_before = ncu_output[max(0, section_start - 5000):section_start]
                
                # 尝试在该部分之前的行中查找内核名称
                kernel_name = None
                for known_kernel in kernel_names:
                    # 查找内核名称模式：kernel_name@、kernel_name( 或 [时间戳] kernel_name
                    # 转义内核名称中的特殊正则表达式字符
                    escaped_name = re.escape(known_kernel)
                    kernel_patterns = [
                        rf"{escaped_name}@",  # Kernel 名称格式示例：kernel_name@...
                        rf"{escaped_name}\(",  # Kernel 名称格式示例：kernel_name(...
                        rf"\[.*?\]\s+{escaped_name}",  # [时间戳] kernel_name
                        rf"==PROF==.*?{escaped_name}",  # ==教授== ... kernel_name
                    ]
                    for pattern in kernel_patterns:
                        if re.search(pattern, lines_before, re.IGNORECASE | re.MULTILINE):
                            kernel_name = known_kernel
                            break
                    if kernel_name:
                        break
                
                # 如果我们无法匹配，请使用基于索引的匹配作为后备
                if kernel_name is None and i < len(kernel_names):
                    kernel_name = kernel_names[i]
                elif kernel_name is None:
                    kernel_name = f"kernel_{i}"
                
                # 提取部分内容
                section_end = section_match.end()
                if i + 1 < len(section_matches):
                    next_section_start = section_matches[i + 1].start()
                    section_content = ncu_output[section_end:next_section_start]
                else:
                    section_content = ncu_output[section_end:]
                
                kernel_blocks.append((kernel_name, section_content))
        
        # 处理每个内核块
        for kernel_name, kernel_log in kernel_blocks:
            # 在此内核日志中查找“部分：GPU 光速吞吐量”部分
            pattern = r"Section: GPU Speed Of Light Throughput\n(.*?)(?=\n\s+Section:|==PROF==|\Z|\[Kernel:)"
            matches = list(re.finditer(pattern, kernel_log, re.DOTALL | re.MULTILINE))
            
            for match in matches:
                table_content = match.group(1)
                # 提取行直到到达表格末尾
                lines = table_content.split('\n')
                table_lines = []
                
                # 始终添加内核名称标头
                table_lines.append(f"Kernel: {kernel_name}")
                table_lines.append("Section: GPU Speed Of Light Throughput")
                
                separator_count = 0
                found_metrics = False
                
                for line in lines:
                    # 检查这是否是分隔线（主要是破折号和空格）
                    is_separator = bool(re.match(r'^[\s-]+$', line))
                    
                    if is_separator:
                        separator_count += 1
                        table_lines.append(line)
                        # 当我们看到指标并点击另一个分隔符后，我们就完成了
                        if found_metrics and separator_count >= 3:
                            break
                    elif separator_count >= 2:
                        # 我们已经过了标题分隔符，现在处于指标中
                        found_metrics = True
                        table_lines.append(line)
                        # 如果我们在指标之后遇到空行（表末尾），则停止
                        if not line.strip() and found_metrics:
                            break
                    elif separator_count == 1:
                        # 标题行（指标名称、指标单位、指标值）
                        table_lines.append(line)
                    else:
                        # 在第一个分隔符之前 - 跳过任何额外的内容
                        continue
                
                # 仅当我们找到实际的表格内容时才添加
                if len(table_lines) > 3:  # 标题 + 至少 2 条分隔线
                    sections.append('\n'.join(table_lines))
        
        if not sections:
            # 后备：尝试更简单的提取 - 只需获取每个节标题后的前 15 行
            pattern = r"Section: GPU Speed Of Light Throughput"
            matches = list(re.finditer(pattern, ncu_output))
            for i, match in enumerate(matches):
                start_pos = match.end()
                # 获取接下来的 15 行
                remaining = ncu_output[start_pos:]
                lines = remaining.split('\n')[:15]
                if lines:
                    kernel_label = f"Kernel: {kernel_names[i] if i < len(kernel_names) else 'unknown'}\n" if kernel_names else ""
                    sections.append(kernel_label + "Section: GPU Speed Of Light Throughput\n" + '\n'.join(lines))
        
        if not sections:
            # 最后的手段：返回最少的循环信息。
            # 降级为信息 – 当使用 --csv 详细信息输出运行时，这是预期的。
            self.agent_logger.info(
                "Could not extract Speed Of Light sections from NCU text; using minimal cycles-only summary"
            )
            simplified = []
            for kernel_name in kernel_names:
                # 尝试找到该内核的循环 - 转义特殊的正则表达式字符
                escaped_name = re.escape(kernel_name)
                cycles_pattern = rf"{escaped_name}.*?Elapsed Cycles\s+\w+\s+(\d+)"
                cycles_match = re.search(cycles_pattern, ncu_output, re.DOTALL | re.IGNORECASE)
                if cycles_match:
                    simplified.append(f"Kernel: {kernel_name}\nElapsed Cycles: {cycles_match.group(1)}")
            if simplified:
                return "\n\n".join(simplified)
            # 如果仍然没有任何内容，则返回空字符串以完全省略该部分
            return ""
        
        return "\n\n".join(sections)

    async def run_rollout(self, initial_code: str, initial_state: str) -> Trajectory:
        """
        运行一条优化 rollout 轨迹，并记录每一步候选、反馈与奖励。

        参数:
        initial_code: 调用方提供的 `initial_code` 参数。
        initial_state: 调用方提供的 `initial_state` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        import json as _json, random, uuid as _uuid
        from dataclasses import asdict

        # --------------------------------------------------------------
        # 为每条轨迹建立独立目录，隔离日志和中间产物。
        # --------------------------------------------------------------
        async with self._trajectory_lock:
            self._next_trajectory_id += 1
            trajectory_index = self._next_trajectory_id

        # 使用 uuid 后缀避免并发运行中的文件夹名称冲突
        _uid = _uuid.uuid4().hex[:8]
        trajectory_dir = self.folder / f"trajectory_{trajectory_index}_{_uid}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        # 初始化轨迹容器
        trajectory = Trajectory()

        current_code: str = initial_code
        current_state: str = initial_state
        current_measurement: Measurement | None = self.initial_measurement
        last_ncu_log: str = getattr(self, "last_ncu_log", "")
        
        self.agent_logger.info(f"Starting rollout from state: {current_state}")
        

        for step in range(self.max_rollout_steps):
            # ----------------------------------------------------------
            # 1) 使用LLM助手分析当前的性能状态
            # ----------------------------------------------------------
            metrics = parse_ncu_metrics(last_ncu_log)
            # print("当前代码：", current_code)
            # 退出(0)
            try:
                profile = await self.database.analyze_performance_state(
                    last_ncu_log,
                    metrics,
                    current_code,
                    elapsed_cycles=(
                        int(current_measurement.value)
                        if current_measurement is not None
                        and current_measurement.unit is MeasurementUnit.CYCLES
                        else None
                    ),
                )
                analysis_json_str = _json.dumps(asdict(profile), indent=2)

                # --------------------------------------------------
                # 2）要求DB生成排名优化计划
                # --------------------------------------------------
                # 随 rollout 步骤动态调整 top_n；cur_iter 使用从 1 开始的计数。
                cur_iter = step + 1
                plan = await self.database.generate_optimization_plan(
                    analysis_json_str, current_code, top_n= max(4,(self.max_rollout_steps-int(cur_iter))))
                
            except Exception as exc:
                self.agent_logger.warning(f"Plan generation failed, falling back: {exc}")
                plan = []

            # ----------------------------------------------------------
            # 3) 选择一种按相关性得分随机加权的技术
            # ----------------------------------------------------------
            optimization_entry = None
            if plan:
                def _safe_rel(x):
                    """
                    处理 `safe_rel` 所表示的内部步骤；该函数不属于稳定的公开接口。

                    参数:
                        x: 调用方提供的 `x` 参数。

                    返回:
                        当前操作产生的结果；具体类型由返回注解和调用约定确定。
                    """
                    try:
                        r = float(x)
                    except (TypeError, ValueError):
                        r = 0.05
                    return min(max(r, 0.0), 1.0)
                # 用于再现性/调试的可选确定性选择。
                # 如果设置，则选择单个最高相关性项目而不是采样。
                force_top1 = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in (
                    "1",
                    "true",
                    "True",
                    "yes",
                    "YES",
                    "y",
                    "on",
                    "ON",
                )
                if force_top1:
                    chosen_plan = max(
                        plan,
                        key=lambda p: _safe_rel(p.get("relevance_score", 0.05)),
                    )
                    self.agent_logger.info(
                        f"KERNELAGENT_DB_FALLBACK_TOP1 is set; deterministically selecting top-1 from LLM plan: "
                        f"{chosen_plan.get('technique')} (relevance {chosen_plan.get('relevance_score', 0.0)})"
                    )
                else:
                    # 减少低相关性选项的相关性
                    weights = [max(_safe_rel(p.get("relevance_score", 0.05)) ** 3, 0.001) for p in plan]
                    chosen_plan = random.choices(plan, weights=weights, k=1)[0]
                technique_name = chosen_plan.get("technique")

                # 帮助程序在数据库中找到相应的条目
                optimization_entry = self._lookup_optim_entry_by_name(technique_name)
                strategy_description = chosen_plan.get("description", "")

                self.agent_logger.info(
                    f"Selected technique from optimisation plan: {technique_name} (relevance {chosen_plan.get('relevance_score', 0.0):.2f})"
                )

            # ----------------------------------------------------------
            # 4) 如果需要，回退到旧数据库选择器
            # ----------------------------------------------------------
            if optimization_entry is None:
                optimization_entry = self.database.select_best_optimization(current_state)
                if optimization_entry is None:
                    # 尝试找到未使用的优化
                    optimization_entry = self.database.select_best_optimization(current_state, exclude_used=True)
                    if optimization_entry is None:
                        # 尝试从所有状态中找到任何优化作为后备
                        all_states_with_optimizations = [
                            state for state, state_data in self.database.optimization_strategies.items()
                            if len(state_data.get("optimizations", [])) > 0
                        ]
                        
                        for state_name in all_states_with_optimizations:
                            optimization_entry = self.database.select_best_optimization(state_name)
                            if optimization_entry is not None:
                                self.agent_logger.info(f"Using fallback optimization from state: {state_name}")
                                break
                        
                        if optimization_entry is None:
                            # 最后的手段：尝试为发现的状态添加默认优化
                            if self._try_add_default_optimizations(current_state):
                                optimization_entry = self.database.select_best_optimization(current_state)
                                if optimization_entry is not None:
                                    self.agent_logger.info(f"Using default optimization for new state: {current_state}")
                    
                    if optimization_entry is None:
                        self.agent_logger.warning(f"No optimization found for state: {current_state}, stopping rollout at step {step}")
                        break
                else:
                    self.agent_logger.info(f"Using unused optimization for state: {current_state}")
            else:
                self.agent_logger.info(f"Using best optimization for state: {current_state}")

            # 根据优化类型获取技术名称
            if isinstance(optimization_entry, CompositeOptimization):
                technique_name = optimization_entry.get_composite_id()
            elif hasattr(optimization_entry, "technique"):
                technique_name = optimization_entry.technique
            else:
                technique_name = str(optimization_entry)
            
            # 测井的安全预测值
            _pred_impr = getattr(optimization_entry, "predicted_improvement", None)
            pred_log = f" (predicted: {_pred_impr}%)" if _pred_impr is not None else ""
            self.agent_logger.info(
                f"Step {step}: Applying {technique_name}{pred_log} | entry_type={type(optimization_entry).__name__}"
            )
            
            try:
                # 应用优化
                with event_context(
                    rollout_id=trajectory_index,
                    stage=f"rollout_step_{step}",
                    candidate_id=f"trajectory_{trajectory_index}_step_{step}",
                ):
                    optimized_code, new_measurement, new_state, new_ncu_log = await self.apply_optimization(
                        current_code,
                        optimization_entry,
                        step,
                        trajectory_dir,
                        strategy_description if 'strategy_description' in locals() else "",
                    )
                
                # 计算实际改进
                try:
                    actual_improvement = (
                        new_measurement.speedup_over(current_measurement) - 1.0
                    ) * 100.0 if current_measurement is not None else 0.0
                    is_faster = (
                        new_measurement.is_faster_than(current_measurement)
                        if current_measurement is not None else False
                    )
                except MeasurementComparisonError as error:
                    self.agent_logger.warning(f"Step {step} cannot be ranked: {error}")
                    actual_improvement = 0.0
                    is_faster = False
                reward = self.calculate_reward(
                    getattr(optimization_entry, "predicted_improvement", None), 
                    actual_improvement,
                    is_faster,
                )
                
                # 创建轨迹步骤
                action_name = (
                    optimization_entry.get_composite_id()
                    if isinstance(optimization_entry, CompositeOptimization)
                    else getattr(optimization_entry, "technique", str(optimization_entry))
                )
                
                traj_step = TrajectoryStep(
                    state=current_state,
                    action=action_name,
                    code=optimized_code,
                    measurement=new_measurement,
                    predicted_improvement=getattr(optimization_entry, "predicted_improvement", 0.0),
                    actual_improvement=actual_improvement,
                    reward=reward
                )
                self.agent_logger.info(f"Adding trajectory step: {traj_step}")
                trajectory.add_step(traj_step)

                self.agent_logger.info(f"Updating database with actual results for {technique_name} in state {current_state} with actual improvement {actual_improvement}")
                # 用实际结果更新数据库
                if isinstance(optimization_entry, CompositeOptimization):
                    self.database.update_composite_optimization_result(
                        current_state,
                        technique_name,
                        actual_improvement,
                        measurement=new_measurement,
                    )
                else:
                    # 用记录器记录这个
                    self.agent_logger.info(f"Updating optimization result for {technique_name} in state {current_state} with actual improvement {actual_improvement}")
                    self.database.update_optimization_result(
                        current_state, 
                        technique_name,
                        actual_improvement,
                        measurement=new_measurement,
                    )
                
                self.agent_logger.info(
                    f"Step {step} result: {format_measurement(new_measurement)} "
                    f"({actual_improvement:.1f}% improvement, reward: {reward:.2f})"
                )
                
                # 更新下一步
                current_code = optimized_code
                current_state = self.next_performance_state(current_state, new_state)
                current_measurement = new_measurement
                last_ncu_log = new_ncu_log or last_ncu_log  # 保留下一次迭代
                
                # 如果严重退化则提前停止（从-20%放宽至-50%至-500%）
                if actual_improvement < -500:  # 速度慢 500% 以上
                    self.agent_logger.warning(f"Stopping rollout due to severe degradation: {actual_improvement:.1f}%")
                    break
                    
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                # 记录详细的回溯和上下文
                try:
                    self.agent_logger.error(
                        f"Error in step {step}: {e}\n"
                        f"Technique: {technique_name} | Entry type: {type(optimization_entry).__name__}\n"
                        f"Raw optimization entry: {optimization_entry}\n"
                        f"Traceback:\n{tb}"
                    )
                except Exception:
                    # 如果记录器格式化失败则回退
                    print(f"Error in step {step}: {e}\n{tb}")
                break
        
        return trajectory

    async def _record_completed_trajectory(self, trajectory: Trajectory) -> None:
        """
        仅记录一次完整的轨迹并触发策略更新。

        参数:
        trajectory: 调用方提供的 `trajectory` 参数。
        """

        self.replay_buffer.add_trajectory(trajectory)
        async with self._trajectory_lock:
            self.total_trajectories += 1
            update_due = (
                self.update_frequency > 0
                and self.total_trajectories % self.update_frequency == 0
            )

        if update_due:
            async with self._policy_lock:
                self.iteration_count += 1
                await self.policy_update_cycle()

    # ------------------------------------------------------------------
    # 通过技术/复合 ID 查找优化条目的帮助器
    # ------------------------------------------------------------------
    def _lookup_optim_entry_by_name(
        self, technique_name: str
    ) -> Optional[OptimizationEntry | CompositeOptimization]:
        # 搜索个人技术
        """
        处理 `lookup_optim_entry_by_name` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
        technique_name: 调用方提供的 `technique_name` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        for state_data in self.database.optimization_strategies.values():
            for opt in state_data.get("optimizations", []):
                if opt.technique == technique_name:
                    return opt

        # 搜索复合优化
        for comps in self.database.composite_optimizations.values():
            for comp in comps:
                if comp.get_composite_id() == technique_name:
                    return comp

        return None

    async def apply_optimization(
        self,
        code: str,
        optimization_entry: OptimizationEntry | CompositeOptimization,
        step: int,
        trajectory_dir: Path | None = None,
        strategy_description: str = "",
    ) -> Tuple[str, int, str, str]:
        """
        应用特定的优化并返回优化的代码、周期、新状态。

        参数:
        code: 待处理的源码文本。
        optimization_entry: 调用方提供的 `optimization_entry` 参数。
        step: 调用方提供的 `step` 参数。
        trajectory_dir: 调用方提供的 `trajectory_dir` 参数。
        strategy_description: 调用方提供的 `strategy_description` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # --------------------------------------------------------------
        # 帮助保存提示/响应对以进行代理检查
        # --------------------------------------------------------------
        def _save_agentic_log(label: str, prompt_text: str, response_text: str):
            """
            保存 `save_agentic_log` 所表示的内部步骤；该函数不属于稳定的公开接口。

            参数:
            label: 调用方提供的 `label` 参数。
            prompt_text: 调用方提供的 `prompt_text` 参数。
            response_text: 调用方提供的 `response_text` 参数。
            """
            if trajectory_dir is None:
                return  # 如果未提供目录，则禁用日志记录
            log_fp = trajectory_dir / "agentic_steps_log.txt"
            with open(log_fp, "a", encoding="utf-8") as f:
                f.write(f"=== {label} ===\n")
                f.write("--- PROMPT ---\n")
                f.write(prompt_text.rstrip() + "\n")
                f.write("--- RESPONSE ---\n")
                f.write(response_text.rstrip() + "\n\n")
        
        # 为此优化尝试创建临时文件
        if isinstance(optimization_entry, CompositeOptimization):
            technique_name = optimization_entry.get_composite_id()
        else:
            technique_name = getattr(optimization_entry, "technique", str(optimization_entry))
        base_label = f"step_{step}_{technique_name}"
        # 若存在轨迹目录，将当前步骤的所有产物统一写入该目录。
        base_dir = trajectory_dir if trajectory_dir is not None else self.folder
        temp_file = base_dir / f"step_{step}_{technique_name}.cu"
        temp_file.write_text(code)
        
        # 收集当前的分析数据；容忍数字验证失败
        try:
            annotated_ncu, ncu_log, _, _ = await self._profile_candidate(temp_file)
        except FeedbackError as prof_err:
            self.agent_logger.warning(
                f"Profiling failed at step {step} with FeedbackError; using empty NCU log. Details: {prof_err}"
            )
            annotated_ncu, ncu_log = "", ""
        except Exception as prof_other:
            self.agent_logger.warning(
                f"Unexpected profiling error at step {step}: {prof_other}; continuing with empty logs."
            )
            annotated_ncu, ncu_log = "", ""
        
        # 生成包含完整数据库内容的策略引导提示
        try:
            database_content = self.database.get_database_md_text()
            # 如果完整数据库为空，则回退到页脚
            if not database_content or database_content.strip() == "":
                self.agent_logger.warning("Database markdown is empty, trying footer")
                database_content = self.database.get_database_footer_text()
                if not database_content or database_content.strip() == "":
                    self.agent_logger.warning("Database footer is also empty, using GPU optimization knowledge")
                    # 最后的后备方案：使用 GPU 优化报告
                    database_content = getattr(self.database, 'gpu_optimization_knowledge', '')[:6000] or ""
        except Exception as e:
            self.agent_logger.warning(f"Failed to load database content: {e}")
            try:
                database_content = self.database.get_database_footer_text()
            except Exception:
                # 最后的后备方案
                database_content = getattr(self.database, 'gpu_optimization_knowledge', '')[:6000] or ""
        
        # 用于调试的日志数据库内容大小
        if database_content:
            self.agent_logger.debug(f"Using database content: {len(database_content)} characters")
        else:
            self.agent_logger.warning("No database content available for prompt")
        prompt = generate_strategy_guided_prompt(
            optimization_entry,
            annotated_ncu,
            ncu_log,
            database_content,
            override_description=strategy_description or None,
            original_code=code,  # 当 annotated_ncu 为空时传递原始代码作为后备
        )

        # 保留初始提示/响应
        # （LLM 响应可用后进行日志记录）
        
        # 从 LLM 获取优化代码
        from .utils import generate_code_retry
        response = await generate_code_retry(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            logger=self.agent_logger,
            max_retries=3
        )

        # 保留初始提示/响应
        _save_agentic_log(f"{base_label}_initial", prompt, response.generations[0])
        
        # 提取并测试优化的代码
        optimized_code, filepath = self.get_code_from_response(
            response.generations[0], step, 0, self.agent_logger
        )
        # 将get_code_from_response生成的中间文件重新定位到轨迹文件夹中以避免碰撞
        try:
            target_fp = base_dir / f"{base_label}_initial.cu"
            # 如果源和目标不同，则移动内容
            if filepath != target_fp:
                # 更喜欢重命名；如果跨设备则回退重写
                try:
                    filepath.rename(target_fp)
                except Exception:
                    target_fp.write_text(optimized_code)
                    try:
                        filepath.unlink()
                    except Exception:
                        pass
            filepath = target_fp
        except Exception:
            # 尽力而为；即使搬迁失败也继续
            pass

        # --------------------------------------------------------------
        # 2) 编译/运行分析并进行自动修复尝试
        # --------------------------------------------------------------
        MAX_FIX_ATTEMPTS = 4  # 尝试自动修复多少次

        attempt_idx = 0
        compile_success = False
        run_success = False
        new_measurement: Measurement | None = None
        new_ncu_log = ""

        while attempt_idx < MAX_FIX_ATTEMPTS:
            # 将（可能固定的）代码写入唯一的文件
            filepath = base_dir / f"{base_label}_attempt{attempt_idx}.cu"
            filepath.write_text(optimized_code)

            try:
                # 分析优化的代码（这会隐式编译+运行它）
                _, new_ncu_log, _, new_measurement = await self._profile_candidate(filepath)

                # 如果到这里就说明编译运行成功了
                compile_success = True
                run_success = True

                # 记录编译/运行成功以供检查
                if trajectory_dir is not None:
                    log_fp = trajectory_dir / "agentic_steps_log.txt"
                    with open(log_fp, "a", encoding="utf-8") as f:
                        f.write(f"Compile success: {compile_success}\n")
                        f.write(f"Run success    : {run_success}\n")
                        f.write(f"Measurement     : {format_measurement(new_measurement)}\n\n")

                break  # 退出重试循环 – 成功

            except Exception as e:
                # 编译或运行时失败 – 捕获错误消息
                error_msg = str(e)

                # 将失败信息附加到代理日志
                if trajectory_dir is not None:
                    log_fp = trajectory_dir / "agentic_steps_log.txt"
                    with open(log_fp, "a", encoding="utf-8") as f:
                        f.write(f"Compile/Run failed on attempt {attempt_idx}: {error_msg}\n\n")

                attempt_idx += 1
                if attempt_idx >= MAX_FIX_ATTEMPTS:
                    # 放弃并传播错误——外部调用者将处理
                    raise

                # ------------------------------------------------------
                # 使用错误消息构建 LLM 的修复提示
                # ------------------------------------------------------
                # 尝试包含优化数据库页脚（包含有用的代码片段）
                db_footer_text = ""
                try:
                    if hasattr(self.database, "optimization_db_footer_path") and self.database.optimization_db_footer_path.exists():
                        db_footer_text = self.database.optimization_db_footer_path.read_text(encoding="utf-8")
                except Exception:
                    db_footer_text = ""

                fix_prompt_parts = [
                    "The previously generated CUDA code failed to compile or run.\n\n",
                    "COMPILER / RUNTIME ERROR LOG:\n```\n",
                    f"{error_msg}\n",
                    "```\n\n",
                    "ORIGINAL CUDA CODE (for reference – please modify in place):\n```cpp\n",
                    f"{optimized_code}\n",
                    "```\n\n",
                ]
                if db_footer_text:
                    fix_prompt_parts.append("OPTIMIZATION DATABASE FOOTER (reference snippets):\n```\n")
                    fix_prompt_parts.append(db_footer_text)
                    fix_prompt_parts.append("\n```\n\n")
                fix_prompt_parts.extend([
                    "Please provide a corrected, fully compilable version of the kernel. Return **complete CUDA code** in one ```cpp``` block.",
                    " Please keep the code structure otherwise unchanged; it is compiled together with separate test code, so do NOT add a main function.\n\n",
                    "Include ALL necessary components:\n",
                    "   - #include statements (cuda_fp16.h, cuda_runtime.h, etc.)\n",
                    "   - #define constants – DEFINE ALL CONSTANTS BEFORE USING THEM\n",
                    "   - Complete __global__ kernel function with proper signature\n",
                    "   - Complete launch_gpu_implementation(void*, void*, void*, int64_t) function\n",
                ])
                fix_prompt = "".join(fix_prompt_parts)

                # 请求LLM修复代码
                fix_response = await generate_code_retry(
                    messages=[{"role": "user", "content": fix_prompt}],
                    model=self.model,
                    logger=self.agent_logger,
                    max_retries=2,
                )

                # 记录修复尝试提示/响应
                _save_agentic_log(
                    f"{base_label}_fix_attempt_{attempt_idx}",
                    fix_prompt,
                    fix_response.generations[0],
                )

                # 为下一次迭代提取新代码
                optimized_code, fix_fp = self.get_code_from_response(
                    fix_response.generations[0], step, attempt_idx, self.agent_logger
                )
                # 将中间修复文件重新定位到轨迹目录以避免碰撞
                try:
                    fix_target_fp = base_dir / f"{base_label}_attempt{attempt_idx}_llm.cu"
                    if fix_fp != fix_target_fp:
                        try:
                            fix_fp.rename(fix_target_fp)
                        except Exception:
                            fix_target_fp.write_text(optimized_code)
                            try:
                                fix_fp.unlink()
                            except Exception:
                                pass
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # 3）确定新状态（仅当编译/运行成功时）
        # ------------------------------------------------------------------
        new_metrics = parse_ncu_metrics(new_ncu_log)
        # new_state = 等待 self.database.get_state_from_ncu_report(new_ncu_log, new_metrics)
        try:
            new_state = await self._classify_profile_state(
                new_ncu_log,
                new_metrics,
                optimized_code,
                new_measurement,
            )
        except Exception as state_error:
            self.agent_logger.warning(
                f"Could not classify the optimized candidate state: {state_error}"
            )
            new_state = "events_only/unknown" if not new_ncu_log else "unknown"
        if new_measurement is None:
            raise ProfilerUnavailable("Optimization completed without a measurement")
        return optimized_code, new_measurement, new_state, new_ncu_log

    def calculate_reward(self, predicted_improvement: Optional[float], actual_improvement: float, 
                        is_faster: bool) -> float:
        """
        根据预测准确性和实际表现计算奖励。
        通过跳过精度奖励来安全地处理无/零 predicted_improvement。

        参数:
        predicted_improvement: 调用方提供的 `predicted_improvement` 参数。
        actual_improvement: 调用方提供的 `actual_improvement` 参数。
        is_faster: 调用方提供的 `is_faster` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        
        # 改进的基本奖励
        base_reward = actual_improvement / 100.0  # 将百分比转换为分数
        
        # 预测准确性奖励
        try:
            safe_predicted = float(predicted_improvement) if predicted_improvement is not None else 0.0
        except (TypeError, ValueError):
            safe_predicted = 0.0
        
        if safe_predicted > 0.0:
            accuracy = min(actual_improvement / safe_predicted, 2.0)
            if 0.8 <= accuracy <= 1.2:  # 好的预测
                accuracy_bonus = 0.2
            else:  # 预测不佳
                accuracy_bonus = -0.1 * abs(accuracy - 1.0)
        else:
            accuracy_bonus = 0.0
        
        # 让事情变得更糟的惩罚
        penalty = -0.5 if not is_faster else 0.0
        
        return base_reward + accuracy_bonus + penalty

    async def policy_update_cycle(self):
        """运行策略评估和更新周期。"""
        if len(self.replay_buffer.trajectories) < 3:
            return  # 需要一些轨迹来分析
        
        self.agent_logger.info("Running policy update cycle...")
        
        try:
            # 政策评估
            evaluation_result = await self.policy_evaluation_agent.evaluate_policy(
                self.replay_buffer, self.database
            )
            
            # 收集最近的故障以进行差距分析
            recent_failures = []
            for traj in self.replay_buffer.get_recent_trajectories(5):
                for step in traj.steps:
                    predicted = step.predicted_improvement or 0.0
                    if step.reward < 0 or step.actual_improvement < predicted * 0.5:
                        recent_failures.append(step)
            
            # 绩效差距分析
            gap_analysis = await self.perf_gap_analysis_agent.analyze_performance_gaps(
                evaluation_result, recent_failures
            )
            
            # 参数更新
            updates = await self.parameter_update_agent.update_parameters(
                gap_analysis, self.database
            )
            
            # 保存分析结果
            analysis_file = self.folder / f"analysis_iteration_{self.iteration_count}.json"
            analysis_data = {
                'iteration': self.iteration_count,
                'evaluation_result': evaluation_result,
                'gap_analysis': gap_analysis,
                'updates': updates,
                'buffer_stats': self.replay_buffer.get_statistics(),
                'database_stats': self.database.get_database_stats()
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            self.agent_logger.info(f"Policy update completed. Analysis saved to {analysis_file}")
            
        except Exception as e:
            self.agent_logger.error(f"Error in policy update cycle: {e}")

    async def get_feedback(self, response, attempt_id, task_id, logger) -> Feedback:
        """
        实现 RL 算法的主反馈回路。

        参数:
        response: 需要解析或规范化的服务响应。
        attempt_id: 调用方提供的 `attempt_id` 参数。
        task_id: 调用方分配的任务唯一标识。
        logger: 记录诊断信息和任务进度的日志器。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        
        # 如果这是第一次调用则初始化
        if self.initial_measurement is None:
            await self.initialize()
        
        # 为这个任务开启一个新的轨迹
        logger.info(f"Starting RL optimization trajectory for task {task_id}")
        
        # 获取初始状态
        temp_file = self.folder / f"temp_task_{task_id}.cu"
        code, filepath = self.get_code_from_response(response, attempt_id, task_id, logger)
        
        try:
            # 分析初始代码，建立 rollout 的起始性能状态。
            annotated_ncu, ncu_log, _, measurement = await self._profile_candidate(filepath)
            metrics = parse_ncu_metrics(ncu_log)
            initial_state = await self.database.get_state_from_ncu_report(
                ncu_log,
                metrics,
                code,
                elapsed_cycles=(
                    int(measurement.value)
                    if measurement.unit is MeasurementUnit.CYCLES else None
                ),
            )
            
            # 运行优化 rollout。
            trajectory = await self.run_rollout(code, initial_state)
            
            await self._record_completed_trajectory(trajectory)
            
            # 更新最佳表现
            try:
                is_faster = (
                    trajectory.final_measurement is not None
                    and (
                        self.best_measurement is None
                        or trajectory.final_measurement.is_faster_than(self.best_measurement)
                    )
                )
            except MeasurementComparisonError:
                is_faster = False
            if is_faster:
                self.best_measurement = trajectory.final_measurement
            
            # 准备反馈消息
            if trajectory.steps:
                best_step = trajectory.steps[0]
                for candidate in trajectory.steps[1:]:
                    try:
                        if candidate.measurement.is_faster_than(best_step.measurement):
                            best_step = candidate
                    except MeasurementComparisonError:
                        continue
                try:
                    improvement_pct = (
                        (best_step.measurement.speedup_over(self.initial_measurement) - 1.0) * 100.0
                        if self.initial_measurement is not None else 0.0
                    )
                except MeasurementComparisonError:
                    improvement_pct = 0.0
                
                feedback_msg = f"""Optimization trajectory completed with {len(trajectory.steps)} steps.

BEST RESULT:
- Measurement: {format_measurement(best_step.measurement)} (vs initial: {format_measurement(self.initial_measurement)})
- Overall improvement: {improvement_pct:.1f}%
- Best technique: {best_step.action}
- Total reward: {trajectory.total_reward:.2f}

FINAL OPTIMIZED CODE:
```
{best_step.code}
```

The optimization process is learning and adapting. Continue with further optimizations."""
                
                new_messages = [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": feedback_msg}
                ]
                
                # 保存最佳结果
                best_file = self.folder / f"best_task_{task_id}.cu"
                best_file.write_text(best_step.code)
                
                return RLNCUFeedback(
                    new_messages=new_messages,
                    success=True,  # 如果我们完成了一条轨迹，则视为成功
                    filename=str(best_file),
                    contents=best_step.code,
                    elapsed_cycles=best_step.cycles,
                    measurement=best_step.measurement,
                    ncu_log=ncu_log,
                    annotated_ncu=annotated_ncu,
                    optimization_technique=best_step.action,
                    predicted_improvement=best_step.predicted_improvement,
                    actual_improvement=best_step.actual_improvement,
                    state=initial_state
                )
            else:
                # 没有成功的优化步骤
                return RLNCUFeedback(
                    new_messages=[
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": "No successful optimization steps completed. Please try a different approach."}
                    ],
                    success=False,
                    elapsed_cycles=(
                        int(measurement.value)
                        if measurement.unit is MeasurementUnit.CYCLES else None
                    ),
                    measurement=measurement,
                    ncu_log=ncu_log,
                    annotated_ncu=annotated_ncu,
                    state=initial_state
                )
                
        except FeedbackError as e:
            logger.error(f"Error in RL optimization: {e}")
            return Feedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"Optimization failed: {str(e)}. Please fix the issues and try again."}
                ],
                success=False,
                feedback=e.feedback if hasattr(e, 'feedback') else str(e)
            )

    def _try_add_default_optimizations(self, current_state: str) -> bool:
        """
        尝试根据发现的状态的特征为其添加默认优化。

        这是未找到优化时的后备机制。

        参数:
        current_state: 调用方提供的 `current_state` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        try:
            # 根据常见模式定义默认优化
            default_optimizations = {
                "memory_bound": [
                    ("memory_coalescing_optimization", 20.0),
                    ("shared_memory_tiling", 25.0),
                    ("vectorized_processing", 15.0)
                ],
                "compute_bound": [
                    ("instruction_level_parallelism", 30.0),
                    ("fast_math_optimization", 20.0),
                    ("vectorized_operations", 25.0)
                ],
                "latency_bound": [
                    ("occupancy_optimization", 35.0),
                    ("register_pressure_reduction", 30.0),
                    ("work_per_thread_increase", 25.0)
                ],
                "hybrid_bound": [
                    ("memory_compute_overlap", 40.0),
                    ("algorithmic_optimization", 35.0),
                    ("adaptive_tiling", 30.0)
                ]
            }
            
            # 从状态名称中提取主要瓶颈
            primary_bottleneck = None
            for bottleneck in default_optimizations.keys():
                if bottleneck in current_state:
                    primary_bottleneck = bottleneck
                    break
            
            if primary_bottleneck and primary_bottleneck in default_optimizations:
                # 为该状态添加默认优化
                for technique, improvement in default_optimizations[primary_bottleneck]:
                    self.database.add_new_optimization(current_state, technique, improvement)
                
                self.agent_logger.info(f"Added {len(default_optimizations[primary_bottleneck])} default optimizations for state: {current_state}")
                return True
            
        except Exception as e:
            self.agent_logger.error(f"Error adding default optimizations: {e}")
        
        return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取全面的绩效总结。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return {
            'total_trajectories': self.total_trajectories,
            'iteration_count': self.iteration_count,
            'initial_measurement': (
                self.initial_measurement.to_dict() if self.initial_measurement is not None else None
            ),
            'best_measurement': (
                self.best_measurement.to_dict() if self.best_measurement is not None else None
            ),
            'overall_improvement': (
                (self.best_measurement.speedup_over(self.initial_measurement) - 1.0) * 100.0
                if self.initial_measurement is not None
                and self.best_measurement is not None
                and self.initial_measurement.comparison_key == self.best_measurement.comparison_key
                else 0
            ),
            'buffer_stats': self.replay_buffer.get_statistics(),
            'database_stats': self.database.get_database_stats()
        }
