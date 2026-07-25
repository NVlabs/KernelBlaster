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

"""组织单个 Kernel 优化任务的状态图执行、超时处理和终态产物。"""

from __future__ import annotations
import time
import asyncio
import loguru
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator
import shutil

from ..graph import build_graph
from ..config import config, WorkflowConfig
from ..graph.state import save_state_to_json
from ..outcomes import RunOutcome, RunStatus

__all__ = ["WorkflowResult", "run_workflow"]


@dataclass
class WorkflowResult:
    """汇总一次工作流的配置、标准终态和可供调用方读取的成功产物。"""
    config: WorkflowConfig
    rl_cuda_perf_filepath: Path = None  # RL 优化的 CUDA 代码
    outcome: RunOutcome = field(
        default_factory=lambda: RunOutcome(
            status=RunStatus.FAILED,
            reason="Failed code generation due to an error or reaching the maximum number of attempts.",
        )
    )

    @property
    def error(self) -> str:
        """返回明确的失败原因；未提供原因时退回终态名称。"""
        return self.outcome.reason or self.outcome.status.value

    @property
    def timeout(self) -> bool:
        """判断工作流是否因超过顶层时限而结束。"""
        return self.outcome.status is RunStatus.TIMEOUT

    def set_outcome(self, outcome: RunOutcome):
        """
        设置标准终态，并仅在成功时暴露对应 CUDA 产物路径。

        参数:
            outcome: 工作流或异常处理分支产生的标准结果。
        """
        self.outcome = outcome
        self.rl_cuda_perf_filepath = outcome.artifact_path if outcome.success else None

    @property
    def success(self) -> bool:
        """判断工作流是否得到一个存在且可读取的改进产物。"""
        return self.outcome.success

    def agents(self) -> Iterator[str]:
        """
        返回结果对象支持的所有 Agent 名称。
        当前仅支持 RL 优化 Agent。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        if hasattr(self, "rl_cuda_perf_filepath"):
            yield "rl_cuda_perf"

    def running_agents(self) -> Iterator[str]:
        """
        返回按当前实现应当运行的 Agent 名称。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 如果启用，RL 优化始终运行
        yield "rl_cuda_perf"

    @property
    def generated_codes(
        self,
    ) -> dict[str, str]:
        """
        处理 `generated_codes` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        def stringify(filepath: Path | None) -> str | None:
            """
            处理 `stringify` 对应的领域操作，并返回调用方所需的标准化结果。

            参数:
                filepath: 目标文件路径。

            返回:
                当前操作产生的结果；具体类型由返回注解和调用约定确定。
            """
            if filepath is None:
                return None
            return str(filepath)

        # 返回带有 RL 优化的 CUDA 代码文件路径的字典
        return {"rl_cuda_perf": stringify(self.rl_cuda_perf_filepath)}

    def write_failures(
        self,
        folder: str,
    ):
        """
        写入 `write_failures` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            folder: 保存当前任务中间状态和最终产物的目录。
        """
        if not self.success:
            (folder / "failed_rl_cuda_perf").write_text(self.error, encoding="utf-8")
            finished = folder / "rl_ncu" / ".finished"
            finished.parent.mkdir(parents=True, exist_ok=True)
            finished.write_text(self.outcome.status.value + "\n", encoding="utf-8")

    def remove_existing_files(self, folder: Path):
        """
        删除 `remove_existing_files` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            folder: 保存当前任务中间状态和最终产物的目录。
        """
        failed_file = folder / "failed_rl_cuda_perf"
        if failed_file.exists() and self.config.retry_failed:
            # 如果设置了 retry_failed 标志并且代理失败，请删除代理文件夹。
            shutil.rmtree(folder / "rl_ncu", ignore_errors=True)
        # 无论 retry_failed 标志如何，都应删除此文件。如果代理的文件夹不包含成功的文件，它将由代理自己重新创建。
        failed_file.unlink(missing_ok=True)


async def run_workflow(
    task_id: str,
    user_message: str,
    reference_code: str,
    folder: Path,
    workflow_config: WorkflowConfig,
    job_logger: loguru.Logger,
    timeout_seconds: int,
    shared_database=None,
) -> WorkflowResult:

    """
    在给定超时内运行优化状态图，并统一收敛成功、失败和超时结果。

    参数:
        task_id: 调用方分配的任务唯一标识。
        user_message: 调用方提供的 `user_message` 参数。
        reference_code: 调用方提供的 `reference_code` 参数。
        folder: 保存当前任务中间状态和最终产物的目录。
        workflow_config: 本次优化任务使用的工作流配置。
        job_logger: 绑定当前任务上下文的日志器。
        timeout_seconds: 允许工作流运行的最长秒数。
        shared_database: 可由多个任务复用的优化数据库实例。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    folder.mkdir(exist_ok=True, parents=True)
    start = time.time()

    job_logger.info(f"Starting workflow for task {task_id}.")
    config.print_config(job_logger)

    result = WorkflowResult(config=workflow_config)

    # 准备运行的输出目录
    result.remove_existing_files(folder)

    workflow = build_graph()
    workflow_input = {
        "user_message": user_message,
        "reference_code": reference_code,
        "folder": folder,
        "logger": job_logger,
        "model": workflow_config.model,
        # 直接从调用者（运行者）传递共享数据库
        "shared_optimization_database": shared_database,
        **workflow_config.dict(),
    }

    try:
        final_state = await asyncio.wait_for(
            workflow.ainvoke(workflow_input),
            timeout=timeout_seconds,
        )
        save_state_to_json(final_state, folder / "state.json")
        outcome_payload = final_state.get("run_outcome")
        outcome = (
            RunOutcome.from_dict(outcome_payload)
            if outcome_payload
            else RunOutcome(
                status=RunStatus.FAILED,
                reason="Workflow completed without a terminal run outcome.",
            )
        )
        result = WorkflowResult(
            config=workflow_config,
            rl_cuda_perf_filepath=(outcome.artifact_path if outcome.success else None),
            outcome=outcome,
        )
    except asyncio.TimeoutError:
        result.set_outcome(
            RunOutcome(
                status=RunStatus.TIMEOUT,
                reason=f"Timeout after {timeout_seconds / 60} minutes",
            )
        )
    except Exception as error:
        job_logger.exception(f"Workflow failed for task {task_id}: {error}")
        result.set_outcome(
            RunOutcome(
                status=RunStatus.FAILED,
                reason=f"{type(error).__name__}: {error}",
            )
        )

    # 成功产物由各 Agent 自行写入。
    # 我们将故障记录在这里，而不是在代理内部，以防出现异常或超时。
    result.write_failures(folder)
    duration = time.time() - start
    job_logger.info(f"Workflow completed in {duration:0.2f} seconds")
    return result
