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

"""定义候选代码反馈循环的基础配置、指标记录和通用 Agent 行为。"""

from __future__ import annotations

import asyncio
from pathlib import Path
import loguru
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
import time
import re

from ..config import config, GPUType
from .utils import (
    generate_code_retry,
    LLMResponse,
    NamedTimer,
    write_jsonl,
    extract_code_from_response,
    write_code_to_file,
    FeedbackError,
    convert_messages_to_string,
)
from .utils.query import (
    extract_code_from_response,
    process_messages,
    trim_messages,
    TrimError,
)

__all__ = ["FeedbackConfig", "FeedbackAgent", "FeedbackError"]


@dataclass
class FeedbackConfig:
    """FeedbackAgent 及其子类的配置。"""

    agent_name: str
    base_folder: Path
    logger: loguru.Logger
    init_user_prompt: str
    model: str
    gpu: GPUType
    test_code_fp: Optional[Path] = None
    retry_failed: bool = False
    num_pgen: int = config.NUM_PARALLEL_GENERATIONS_PER_ATTEMPT
    max_attempts: int = config.MAX_ATTEMPTS
    system_prompt: Optional[str] = None
    file_rules: list[Callable] = field(default_factory=list)


@dataclass
class Feedback:
    """封装 `Feedback` 对应的领域状态与操作。"""
    new_messages: list[dict] = field(default_factory=list)
    llm_calls: list[LLMResponse] = field(default_factory=list)
    success: bool = False
    filename: str = None
    contents: str = None
    # 代理中各个步骤所花费时间的字典。
    # 键是步骤的名称，值是以秒为单位的时间。
    # 使用 task_timer 会自动将所花费的时间添加到持续时间字典中。
    durations: dict[str, float] = field(default_factory=dict)
    feedback: str = None


def write_metrics(filepath: Path, threads: dict[int, dict]):
    """
    写入 `write_metrics` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        filepath: 目标文件路径。
        threads: 调用方提供的 `threads` 参数。
    """
    metrics_file = [
        {
            "attempt_id": attempt_id,
            "thread_id": thread_id,
            **asdict(feedback),
            "version": 1.2,
        }
        for thread_id in threads
        for attempt_id, feedback in enumerate(threads[thread_id]["feedbacks"])
    ]
    metrics_file = list(
        sorted(metrics_file, key=lambda x: (x["attempt_id"], x["thread_id"]))
    )
    write_jsonl(filepath, metrics_file)


class FeedbackAgent:
    """封装候选生成、编译运行、正确性验证和指标反馈的通用循环。"""
    def __init__(
        self,
        fb_config: FeedbackConfig,
    ):
        """
        初始化 FeedbackAgent 实例，并保存后续流程所需的配置与依赖。

        参数:
            fb_config: 调用方提供的 `fb_config` 参数。
        """
        self.fb_config = fb_config

        self.agent_name = fb_config.agent_name
        self.base_folder = Path(fb_config.base_folder)
        self.folder = self.base_folder / self.agent_name
        self.system_prompt = fb_config.system_prompt
        self.init_user_prompt = fb_config.init_user_prompt
        self.model = fb_config.model
        self.num_pgen = fb_config.num_pgen
        self.max_attempts = fb_config.max_attempts

        self.folder.mkdir(exist_ok=True, parents=True)
        self.timers = []

        # 为此代理添加自定义记录器
        self.agent_logger = fb_config.logger.bind(
            agent_name=self.agent_name, folder=str(self.folder)
        )

        self.task_loggers = []
        self.file_rules = fb_config.file_rules
        self.retry_failed = fb_config.retry_failed
        self.gpu = fb_config.gpu

    def check_rules(self, code: str):
        """
        检查 `check_rules` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            code: 待处理的源码文本。
        """
        for rule in self.file_rules:
            rule(code)

    def get_intermediate_filepath(self, attempt_id, task_id) -> Path:
        """
        获取 `get_intermediate_filepath` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            attempt_id: 调用方提供的 `attempt_id` 参数。
            task_id: 调用方分配的任务唯一标识。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return self.folder / f"attempt{attempt_id}_task{task_id}.cu"

    def get_ids_from_filepath(self, filepath: Path) -> tuple[int, int]:
        """
        从文件路径获取 attempt_id 和 task_id。
        文件路径的格式应为 *attempt<attempt_id>_task<task_id>.*

        参数:
            filepath: 目标文件路径。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        match = re.search(r"attempt(\d+)_task(\d+)", filepath.stem)
        assert match, f"Failed to parse filepath for attempt and task id: {filepath}"
        assert (
            len(match.groups()) == 2
        ), f"Failed to parse filepath for attempt and task id: {filepath}"
        return int(match.group(1)), int(match.group(2))

    def get_code_from_response(
        self, response, attempt_id, task_id, logger
    ) -> tuple[str, Path]:
        """
        获取 `get_code_from_response` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            response: 需要解析或规范化的服务响应。
            attempt_id: 调用方提供的 `attempt_id` 参数。
            task_id: 调用方分配的任务唯一标识。
            logger: 记录诊断信息和任务进度的日志器。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
            FeedbackError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        code = extract_code_from_response(response)
        if code is None:
            raise FeedbackError(
                "Error: The code should be contained within ```cpp and ``` tags."
            )
        filepath = self.get_intermediate_filepath(attempt_id, task_id)
        write_code_to_file(code, filepath, logger)
        return code, filepath

    async def get_feedback(self, response, attempt_id, task_id) -> Feedback:
        # 在子类中实现
        """
        获取 `get_feedback` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            response: 需要解析或规范化的服务响应。
            attempt_id: 调用方提供的 `attempt_id` 参数。
            task_id: 调用方分配的任务唯一标识。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
            NotImplementedError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        raise NotImplementedError

    async def initialize(self):
        # 可选地在子类中实现
        """初始化 `initialize` 对应的领域操作，并返回调用方所需的标准化结果。"""
        return

    def choose_best_task(self, successful_tasks: list[Path]) -> Path:
        # 可选地在子类中实现
        """
        选择 `choose_best_task` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            successful_tasks: 调用方提供的 `successful_tasks` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return successful_tasks[0]

    @staticmethod
    def raise_numerics_verification_error(
        stdouts: list[str], stderr: list[str], custom_msg=""
    ) -> FeedbackError:
        """
        处理 `raise_numerics_verification_error` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            stdouts: 调用方提供的 `stdouts` 参数。
            stderr: 调用方提供的 `stderr` 参数。
            custom_msg: 调用方提供的 `custom_msg` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
            FeedbackError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        if not custom_msg:
            custom_msg = "The numerics verification failed."
        raise FeedbackError(
            f"{custom_msg}. Please check your implementation carefully and try again. \nstdout:\n{stdouts[0]}\nstderr:\n{stderr[0]}",
        )

    @staticmethod
    def raise_time_measurement_error(
        stdouts: list[str], stderr: list[str]
    ) -> FeedbackError:
        """
        处理 `raise_time_measurement_error` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            stdouts: 调用方提供的 `stdouts` 参数。
            stderr: 调用方提供的 `stderr` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
            FeedbackError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        raise FeedbackError(
            f"The time measurement failed. Please check your implementation carefully and try again:\nstdout:\n{stdouts[0]}\nstderr:\n{stderr[0]}",
        )

    def check_for_existing_run(self) -> str | None:
        """
        检查 `check_for_existing_run` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        finished_fp = self.folder / ".finished"
        successful_files = list(self.folder.glob("success_*"))

        # 这些任务名为 success_attempt<attempt_id>_task<task_id>.cu
        # 我们希望根据 attempt_id 对任务进行排序，然后对 task_id 进行排序
        # 按 attempt_id 和 task_id 对任务进行排序
        successful_files.sort(key=lambda x: self.get_ids_from_filepath(x))

        attempt_ids = [self.get_ids_from_filepath(p)[0] for p in successful_files]

        # 如果我们通过多次尝试找到成功的文件，则仅保留最近的尝试。
        # 当存在先前的成功运行且工作流程重新启动时，可能会发生这种情况
        # （e.g.，retry_failed=True）在同一文件夹中产生新的尝试。我们没有失败，而是
        # 过滤到最新的尝试并继续。
        unique_attempt_ids = set(attempt_ids)
        if len(unique_attempt_ids) > 1:
            latest_attempt_id = max(unique_attempt_ids)
            self.agent_logger.warning(
                f"Detected successful files from multiple attempts {sorted(unique_attempt_ids)} in {self.folder}. "
                f"Using the latest attempt id {latest_attempt_id}."
            )
            successful_files = [
                p for p in successful_files if self.get_ids_from_filepath(p)[0] == latest_attempt_id
            ]
            # 过滤后重新计算attempt_ids
            attempt_ids = [latest_attempt_id] * len(successful_files)

        if finished_fp.exists() and len(successful_files):
            filename = self.choose_best_task(successful_files)
            self.agent_logger.warning(
                f"Found an existing solution for this agent: {filename}"
            )
            return filename

        elif finished_fp.exists() and not len(successful_files):
            if self.retry_failed:
                finished_fp.unlink()
                return None
            else:
                self.agent_logger.warning(
                    f"No successful solutions found for this agent and retry_failed flag is not set. Skipping problem as failed."
                )
                return "__failed__"

        existing_files = list(self.folder.glob("*"))
        if len(existing_files):
            self.agent_logger.warning(
                f"No successful solutions found for this agent. Regenerating..."
            )
            import shutil
            for file in existing_files:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    # 删除目录（如轨迹目录）
                    shutil.rmtree(file)
            return None

    async def __run(self, messages, attempt_id, task_id) -> Feedback:
        """
        处理 `__run` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            messages: 按对话顺序排列的 LLM 消息。
            attempt_id: 调用方提供的 `attempt_id` 参数。
            task_id: 调用方分配的任务唯一标识。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        timer = self.timers[task_id]
        logger = self.task_loggers[task_id]

        prompt_path = self.folder / f"attempt{attempt_id}_task{task_id}_prompt.md"
        prompt_path.write_text(convert_messages_to_string(messages))

        timer.reset()
        timer.start("attempt")
        logger.info(f"Generating response with {self.model}...")
        try:
            response = await generate_code_retry(messages, self.model, logger)
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            return Feedback()
        assert response.generations, "No generations found"
        logger.info(
            f"Response generation completed in {response.elapsed_time:0.2f} seconds"
        )

        generation = response.generations[0]
        prompt_path.write_text(
            convert_messages_to_string(
                messages, response=generation, usage=response.usage
            )
        )
        new_feedback = await self.get_feedback(generation, attempt_id, task_id, logger)
        timer.stop("attempt")

        new_feedback.llm_calls.insert(0, response)
        new_feedback.durations = self.timers[task_id].elapsed.copy()
        return new_feedback

    async def run(self) -> Path:
        """
        运行代理并返回文件名和生成的代码。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        existing_attempt = self.check_for_existing_run()
        if isinstance(existing_attempt, Path):
            return existing_attempt
        elif existing_attempt == "__failed__":
            return None

        successful_tasks = []
        chosen_task = None
        start_time = time.time()

        custom_logger_id = self.agent_logger.add(
            self.folder / "run.log",
            level=config.LOG_LEVEL,
            backtrace=True,
            diagnose=True,
            format=config.CUSTOM_LOGGER_FORMAT,
            filter=lambda record: record["extra"].get("folder") == str(self.folder),
        )
        self.agent_logger.info(f"Running {self.folder}...")

        try:
            await self.initialize()
        except FeedbackError as e:
            self.agent_logger.error(
                f"Failed to initialize agent for {self.folder}: {e}"
            )
            return None

        initial_messages = [
            {
                "role": "system",
                "content": self.system_prompt if self.system_prompt else "",
            },
            {
                "role": "user",
                "content": self.init_user_prompt if self.init_user_prompt else "",
            },
        ]
        threads = {
            i: {
                "messages": process_messages(initial_messages, self.model),
                "feedbacks": [],
                "running": True,
            }
            for i in range(self.num_pgen)
        }
        for attempt in range(self.max_attempts):
            tasks = {}
            self.task_loggers.clear()
            self.timers.clear()

            for i in range(self.num_pgen):
                if not threads[i]["running"]:
                    continue
                try:
                    threads[i]["messages"] = trim_messages(
                        threads[i]["messages"],
                        logger=self.agent_logger,
                    )
                    self.task_loggers.append(
                        self.agent_logger.bind(attempt_id=attempt, task_id=i)
                    )
                    self.timers.append(NamedTimer())
                    tasks[i] = asyncio.create_task(
                        self.__run(
                            deepcopy(threads[i]["messages"]),
                            attempt,
                            i,
                        )
                    )
                except TrimError as e:
                    self.agent_logger.warning(
                        f"Failed to trim messages: {e}, dropping this thread"
                    )
                    threads[i]["running"] = False
                    continue

            feedbacks = await asyncio.gather(
                *[task for task in tasks.values() if task is not None]
            )
            for i, feedback in enumerate(feedbacks):
                if not threads[i]["running"]:
                    continue
                threads[i]["feedbacks"].append(feedback)
                for message in feedback.new_messages:
                    threads[i]["messages"].append(message)
                if feedback.success:
                    successful_tasks.append(feedback.filename)

            # 将指标保存到文件中
            write_metrics(self.folder / "metrics.jsonl", threads)

            if len(successful_tasks):
                # 在这次尝试中发现了成功的任务。选择最好的一个并打破
                chosen_task = self.choose_best_task(successful_tasks)
                self.agent_logger.info(
                    f"Successfully generated and verified code in task {chosen_task}"
                )
                break

        (self.folder / ".finished").write_text("")
        duration = time.time() - start_time
        self.agent_logger.info(f"Agent completed in {duration:0.2f} seconds.")
        self.agent_logger.remove(custom_logger_id)
        if chosen_task is None:
            self.agent_logger.error(
                "Failed to generate and verify correct code after multiple attempts."
            )
            return None
        return chosen_task
