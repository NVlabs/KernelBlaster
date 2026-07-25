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

"""定义大语言模型 Provider 的抽象协议、标准响应和预算异常。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """现有优化代理消耗的提供者中立响应。"""

    input_messages: list[dict]
    generations: list[str]
    usage: dict
    model: str
    num_tasks: int
    elapsed_time: float
    provider: str = ""
    request_ids: list[str] = field(default_factory=list)
    attempts: int = 0
    usage_source: str = "provider"
    response_models: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """
        处理 `__str__` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return str(asdict(self))

    @property
    def response(self) -> str:
        """
        返回遗留单一响应调用者的第一代。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return self.generations[0] if self.generations else ""


class LLMProvider(ABC):
    """代码生成提供程序的最小异步接口。"""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        n: int = 1,
    ) -> LLMResponse:
        """
        为一次对话生成“`n`”候选者。

        参数:
            messages: 按对话顺序排列的 LLM 消息。
            model: 生成候选时使用的模型标识。
            n: 调用方提供的 `n` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """

    @abstractmethod
    def public_config(self) -> dict[str, Any]:
        """
        返回适用于清单和日志的非秘密配置。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """


class LLMConfigurationError(RuntimeError):
    """当 LLM 提供商缺少所需配置时引发。"""


class LLMBudgetExceeded(RuntimeError):
    """在超出配置的硬预算的请求之前提出。"""
