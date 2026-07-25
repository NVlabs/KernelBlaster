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

"""根据系统配置延迟创建并缓存统一的 LLM Provider。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LLMConfigurationError, LLMProvider
if TYPE_CHECKING:
    from ..config.config import SystemConfig


_provider: LLMProvider | None = None


def get_llm_provider(config: type[SystemConfig]) -> LLMProvider:
    """
    返回程序启动前配置的进程范围提供程序。

    参数:
        config: 控制当前组件行为的配置对象。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        LLMConfigurationError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    global _provider
    if _provider is not None:
        return _provider

    if config.LLM_PROVIDER != "openai_compatible":
        raise LLMConfigurationError(
            f"Unsupported KERNELBLASTER_LLM_PROVIDER: {config.LLM_PROVIDER}"
        )

    # Keep the optional OpenAI client out of CPU-only imports. This lets
    # profiling and artifact tooling run without the ``llm`` extra installed.
    from .openai_compatible import OpenAICompatibleProvider, OpenAICompatibleSettings

    _provider = OpenAICompatibleProvider(
        OpenAICompatibleSettings(
            base_url=config.LLM_BASE_URL,
            api_key=config.API_KEY,
            timeout_seconds=config.LLM_REQUEST_TIMEOUT_SECONDS,
            max_concurrency=config.LLM_MAX_CONCURRENCY,
            max_retries=config.LLM_MAX_RETRIES,
            max_requests=config.LLM_MAX_REQUESTS,
            max_total_tokens=config.LLM_MAX_TOTAL_TOKENS,
            max_completion_tokens=config.LLM_MAX_COMPLETION_TOKENS,
            reasoning_effort=config.LLM_REASONING_EFFORT,
            stream=config.STREAM.lower() in ("true", "1", "yes", "y", "on"),
            log_content=config.LLM_LOG_CONTENT,
        )
    )
    return _provider


def reset_llm_provider() -> None:
    """清除单例，以便将来的隔离测试可以重建配置。"""
    global _provider
    _provider = None
