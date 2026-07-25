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

"""导出统一的 LLM Provider 接口、配置异常与工厂函数。"""

from .base import LLMBudgetExceeded, LLMConfigurationError, LLMProvider, LLMResponse
from .factory import get_llm_provider, reset_llm_provider

__all__ = [
    "LLMBudgetExceeded",
    "LLMConfigurationError",
    "LLMProvider",
    "LLMResponse",
    "OpenAICompatibleProvider",
    "OpenAICompatibleSettings",
    "get_llm_provider",
    "reset_llm_provider",
]


def __getattr__(name: str):
    if name in {"OpenAICompatibleProvider", "OpenAICompatibleSettings"}:
        from .openai_compatible import OpenAICompatibleProvider, OpenAICompatibleSettings

        return {
            "OpenAICompatibleProvider": OpenAICompatibleProvider,
            "OpenAICompatibleSettings": OpenAICompatibleSettings,
        }[name]
    raise AttributeError(name)
