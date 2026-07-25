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

"""实现兼容 OpenAI Chat Completions 协议的并发请求、预算和重试控制。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import math
import random
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import openai
from loguru import logger

from .base import LLMBudgetExceeded, LLMConfigurationError, LLMProvider, LLMResponse
from ..observability import prompt_metadata, record_event


@dataclass(frozen=True)
class OpenAICompatibleSettings:
    """封装 `OpenAICompatibleSettings` 对应的领域状态与操作。"""
    base_url: str
    api_key: str
    timeout_seconds: float = 1800
    max_concurrency: int = 4
    max_retries: int = 3
    max_requests: int = 0
    max_total_tokens: int = 0
    max_completion_tokens: int = 12288
    reasoning_effort: str = ""
    stream: bool = False
    log_content: bool = False

    def validate(self) -> None:
        """
        校验 `validate` 对应的领域操作，并返回调用方所需的标准化结果。

        异常:
            LLMConfigurationError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        if not self.api_key:
            raise LLMConfigurationError(
                "No API key configured. Set KERNELBLASTER_LLM_API_KEY or OPENAI_API_KEY."
            )
        if not self.base_url.startswith(("http://", "https://")):
            raise LLMConfigurationError(
                "KERNELBLASTER_LLM_BASE_URL must start with http:// or https://."
            )
        if self.max_concurrency < 1:
            raise LLMConfigurationError("LLM_MAX_CONCURRENCY must be at least 1.")
        if self.max_retries < 0:
            raise LLMConfigurationError("LLM_MAX_RETRIES cannot be negative.")
        if self.max_requests < 0 or self.max_total_tokens < 0:
            raise LLMConfigurationError("LLM budgets cannot be negative.")
        if self.max_completion_tokens < 1:
            raise LLMConfigurationError(
                "LLM_MAX_COMPLETION_TOKENS must be at least 1."
            )
        if self.reasoning_effort not in {
            "",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise LLMConfigurationError(
                "LLM_REASONING_EFFORT must be empty or a supported effort."
            )


@dataclass
class _CandidateResult:
    """保存一次操作的标准化结果及其诊断信息。"""
    content: str
    usage: dict[str, Any]
    request_id: str
    attempts: int
    elapsed_time: float
    usage_source: str
    response_model: str


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """
    处理 `to_plain_dict` 所表示的内部步骤；该函数不属于稳定的公开接口。

    参数:
        value: 需要转换、保存或校验的值。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _estimate_tokens(text: str) -> int:
    """
    对网关响应使用确定性的无标记器估计。

    参数:
        text: 调用方提供的 `text` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """
    返回保守的无标记器提示保留。

    UTF-8 字节上限是普通文本的字节对标记的数量，
    而每条消息的少量津贴涵盖了角色和框架令牌。

    参数:
        messages: 按对话顺序排列的 LLM 消息。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    serialized = "".join(
        f"{message.get('role', '')}:{message.get('content', '')}\n"
        for message in messages
    )
    return len(serialized.encode("utf-8")) + (8 * len(messages))


def _sanitize_base_url(base_url: str) -> str:
    """
    清理 `sanitize_base_url` 所表示的内部步骤；该函数不属于稳定的公开接口。

    参数:
        base_url: 调用方提供的 `base_url` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    parsed = urlsplit(base_url)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path.rstrip("/"), "", ""))


def _merge_usage(items: list[dict[str, Any]]) -> dict[str, int]:
    """
    处理 `merge_usage` 所表示的内部步骤；该函数不属于稳定的公开接口。

    参数:
        items: 调用方提供的 `items` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        key: sum(int(item.get(key, 0) or 0) for item in items)
        for key in keys
    }


class OpenAICompatibleProvider(LLMProvider):
    """通过 OpenAI 兼容接口并发生成候选，并统一管理重试、预算与用量。"""

    name = "openai_compatible"

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        client: openai.AsyncOpenAI | openai.AsyncAzureOpenAI | None = None,
    ) -> None:
        """
        初始化 OpenAICompatibleProvider 实例，并保存后续流程所需的配置与依赖。

        参数:
            settings: 调用方提供的 `settings` 参数。
            client: 调用方提供的 `client` 参数。
        """
        settings.validate()
        self.settings = settings
        self._client = client or openai.AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._budget_lock = asyncio.Lock()
        self._request_count = 0
        self._total_tokens = 0
        self._reserved_tokens = 0

    def public_config(self) -> dict[str, Any]:
        """
        处理 `public_config` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return {
            "provider": self.name,
            "base_url": _sanitize_base_url(self.settings.base_url),
            "timeout_seconds": self.settings.timeout_seconds,
            "max_concurrency": self.settings.max_concurrency,
            "max_retries": self.settings.max_retries,
            "max_requests": self.settings.max_requests,
            "max_total_tokens": self.settings.max_total_tokens,
            "max_completion_tokens": self.settings.max_completion_tokens,
            "reasoning_effort": self.settings.reasoning_effort or None,
            "stream": self.settings.stream,
            "fanout_mode": "client",
            "api_key_configured": bool(self.settings.api_key),
            "log_content": self.settings.log_content,
        }

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        n: int = 1,
    ) -> LLMResponse:
        """
        生成 `generate` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            messages: 按对话顺序排列的 LLM 消息。
            model: 生成候选时使用的模型标识。
            n: 调用方提供的 `n` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
            LLMConfigurationError: 输入、外部调用或状态不满足执行要求时抛出。
            ValueError: 输入、外部调用或状态不满足执行要求时抛出。
            errors[0]: 输入、外部调用或状态不满足执行要求时抛出。
        """
        if not model:
            raise LLMConfigurationError("A model identifier is required.")
        if n < 1:
            raise ValueError("n must be at least 1.")

        started = time.monotonic()
        tasks = [
            asyncio.create_task(self._generate_candidate(messages, model))
            for _ in range(n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [item for item in results if isinstance(item, BaseException)]
        if errors:
            record_event(
                "llm_fanout_failed",
                status="error",
                data={
                    "requested": n,
                    "completed": sum(
                        not isinstance(item, BaseException) for item in results
                    ),
                    "failed": len(errors),
                    "error_types": sorted({type(error).__name__ for error in errors}),
                },
            )
            raise errors[0]
        candidates = [item for item in results if isinstance(item, _CandidateResult)]

        usage_sources = {candidate.usage_source for candidate in candidates}
        return LLMResponse(
            input_messages=deepcopy(messages),
            generations=[candidate.content for candidate in candidates],
            usage=_merge_usage([candidate.usage for candidate in candidates]),
            model=model,
            num_tasks=n,
            elapsed_time=time.monotonic() - started,
            provider=self.name,
            request_ids=[
                candidate.request_id
                for candidate in candidates
                if candidate.request_id
            ],
            attempts=sum(candidate.attempts for candidate in candidates),
            usage_source=(usage_sources.pop() if len(usage_sources) == 1 else "mixed"),
            response_models=[candidate.response_model for candidate in candidates],
        )

    async def _generate_candidate(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> _CandidateResult:
        """
        生成 `generate_candidate` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
            messages: 按对话顺序排列的 LLM 消息。
            model: 生成候选时使用的模型标识。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
            RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        started = time.monotonic()
        last_error: Exception | None = None
        prompt = prompt_metadata(messages, include_content=self.settings.log_content)

        for attempt in range(self.settings.max_retries + 1):
            attempt_started = time.monotonic()
            reservation = 0
            try:
                reservation = (
                    _estimate_message_tokens(messages)
                    + self.settings.max_completion_tokens
                )
                await self._reserve_request(reservation)
                record_event(
                    "llm_request_started",
                    attempt=attempt + 1,
                    data={
                        "provider": self.name,
                        "model": model,
                        "prompt": prompt,
                    },
                )
                async with self._semaphore:
                    response = await self._create_completion(messages, model)
                (
                    content,
                    usage,
                    request_id,
                    usage_source,
                    response_model,
                ) = self._normalize_response(response, messages)
                await self._settle_tokens(
                    reservation,
                    int(usage.get("total_tokens", 0) or 0),
                )
                reservation = 0
                latency = time.monotonic() - attempt_started
                record_event(
                    "llm_request_completed",
                    attempt=attempt + 1,
                    data={
                        "provider": self.name,
                        "model": model,
                        "request_id": request_id,
                        "usage": usage,
                        "usage_source": usage_source,
                        "latency_seconds": latency,
                    },
                )
                return _CandidateResult(
                    content=content,
                    usage=usage,
                    request_id=request_id,
                    attempts=attempt + 1,
                    elapsed_time=time.monotonic() - started,
                    usage_source=usage_source,
                    response_model=response_model,
                )
            except Exception as error:
                if reservation:
                    await self._release_reservation(reservation)
                last_error = error
                if attempt >= self.settings.max_retries or not self._is_retryable(error):
                    record_event(
                        "llm_request_failed",
                        status="error",
                        attempt=attempt + 1,
                        data={
                            "provider": self.name,
                            "model": model,
                            **self._error_metadata(error),
                        },
                    )
                    raise
                delay = min(2**attempt, 30) + random.uniform(0, 1)
                record_event(
                    "llm_retry_scheduled",
                    status="retry",
                    attempt=attempt + 1,
                    data={
                        "provider": self.name,
                        "model": model,
                        "delay_seconds": delay,
                        **self._error_metadata(error),
                    },
                )
                logger.warning(
                    "OpenAI-compatible request failed on attempt {}/{}; retrying in {:.2f}s ({})",
                    attempt + 1,
                    self.settings.max_retries + 1,
                    delay,
                    type(error).__name__,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("OpenAI-compatible request failed") from last_error

    async def _create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> Any:
        """
        创建 `create_completion` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
            messages: 按对话顺序排列的 LLM 消息。
            model: 生成候选时使用的模型标识。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        args: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": self.settings.stream,
            "n": 1,
        }
        request_model = model

        if re.search(r"claude-.*-thinking", model.lower()):
            args["extra_body"] = {
                "thinking": {"type": "enabled", "budget_tokens": 16384}
            }
            request_model = model.replace("-thinking", "")
        elif re.search(r"deepseek-r1", model.lower()):
            args["top_p"] = 0.95
            args["max_tokens"] = 12288
            args["extra_body"] = {"enable_thinking": True}
        elif re.search("qwen", model.lower()) or re.search("kevin", model.lower()):
            args["top_p"] = 0.9
            args["max_tokens"] = 12288
            args["extra_body"] = {"enable_thinking": True}
        elif re.search(r".*-nemotron-.*-thinking", model.lower()):
            request_model = model.replace("-thinking", "")
            args["max_tokens"] = 12288
        else:
            args["max_completion_tokens"] = self.settings.max_completion_tokens

        if self.settings.reasoning_effort:
            args["reasoning_effort"] = self.settings.reasoning_effort

        args["model"] = request_model
        if not self.settings.stream:
            return await self._client.chat.completions.create(**args)

        chunks = []
        stream = await self._client.chat.completions.create(**args)
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    def _normalize_response(
        self,
        response: Any,
        messages: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any], str, str, str]:
        """
        规范化 `normalize_response` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
            response: 需要解析或规范化的服务响应。
            messages: 按对话顺序排列的 LLM 消息。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
            RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        if isinstance(response, list):
            content_parts: list[str] = []
            usage: dict[str, Any] = {}
            request_id = ""
            response_model = ""
            for chunk in response:
                request_id = request_id or str(
                    getattr(chunk, "_request_id", "") or ""
                )
                response_model = response_model or str(
                    getattr(chunk, "model", "") or ""
                )
                chunk_usage = _to_plain_dict(getattr(chunk, "usage", None))
                if chunk_usage:
                    usage = chunk_usage
                for choice in getattr(chunk, "choices", []) or []:
                    delta = getattr(choice, "delta", None)
                    text = getattr(delta, "content", None) if delta else None
                    if text:
                        content_parts.append(text)
            content = "".join(content_parts)
        else:
            choices = getattr(response, "choices", []) or []
            if not choices:
                raise RuntimeError("The provider returned no completion choices.")
            content = getattr(choices[0].message, "content", None) or ""
            usage = _to_plain_dict(getattr(response, "usage", None))
            request_id = str(getattr(response, "_request_id", "") or "")
            response_model = str(getattr(response, "model", "") or "")

        if usage:
            usage_source = "provider"
        else:
            prompt_text = "".join(
                str(message.get("content", "")) for message in messages
            )
            prompt_tokens = _estimate_tokens(prompt_text)
            completion_tokens = _estimate_tokens(content)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            usage_source = "estimated"

        return content, usage, request_id, usage_source, response_model

    async def _reserve_request(self, token_reservation: int) -> None:
        """
        处理 `reserve_request` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
            token_reservation: 调用方提供的 `token_reservation` 参数。

        异常:
            LLMBudgetExceeded: 输入、外部调用或状态不满足执行要求时抛出。
        """
        async with self._budget_lock:
            if (
                self.settings.max_requests
                and self._request_count >= self.settings.max_requests
            ):
                raise LLMBudgetExceeded(
                    f"LLM request budget exhausted ({self.settings.max_requests})."
                )
            if (
                self.settings.max_total_tokens
                and self._total_tokens
                + self._reserved_tokens
                + token_reservation
                > self.settings.max_total_tokens
            ):
                raise LLMBudgetExceeded(
                    "LLM token budget exhausted "
                    f"({self.settings.max_total_tokens})."
                )
            self._request_count += 1
            self._reserved_tokens += token_reservation

    async def _settle_tokens(self, reservation: int, total_tokens: int) -> None:
        """
        处理 `settle_tokens` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
            reservation: 调用方提供的 `reservation` 参数。
            total_tokens: 调用方提供的 `total_tokens` 参数。
        """
        async with self._budget_lock:
            self._reserved_tokens = max(0, self._reserved_tokens - reservation)
            self._total_tokens += max(0, total_tokens)

    async def _release_reservation(self, reservation: int) -> None:
        """
        处理 `release_reservation` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
            reservation: 调用方提供的 `reservation` 参数。
        """
        async with self._budget_lock:
            self._reserved_tokens = max(0, self._reserved_tokens - reservation)

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """
        判断 `is_retryable` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
            error: 调用方提供的 `error` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        if isinstance(
            error,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        ):
            return True
        if isinstance(error, openai.APIStatusError):
            return error.status_code == 429 or error.status_code >= 500
        return False

    @staticmethod
    def _error_metadata(error: Exception) -> dict[str, Any]:
        """
        处理 `error_metadata` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
            error: 调用方提供的 `error` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        status_code = getattr(error, "status_code", None)
        return {
            "error_type": type(error).__name__,
            "status_code": status_code,
        }
