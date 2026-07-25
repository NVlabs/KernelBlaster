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

"""统一处理 LLM 消息裁剪、响应解析、重试和本地/远端生成路由。"""

from copy import deepcopy
from typing import List, Tuple
try:
    import openai
except ModuleNotFoundError:  # The client is an optional runtime dependency.
    openai = None
import os
import re
import time
import asyncio
import json
import random
from loguru import logger

from ...config import config
from ...llm import LLMResponse, get_llm_provider

__all__ = [
    "generate_code_openai",
    "generate_code",
    "generate_code_retry",
    "convert_messages_to_string",
    "LLMResponse",
    "trim_messages",
    "to_dict_recursive",
    "extract_code_from_response",
]

# 如果不使用本地模型，则延迟导入以避免加载大量依赖项
_local_llm_module = None

TIMEOUT = 1800
# 由于标记化成本高昂，因此粗略估计最大标记数
MAX_CHARACTERS_TOTAL = 340_000

# 通过 PERFLAB_KEY 配置的 Perflab 客户端（后备选项）。
perflab_client = None
if os.getenv("PERFLAB_KEY") and openai is not None:
    perflab_client = openai.AsyncAzureOpenAI(
        azure_endpoint="https://llm-proxy.perflab.nvidia.com",
        api_version="2024-12-01-preview",
        api_key=os.getenv("PERFLAB_KEY"),
        timeout=None,
    )


class TrimError(Exception):
    """表示该领域内可被调用方识别和处理的失败。"""
    pass


class NoResponseError(Exception):
    """表示该领域内可被调用方识别和处理的失败。"""
    pass


def trim_messages(messages, logger):
    """
    裁剪 `trim_messages` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        messages: 按对话顺序排列的 LLM 消息。
        logger: 记录诊断信息和任务进度的日志器。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        TrimError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    total_characters = sum(
        len(message["content"]) if message["content"] is not None else 0
        for message in messages
    )
    while total_characters > MAX_CHARACTERS_TOTAL and len(messages) >= 3:
        logger.warning(
            f"Trimming messages. Total characters: {total_characters}. Max characters: {MAX_CHARACTERS_TOTAL}"
        )
        # 保留第一条系统消息和第一条用户消息
        # 一次弹出两条消息以确保保留替代角色
        messages.pop(2)  # 删除第一助手
        if len(messages) >= 3:
            messages.pop(2)  # 删除第二条用户消息

        total_characters = sum(
            len(message["content"]) if message["content"] is not None else 0
            for message in messages
        )
        logger.warning(
            f"Trimmed messages. Total characters: {total_characters}. Max characters: {MAX_CHARACTERS_TOTAL}"
        )

    if total_characters > MAX_CHARACTERS_TOTAL:
        logger.error(
            f"Failed to trim messages. Total characters: {total_characters}. Max characters: {MAX_CHARACTERS_TOTAL}"
        )
        raise TrimError(
            f"Failed to trim messages. Total characters: {total_characters}. Max characters: {MAX_CHARACTERS_TOTAL}"
        )

    return messages


def to_dict_recursive(obj):
    """
    处理 `to_dict_recursive` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        obj: 调用方提供的 `obj` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    obj = dict(obj)
    for k, v in obj.items():
        obj[k] = to_dict_recursive(v) if hasattr(v, "__dict__") else v
    return obj


def extract_code_from_response(response_text, tag="cpp") -> str | None:
    # 首先删除 <think></think> 标签内的所有内容（如果存在）
    # 以避免在推理过程中拾取任何代码内容。
    # Deepseek-R1 在聊天模板本身中引入了 <think> 开始标签
    # 所以它在生成过程中不再返回。
    """
    提取 `extract_code_from_response` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        response_text: 调用方提供的 `response_text` 参数。
        tag: 调用方提供的 `tag` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    cleaned_text = re.sub(r".*?</think>", "", response_text, flags=re.DOTALL)
    code_blocks = re.findall(f"```{tag}\n(.*?)```", cleaned_text, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[0]


def process_messages(messages: List[dict], model: str) -> List[dict]:
    """
    处理 `process_messages` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        messages: 按对话顺序排列的 LLM 消息。
        model: 生成候选时使用的模型标识。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    messages = deepcopy(messages)
    if re.search(r".*-nemotron-.*-thinking", model.lower()):
        # 通过使用特殊的系统提示启用nemotron模型的思维模式
        formatted_messages = messages
        if messages[0]["role"] == "system":
            formatted_messages[0]["content"] = "detailed thinking on"
            formatted_messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
        return formatted_messages

    if "o1" in model.lower():
        formatted_messages = []
        sys_message = ""
        for message in messages:
            if message["role"] == "system":
                sys_message += message["content"] + "\n\n"
            else:
                formatted_messages.append(deepcopy(message))
        if len(sys_message) > 0:
            formatted_messages.insert(0, {"role": "user", "content": sys_message})
        return formatted_messages

    return messages


async def generate_code_openai(client, messages, n_tasks, model: str) -> LLMResponse:
    """
    生成 `generate_code_openai` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        client: 调用方提供的 `client` 参数。
        messages: 按对话顺序排列的 LLM 消息。
        n_tasks: 调用方提供的 `n_tasks` 参数。
        model: 生成候选时使用的模型标识。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    args = {
        "model": model,
        "messages": messages,
        "stream": config.STREAM
        and config.STREAM.lower() in ("true", "1", "yes", "y", "on"),
        "n": n_tasks,
    }

    if re.search(r"claude-.*-thinking", model.lower()):
        args["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": 16384}}
        args["model"] = model.replace("-thinking", "")
    elif re.search(r"deepseek-r1", model.lower()):
        # 思考模式应默认启用
        args["top_p"] = 0.95
        args["max_tokens"] = 12288
        args["enable_thinking"] = True
    elif re.search("qwen", model.lower()) or re.search("kevin", model.lower()):
        args["top_p"] = 0.9
        args["enable_thinking"] = True
        args["max_tokens"] = 12288
    elif re.search(r".*-nemotron-.*-thinking", model.lower()):
        args["model"] = model.replace("-thinking", "")
        args["max_tokens"] = 12288

    start = time.time()

    if not args.get("stream", False):
        # 处理非流式响应
        response = await client.chat.completions.create(**args)
        outputs = [choice.message.content for choice in response.choices]

        end = time.time()
        return LLMResponse(
            deepcopy(messages),
            outputs,
            to_dict_recursive(response.usage),
            model,
            n_tasks,
            end - start,
        )
    else:
        # 处理流响应
        outputs = [""] * n_tasks
        usage_info = None

        try:
            stream = await client.chat.completions.create(**args)
            async for chunk in stream:
                # 检查该块中的使用信息是否可用
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_info = to_dict_recursive(chunk.usage)

                for i, choice in enumerate(chunk.choices):
                    if i >= len(outputs):
                        # 以防万一返回的选择多于请求的数量
                        continue

                    if (
                        hasattr(choice, "delta")
                        and hasattr(choice.delta, "content")
                        and choice.delta.content
                    ):
                        outputs[i] += choice.delta.content
                    elif (
                        hasattr(choice, "message")
                        and hasattr(choice.message, "content")
                        and choice.message.content
                    ):
                        # 即使在流模式下，某些客户端也可能返回完整消息
                        outputs[i] = choice.message.content

            # 获取最终时间
            end = time.time()

            # 如果我们没有从流中获取使用信息，请估计它
            if not usage_info:
                usage_info = {
                    "completion_tokens": sum(
                        len(output.split()) * 4 // 3 for output in outputs
                    ),
                    "prompt_tokens": sum(
                        len(message["content"].split()) * 4 // 3 for message in messages
                    ),
                    "total_tokens": 0,
                }
                usage_info["total_tokens"] = (
                    usage_info["completion_tokens"] + usage_info["prompt_tokens"]
                )

            return LLMResponse(
                deepcopy(messages),
                outputs,
                usage_info,
                model,
                n_tasks,
                end - start,
            )
        except Exception as e:
            logger.error(f"Error during streaming response: {e}")
            raise


async def generate_code(messages, n_tasks=1, model=None) -> LLMResponse:
    """
    使用 OpenAI REST 接口或本地模型生成代码的主要功能。

    支持：
    - 通过 OPENAI_API_KEY 配置 OpenAI 兼容客户端
    - 当模型名称指示本地使用时，本地模型（e.g.、Qwen2.5-Coder-32B-Instruct）
    - 通过批处理队列自动批处理本地模型

    参数:
        messages: 按对话顺序排列的 LLM 消息。
        n_tasks: 调用方提供的 `n_tasks` 参数。
        model: 生成候选时使用的模型标识。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    # 检查这是否是本地模型
    global _local_llm_module
    if _local_llm_module is None:
        try:
            from . import local_llm as _local_llm_module
        except ImportError:
            _local_llm_module = False  # 标记为不可用
    
    # 对于本地模型，如果启用，请使用批处理队列
    if _local_llm_module and _local_llm_module.is_local_model(model):
        # 检查是否启用了批处理
        batch_enabled = os.getenv("LLM_BATCH_ENABLED", "true").lower() == "true"
        batch_only_local = os.getenv("LLM_BATCH_ONLY_LOCAL", "true").lower() == "true"
        
        if batch_enabled and batch_only_local:
            # 对本地模型使用批处理队列
            from .batch_queue import get_batch_queue
            batch_queue = get_batch_queue()
            return await batch_queue.submit(
                messages=messages,
                model=model,
                n_tasks=n_tasks,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                use_4bit=True,
            )
        else:
            # 直接调用（禁用批处理）
            logger.info(f"Using local model (direct): {model}")
            return await _local_llm_module.generate_code_local(
                messages,
                n_tasks,
                model,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                use_4bit=True,
            )
    
    # 更喜欢可配置的提供程序。候选扇出、重试、并发、
    # 预算执行发生在该提供商边界内。
    if config.API_KEY:
        provider = get_llm_provider(type(config))
        return await provider.generate(messages, model=model, n=n_tasks)

    # 配置后保留 NVIDIA Perflab 开发后备。
    if perflab_client is not None:
        return await generate_code_openai(
            perflab_client, messages, n_tasks=n_tasks, model=model
        )

    if not config.API_KEY:
        raise RuntimeError(
            "No LLM client configured. Set KERNELBLASTER_LLM_API_KEY "
            "(or OPENAI_API_KEY/PERFLAB_KEY), or use a local model."
        )


async def generate_code_retry(
    messages,
    model,
    logger=None,
    n_tasks: int = 1,
    max_retries: int | None = config.LLM_MAX_RETRIES,
    **kwargs,
) -> LLMResponse:
    """
    generate_code 周围的强大包装器可自动重试并
    支持向后兼容的可选参数（e.g。*system_prompt*，
    *max_attempts*)。

    参数
    ----------
    消息：列表[字典]
    聊天完成消息（与 OpenAI/其他提供商的格式相同）。
    型号：str
    LLM模型名称。
    记录器：logging.Logger |无，可选
    调试/警告消息的记录器。如果满足以下条件，将创建默认记录器
    提供“`None`”，以便未传递记录器的遗留调用者
    继续工作。
    n_tasks ：整数，默认1
    请求的完成数量。
    max_retries : 整数 |无，可选
    放弃之前重试。如果“`None`”，调用将不断重试
    无限期地。
    **kwargs：任何
    默默地忽略额外的关键字参数。目前有以下键
    被解释为：

    - ``system_prompt`` (str)：将作为 *messages* 的前缀
    如果提供了“`{"role": "system", "content": system_prompt}`”条目。
    - ``max_attempts`` (int)：为历史保留 *max_retries* 的别名
    原因。

    参数:
        messages: 按对话顺序排列的 LLM 消息。
        model: 生成候选时使用的模型标识。
        logger: 记录诊断信息和任务进度的日志器。
        n_tasks: 调用方提供的 `n_tasks` 参数。
        max_retries: 调用方提供的 `max_retries` 参数。
        kwargs: 调用方提供的 `kwargs` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
    """

    # ------------------------------------------------------------------
    # 新引入的 kwargs 的向后兼容性处理
    # ------------------------------------------------------------------
    system_prompt: str | None = kwargs.pop("system_prompt", None)
    max_attempts = kwargs.pop("max_attempts", None)
    if max_attempts is not None:
        max_retries = max_attempts

    # 如果还有任何其他意想不到的 kwargs，只需忽略它们，但发出一条痕迹。
    if kwargs:
        import warnings as _warnings
        _warnings.warn(
            f"[generate_code_retry] Ignoring unrecognised kwargs: {list(kwargs.keys())}",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 确保我们始终有一个记录器实例。
    # ------------------------------------------------------------------
    if logger is None:
        import logging as _logging
        logger = _logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 如果提供的话，请在前面添加系统提示（LLM API 期望将其作为第一条消息）
    # ------------------------------------------------------------------
    if system_prompt:
        # 避免就地改变调用者提供的列表
        messages = [{"role": "system", "content": system_prompt}] + list(messages)

    # 远程提供者拥有每个候选重试语义，因此部分扇出是
    # 这个传统包装器从未重复批发过。本地路径和 Perflab 路径
    # 保留下面的历史外部重试循环。
    if config.API_KEY:
        return await generate_code(messages, n_tasks=n_tasks, model=model)

    response = None
    tries = 0
    while max_retries is None or tries < max_retries:
        delay = 5
        try:
            response = await generate_code(messages, n_tasks=n_tasks, model=model)
            break
        except openai.RateLimitError as e:
            # 带抖动的指数退避以平滑并发请求
            delay = (2**tries) + random.uniform(0, 60)
            logger.warning(
                f"Rate limit error on attempt {tries}/{max_retries}. "
                f"Retrying in {delay:.2f} seconds: {e}"
            )
        except openai.AuthenticationError as e:
            logger.error(
                f"Failed to generate code due to 401 error. Trying again ({e})"
            )
        except openai.APITimeoutError as e:
            logger.error(
                f"Failed to generate code due to timeout error. Trying again ({e})"
            )
        except openai.InternalServerError as e:
            logger.error(
                f"Failed to generate code due to internal server error. Trying again ({e})"
            )
        except openai.APIConnectionError as e:
            logger.error(
                f"Failed to generate code due to connection error. Trying again ({e})"
            )
        except NoResponseError as e:
            logger.error(f"No response from LLM. Response: {e}")
        except Exception as e:
            if "timeout" in str(e).lower() or "time-out" in str(e).lower():
                delay = (2**tries) + random.uniform(0, 60)
                logger.error(
                    f"Failed to generate code due to timeout. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "502" in str(e) or "bad gateway" in str(e).lower():
                logger.error(
                    f"Failed to generate code due to bad gateway. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "rate_limited" in str(e).lower() or "rate limit" in str(e).lower():
                delay = (2**tries) + random.uniform(0, 60)
                logger.warning(
                    f"Failed to generate code due to rate limit. Retrying in {delay:.2f} seconds (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "invalid client" in str(e).lower():
                logger.error(
                    f"Failed to generate code due to invalid client bug. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "inference connection error" in str(e).lower():
                logger.error(
                    f"Failed to generate code due to inference connection error. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            elif "device or resource busy" in str(e).lower():
                delay = (2**tries) + random.uniform(0, 60)
                tries += 1
                logger.warning(
                    f"Device or resource busy. Retrying in {delay:.2f} seconds: {e}"
                )
                continue
            else:
                logger.error(
                    f"Failed to generate code due to unknown error. Trying again (attempt {tries}/{max_retries}, error: {e})"
                )
            await asyncio.sleep(delay)
            tries += 1
    if not response:
        logger.error("Failed to generate code exceeding max retries.")
        raise RuntimeError("Exceeded max retries")
    return response


def convert_messages_to_string(
    messages: list[dict], response: str = None, usage: dict = None
):
    """
    转换 `convert_messages_to_string` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        messages: 按对话顺序排列的 LLM 消息。
        response: 需要解析或规范化的服务响应。
        usage: 调用方提供的 `usage` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    def to_string(role, content):
        """
        处理 `to_string` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            role: 调用方提供的 `role` 参数。
            content: 调用方提供的 `content` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        r = "# " + "=" * 20 + role.upper() + "=" * 20 + "\n"
        r += content
        return r

    string = ""
    for message in messages:
        string += "\n" + to_string(message["role"], message["content"])
    if response:
        string += "\n" + to_string("assistant", response)
    if usage:
        string += "\n" + to_string("usage", json.dumps(usage, indent=2))
    return string
