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
本地 LLM 包装器，用于在本地运行具有量化支持的模型。

该模块提供对本地运行 LLM 的支持，特别是
Qwen2.5-Coder-32B-使用位和字节进行 4 位量化的指令。
"""
import os
import time
import asyncio
from typing import List, Dict, Optional
from copy import deepcopy
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from loguru import logger

from .query import LLMResponse, to_dict_recursive

__all__ = [
    "generate_code_local",
    "generate_code_local_batch",
    "get_local_model",
    "is_local_model",
]

# 全局模型缓存以避免重新加载
_model_cache: Dict[str, tuple] = {}  # model_name ->（分词器，模型）


def is_local_model(model_name: str) -> bool:
    """
    检查模型名称是否表明它应该在本地运行。

    参数:
        model_name: 调用方提供的 `model_name` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    if not model_name:
        return False
    local_model_patterns = [
        "qwen2.5-coder-32b-instruct",
        "qwen2.5-coder-32b",
        "qwen/qwen2.5-coder-32b-instruct",
        "local-qwen",
    ]
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in local_model_patterns)


def get_local_model(
    model_name: str,
    use_4bit: bool = True,
    device_map: str = None,
    trust_remote_code: bool = True,
):
    """
    加载具有可选 4 位量化的本地模型。

    参数
    ----------
    model_name : 字符串
    HuggingFace 模型标识符（e.g.、“Qwen/Qwen2.5-Coder-32B-Instruct”）
    use_4bit : 布尔值，默认 True
    是否通过bitsandbytes使用4位量化
    device_map : 字符串 |无，默认无
    模型加载的设备映射策略。选项：
    - 无：使用 LOCAL_LLM_DEVICE_MAP 环境变量或“auto”
    - “auto”：自动分配到可用的 GPU
    - “single”：强制使用单个 GPU (cuda:0)
    - “平衡”：所有 GPU 之间的平衡
    - “balanced_low_0”：第一个 GPU 的平衡变得更少
    - Dict：自定义设备映射
    trust_remote_code : 布尔值，默认 True
    是否信任来自 HuggingFace 的远程代码

    退货
    -------
    元组
    （分词器，模型）元组

    参数:
        model_name: 调用方提供的 `model_name` 参数。
        use_4bit: 调用方提供的 `use_4bit` 参数。
        device_map: 调用方提供的 `device_map` 参数。
        trust_remote_code: 调用方提供的 `trust_remote_code` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    # 从参数或环境变量确定device_map
    if device_map is None:
        device_map = os.getenv("LOCAL_LLM_DEVICE_MAP", "auto")
    
    # 处理特殊的“单一”选项以强制使用单一 GPU
    if device_map == "single":
        device_map = "cuda:0"
        logger.info("Using single GPU (cuda:0) for model loading")
    # 首先检查缓存
    cache_key = f"{model_name}_{use_4bit}"
    if cache_key in _model_cache:
        logger.info(f"Using cached model: {cache_key}")
        return _model_cache[cache_key]
    
    logger.info(f"Loading local model: {model_name} (4-bit: {use_4bit}, device_map: {device_map})")
    
    # 确定实际模型路径
    model_lower = model_name.lower()
    if "qwen2.5-coder-32b" in model_lower:
        # 使用官方 HuggingFace 模型名称
        actual_model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    elif model_name.startswith("Qwen/") or "/" in model_name:
        # 已经是完整的 HuggingFace 路径
        actual_model_name = model_name
    else:
        actual_model_name = model_name
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        actual_model_name,
        trust_remote_code=trust_remote_code,
    )
    
    # 如果需要，配置量化
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4-bit quantization (NF4)")
    
    # 负载模型
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if not use_4bit else None,
    )
    
    # 缓存模型
    _model_cache[cache_key] = (tokenizer, model)
    logger.info(f"Model loaded and cached: {cache_key}")
    
    return tokenizer, model


def format_messages_for_qwen(messages: List[Dict]) -> str:
    """
    将消息格式化为 Qwen 模型所需的聊天格式。

    Qwen 模型使用特定的聊天模板格式。这个功能
    将 OpenAI 风格的消息转换为 Qwen 格式。

    参数:
        messages: 按对话顺序排列的 LLM 消息。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    # 提取系统消息（如果存在）
    system_message = None
    conversation_messages = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            system_message = content
        else:
            conversation_messages.append({"role": role, "content": content})
    
    # 如果可用，请使用分词器的聊天模板，否则手动格式化
    # 我们将使用标记器的 apply_chat_template 方法
    return conversation_messages, system_message


async def generate_code_local(
    messages: List[Dict],
    n_tasks: int,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_4bit: bool = True,
) -> LLMResponse:
    """
    使用本地模型生成代码。

    参数
    ----------
    消息：列表[字典]
    OpenAI 格式的聊天消息
    n_tasks：整数
    生成的完成数
    型号：str
    型号名称/标识符
    max_tokens ：整数，默认 4096
    生成的最大代币数量
    温度：浮动，默认0.7
    取样温度
    top_p ：浮动，默认0.9
    细胞核采样参数
    use_4bit : 布尔值，默认 True
    是否使用4位量化

    退货
    -------
    LLM回应
    具有世代的响应对象

    参数:
        messages: 按对话顺序排列的 LLM 消息。
        n_tasks: 调用方提供的 `n_tasks` 参数。
        model: 生成候选时使用的模型标识。
        max_tokens: 调用方提供的 `max_tokens` 参数。
        temperature: 调用方提供的 `temperature` 参数。
        top_p: 调用方提供的 `top_p` 参数。
        use_4bit: 调用方提供的 `use_4bit` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    start_time = time.time()
    
    try:
        # 加载模型（如果已经加载，将使用缓存）
        # 允许通过环境变量配置device_map
        device_map = os.getenv("LOCAL_LLM_DEVICE_MAP", None)
        tokenizer, model_obj = get_local_model(
            model,
            use_4bit=use_4bit,
            device_map=device_map,
        )
        
        # 设置 Qwen 消息的格式
        conversation_messages, system_message = format_messages_for_qwen(messages)
        
        # 应用聊天模板
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            # 使用 tokenizer 的聊天模板
            if system_message:
                # 如果需要，将系统消息添加到第一条用户消息之前
                if conversation_messages and conversation_messages[0]["role"] == "user":
                    conversation_messages[0]["content"] = (
                        system_message + "\n\n" + conversation_messages[0]["content"]
                    )
            
            prompt = tokenizer.apply_chat_template(
                conversation_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # 后备：手动格式化
            prompt_parts = []
            if system_message:
                prompt_parts.append(f"System: {system_message}\n\n")
            for msg in conversation_messages:
                role = msg["role"].capitalize()
                prompt_parts.append(f"{role}: {msg['content']}\n\n")
            prompt_parts.append("Assistant: ")
            prompt = "".join(prompt_parts)
        
        # 标记化
        inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
        
        # 使用批量推理生成多个完成
        # 在执行器中运行以避免阻塞异步事件循环
        loop = asyncio.get_event_loop()
        
        def generate_batched():
            """Synchronous batched generation function to run in executor."""
            with torch.no_grad():
                # 当 n_tasks > 1 时使用批量生成
                # 这比处理时顺序生成更有效
                # 前向传递过程中所有序列并行
                if n_tasks > 1:
                    # 将输入扩展至批量大小 n_tasks
                    # 对每个任务重复 input_ids 和 attention_mask
                    batch_input_ids = inputs["input_ids"].repeat(n_tasks, 1)
                    if "attention_mask" in inputs:
                        batch_attention_mask = inputs["attention_mask"].repeat(n_tasks, 1)
                        batch_inputs = {
                            "input_ids": batch_input_ids,
                            "attention_mask": batch_attention_mask,
                        }
                    else:
                        batch_inputs = {"input_ids": batch_input_ids}
                    
                    logger.info(f"Generating {n_tasks} completions in batch with local model")
                    generated_ids = model_obj.generate(
                        **batch_inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    # 单代
                    logger.info(f"Generating single completion with local model")
                    generated_ids = model_obj.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    # 添加批次维度以实现一致的处理
                    if generated_ids.dim() == 1:
                        generated_ids = generated_ids.unsqueeze(0)
            
            # 解码所有生成的序列
            input_length = inputs["input_ids"].shape[1]
            outputs = []
            batch_size = generated_ids.shape[0]
            for i in range(min(batch_size, n_tasks)):
                generated_text = tokenizer.decode(
                    generated_ids[i][input_length:],
                    skip_special_tokens=True,
                )
                outputs.append(generated_text)
            
            return outputs
        
        # 在线程池中运行批量生成以避免阻塞
        outputs = await loop.run_in_executor(None, generate_batched)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 估计代币使用情况（粗略估计）
        total_input_tokens = inputs["input_ids"].shape[1]
        total_output_tokens = sum(
            len(tokenizer.encode(output, add_special_tokens=False))
            for output in outputs
        )
        
        usage_info = {
            "prompt_tokens": total_input_tokens,
            "completion_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        
        return LLMResponse(
            deepcopy(messages),
            outputs,
            usage_info,
            model,
            n_tasks,
            elapsed_time,
        )
        
    except Exception as e:
        logger.error(f"Error generating code with local model: {e}")
        raise


async def generate_code_local_batch(
    prompts: List[List[Dict]],
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_4bit: bool = True,
) -> List[LLMResponse]:
    """
    在单个批量前向传递中为多个不同提示生成代码。

    这比多次调用 generate_code_local 更有效，因为
    在一次前向传递过程中并行处理所有提示。

    参数
    ----------
    提示：列表[列表[字典]]
    聊天消息列表列表（每个提示一个）
    型号：str
    型号名称/标识符
    max_tokens ：整数，默认 4096
    每个提示生成的最大令牌数
    温度：浮动，默认0.7
    取样温度
    top_p ：浮动，默认0.9
    细胞核采样参数
    use_4bit : 布尔值，默认 True
    是否使用4位量化

    退货
    -------
    list[LLMResponse]
    响应对象列表，每个提示一个

    参数:
        prompts: 调用方提供的 `prompts` 参数。
        model: 生成候选时使用的模型标识。
        max_tokens: 调用方提供的 `max_tokens` 参数。
        temperature: 调用方提供的 `temperature` 参数。
        top_p: 调用方提供的 `top_p` 参数。
        use_4bit: 调用方提供的 `use_4bit` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    start_time = time.time()
    
    try:
        # 加载模型（如果已经加载，将使用缓存）
        tokenizer, model_obj = get_local_model(
            model,
            use_4bit=use_4bit,
        )
        
        # 设置所有提示的格式
        formatted_prompts = []
        for messages in prompts:
            conversation_messages, system_message = format_messages_for_qwen(messages)
            
            # 应用聊天模板
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                if system_message:
                    if conversation_messages and conversation_messages[0]["role"] == "user":
                        conversation_messages[0]["content"] = (
                            system_message + "\n\n" + conversation_messages[0]["content"]
                        )
                
                prompt = tokenizer.apply_chat_template(
                    conversation_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # 后备：手动格式化
                prompt_parts = []
                if system_message:
                    prompt_parts.append(f"System: {system_message}\n\n")
                for msg in conversation_messages:
                    role = msg["role"].capitalize()
                    prompt_parts.append(f"{role}: {msg['content']}\n\n")
                prompt_parts.append("Assistant: ")
                prompt = "".join(prompt_parts)
            
            formatted_prompts.append(prompt)
        
        # 对所有提示进行标记
        tokenized = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,  # 合理的最大长度
        ).to(model_obj.device)
        
        # 在执行器中运行以避免阻塞
        loop = asyncio.get_event_loop()
        
        def generate_batched():
            """Synchronous batched generation function."""
            with torch.no_grad():
                logger.info(
                    f"Generating batch of {len(prompts)} different prompts with local model"
                )
                generated_ids = model_obj.generate(
                    **tokenized,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # 独立解码每个序列
            input_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
            outputs = []
            
            for i, (gen_ids, input_len) in enumerate(zip(generated_ids, input_lengths)):
                generated_text = tokenizer.decode(
                    gen_ids[input_len:],
                    skip_special_tokens=True,
                )
                outputs.append(generated_text)
            
            return outputs, input_lengths
        
        # 运行批量生成
        outputs, input_lengths = await loop.run_in_executor(None, generate_batched)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 为每个提示创建 LLMResponse
        responses = []
        for i, (prompt_messages, output) in enumerate(zip(prompts, outputs)):
            # 估计代币使用情况
            input_tokens = input_lengths[i]
            output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
            
            usage_info = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            
            response = LLMResponse(
                deepcopy(prompt_messages),
                [output],
                usage_info,
                model,
                1,  # 根据提示 n_tasks
                elapsed_time / len(prompts),  # 每次提示的平均时间
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error generating batched code with local model: {e}")
        raise
