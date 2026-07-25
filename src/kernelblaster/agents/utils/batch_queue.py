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
用于聚合来自多个轨迹/代理的 LLM 请求的批处理队列。

该模块提供批处理中间件，用于收集某个时间窗口内的请求
并将它们批处理在一起以实现更高效的处理，特别是对于本地模型。
"""
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy
import os

from loguru import logger

from .local_llm import is_local_model, generate_code_local, generate_code_local_batch
from .query import LLMResponse

__all__ = ["LLMBatchQueue", "get_batch_queue"]


@dataclass
class QueuedRequest:
    """等待批处理的单个 LLM 请求。"""
    messages: List[Dict]
    model: str
    n_tasks: int
    max_tokens: int
    temperature: float
    top_p: float
    future: asyncio.Future
    timestamp: float
    use_4bit: bool = True


class LLMBatchQueue:
    """
    用于批量处理本地模型的 LLM 请求的队列。

    收集一段时间窗口内的请求并批量处理它们
    提高 GPU 利用率并减少延迟。
    """
    
    def __init__(
        self,
        window_ms: int = 100,
        max_batch_size: int = 8,
        enabled: bool = True,
    ):
        """
        初始化批处理队列。

        参数
        ----------
        window_ms ：整数，默认100
        处理批次之前等待的最长时间（以毫秒为单位）
        max_batch_size ：整数，默认8
        单个批次中包含的最大请求数
        启用：布尔值，默认 True
        是否启用批处理（可以通过环境变量禁用）

        参数:
            window_ms: 调用方提供的 `window_ms` 参数。
            max_batch_size: 调用方提供的 `max_batch_size` 参数。
            enabled: 调用方提供的 `enabled` 参数。
        """
        self.window_ms = window_ms / 1000.0  # 转换为秒
        self.max_batch_size = max_batch_size
        self.enabled = enabled and (os.getenv("LLM_BATCH_ENABLED", "true").lower() == "true")
        
        self.queue: List[QueuedRequest] = []
        self.lock = asyncio.Lock()
        self.processing = False
        self._pending_timer: Optional[asyncio.Task] = None
        
        if self.enabled:
            logger.info(
                f"LLM batch queue enabled: window={window_ms}ms, max_batch={max_batch_size}"
            )
        else:
            logger.info("LLM batch queue disabled")
    
    async def submit(
        self,
        messages: List[Dict],
        model: str,
        n_tasks: int = 1,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_4bit: bool = True,
    ) -> LLMResponse:
        """
        向批处理队列提交请求。

        对于本地模型，请求被批量处理在一起。对于 API 模型，
        请求立即通过（无批处理）。

        参数
        ----------
        消息：列表[字典]
        OpenAI 格式的聊天消息
        型号：str
        型号名称/标识符
        n_tasks ：整数，默认1
        生成的完成数
        max_tokens ：整数，默认 4096
        生成的最大代币数量
        温度：浮动，默认0.7
        取样温度
        top_p ：浮动，默认0.9
        细胞核采样参数
        use_4bit : 布尔值，默认 True
        是否使用4位量化（针对本地模型）

        退货
        -------
        LLM回应
        具有世代的响应对象

        参数:
            messages: 按对话顺序排列的 LLM 消息。
            model: 生成候选时使用的模型标识。
            n_tasks: 调用方提供的 `n_tasks` 参数。
            max_tokens: 调用方提供的 `max_tokens` 参数。
            temperature: 调用方提供的 `temperature` 参数。
            top_p: 调用方提供的 `top_p` 参数。
            use_4bit: 调用方提供的 `use_4bit` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 如果批处理被禁用或不是本地模型，请立即处理
        if not self.enabled or not is_local_model(model):
            # 对于 API 模型或禁用时，请使用直接调用
            from .query import generate_code
            return await generate_code(messages, n_tasks, model)
        
        # 为异步结果创建 Future，并由批处理 Worker 在完成后设置结果。
        future = asyncio.Future()
        request = QueuedRequest(
            messages=deepcopy(messages),
            model=model,
            n_tasks=n_tasks,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            future=future,
            timestamp=time.time(),
            use_4bit=use_4bit,
        )
        
        async with self.lock:
            self.queue.append(request)
            
            # 批次已满立即处理
            if len(self.queue) >= self.max_batch_size:
                if not self.processing:
                    asyncio.create_task(self._process_batch())
            # 否则，在窗口延迟后安排处理
            elif not self.processing and self._pending_timer is None:
                self._pending_timer = asyncio.create_task(self._process_batch_after_delay())
        
        # 等待结果
        return await future
    
    async def _process_batch_after_delay(self):
        """在窗口延迟后处理批次。"""
        await asyncio.sleep(self.window_ms)
        await self._process_batch()
    
    async def _process_batch(self):
        """处理一批排队的请求。"""
        # 从队列中提取批次
        async with self.lock:
            if self.processing or not self.queue:
                if self._pending_timer:
                    self._pending_timer = None
                return
            
            self.processing = True
            self._pending_timer = None
            
            # 最多可处理 max_batch_size 请求
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]
            
            batch_size = len(batch)
            model = batch[0].model  # 假设批次相同型号
        
        try:
            logger.info(f"Processing batch of {batch_size} requests for model {model}")
            
            # 按参数对请求进行分组（相同型号、温度、top_p等）
            # 目前，如果它们具有兼容的参数，我们将一起处理
            # 未来我们可以更加智能地分组
            
            # 从批处理中提取提示
            prompts = [req.messages for req in batch]
            
            # 按兼容参数对请求进行分组
            # 对于真正的配料，我们需要相同的型号、温度、top_p、max_tokens、use_4bit
            # 现在，如果它们是相同的型号，我们会将它们一起批处理
            # （可以更智能地按参数分组）
            
            first_req = batch[0]
            
            # 检查所有请求是否具有兼容的参数以实现真正的批处理
            compatible = all(
                req.model == first_req.model
                and req.temperature == first_req.temperature
                and req.top_p == first_req.top_p
                and req.max_tokens == first_req.max_tokens
                and req.use_4bit == first_req.use_4bit
                and req.n_tasks == 1  # 目前仅批量处理单任务请求
                for req in batch
            )
            
            if compatible and len(batch) > 1:
                # 真正的批处理：所有提示的单前向传递
                prompts = [req.messages for req in batch]
                responses = await generate_code_local_batch(
                    prompts,
                    first_req.model,
                    max_tokens=first_req.max_tokens,
                    temperature=first_req.temperature,
                    top_p=first_req.top_p,
                    use_4bit=first_req.use_4bit,
                )
            else:
                # Fallback：单独但并行处理
                # （对于具有不同参数或 n_tasks > 1 的请求）
                tasks = []
                for req in batch:
                    task = asyncio.create_task(
                        generate_code_local(
                            req.messages,
                            req.n_tasks,
                            req.model,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            use_4bit=req.use_4bit,
                        )
                    )
                    tasks.append(task)
                
                # 等待全部完成
                responses = await asyncio.gather(*tasks)
            
            # 将响应路由回 future
            for req, response in zip(batch, responses):
                if not req.future.done():
                    req.future.set_result(response)
            
            logger.info(
                f"Completed batch of {batch_size} requests in "
                f"{sum(r.elapsed_time for r in responses) / batch_size:.2f}s average"
            )
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # 将错误路由回 future
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
        finally:
            async with self.lock:
                self.processing = False
                # 如果有更多请求，则处理剩余队列
                if self.queue:
                    if len(self.queue) >= self.max_batch_size:
                        asyncio.create_task(self._process_batch())
                    else:
                        self._pending_timer = asyncio.create_task(
                            self._process_batch_after_delay()
                        )
    
    async def flush(self):
        """立即强制处理所有待处理的请求。"""
        async with self.lock:
            if self.queue and not self.processing:
                await self._process_batch()


# 全局队列实例
_global_batch_queue: Optional[LLMBatchQueue] = None


def get_batch_queue() -> LLMBatchQueue:
    """
    获取或创建全局批处理队列实例。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    global _global_batch_queue
    
    if _global_batch_queue is None:
        window_ms = int(os.getenv("LLM_BATCH_WINDOW_MS", "100"))
        max_batch = int(os.getenv("LLM_BATCH_MAX_SIZE", "8"))
        enabled = os.getenv("LLM_BATCH_ENABLED", "true").lower() == "true"
        
        _global_batch_queue = LLMBatchQueue(
            window_ms=window_ms,
            max_batch_size=max_batch,
            enabled=enabled,
        )
    
    return _global_batch_queue
