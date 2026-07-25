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

"""使用上下文变量传递当前运行记录器，并为事件补充公共元数据。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from .recorder import RunRecorder


_active_recorder: ContextVar[RunRecorder | None] = ContextVar(
    "kernelblaster_run_recorder", default=None
)
_event_context: ContextVar[dict[str, Any]] = ContextVar(
    "kernelblaster_event_context", default={}
)


def get_run_recorder() -> RunRecorder | None:
    """
    获取 `get_run_recorder` 对应的领域操作，并返回调用方所需的标准化结果。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return _active_recorder.get()


def set_run_recorder(recorder: RunRecorder | None) -> None:
    """
    设置 `set_run_recorder` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        recorder: 调用方提供的 `recorder` 参数。
    """
    _active_recorder.set(recorder)


@contextmanager
def event_context(**values: Any) -> Iterator[None]:
    """
    处理 `event_context` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        values: 调用方提供的 `values` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    merged = {**_event_context.get(), **{k: v for k, v in values.items() if v is not None}}
    token = _event_context.set(merged)
    try:
        yield
    finally:
        _event_context.reset(token)


def record_event(event_type: str, **kwargs) -> None:
    """
    记录 `record_event` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        event_type: 调用方提供的 `event_type` 参数。
        kwargs: 调用方提供的 `kwargs` 参数。
    """
    recorder = get_run_recorder()
    if recorder is not None:
        context = _event_context.get()
        for key in ("task_id", "rollout_id", "stage", "candidate_id"):
            if key not in kwargs and key in context:
                kwargs[key] = context[key]
        recorder.record_event(event_type, **kwargs)
