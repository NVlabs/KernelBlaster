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

"""提供可按名称累计耗时的轻量级上下文计时器。"""

import time


class NamedTimer:
    """封装 `NamedTimer` 对应的领域状态与操作。"""
    def __init__(self):
        """初始化 NamedTimer 实例，并保存后续流程所需的配置与依赖。"""
        self.starts = {}
        self.elapsed = {}

    def start(self, key=""):
        """
        启动 `start` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            key: 调用方提供的 `key` 参数。
        """
        self.starts[key] = time.time()

    def stop(self, key=""):
        """
        停止 `stop` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            key: 调用方提供的 `key` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        assert key in self.starts
        self.elapsed[key] = time.time() - self.starts[key]
        return self.elapsed[key]

    def reset(self):
        """重置 `reset` 对应的领域操作，并返回调用方所需的标准化结果。"""
        self.starts.clear()
        self.elapsed.clear()
