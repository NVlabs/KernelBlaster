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

"""提供基于 TCP 探活的轻量级远端资源客户端。"""

import aiohttp


class TCPClient:
    """封装对远端资源的连接与调用。"""
    _session = None

    @classmethod
    def get_session(cls):
        """
        获取 `get_session` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        if cls._session is None:
            connector = aiohttp.TCPConnector(limit=1024)
            cls._session = aiohttp.ClientSession(connector=connector)
        return cls._session

    @classmethod
    async def close_session(cls):
        """处理 `close_session` 对应的领域操作，并返回调用方所需的标准化结果。"""
        if cls._session:
            await cls._session.close()
            cls._session = None
