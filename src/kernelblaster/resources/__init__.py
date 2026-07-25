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

"""导出编译与 GPU 资源客户端和受管服务包装器。"""

from .client import TCPClient

__all__ = ["TCPClient", "CompileServer", "GPUServer"]


def __getattr__(name: str):
    if name in {"CompileServer", "GPUServer"}:
        from .servers import CompileServer, GPUServer

        return {"CompileServer": CompileServer, "GPUServer": GPUServer}[name]
    raise AttributeError(name)
