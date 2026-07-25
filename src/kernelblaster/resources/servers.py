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

"""管理编译服务和 GPU 服务的启动、健康检查与退出清理。"""

from pathlib import Path
from typing import Optional
from ..servers.management import (
    initialize_compiler_server,
    initialize_gpu_server,
    test_server_connection,
)
from ..config import config
from ..config import GPUType


class ManagedServer:
    """管理远端计算资源的生命周期和客户端连接。"""
    def __init__(self, logger, log_path: Path):
        """
        初始化 ManagedServer 实例，并保存后续流程所需的配置与依赖。

        参数:
            logger: 记录诊断信息和任务进度的日志器。
            log_path: 调用方提供的 `log_path` 参数。
        """
        self.logger = logger
        self.log_path = log_path
        self.log_file_handle = open(log_path, "w")
        self.process = None
        self.url = None

    def __del__(self):
        """处理 `__del__` 对应的领域操作，并返回调用方所需的标准化结果。"""
        self.cleanup()

    @property
    def is_managed(self):
        """
        判断 `is_managed` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return self.process is not None

    def cleanup(self):
        """清理 `cleanup` 对应的领域操作，并返回调用方所需的标准化结果。"""
        if self.is_managed:
            self.process.terminate()
            self.process.wait()
            self.process = None
        if self.log_file_handle:
            self.log_file_handle.close()

    def _log_error_output(self):
        """服务器进程的日志错误输出"""
        try:
            log_content = self.log_path.read_text()
            self.logger.error(f"Server Logs:\n{log_content}")
        except Exception as e:
            self.logger.error(f"Failed to read log file: {e}")

    def wait_for_connection(self, timeout: int = 5):
        """
        等待服务器启动

        参数:
            timeout: 允许操作等待的最长秒数。

        异常:
            RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        test_result = test_server_connection(self.process, self.url, timeout)
        if not test_result:
            self._log_error_output()
            raise RuntimeError(f"Failed to initialize server at {self.url}")


class CompileServer(ManagedServer):
    """管理远端计算资源的生命周期和客户端连接。"""
    def __init__(
        self,
        logger,
        experiment_dir: Path,
        artifacts_dir: str = config.TEMP_DIRECTORY,
        port: int = None,
    ):
        """
        创建一个新的编译服务器。
        如果 port 不是 None，服务器将使用新端口进行初始化。

        参数:
            logger: 记录诊断信息和任务进度的日志器。
            experiment_dir: 调用方提供的 `experiment_dir` 参数。
            artifacts_dir: 调用方提供的 `artifacts_dir` 参数。
            port: 远端服务监听或连接的端口。
        """
        super().__init__(logger, experiment_dir / "compile_server.log")
        self.artifacts_dir = artifacts_dir
        self.__initialize(port)

    def __initialize(self, port: int = None):
        """
        处理 `__initialize` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            port: 远端服务监听或连接的端口。
        """
        self.process, self.url = initialize_compiler_server(
            self.log_file_handle,
            config.COMPILE_SERVER_URL,
            Path(self.artifacts_dir),
            port,
        )


class GPUServer(ManagedServer):
    """管理远端计算资源的生命周期和客户端连接。"""
    def __init__(
        self,
        logger,
        experiment_dir: Path,
        gpu: Optional[GPUType] = None,
        port: int = None,
    ):
        """
        初始化 GPUServer 实例，并保存后续流程所需的配置与依赖。

        参数:
            logger: 记录诊断信息和任务进度的日志器。
            experiment_dir: 调用方提供的 `experiment_dir` 参数。
            gpu: 执行或分析任务使用的 GPU 配置。
            port: 远端服务监听或连接的端口。
        """
        super().__init__(logger, experiment_dir / "gpu_server.log")
        self.__initialize(gpu, port)

    def __initialize(self, gpu: Optional[GPUType], port: int = None):
        """
        处理 `__initialize` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            gpu: 执行或分析任务使用的 GPU 配置。
            port: 远端服务监听或连接的端口。
        """
        self.logger.info(
            f"Initializing GPU server for {gpu if gpu else 'current GPU'}..."
        )
        self.process, self.url = initialize_gpu_server(
            self.log_file_handle,
            gpu,
            port,
        )
