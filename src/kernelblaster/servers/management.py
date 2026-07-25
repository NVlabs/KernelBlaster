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

"""在本机启动、探活并配置编译服务和 GPU Worker。"""

import requests
import os
import subprocess
from pathlib import Path
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)
from io import TextIOWrapper
import time
import psutil
import socket
from ..config import GPUType, config
from .security import sanitized_worker_environment
from typing import Optional


def _worker_environment() -> dict[str, str]:
    """
    处理 `worker_environment` 所表示的内部步骤；该函数不属于稳定的公开接口。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    environment = sanitized_worker_environment(dict(os.environ))
    environment["KERNELBLASTER_WORKER_TOKEN"] = config.WORKER_TOKEN
    return environment


def test_server_connection(process, url, timeout: int = 5):
    """
    测试与服务器的连接并返回它是否可用。

    参数:
        process: 调用方提供的 `process` 参数。
        url: 目标服务或资源的 URL。
        timeout: 允许操作等待的最长秒数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """

    poll_interval = 0.25
    start_time = time.time()
    warned_exited = False
    while time.time() - start_time < timeout:
        # 注意：我们故意不会在启动过程中立即失败
        # 看来已经退出了。实际上，这可能会发生在陈旧/僵尸的情况下
        # 进程或包装启动器，而服务器端点仍然是
        # 可访问（e.g.，该端口上已运行的服务器）。
        if process is not None and process.poll() is not None and not warned_exited:
            warned_exited = True
            try:
                rc = process.poll()
            except Exception:
                rc = None
            logger.warning(
                f"⚠️ Server process for {url} appears to have exited (returncode={rc}); "
                f"continuing to probe {url}/health for up to {timeout}s."
            )
            # 停止检查进程对象以避免重复警告。
            process = None

        try:
            # 只需尝试连接到服务器而不发出真正的请求
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code < 500:
                # 根路径可能不存在，但服务器已启动。
                # 或者，任何非服务器错误都意味着服务器已启动
                logger.info(f"✅ Server at {url} is running")
                return True
            else:
                logger.error(
                    f"❌ Server at {url} returned error: {response.status_code}"
                )
                return False
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            logger.error(f"❌ Error connecting to {url}: {e}")
            return False
        time.sleep(poll_interval)

    logger.error(f"❌ Server at {url} failed to start after {timeout} seconds")
    return False


def initialize_compiler_server(
    log_file: TextIOWrapper,
    compile_server_url: str | None,
    artifacts_dir: Path,
    port: int | None,
):
    """
    初始化服务器并返回 URL。

    参数:
        log_file: 调用方提供的 `log_file` 参数。
        compile_server_url: 调用方提供的 `compile_server_url` 参数。
        artifacts_dir: 调用方提供的 `artifacts_dir` 参数。
        port: 远端服务监听或连接的端口。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """

    if compile_server_url is not None:
        logger.info(f"Using existing compile server at {compile_server_url}")
        return None, compile_server_url

    if port is None:
        port = find_free_port(start_port=2001)
        logger.info(f"🎯 Auto-assigned compiler server port: {port}")

    # 启动编译服务器
    compiler_server_process = None
    # 检查 libtorch 是否存在
    try:
        from torch.utils import cmake_prefix_path
    except ImportError:
        logger.error("PyTorch is required to start the compilation server.")
        return False
    if not Path(cmake_prefix_path).exists():
        logger.error(
            f"Libtorch CMake prefix path {cmake_prefix_path} does not exist! Please install pytorch."
        )
        return False

    compiler_server_cmd = [
        "python",
        "-m",
        "src.kernelblaster.servers.compile",
        "--port",
        str(port),
        "--num-workers",
        str(psutil.cpu_count(logical=False) - 1),  # 物理CPU核心
        "--artifacts-dir",
        str(artifacts_dir),
        "--host",
        "127.0.0.1",
    ]

    # 对 stdout 和 stderr 使用单个文件
    compiler_server_process = subprocess.Popen(
        compiler_server_cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        env=_worker_environment(),
    )

    compile_server_url = f"http://localhost:{port}"
    logger.info(
        f"Starting the compilation server at {compile_server_url}: {' '.join(compiler_server_cmd)}"
    )
    return compiler_server_process, compile_server_url


def initialize_gpu_server(
    log_file: TextIOWrapper,
    gpu: Optional[GPUType],
    port: int | None,
):
    """
    初始化GPU服务器并返回URL。

    参数:
        log_file: 调用方提供的 `log_file` 参数。
        gpu: 执行或分析任务使用的 GPU 配置。
        port: 远端服务监听或连接的端口。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """

    if gpu is None:
        gpu = GPUType.current()

    gpu_server_url = config.get_gpu_server_url(gpu)

    if gpu_server_url is not None:
        logger.info(f"Using existing GPU server at {gpu_server_url} for {gpu}")
        return None, gpu_server_url

    if port is None:
        port = find_free_port(start_port=2002)
        logger.info(f"🎯 Auto-assigned GPU server port: {port}")

    gpu_server_cmd = [
        "python",
        "-m",
        "src.kernelblaster.servers.gpu",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]

    # 对 stdout 和 stderr 使用单个文件
    gpu_server_process = subprocess.Popen(
        gpu_server_cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        env=_worker_environment(),
    )

    gpu_server_url = f"http://localhost:{port}"
    logger.info(
        f"Starting the GPU command server at {gpu_server_url}: {' '.join(gpu_server_cmd)}"
    )
    config.set_gpu_server_url(gpu, gpu_server_url)
    return gpu_server_process, gpu_server_url


def find_free_port(start_port: int = 2001) -> int:
    """
    查找从 start_port 开始的可用端口。

    参数:
        start_port: 调用方提供的 `start_port` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port starting from {start_port}")


def start_standalone_gpu_server(port: int = None, log_file_path: str = None) -> tuple[subprocess.Popen, str]:
    """
    启动独立的 GPU 服务器并返回进程和 URL。

    参数:
        port: 远端服务监听或连接的端口。
        log_file_path: 调用方提供的 `log_file_path` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    
    if port is None:
        port = find_free_port(start_port=2002)
        
    gpu_server_cmd = [
        "python",
        "-m",
        "src.kernelblaster.servers.gpu",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    
    # 设置日志记录
    if log_file_path:
        # 确保日志目录存在
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # 将日志文件路径传递给 GPU 服务器，以便 uvicorn 可以配置文件日志记录
        gpu_server_cmd.extend(["--log_path", str(log_file_path)])
        # 还可以使用行缓冲重定向 stdout/stderr 作为备份
        log_file = open(log_file_path, 'a', buffering=1)  # 行缓冲
        stdout_file = log_file
        stderr_file = log_file
    else:
        stdout_file = subprocess.PIPE
        stderr_file = subprocess.PIPE
    
    gpu_server_process = subprocess.Popen(
        gpu_server_cmd,
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,
        env=_worker_environment(),
    )
    
    gpu_server_url = f"http://localhost:{port}"
    logger.info(f"Starting standalone GPU server at {gpu_server_url}: {' '.join(gpu_server_cmd)}")
    
    # 测试连接
    if not test_server_connection(gpu_server_process, gpu_server_url, timeout=10):
        logger.error(f"Failed to start GPU server at {gpu_server_url}")
        gpu_server_process.terminate()
        raise RuntimeError(f"GPU server failed to start at {gpu_server_url}")
    
    return gpu_server_process, gpu_server_url
