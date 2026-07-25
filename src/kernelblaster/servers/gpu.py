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

"""实现 GPU 执行与性能分析服务，限制上传、环境变量、命令和临时文件。"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from enum import Enum
import os
import shlex
import psutil
from fastapi import Depends, FastAPI, HTTPException, File, UploadFile, Form
import logging
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import json
import tempfile
import stat
from typing import Optional

from .server_logging import get_log_config
from .security import sanitized_worker_environment, validate_worker_environment
from .utils import safe_kill_process
from .auth import require_worker_token, validate_worker_token
from ..config import config

env = None

QUEUE = asyncio.Queue()

logger = logging.getLogger("uvicorn")

# 所有操作的通用临时目录
WORKING_DIR = None

# 多 GPU 工作线程配置（启动时填充）
GPU_IDS: list[str] | None = None

ALLOWED_ENVIRONMENT_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_TF32_OVERRIDE",
    "CUDA_LAUNCH_BLOCKING",
    "OMP_NUM_THREADS",
}
class Profiler(str, Enum):
    """采集并规范化 Kernel 的性能指标。"""
    NCU = "ncu"
    NSYS = "nsys"


ALLOWED_PROFILERS = {profiler.value for profiler in Profiler}
FORBIDDEN_ARGUMENT_TOKENS = {";", "|", "||", "&&", ">", ">>", "<", "2>", "2>&1"}
async def read_upload_with_limit(upload: UploadFile, limit: int) -> bytes:
    """
    读取 `read_upload_with_limit` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        upload: 调用方提供的 `upload` 参数。
        limit: 调用方提供的 `limit` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        HTTPException: 输入、外部调用或状态不满足执行要求时抛出。
    """
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, limit + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail="Binary upload exceeds size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def validated_environment(values: Optional[dict]) -> dict[str, str]:
    """
    处理 `validated_environment` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        values: 调用方提供的 `values` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    result: dict[str, str] = {}
    for raw_key, raw_value in (values or {}).items():
        key = str(raw_key)
        value = str(raw_value)
        if key not in ALLOWED_ENVIRONMENT_KEYS:
            raise ValueError(f"Environment key is not allowed: {key}")
        if "\x00" in value:
            raise ValueError(f"Environment value contains NUL: {key}")
        result[key] = value
    return result


def build_execution_argv(
    binary_path: str,
    args: str = "",
    prefix_command: Optional[str | list[str]] = None,
) -> tuple[list[str], dict[str, str]]:
    """
    构建 argv 向量而不调用 shell。

    参数:
        binary_path: 调用方提供的 `binary_path` 参数。
        args: 调用方提供的 `args` 参数。
        prefix_command: 调用方提供的 `prefix_command` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """

    prefix = (
        [str(value) for value in prefix_command]
        if isinstance(prefix_command, list)
        else shlex.split(prefix_command or "")
    )
    prefix_environment: dict[str, str] = {}
    while prefix and "=" in prefix[0] and not prefix[0].startswith("-"):
        key, value = prefix.pop(0).split("=", 1)
        if key != "NVIDIA_TF32_OVERRIDE":
            raise ValueError(f"Profiler environment assignment is not allowed: {key}")
        prefix_environment[key] = value

    if prefix:
        profiler = Path(prefix[0]).name
        if profiler not in ALLOWED_PROFILERS:
            raise ValueError(f"Profiler is not allowed: {profiler}")
        for value in prefix[1:]:
            if value in FORBIDDEN_ARGUMENT_TOKENS or "\x00" in value:
                raise ValueError(f"Profiler argument is not allowed: {value!r}")

    return [*prefix, binary_path, *shlex.split(args or "")], prefix_environment


def get_temp_dir():
    """
    获取或创建所有 GPU 操作的通用临时目录

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    global WORKING_DIR
    if WORKING_DIR is None or not os.path.exists(WORKING_DIR):
        WORKING_DIR = tempfile.mkdtemp(prefix="kernelblaster_gpu_")
    return WORKING_DIR


# 在后台启动工作任务
@asynccontextmanager
async def lifespan(app):
    """
    处理 `lifespan` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        app: 调用方提供的 `app` 参数。
    """
    global logger, env, GPU_IDS

    # 该服务器启动的子进程的基础环境。
    # 注意：每个工作线程的 GPU 固定是在执行时通过环境变量应用的。
    env = sanitized_worker_environment()
    env.setdefault("NVIDIA_TF32_OVERRIDE", "0")

    # 确定要使用哪些 GPU（以及多少个工作线程）。
    # 示例：
    # 环境变量配置：KERNELBLASTER_GPU_SERVER_GPU_IDS="0,1,2,3"
    # 环境变量配置：KERNELBLASTER_GPU_SERVER_NUM_WORKERS=4
    gpu_ids_raw = os.getenv("KERNELBLASTER_GPU_SERVER_GPU_IDS", "").strip()
    if gpu_ids_raw:
        GPU_IDS = [s.strip() for s in gpu_ids_raw.split(",") if s.strip()]
    else:
        num_workers = int(os.getenv("KERNELBLASTER_GPU_SERVER_NUM_WORKERS", "1"))
        GPU_IDS = [str(i) for i in range(max(1, num_workers))]

    logger.info(
        f"GPU Server worker config: num_workers={len(GPU_IDS)} GPU_IDS={GPU_IDS} "
        f"(override via KERNELBLASTER_GPU_SERVER_NUM_WORKERS / KERNELBLASTER_GPU_SERVER_GPU_IDS)"
    )

    # 服务器启动时打印当前用户 (whoami)
    logger.info(f"GPU Server running as user: {os.getuid()}")
    logger.info(f"GPU Server running as user: {os.geteuid()}")

    stdout, stderr = await exec_command("whoami")
    logger.info(f"GPU Server running as user: {stdout}\n{stderr}")

    stdout, stderr = await exec_command("groups")
    logger.info(f"User groups: {stdout}\n{stderr}")
    
    # 启动服务器前打印 nvidia-smi 信息
    await print_nvidia_smi(logger)

    # 检查预先存在的 GPU 进程
    await check_gpu_processes()
    # 启动时启动工作任务（每个 GPU id 一个）
    for wid in range(len(GPU_IDS)):
        _ = asyncio.create_task(gpu_worker(wid))
    yield


APP = FastAPI(lifespan=lifespan)


class GpuExecutionRequest(BaseModel):
    """GPU二进制执行的请求模型"""

    args: Optional[str] = ""  # 二进制文件的命令行参数


class GpuCommandResult(BaseModel):
    """保存一次操作的标准化结果及其诊断信息。"""
    stdout: str | list[str] = []
    stderr: str | list[str] = []
    success: bool = False
    message: str = None


class GpuCommandError(Exception):
    """表示该领域内可被调用方识别和处理的失败。"""
    def __init__(self, error_message: str):
        """
        初始化 GpuCommandError 实例，并保存后续流程所需的配置与依赖。

        参数:
            error_message: 调用方提供的 `error_message` 参数。
        """
        self.error_message = error_message
        super().__init__(self.error_message)


async def print_nvidia_smi(logger):
    """
    打印 nvidia-smi 信息

    参数:
        logger: 记录诊断信息和任务进度的日志器。
    """
    try:
        nvidia_smi_stdout, nvidia_smi_stderr = await exec_command("nvidia-smi")
        logger.info(f"GPU Server Startup - nvidia-smi output:\n{nvidia_smi_stdout}")
        if nvidia_smi_stderr:
            logger.warning(
                f"GPU Server Startup - nvidia-smi stderr:\n{nvidia_smi_stderr}"
            )
    except Exception as nvidia_smi_error:
        logger.warning(
            f"GPU Server Startup - Failed to execute nvidia-smi: {str(nvidia_smi_error)}"
        )


async def check_gpu_processes():
    """
    检查 NVIDIA GPU 上是否存在任何预先存在的进程。

    过滤掉过时或不存在的 PID 以及进程名称所在的条目
    nvidia-smi 报告为“[未找到]”，以避免误报。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
        e: 输入、外部调用或状态不满足执行要求时抛出。
    """
    try:
        stdout, _ = await exec_command(
            "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader"
        )

        active_processes: list[str] = []
        for raw_line in stdout.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue

            pid_str = parts[0]
            proc_name = parts[1] if len(parts) > 1 else ""

            # 跳过 PID 格式无效的条目
            try:
                pid = int(pid_str)
            except ValueError:
                continue

            # 忽略过时的条目或无法解析进程名称的地方
            if proc_name == "[Not Found]" or not psutil.pid_exists(pid):
                continue

            active_processes.append(f"{pid}, {proc_name or '[Unknown]'}")

        if active_processes:
            raise RuntimeError(
                f"Found pre-existing GPU processes:\n{json.dumps(active_processes, indent=2)}"
            )

    except Exception as e:
        if "nvidia-smi: not found" in str(e):
            raise RuntimeError(
                "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
            )
        raise e


async def exec_command(
    cmd: str | list[str],
    timeout=3600,
    env_vars: Optional[dict] = None,
    n_runs: Optional[int] = 1,
) -> tuple[list[str], list[str]] | tuple[str, str]:
    """
    执行外壳命令

    参数:
        cmd: 调用方提供的 `cmd` 参数。
        timeout: 允许操作等待的最长秒数。
        env_vars: 调用方提供的 `env_vars` 参数。
        n_runs: 调用方提供的 `n_runs` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        GpuCommandError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    # 准备环境
    process_env = env.copy() if env else os.environ.copy()
    process_env.update(validated_environment(env_vars))
    argv = shlex.split(cmd) if isinstance(cmd, str) else [str(item) for item in cmd]
    if not argv:
        raise GpuCommandError("No command was provided")

    # 使用公共临时目录作为工作目录
    working_dir = get_temp_dir()

    stdout_list = []
    stderr_list = []

    for _ in range(n_runs):
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_env,
            start_new_session=True,
            cwd=working_dir,
        )
        try:
            # 等待进程超时
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            stdout_list.append(stdout.decode())
            stderr_list.append(stderr.decode())
            if proc.returncode != 0:
                raise GpuCommandError(
                    f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
                )
        except asyncio.TimeoutError:
            # 如果超时则终止该进程
            logger.error(f"TIMEOUT: {argv[0]}")
            await safe_kill_process(proc, logger)
            raise GpuCommandError(
                f"Timeout: Execution timed out after {timeout} seconds"
            )

    if n_runs == 1:
        return stdout_list[0], stderr_list[0]
    else:
        return stdout_list, stderr_list


async def exec_binary(
    binary_path: str,
    args: str = "",
    timeout=3600,
    env_vars: Optional[dict] = None,
    prefix_command: Optional[str] = None,
    n_runs: Optional[int] = 1,
) -> tuple[list[str], list[str]] | tuple[str, str]:
    """
    使用可选参数、环境变量和前缀命令执行二进制文件

    参数:
        binary_path: 调用方提供的 `binary_path` 参数。
        args: 调用方提供的 `args` 参数。
        timeout: 允许操作等待的最长秒数。
        env_vars: 调用方提供的 `env_vars` 参数。
        prefix_command: 调用方提供的 `prefix_command` 参数。
        n_runs: 调用方提供的 `n_runs` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    argv, prefix_environment = build_execution_argv(
        binary_path,
        args,
        prefix_command,
    )
    effective_environment = validated_environment(env_vars)
    effective_environment.update(prefix_environment)
    return await exec_command(argv, timeout, effective_environment, n_runs)


def save_binary_to_temp(binary_data: bytes, filename: str = "gpu_executable") -> str:
    """
    将二进制数据保存到临时文件并使其可执行

    参数:
        binary_data: 调用方提供的 `binary_data` 参数。
        filename: 调用方提供的 `filename` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    # 使用公共临时目录
    temp_dir = get_temp_dir()
    # 重要提示：切勿写入仅从客户端提供的文件名派生的路径。
    # 我们可以接收并发请求（并且客户端可以重试相同的请求），
    # 否则会导致：
    # - [Errno 26] 文本文件忙（执行时覆盖）
    # - “不存在”（另一个工作人员清理共享路径）
    safe_name = os.path.basename(filename) if filename else "gpu_executable"
    fd, binary_path = tempfile.mkstemp(prefix=f"{safe_name}_", dir=temp_dir)

    # 写入二进制数据
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(binary_data)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        cleanup_temp_file(binary_path)
        raise

    # 使可执行
    os.chmod(
        binary_path,
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
    )

    return binary_path


def cleanup_temp_file(binary_path: str):
    """
    清理临时二进制文件

    参数:
        binary_path: 调用方提供的 `binary_path` 参数。
    """
    try:
        if os.path.exists(binary_path):
            os.remove(binary_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file: {e}")


def complete_future(completion_future: asyncio.Future, result: GpuCommandResult) -> None:
    """
    当 HTTP 客户端已经断开连接时，不要使工作线程崩溃。

    参数:
        completion_future: 调用方提供的 `completion_future` 参数。
        result: 上一步产生并等待进一步处理的结果。
    """
    if not completion_future.done():
        completion_future.set_result(result)


async def gpu_worker(worker_id: int) -> GpuCommandResult:
    """
    处理来自队列的 GPU 执行请求

    参数:
        worker_id: 调用方提供的 `worker_id` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    while True:
        queue_item = await QUEUE.get()
        completion_future = queue_item[-1]  # completion Future 始终位于队列项末尾。

        try:
            if len(queue_item) == 7:
                # 带前缀的二进制执行：（binary_path，args，env_vars，prefix_command，n_runs，超时，completion_future）
                binary_path, args, env_vars, prefix_command, n_runs, timeout, _ = queue_item
            elif len(queue_item) == 6:
                # 向后兼容性：（binary_path、args、env_vars、prefix_command、n_runs、completion_future）
                binary_path, args, env_vars, prefix_command, n_runs, _ = queue_item
                timeout = 3600  # 默认超时
            
            # 6 项和 7 项格式的通用执行代码
            if len(queue_item) in (6, 7):
                # 通过注入 CUDA_VISIBLE_DEVICES 将此工作线程固定到特定 GPU。
                # 如果调用者显式传递了 CUDA_VISIBLE_DEVICES，请尊重它。
                eff_env_vars = dict(env_vars or {})
                if "CUDA_VISIBLE_DEVICES" not in eff_env_vars:
                    gpu_id = str(worker_id)
                    if GPU_IDS and worker_id < len(GPU_IDS):
                        gpu_id = str(GPU_IDS[worker_id])
                    eff_env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id
                # 确保 TF32 覆盖稳定，除非调用者另有要求。
                eff_env_vars.setdefault("NVIDIA_TF32_OVERRIDE", "0")
                gpu_visible = eff_env_vars.get("CUDA_VISIBLE_DEVICES", "<unset>")
                logger.info(
                    f"[Worker {worker_id}]: Assigned GPU CUDA_VISIBLE_DEVICES={gpu_visible} for binary {binary_path}"
                )
                logger.info(
                    f"[Worker {worker_id}]: Executing binary {binary_path} with args: {args}, env_vars: {eff_env_vars}, prefix: {prefix_command}, n_runs: {n_runs}, timeout: {timeout}"
                )

                stdout_list, stderr_list = await exec_binary(
                    binary_path,
                    args,
                    timeout=timeout,
                    env_vars=eff_env_vars,
                    prefix_command=prefix_command,
                    n_runs=n_runs,
                )
                logger.info(
                    f"[Worker {worker_id}]: Successfully executed binary on CUDA_VISIBLE_DEVICES={gpu_visible}: {f'{prefix_command} ' if prefix_command else ''}{binary_path} with {n_runs} runs"
                )

                # 清理临时二进制文件
                cleanup_temp_file(binary_path)

            else:
                raise ValueError(f"Invalid queue item format: {len(queue_item)} items")

            complete_future(
                completion_future,
                GpuCommandResult(success=True, stdout=stdout_list, stderr=stderr_list),
            )

        except GpuCommandError as e:
            if len(queue_item) in (6, 7):
                binary_path = queue_item[0]
                # 尽最大努力 GPU 归因失败
                gpu_visible = None
                try:
                    gpu_visible = (env_vars or {}).get("CUDA_VISIBLE_DEVICES")
                except Exception:
                    gpu_visible = None
                logger.error(
                    f"[Worker {worker_id}]: Error executing binary {binary_path}"
                    f"{' on CUDA_VISIBLE_DEVICES=' + str(gpu_visible) if gpu_visible is not None else ''}: {e.error_message}"
                )
                # 也清理错误
                cleanup_temp_file(binary_path)
            complete_future(
                completion_future,
                GpuCommandResult(success=False, message=e.error_message),
            )
        except Exception as e:
            logger.error(f"[Worker {worker_id}]: Unexpected error: {str(e)}")

            # 如果这是二进制执行，则清理二进制文件
            if len(queue_item) >= 4:
                cleanup_temp_file(queue_item[0])

            complete_future(
                completion_future,
                GpuCommandResult(
                    success=False,
                    message=f"Internal error: {str(e)}",
                ),
            )
        finally:
            QUEUE.task_done()


@APP.get("/health")
async def health_check():
    """
    健康检查端点

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return {"status": "healthy", "service": "gpu-server"}


@APP.get("/ready")
async def ready_check(_authorized: None = Depends(require_worker_token)):
    """Authenticated readiness endpoint used to verify worker-token audience."""
    return {"status": "ready", "service": "gpu-server"}


@APP.post("/gpu/binary", response_model=GpuCommandResult)
async def execute_gpu_binary(
    binary: UploadFile = File(..., description="Binary executable to run on GPU"),
    args: Optional[str] = Form("", description="Command line arguments for the binary"),
    env_vars: Optional[str] = Form(
        None, description="Environment variables in JSON format"
    ),
    prefix_command: Optional[str] = Form(
        None,
        description="Deprecated string profiler prefix; use profiler/profiler_args.",
    ),
    profiler: Optional[Profiler] = Form(
        None,
        description="Enumerated profiler executable.",
    ),
    profiler_args: Optional[str] = Form(
        None,
        description="JSON list of profiler arguments.",
    ),
    n_runs: Optional[int] = Form(
        1,
        description="Number of times to run the binary",
    ),
    timeout: Optional[float] = Form(
        3600,
        description="Timeout in seconds for command execution",
    ),
    _authorized: None = Depends(require_worker_token),
):
    """
    在GPU服务器上执行二进制文件

    参数:
        binary: 调用方提供的 `binary` 参数。
        args: 调用方提供的 `args` 参数。
        env_vars: 调用方提供的 `env_vars` 参数。
        prefix_command: 调用方提供的 `prefix_command` 参数。
        profiler: 调用方提供的 `profiler` 参数。
        profiler_args: 调用方提供的 `profiler_args` 参数。
        n_runs: 调用方提供的 `n_runs` 参数。
        timeout: 允许操作等待的最长秒数。
        _authorized: 调用方提供的 `_authorized` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        HTTPException: 输入、外部调用或状态不满足执行要求时抛出。
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """

    logger.info(
        f"/gpu/binary - Binary: {binary.filename}, Prefix: {prefix_command}, "
        f"Timeout: {timeout}s, Queue backlog: {QUEUE.qsize()}"
    )

    try:
        # 读取二进制数据
        binary_data = await read_upload_with_limit(
            binary,
            config.MAX_GPU_BINARY_BYTES,
        )
        if not binary_data:
            raise HTTPException(status_code=400, detail="Empty binary file provided")
        if n_runs is None or not 1 <= n_runs <= 100:
            raise HTTPException(status_code=400, detail="n_runs must be between 1 and 100")
        if timeout is None or not 0 < timeout <= 3600:
            raise HTTPException(status_code=400, detail="timeout must be in (0, 3600]")

        try:
            if profiler is not None and prefix_command:
                raise ValueError("Use either profiler or deprecated prefix_command, not both")
            effective_prefix: str | list[str] | None = prefix_command
            if profiler is not None:
                decoded_profiler_args = json.loads(profiler_args or "[]")
                if not isinstance(decoded_profiler_args, list) or not all(
                    isinstance(value, str) for value in decoded_profiler_args
                ):
                    raise ValueError("profiler_args must be a JSON list of strings")
                effective_prefix = [profiler.value, *decoded_profiler_args]
            build_execution_argv("/tmp/probe", args or "", effective_prefix)
        except (json.JSONDecodeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        # 解析环境变量（如果提供）
        parsed_env_vars = None
        if env_vars:
            try:
                decoded_env_vars = json.loads(env_vars)
                if not isinstance(decoded_env_vars, dict):
                    raise ValueError("Environment variables must be a JSON object")
                parsed_env_vars = validated_environment(decoded_env_vars)
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid environment variables JSON: {str(e)}",
                )

        # 将二进制文件保存到临时位置
        binary_path = save_binary_to_temp(
            binary_data, binary.filename or "gpu_executable"
        )

        # 创建 Future，使请求处理协程可以等待 GPU Worker 完成。
        completion_future = asyncio.Future()

        # 添加到执行队列（7项元组格式：binary_path、args、env_vars、prefix_command、n_runs、超时、completion_future）
        await QUEUE.put(
            (
                binary_path,
                args,
                parsed_env_vars,
                effective_prefix,
                n_runs,
                timeout,
                completion_future,
            )
        )

        # 等待完成
        await completion_future
        return completion_future.result()

    except asyncio.CancelledError:
        logger.info(f"Request for binary {binary.filename} was cancelled")
        raise HTTPException(status_code=500, detail="Request was cancelled")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing binary execution request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def run_server(host: str, port: int, log_filepath: str = None):
    """
    使用 REST API 运行编译服务器

    参数：
    host：要绑定服务器的主机
    port：服务器绑定的端口
    log_filepath：用于 uvicorn 日志记录的日志文件的可选路径

    参数:
        host: 远端服务监听或连接的主机名。
        port: 远端服务监听或连接的端口。
        log_filepath: 调用方提供的 `log_filepath` 参数。
    """
    # 运行 FastAPI 服务器
    log_config = get_log_config(log_filepath=log_filepath)
    uvicorn.run(
        APP, host=host, port=port, log_config=log_config, timeout_graceful_shutdown=0.1
    )


def main(args):
    validate_worker_environment()
    validate_worker_token()
    # 如果提供了日志路径，请确保日志目录存在
    """
    处理 `main` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        args: 调用方提供的 `args` 参数。
    """
    if args.log_path:
        log_dir = args.log_path.parent
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行 REST API 编译服务器
    run_server(
        args.host,
        args.port,
        log_filepath=str(args.log_path) if args.log_path else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2002)
    parser.add_argument(
        "--log_path", type=Path, default=Path("/tmp/kernelblaster/gpu_server.log")
    )
    args = parser.parse_args()

    # 定义 GPU 子进程的基本环境变量。
    # 每个工作线程 CUDA_VISIBLE_DEVICES 固定应用于 gpu_worker()。
    env = sanitized_worker_environment()
    env.setdefault("NVIDIA_TF32_OVERRIDE", "0")

    main(args)
