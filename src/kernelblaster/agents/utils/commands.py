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

"""封装 CUDA 源码编译、GPU 可执行文件运行及远端服务调用。"""

import aiohttp
import asyncio
from typing import Optional
from pathlib import Path
import os
import json
import shlex
import time

from loguru import logger
from .error import FeedbackError
from ...config import config, GPUType
from ...observability import record_event
from ...resources import TCPClient

__all__ = ["run_gpu_executable", "compile_cu", "compile_and_run_cu_file"]


async def _run_gpu_binary(
    binary_path,
    url,
    timeout,
    job_name,
    env_vars=None,
    prefix_command=None,
    n_runs=1,
    attempt=0,
):
    """
    通过新的 GPU 二进制文件上传端点执行二进制文件。

    参数:
        binary_path: 调用方提供的 `binary_path` 参数。
        url: 目标服务或资源的 URL。
        timeout: 允许操作等待的最长秒数。
        job_name: 调用方提供的 `job_name` 参数。
        env_vars: 调用方提供的 `env_vars` 参数。
        prefix_command: 调用方提供的 `prefix_command` 参数。
        n_runs: 调用方提供的 `n_runs` 参数。
        attempt: 调用方提供的 `attempt` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        FeedbackError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    try:
        # 首先读取二进制文件以获取大小信息
        with open(binary_path, "rb") as f:
            binary_data = f.read()
        
        binary_size = len(binary_data)
        binary_filename = os.path.basename(binary_path)
        
        # 检查这是否是 init.cu 文件以获取其他上下文
        is_init_cu = "init.cu" in str(job_name) if job_name else False
        init_cu_note = " (init.cu file)" if is_init_cu else ""
        
        logger.info(
            f"Uploading and executing binary to {url}/gpu/binary{init_cu_note} - "
            f"binary_path: {binary_path}, binary_filename: {binary_filename}, "
            f"binary_size: {binary_size} bytes, prefix_command: {prefix_command}, "
            f"n_runs: {n_runs}, job_name: {job_name}, attempt: {attempt}"
        )

        # 准备表单数据
        data = aiohttp.FormData()
        data.add_field(
            "binary",
            binary_data,
            filename=os.path.basename(binary_path),
            content_type="application/octet-stream",
        )
        data.add_field("n_runs", str(n_runs))
        data.add_field("timeout", str(timeout))
        if prefix_command:
            prefix = shlex.split(prefix_command)
            prefix_environment = {}
            while prefix and "=" in prefix[0] and not prefix[0].startswith("-"):
                key, value = prefix.pop(0).split("=", 1)
                if key != "NVIDIA_TF32_OVERRIDE":
                    raise FeedbackError(
                        f"Profiler environment assignment is not allowed: {key}"
                    )
                prefix_environment[key] = value
            if not prefix:
                raise FeedbackError("Profiler prefix did not name a profiler")
            effective_env = dict(env_vars or {})
            effective_env.update(prefix_environment)
            data.add_field("env_vars", json.dumps(effective_env))
            data.add_field("profiler", os.path.basename(prefix[0]))
            data.add_field("profiler_args", json.dumps(prefix[1:]))
        elif env_vars:
            data.add_field("env_vars", json.dumps(env_vars))

        if url and "api.nvcf.nvidia.com" in url and os.getenv("NVCF_API_KEY"):
            headers = {"Authorization": f"Bearer {os.getenv('NVCF_API_KEY')}"}
        else:
            headers = {"Authorization": f"Bearer {config.WORKER_TOKEN}"}
        async with TCPClient.get_session().post(
            f"{url}/gpu/binary", data=data, timeout=timeout, headers=headers
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.warning(
                    f"GPU server returned status {response.status}, response: {response_text}"
                )
                raise FeedbackError(
                    f"GPU execution failed for {job_name}: {response_text}"
                )

            try:
                result = await response.json()
            except Exception as json_error:
                # 如果 JSON 解析失败，尝试获取文本响应
                response_text = await response.text()
                raise FeedbackError(
                    f"GPU server returned invalid JSON for {job_name}: {json_error}. Response: {response_text[:500]}"
                )
            
            success = result.get("success", False)
            if not success:
                error_message = result.get("message", result.get("detail", "Unknown error"))
                raise FeedbackError(
                    f"Execution failed for {job_name}: {error_message}"
                )
            return result.get("stdout", ""), result.get("stderr", "")
    except aiohttp.ClientError as e:
        error_msg = str(e).lower()
        # 获取错误日志的二进制信息
        try:
            binary_size = os.path.getsize(binary_path) if os.path.exists(binary_path) else "unknown"
            binary_filename = os.path.basename(binary_path)
        except:
            binary_size = "unknown"
            binary_filename = os.path.basename(binary_path) if binary_path else "unknown"
        
        # 重试暂时性连接错误（例如“无法写入请求正文”）
        if ("can not write request body" in error_msg or "connection" in error_msg) and attempt == 0:
            # 检查这是否是 init.cu 文件以获取其他上下文
            is_init_cu = "init.cu" in str(job_name) if job_name else False
            init_cu_note = " (init.cu file)" if is_init_cu else ""
            
            logger.warning(
                f"Transient connection error for {job_name}{init_cu_note}: {e}. "
                f"Attempting to write binary_path: {binary_path}, binary_filename: {binary_filename}, "
                f"binary_size: {binary_size} bytes, url: {url}/gpu/binary, "
                f"prefix_command: {prefix_command}, n_runs: {n_runs}. Retrying once after 1s..."
            )
            await asyncio.sleep(1.0)
            # 重试整个操作一次
            return await _run_gpu_binary(binary_path, url, timeout, job_name, env_vars, prefix_command, n_runs, attempt=1)
        # 检查这是否是 init.cu 文件以获取其他上下文
        is_init_cu = "init.cu" in str(job_name) if job_name else False
        init_cu_note = " (init.cu file)" if is_init_cu else ""
        
        raise FeedbackError(
            f"Error connecting to GPU server for {job_name}{init_cu_note}: {e}. "
            f"binary_path: {binary_path}, binary_filename: {binary_filename}, "
            f"binary_size: {binary_size} bytes, url: {url}/gpu/binary, prefix_command: {prefix_command}"
        )
    except asyncio.TimeoutError as e:
        try:
            binary_size = os.path.getsize(binary_path) if os.path.exists(binary_path) else "unknown"
            binary_filename = os.path.basename(binary_path)
        except:
            binary_size = "unknown"
            binary_filename = os.path.basename(binary_path) if binary_path else "unknown"
        
        # 检查这是否是 init.cu 文件以获取其他上下文
        is_init_cu = "init.cu" in str(job_name) if job_name else False
        init_cu_note = " (init.cu file)" if is_init_cu else ""
        
        if attempt == 0:
            logger.warning(
                f"Timeout for {job_name}{init_cu_note}. binary_path: {binary_path}, binary_filename: {binary_filename}, "
                f"binary_size: {binary_size} bytes, url: {url}/gpu/binary, prefix_command: {prefix_command}, "
                f"n_runs: {n_runs}. Retrying once after 1s..."
            )
            await asyncio.sleep(1.0)
            return await _run_gpu_binary(binary_path, url, timeout, job_name, env_vars, prefix_command, n_runs, attempt=1)
        raise FeedbackError(
            f"Timeout: failed to receive a result for {job_name}{init_cu_note} after {timeout} seconds. "
            f"binary_path: {binary_path}, binary_filename: {binary_filename}, binary_size: {binary_size} bytes, "
            f"url: {url}/gpu/binary, prefix_command: {prefix_command}, n_runs: {n_runs}"
        )
    except IOError as e:
        logger.error(f"Error reading binary file {binary_path}: {e}")
        exit(1)


async def run_gpu_executable(
    executable_path: Path,
    gpu: GPUType,
    timeout: float,
    job_name: str,
    prefix_command: Optional[str] = None,
    n_runs: int = 1,
) -> tuple[list[str], list[str]]:
    """
    运行 `run_gpu_executable` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        executable_path: 调用方提供的 `executable_path` 参数。
        gpu: 执行或分析任务使用的 GPU 配置。
        timeout: 允许操作等待的最长秒数。
        job_name: 调用方提供的 `job_name` 参数。
        prefix_command: 调用方提供的 `prefix_command` 参数。
        n_runs: 调用方提供的 `n_runs` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    url = config.get_gpu_server_url(gpu)
    started = time.monotonic()
    is_ncu = bool(prefix_command and "ncu" in prefix_command.lower())
    try:
        result = await _run_gpu_binary(
            executable_path,
            url,
            timeout,
            job_name,
            prefix_command=prefix_command,
            n_runs=n_runs,
        )
    except Exception as error:
        if is_ncu:
            record_event(
                "cuda_profile_failed",
                status="error",
                data={
                    "job_name": Path(job_name).name,
                    "gpu": gpu.value,
                    "profiler": "ncu",
                    "latency_seconds": time.monotonic() - started,
                    "error_type": type(error).__name__,
                },
            )
        raise

    if is_ncu:
        record_event(
            "cuda_profile_completed",
            data={
                "job_name": Path(job_name).name,
                "gpu": gpu.value,
                "profiler": "ncu",
                "runs": n_runs,
                "latency_seconds": time.monotonic() - started,
            },
        )
    return result


async def _compile_cu(
    main_filepath: Path,
    cuda_filepath: Optional[Path],
    gpu: GPUType,
    url: str,
    timeout: float,
    job_name: str,
    persistent_artifacts: bool,
):
    """
    编译 `compile_cu` 所表示的内部步骤；该函数不属于稳定的公开接口。

    参数:
        main_filepath: 调用方提供的 `main_filepath` 参数。
        cuda_filepath: 调用方提供的 `cuda_filepath` 参数。
        gpu: 执行或分析任务使用的 GPU 配置。
        url: 目标服务或资源的 URL。
        timeout: 允许操作等待的最长秒数。
        job_name: 调用方提供的 `job_name` 参数。
        persistent_artifacts: 调用方提供的 `persistent_artifacts` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        FeedbackError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    try:
        # 确保编译服务器的路径是绝对的
        main_filepath_abs = main_filepath.resolve()
        cuda_filepath_abs = cuda_filepath.resolve() if cuda_filepath else None
        
        logger.info(f"Compile request - job_name: {job_name}")
        logger.info(f"  main_filepath (original): {main_filepath}")
        logger.info(f"  main_filepath (resolved): {main_filepath_abs}")
        if cuda_filepath:
            logger.info(f"  cuda_filepath (original): {cuda_filepath}")
            logger.info(f"  cuda_filepath (resolved): {cuda_filepath_abs}")
        logger.info(f"  persistent_artifacts: {persistent_artifacts}")
        logger.info(f"  sm_version: {gpu.sm}")
        
        logger.info(f"Submitted {job_name} to {url}/compile")
        async with TCPClient.get_session().get(
            f"{url}/compile",
            params={
                "job_name": job_name,
                "main_file": str(main_filepath_abs),
                "cuda_file": str(cuda_filepath_abs) if cuda_filepath_abs else "",
                "persistent_artifacts": int(persistent_artifacts),
                "sm_version": gpu.sm,
            },
            timeout=timeout,
            headers={"Authorization": f"Bearer {config.WORKER_TOKEN}"},
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.warning(
                    f"Compilation server returned status {response.status}, response: {response_text}"
                )
                raise FeedbackError(
                    f"Failed to compile the file {job_name}: {response_text}"
                )

            result = await response.json()
            if not result["success"]:
                raise FeedbackError(
                    f"Failed to compile the file {job_name}: {result['message']}"
                )
            return result["output_path"]
    except aiohttp.ClientError as e:
        raise FeedbackError(f"Error connecting to compilation server: {e}")
    except asyncio.TimeoutError as e:
        raise FeedbackError(
            f"Timeout: failed to compile {job_name} after {timeout} seconds"
        )


async def compile_cu(
    main_filepath: Path,
    cuda_filepath: Optional[Path],
    gpu: GPUType,
    timeout: float = 120,
    job_name: str = "",
    persistent_artifacts: bool = False,
) -> str:
    """
    通过编译服务器编译CUDA文件。

    参数：
    job_name：作业名称
    main_file：主 .cu 文件的路径
    cuda_file：CUDA .cuh 文件的路径
    gpu：编译文件的 GPU 类型
    超时：超时（以秒为单位）
    url：编译服务器的URL。如果没有，将使用默认 URL。
    persistent_artifacts：为 True 时，编译服务把 CUDA 源码产物保存在唯一目录中，避免被并行编译覆盖。

    返回：
    编译后的二进制文件的路径

    参数:
        main_filepath: 调用方提供的 `main_filepath` 参数。
        cuda_filepath: 调用方提供的 `cuda_filepath` 参数。
        gpu: 执行或分析任务使用的 GPU 配置。
        timeout: 允许操作等待的最长秒数。
        job_name: 调用方提供的 `job_name` 参数。
        persistent_artifacts: 调用方提供的 `persistent_artifacts` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return await _compile_cu(
        main_filepath,
        cuda_filepath,
        gpu,
        config.COMPILE_SERVER_URL,
        timeout,
        job_name,
        persistent_artifacts,
    )


async def compile_and_run_cu_file(
    main_filepath: Path,
    cuda_filepath: Path,
    gpu: GPUType,
    timer,
    logger,
    persistent_artifacts=False,
    timeout=1200,
    num_runs=5,
    passed_keyword=None,
    prefix_command: Optional[str] = None,
) -> tuple[list[str], list[str], Path, bool]:
    """
    使用新的二进制上传方法多次编译并运行 CUDA 文件。

    参数：
    main_filepath：主 CUDA 文件的路径
    cuda_filepath：CUDA头文件的路径
    定时器：定时器对象
    记录器：记录器对象
    persistent_artifacts：为 True 时，编译服务把 CUDA 源码产物保存在唯一目录中，避免被并行编译覆盖。
    超时：超时（以秒为单位）
    num_runs：运行内核的次数
    passed_keyword：如果提供，请检查此关键字是否在每次运行的标准输出中。如果没有找到则停止运行。
    prefix_command：在二进制文件之前添加前缀的命令（e.g.、'ncu'、'nsys profile'）

    返回：
    一个元组包含：
    - 每次运行的标准输出字符串列表
    - 每次运行的 stderr 字符串列表
    - 编译后的二进制文件的路径
    - 内核执行是否成功

    参数:
        main_filepath: 调用方提供的 `main_filepath` 参数。
        cuda_filepath: 调用方提供的 `cuda_filepath` 参数。
        gpu: 执行或分析任务使用的 GPU 配置。
        timer: 调用方提供的 `timer` 参数。
        logger: 记录诊断信息和任务进度的日志器。
        persistent_artifacts: 调用方提供的 `persistent_artifacts` 参数。
        timeout: 允许操作等待的最长秒数。
        num_runs: 调用方提供的 `num_runs` 参数。
        passed_keyword: 调用方提供的 `passed_keyword` 参数。
        prefix_command: 调用方提供的 `prefix_command` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    job_name = str(main_filepath) if cuda_filepath is None else str(cuda_filepath)
    timer.start("compilation")
    try:
        compiled_path = await compile_cu(
            main_filepath, cuda_filepath, gpu, timeout, job_name, persistent_artifacts
        )
        duration = timer.stop("compilation")
        record_event(
            "cuda_compile_completed",
            data={
                "job_name": Path(job_name).name,
                "gpu": gpu.value,
                "sm": gpu.sm,
                "latency_seconds": duration,
                "persistent_artifacts": persistent_artifacts,
            },
        )
    except Exception as error:
        duration = timer.stop("compilation")
        record_event(
            "cuda_compile_failed",
            status="error",
            data={
                "job_name": Path(job_name).name,
                "gpu": gpu.value,
                "sm": gpu.sm,
                "latency_seconds": duration,
                "error_type": type(error).__name__,
            },
        )
        raise

    logger.info(f"File compilation completed in {duration:0.2f} seconds")

    success = True

    timer.start("kernel_executions")

    if num_runs == 1:
        stdout, stderr = await run_gpu_executable(
            executable_path=Path(compiled_path),
            gpu=gpu,
            timeout=timeout,
            job_name=job_name,
            prefix_command=prefix_command,
            n_runs=num_runs,
        )
        stdout_list = [stdout]
        stderr_list = [stderr]
    else:
        stdout_list, stderr_list = await run_gpu_executable(
            executable_path=Path(compiled_path),
            gpu=gpu,
            timeout=timeout,
            job_name=job_name,
            prefix_command=prefix_command,
            n_runs=num_runs,
        )

    logger.info(
        f"Kernel execution of {num_runs} runs completed in {duration:0.2f} seconds"
    )

    duration = timer.stop("kernel_executions")

    # 如果提供了 passed_keyword 并且在标准输出中找不到，请提前停止
    for i, stdout in enumerate(stdout_list):
        if passed_keyword is not None and passed_keyword.lower() not in stdout.lower():
            logger.info(
                f"Keyword '{passed_keyword}' not found in run {i+1}, stopping further runs"
            )
            success = False
            break

    logger.info(
        f"{len(stdout_list)} kernel executions completed in {duration:0.2f} seconds. Success: {success}"
    )

    if passed_keyword is not None:
        record_event(
            "cuda_correctness_completed",
            status="ok" if success else "error",
            data={
                "job_name": Path(job_name).name,
                "gpu": gpu.value,
                "runs": len(stdout_list),
                "passed_keyword": passed_keyword,
                "success": success,
                "latency_seconds": duration,
            },
        )

    return stdout_list, stderr_list, compiled_path, success
