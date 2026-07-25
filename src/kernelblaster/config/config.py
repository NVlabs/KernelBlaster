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

"""集中定义服务地址、实验开关和优化工作流的可配置参数。"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from urllib.parse import urlsplit, urlunsplit

from .gpu_config import GPUType
from typing import Any

load_dotenv()


def _public_url(value: str) -> str:
    """
    处理 `public_url` 所表示的内部步骤；该函数不属于稳定的公开接口。

    参数:
    value: 需要转换、保存或校验的值。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path.rstrip("/"), "", ""))


class ExperimentalFeatures:
    """封装 `ExperimentalFeatures` 对应的领域状态与操作。"""
    OPT_RL_NCU = os.getenv("KERNELBLASTER_OPT_RL_NCU", "0") == "1"

    @staticmethod
    def dict() -> dict[str, bool | int]:
        # 返回实验特征及其值的字典
        """
        处理 `dict` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return {
            k: v
            for k, v in ExperimentalFeatures.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


class SystemConfig:
    # 可插入的远程 LLM 提供商。秘密是故意只针对环境的。
    """封装相关组件的配置项和默认策略。"""
    LLM_PROVIDER = os.getenv(
        "KERNELBLASTER_LLM_PROVIDER", "openai_compatible"
    ).strip().lower()
    LLM_BASE_URL = os.getenv(
        "KERNELBLASTER_LLM_BASE_URL", "https://api.openai.com/v1"
    ).strip()
    LLM_BASE_URL_PUBLIC = _public_url(LLM_BASE_URL)
    API_KEY = os.getenv("KERNELBLASTER_LLM_API_KEY") or os.getenv(
        "OPENAI_API_KEY", ""
    )
    MODEL = os.getenv("MODEL", "gpt-4.1-20250414")
    STREAM = os.getenv("STREAM", "False")
    LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
    LLM_REQUEST_TIMEOUT_SECONDS = float(
        os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "1800")
    )
    LLM_MAX_REQUESTS = int(os.getenv("LLM_MAX_REQUESTS", "0"))
    LLM_MAX_TOTAL_TOKENS = int(os.getenv("LLM_MAX_TOTAL_TOKENS", "0"))
    LLM_MAX_COMPLETION_TOKENS = int(
        os.getenv("LLM_MAX_COMPLETION_TOKENS", "12288")
    )
    LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
    LLM_LOG_CONTENT = os.getenv("LLM_LOG_CONTENT", "false").lower() in (
        "true",
        "1",
        "yes",
        "y",
        "on",
    )
    # Tokens are deliberately separate audiences. Neither may be generated at
    # import time: a service must fail closed when its deployment secret is
    # missing rather than silently accepting a process-local random token.
    CONTROL_TOKEN = os.getenv("KERNELBLASTER_CONTROL_TOKEN", "").strip()
    WORKER_TOKEN = os.getenv("KERNELBLASTER_WORKER_TOKEN", "").strip()
    MAX_GPU_BINARY_BYTES = int(
        os.getenv("KERNELBLASTER_MAX_GPU_BINARY_BYTES", str(256 * 1024 * 1024))
    )
    MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 300))
    NUM_PARALLEL_GENERATIONS_PER_ATTEMPT = int(
        os.getenv("NUM_PARALLEL_GENERATIONS_PER_ATTEMPT", 4)
    )

    # 服务器网址
    COMPILE_SERVER_URL = os.getenv("COMPILE_SERVER_URL", None)
    GPU_SERVER_URLS = {
        gpu: os.getenv(f"GPU_SERVER_URL_{gpu.value.upper()}", None) for gpu in GPUType
    }

    assert (
        COMPILE_SERVER_URL is None or "http" in COMPILE_SERVER_URL
    ), "COMPILE_SERVER_URL must be unset or start with http"
    for __gpu, __url in GPU_SERVER_URLS.items():
        assert (
            __url is None or "http" in __url
        ), f"GPU_SERVER_URL_{__gpu.value.upper()} must be unset or start with http but got {__url}"

    LLM_GATEWAY_CLIENT_ID = os.getenv("LLM_GATEWAY_CLIENT_ID", None)
    LLM_GATEWAY_CLIENT_SECRET = os.getenv("LLM_GATEWAY_CLIENT_SECRET", None)
    LLM_GATEWAY_BASE_URL = os.getenv("LLM_GATEWAY_BASE_URL", None)
    LLM_GATEWAY_TOKEN_URL = os.getenv(
        "LLM_GATEWAY_TOKEN_URL",
        f"{LLM_GATEWAY_BASE_URL}/token" if LLM_GATEWAY_BASE_URL else None
    )

    NVCF_FUNCTION_ID = os.getenv("NVCF_FUNCTION_ID", None)
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

    TEMP_DIRECTORY = "/tmp/kernelblaster"
    EOS_BASE_URL = os.getenv("EOS_BASE_URL", None)

    # NCU 代理必须超过原始 cuda-c 基线的加速数
    # 在它被认为是成功之前。
    OPT_NCU_MIN_IMPROVEMENTS = int(os.getenv("KERNELBLASTER_NCU_MIN_IMPROVEMENTS", "3"))

    EXPERIMENTAL_FEATURES = ExperimentalFeatures()

    @staticmethod
    def set_compile_server_url(url: str):
        """
        设置 `set_compile_server_url` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        url: 目标服务或资源的 URL。
        """
        assert (
            SystemConfig.COMPILE_SERVER_URL is None
        ), "COMPILE_SERVER_URL is already set"
        SystemConfig.COMPILE_SERVER_URL = url

    @staticmethod
    def set_gpu_server_url(gpu: GPUType, url: str):
        """
        设置 `set_gpu_server_url` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        gpu: 执行或分析任务使用的 GPU 配置。
        url: 目标服务或资源的 URL。
        """
        if (
            gpu not in SystemConfig.GPU_SERVER_URLS
            or SystemConfig.GPU_SERVER_URLS[gpu] is None
        ):
            SystemConfig.GPU_SERVER_URLS[gpu] = url
        else:
            logger.warning(
                f"GPU_SERVER_URL_{gpu.value.upper()} is already set to {SystemConfig.GPU_SERVER_URLS[gpu]}"
            )

    @staticmethod
    def get_gpu_server_url(gpu: GPUType) -> str:
        """
        获取 `get_gpu_server_url` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        gpu: 执行或分析任务使用的 GPU 配置。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        assert (
            gpu in SystemConfig.GPU_SERVER_URLS
        ), f"GPU_SERVER_URL_{gpu.value.upper()} is not set"
        return SystemConfig.GPU_SERVER_URLS[gpu]

    @staticmethod
    def get_all_gpu_server_urls() -> dict[GPUType, str]:
        """
        获取 `get_all_gpu_server_urls` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return {
            gpu: url
            for gpu, url in SystemConfig.GPU_SERVER_URLS.items()
            if url is not None
        }

    @staticmethod
    def CUSTOM_LOGGER_FORMAT(record):
        """
        处理 `CUSTOM_LOGGER_FORMAT` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        record: 调用方提供的 `record` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        attempt_id = record["extra"].get("attempt_id")
        if attempt_id is None:
            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<green>{extra[agent_name]}</green> - <level>{message}</level> "
                "\n{exception}"
            )
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<green>{extra[agent_name]}</green> | "
            "<green>Attempt {extra[attempt_id]} Task {extra[task_id]}</green> - <level>{message}</level>"
            "\n{exception}"
        )

    @classmethod
    def print_config(cls, logger):
        """
        输出 `print_config` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        logger: 记录诊断信息和任务进度的日志器。
        """
        gpu_servers_strs = "\n".join(
            [
                f"    - {gpu.value.upper()}: {url}"
                for gpu, url in cls.GPU_SERVER_URLS.items()
            ]
        )
        """Print the various hyperparameters to the logger."""
        config_str = f"""
- Using {cls.MODEL} for generation
- LLM provider: {cls.LLM_PROVIDER}
- LLM endpoint: {cls.LLM_BASE_URL_PUBLIC}
- LLM API key configured: {bool(cls.API_KEY)}
- LLM max concurrency: {cls.LLM_MAX_CONCURRENCY}
- LLM max retries: {cls.LLM_MAX_RETRIES}
- LLM request budget: {cls.LLM_MAX_REQUESTS or 'unlimited'}
- LLM token budget: {cls.LLM_MAX_TOTAL_TOKENS or 'unlimited'}
- LLM max completion tokens: {cls.LLM_MAX_COMPLETION_TOKENS}
- LLM reasoning effort: {cls.LLM_REASONING_EFFORT or 'provider default'}
- Parallel generations per attempt: {cls.NUM_PARALLEL_GENERATIONS_PER_ATTEMPT}
- Maximum attempts: {cls.MAX_ATTEMPTS}
- Compiler server: {cls.COMPILE_SERVER_URL}
- GPU servers:
{cls.get_all_gpu_server_urls()}
- Experimental features: {cls.EXPERIMENTAL_FEATURES.dict()}
"""
        logger.info(f"Config:\n{config_str}")


config = SystemConfig()


@dataclass
class WorkflowConfig:
    """封装相关组件的配置项和默认策略。"""
    model: str
    run_cuda: bool
    run_cuda_perf: bool
    run_cuda_bench: bool
    run_cuda_perf_bench: bool
    retry_failed: bool
    gpu: GPUType
    # 强化学习优化参数
    rl_iterations: int = 10
    rl_rollout_steps: int = 5
    rl_buffer_size: int = 100
    rl_update_frequency: int = 3
    use_baseline_optimization: bool = True
    # 可选的共享 OptimizationDatabase 实例（未序列化）
    shared_optimization_database: Any = None

    def dict(self):
        # 不要使用 dataclasses.asdict()：它会先深度复制值
        # 被删除，并且共享数据库有意拥有 RLock。
        """
        处理 `dict` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return {
            name: value
            for name, value in vars(self).items()
            if name != "shared_optimization_database"
        }

    def should_skip_folder(self, folder: Path):
        """
        检查我们是否应该根据现有文件跳过处理此问题

        参数:
        folder: 保存当前任务中间状态和最终产物的目录。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """

        if not folder.exists():
            return False

        files = os.listdir(folder)

        def should_skip(name: str, folder_key: str, enabled_field: str):
            # 这些文件由各 Agent 自行创建。
            """
            处理 `should_skip` 对应的领域操作，并返回调用方所需的标准化结果。

            参数:
            name: 目标对象或资源的名称。
            folder_key: 调用方提供的 `folder_key` 参数。
            enabled_field: 调用方提供的 `enabled_field` 参数。

            返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
            """
            agent_enabled = getattr(self, enabled_field)
            return (
                not agent_enabled
                or f"final_{name}.cu" in files
                or (
                    f"failed_{name}" in files
                    and (Path(folder) / folder_key / ".finished").exists()
                    and not self.retry_failed
                )
            )

        return should_skip("rl_cuda_perf", "rl_ncu", "run_cuda_perf")


    def validate(self):
        """校验 `validate` 对应的领域操作，并返回调用方所需的标准化结果。"""
        assert isinstance(
            self.gpu, GPUType
        ), f"Please ensure gpu is a GPUType: got {type(self.gpu)}"
