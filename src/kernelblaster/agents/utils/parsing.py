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

"""从 CUDA 源码和 NCU 日志中解析 Kernel 名称、启动信息与周期指标。"""

from pathlib import Path
import re

from ...config import GPUType

# 允许在测试期间使用猴子补丁命令
from . import commands as commands

from ...config import GPUType

__all__ = [
    "find_kernel_names_ncu",
    "find_kernel_names",
    "find_kernel_name",
    "get_elapsed_cycles_ncu_log",
    "find_kernel_launch_header",
]


async def find_kernel_names_ncu(
    executable: Path, source_path: Path, gpu: GPUType, timeout: int
) -> list[str]:
    """
    通过在给定的可执行文件上运行 NCU 并将其与源代码进行比较来查找内核名称。

    参数:
        executable: 调用方提供的 `executable` 参数。
        source_path: 调用方提供的 `source_path` 参数。
        gpu: 执行或分析任务使用的 GPU 配置。
        timeout: 允许操作等待的最长秒数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
    """

    INVALID_KERNEL_NAME = "invalid_magic_kernel_name_here"

    kernels_in_source = find_kernel_names(source_path)

    if not kernels_in_source:
        raise RuntimeError(f"No kernels found in the source code:\n{source_path}")

    # 在可执行文件上运行 ncu
    # 应该在日志中打印可用内核的列表，如下所示：
    # ==PROF== 连接到进程 2138337 (/tmp/kernelagent/compile_env/build/main)
    # ==PROF== 与进程 2138337 断开连接
    # ==警告== 没有分析内核。
    # 可用内核：
    # Kernel 名称示例 1：distribution_elementwise_grid_stride_kernel
    # 2. 内核
    # Kernel 名称示例 3：matmul_fp16_kernel_8x8
    # Kernel 名称示例 4：reduce_kernel
    # Kernel 名称示例 5：vectorized_elementwise_kernel
    stdout, stderr = await commands.run_gpu_executable(
        executable,
        gpu,
        timeout,
        job_name=str(executable),
        prefix_command=f"NVIDIA_TF32_OVERRIDE=0 ncu -k {INVALID_KERNEL_NAME}",
    )

    # 解析标准输出以获取内核名称
    kernel_section = stdout.split("Available Kernels:")[1]
    ncu_kernel_names = re.findall(r"\s*\d+\.\s*(\w+)", kernel_section)
    if not ncu_kernel_names:
        raise RuntimeError(
            f"Failed to find NCU kernel names in:\n stdout: {stdout}\n stderr: {stderr}"
        )

    # 找到内核名称的交集
    kernel_names = list(set(ncu_kernel_names) & set(kernels_in_source))

    # 检查内核名称是否在源代码中
    if not kernel_names:
        raise RuntimeError(
            f"Failed to find kernels running in both the executable and the source code:\n Source code kernels: {kernels_in_source}\n NCU kernels: {ncu_kernel_names}"
        )

    return kernel_names


def find_kernel_names(filename: Path) -> str:
    """
    从给定的 cuda 文件中查找内核名称。

    参数：
    filename：内核代码的文件名。

    返回：
    内核名称。

    参数:
        filename: 调用方提供的 `filename` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    kernel_code = filename.read_text()
    # 内核也可以从 cuda 函数启动，所以
    # 尝试从内核声明中解析内核名称
    kernel_names_launches = re.findall(r"__global__ void (\S+)\(", kernel_code)
    kernel_names_decls = re.findall(r"(\w+)(?:<[^>]*>)?\s*<<<", kernel_code)

    kernel_names = list(set(kernel_names_launches) | set(kernel_names_decls))

    # 过滤掉前缀为 __launch_bounds__ 的名称，因为
    # 这些是启动定义而不是名称
    kernel_names = list(
        filter(lambda x: not x.startswith("__launch_bounds__"), kernel_names)
    )

    if len(kernel_names) == 0:
        raise RuntimeError(
            f"Failed to find kernel name in:\n{kernel_code}\n Please define the kernel with __global__ void kernel_name()"
        )
    return kernel_names


def find_kernel_name(filename: Path) -> str:
    """
    从给定的 cuda 文件中查找唯一的内核名称。

    参数：
    filename：内核代码的文件名。

    返回：
    唯一的内核名称。

    参数:
        filename: 调用方提供的 `filename` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    kernel_names = find_kernel_names(filename)
    if len(kernel_names) > 1:
        raise RuntimeError(
            f"Found multiple kernel names in:\n{filename.read_text()}\n Please generate only one kernel in the output."
        )
    return kernel_names[0]


def get_elapsed_cycles_ncu_log(ncu_log: str) -> int:
    """
    从给定的 ncu 日志中获取经过的周期。

    解析“GPU 光吞吐量速度”部分中的“已用周期”指标。
    支持表格格式和CSV格式。

    参数：
    ncu_log：ncu 日志。

    返回：
    已过去的周期。

    参数:
        ncu_log: 调用方提供的 `ncu_log` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    # 尝试多种模式来匹配不同的 NCU 输出格式
    patterns = [
        # 表格式：“Elapsed Cycles 周期 12675”
        r"Elapsed Cycles\s+\w+\s+(\d[\d,]*)",
        # CSV 格式或其他格式：“Elapsed Cycles,cycle,12675”或“Elapsed Cycles: 12675”
        r"Elapsed Cycles[,\s:]+(?:cycle[,\s]+)?(\d[\d,]*)",
        # 后备：任何带有“已用周期”后跟数字的格式
        r"Elapsed Cycles.*?(\d[\d,]*)",
    ]
    
    for pattern in patterns:
        elapsed_cycles = re.search(pattern, ncu_log, re.IGNORECASE | re.MULTILINE)
        if elapsed_cycles:
            try:
                return int(elapsed_cycles.group(1).replace(",", ""))
            except ValueError:
                continue
    
    raise RuntimeError(f"Failed to find elapsed cycles in NCU log. Patterns tried: {len(patterns)}")


def find_kernel_launch_header(code: str) -> str:
    """
    在给定代码中查找内核启动标头。

    参数：
    代码：代码。

    返回：
    内核启动标头。

    参数:
        code: 待处理的源码文本。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        RuntimeError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    launch_headers = re.findall(
        r"(void launch_gpu_implementation\(.*?\);)", code, flags=re.DOTALL
    )
    if len(launch_headers) == 0:
        raise RuntimeError(f"Failed to find kernel launch header in:\n{code}")
    if len(launch_headers) > 1:
        raise RuntimeError(
            f"Found multiple kernel launch headers in:\n{code}\n Please generate only one kernel in the output."
        )
    return launch_headers[0]
