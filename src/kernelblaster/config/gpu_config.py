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

"""规范化 GPU 名称，并将运行设备映射为受支持的 GPU 枚举。"""

from enum import Enum
import subprocess


# python <3.11中不包含StrEnum类，因此我们在这里定义它
class StrEnum(str, Enum):
    """封装 `StrEnum` 对应的领域状态与操作。"""
    pass


_current_gpu = None

_SM_MAP = {
    # 如果修改此图请修改test_gpu_config.py
    "a100": "sm_80",
    "a6000": "sm_86",
    "rtx3080": "sm_86",
    "l40": "sm_89",
    "l40s": "sm_89",
    "l40g": "sm_89",
    "rtx4090": "sm_89",
    "rtx5000": "sm_89",
    "rtx6000": "sm_89",
    "h100": "sm_90",
    "h200": "sm_90",
    "b200": "sm_100",
}


class GPUType(StrEnum):
    """封装 `GPUType` 对应的领域状态与操作。"""
    A100 = "a100"
    A6000 = "a6000"
    RTX3080 = "rtx3080"
    L40 = "l40"
    L40S = "l40s"
    L40G = "l40g"
    RTX4090 = "rtx4090"
    H100 = "h100"
    H200 = "h200"
    B200 = "b200"
    RTX5000 = "rtx5000"
    RTX6000 = "rtx6000"

    @property
    def sm(self):
        """
        处理 `sm` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        assert self.value in _SM_MAP, f"Unknown GPU type: {self.value}"
        return _SM_MAP[self.value]

    @staticmethod
    def current():
        """
        返回当前的 GPU 类型。
        对此进行缓存以避免重复调用 nvidia-smi。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        global _current_gpu
        if _current_gpu is None:
            name = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=gpu_name",
                        "--format=csv,noheader",
                    ]
                )
                .decode("utf-8")
                .strip()
            )
            name = name.replace(" ", "").lower()
            _current_gpu = _parse_gpu_name(name)
        return _current_gpu


def _parse_gpu_name(nvidia_smi_name: str) -> GPUType:
    """
    从 nvidia-smi 输出中解析 GPU 类型。

    参数:
    nvidia_smi_name: 调用方提供的 `nvidia_smi_name` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
    ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    nvidia_smi_name = nvidia_smi_name.replace(" ", "").lower()

    # 按长度降序对 GPU 类型进行排序，以匹配最长的可能名称。
    # 这涵盖了 GPU 名称是不同 GPU 名称（例如 l40 和 l40s）的子字符串的情况。
    avail_types = sorted(
        [gpu.value for gpu in GPUType], key=lambda x: len(x), reverse=True
    )
    for gpu in avail_types:
        if gpu.lower() in nvidia_smi_name:
            return GPUType(gpu)
    raise ValueError(f"Unknown GPU type: {nvidia_smi_name}")
