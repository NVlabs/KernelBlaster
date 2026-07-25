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

"""定义 LangGraph 节点共享状态及其序列化、恢复和差异比较逻辑。"""

from typing import Optional, TypedDict, Any, Dict
from pathlib import Path
import json

from ..config import GPUType


class GraphState(TypedDict):
    """描述各工作流节点共享且可持久化的任务状态字段。"""
    model: str  # 用于生成的模型
    gpu: GPUType  # 用于生成的 GPU 类型

    run_cuda: bool  # 是否运行CUDA生成代理
    run_cuda_perf: bool  # 是否运行性能代理
    run_cuda_bench: bool  # 是否运行CUDA基准测试代理
    run_cuda_perf_bench: bool  # 是否运行CUDA基准测试代理

    retry_failed: bool  # 是否重试失败的代理
    reference_code: str  # 参考代码
    user_message: str  # 用户留言
    folder: Path  # 保存生成代码的文件夹
    logger: Any  # 问题记录器
    
    # 强化学习优化参数
    rl_iterations: int  # 要运行的 RL 迭代次数
    rl_rollout_steps: int  # 每次 RL 迭代包含的 rollout 步骤数
    rl_buffer_size: int  # RL 重放缓冲区的大小
    rl_update_frequency: int  # 强化学习数据库更新的频率

    filepath: str  # 图表中最近生成的文件的文件名
    test_code_fp: Path  # 为 CUDA 生成的测试代码的路径
    cuda_fp: Path  # 生成的 CUDA 内核代码的路径
    cuda_bench_fp: Path  # 生成的 CUDA 基准代码的路径
    ncu_cuda_fp: Path  # 基于 NCU 分析的优化 CUDA 代码路径
    ncu_cuda_bench_fp: (
        Path  # 基于 NCU 分析和基准测试的优化 CUDA 代码路径
    )
    rl_ncu_cuda_fp: Path  # RL 优化的 CUDA 代码的路径
    run_outcome: Dict[str, Any]  # 序列化 RunOutcome 终端状态


def save_state_to_json(state: GraphState, output_path: str) -> None:
    """
    序列化 GraphState 并将其写入 JSON 文件，处理不可序列化的字段。

    参数：
    state：要序列化的 GraphState
    output_path：输出 JSON 文件的路径

    参数:
        state: 工作流节点读取并按约定更新的共享状态。
        output_path: 调用方提供的 `output_path` 参数。
    """
    # 创建状态字典的可序列化副本
    serializable_state: Dict[str, Any] = {}

    # 比较期间要忽略的字段（例如不可序列化的记录器）
    ignore_fields = {"logger", "shared_optimization_database"}

    for key, value in state.items():
        # 跳过记录器，因为它不可序列化
        if key in ignore_fields:
            continue

        # 将 Path 对象转换为字符串
        if isinstance(value, Path):
            serializable_state[key] = str(value.resolve())
        else:
            # 包括所有其他可序列化值
            serializable_state[key] = value

    # 写入 JSON 文件
    try:
        with open(output_path, "w") as f:
            json.dump(serializable_state, f, indent=2)
    except Exception as e:
        print(f"Error saving state to {output_path}: {e}")


def load_state_from_json(json_path: str, read_fp: bool = False) -> Dict[str, Any]:
    """
    从 JSON 文件加载状态字典并解析文件指针。

    以“_fp”结尾的字段被视为文件路径，其内容
    加载到没有“_fp”后缀的字段中。

    参数：
    json_path：JSON 文件的路径
    read_fp：如果为 True，则将文件内容读取到不带“_fp”后缀的字段中
    返回：
    包含文件内容的加载状态的字典

    例子：
    如果 JSON 包含 {"cuda_fp": "path/to/file.txt"}，
    结果将包括：
    {“cuda_fp”：“路径/到/file.txt”，“cuda”：“<文件内容>”}

    参数:
        json_path: 调用方提供的 `json_path` 参数。
        read_fp: 调用方提供的 `read_fp` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    try:
        # 加载 JSON 文件
        with open(json_path, "r") as f:
            state_dict = json.load(f)

        # 处理以 _fp 结尾的字段
        fp_fields = [key for key in state_dict.keys() if key.endswith("_fp")]

        if read_fp:
            for fp_field in fp_fields:
                file_path = state_dict[fp_field]
                content_field = fp_field[:-3]  # 删除“_fp”后缀

                # 如果路径为空或无则跳过
                if not file_path:
                    continue

                try:
                    # 尝试读取文件内容
                    with open(file_path, "r") as file:
                        state_dict[content_field] = file.read()
                except Exception as e:
                    print(f"Warning: Could not read file at {file_path}: {e}")
                    # 保留具有“无”值的字段以指示尝试加载但失败
                    state_dict[content_field] = None

        return state_dict

    except Exception as e:
        print(f"Error loading state from {json_path}: {e}")
        return {}


def compare_states(state1: Optional[GraphState], state2: Optional[GraphState]) -> bool:
    """
    比较两个 GraphState 字典。

    参数：
    state1：第一个 GraphState 字典
    state2：第二个GraphState字典

    返回：
    如果状态相等则为 True，否则为 False

    参数:
        state1: 调用方提供的 `state1` 参数。
        state2: 调用方提供的 `state2` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    # 处理无情况
    if state1 is None and state2 is None:
        return True
    if state1 is None or state2 is None:
        return False

    # 比较期间要忽略的字段（例如不可序列化的记录器）
    ignore_fields = {"logger"}

    # 比较除 ignore_fields 之外的所有字段
    for key in set(state1.keys()) | set(state2.keys()):
        # 跳过忽略的字段
        if key in ignore_fields:
            continue

        # 检查密钥在两种状态下是否都存在
        if key not in state1 or key not in state2:
            return False

        value1, value2 = state1[key], state2[key]

        # 通过转换为字符串进行比较来处理 Path 对象
        if isinstance(value1, Path) and isinstance(value2, Path):
            if str(value1.resolve()) != str(value2.resolve()):
                return False
        # 处理其中一个是路径而另一个不是的情况
        elif isinstance(value1, Path) or isinstance(value2, Path):
            return False
        # 直接比较其他值
        elif value1 != value2:
            return False

    return True
