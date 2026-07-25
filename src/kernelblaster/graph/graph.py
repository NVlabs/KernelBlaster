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

"""组装 LangGraph 状态图，并连接 KernelBlaster 的优化节点。"""

from langgraph.graph import StateGraph, START, END

from .nodes import optimization_rl_ncu
from .state import GraphState


def build_graph():
    """
    构建 `build_graph` 对应的领域操作，并返回调用方所需的标准化结果。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    graph_builder = StateGraph(GraphState)
    
    # 基线驱动的 RL 优化节点
    graph_builder.add_node("Baseline RL Optimization", optimization_rl_ncu)

    # 工作流程路由
    graph_builder.add_edge(START, "Baseline RL Optimization")

    # 基线优化结束工作流程
    graph_builder.add_edge("Baseline RL Optimization", END)

    return graph_builder.compile()
