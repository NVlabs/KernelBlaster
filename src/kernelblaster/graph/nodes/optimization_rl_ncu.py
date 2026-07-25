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

"""把工作流状态转换为 RL 优化任务，组织输入产物、Profiler 和最终结果。"""

from pathlib import Path
from ...agents import FeedbackConfig
from ...agents.opt_ncu_rl import RLNCUAgent
from ..state import GraphState, save_state_to_json
from ...outcomes import RunOutcome, RunStatus
from ...profiling import CudaEventsRunner, EventsProfilerBackend


async def optimization_rl_ncu(state: GraphState):
    """
    基于强化学习的 NCU 优化节点。
    该节点采用标准工作流程生成的CUDA内核
    并使用 RLNCUAgent 应用基于 RL 的优化。

    需要来自 data/kernelbench-cuda 的精选 CUDA 文件（driver.cpp 和 init.cu）。
    如果这些文件不可用，则会正常跳过该问题。

    参数:
        state: 工作流节点读取并按约定更新的共享状态。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    base_folder = Path(state["folder"])
    base_folder.mkdir(parents=True, exist_ok=True)
    
    # 优先从 data/kernelbench-cuda 加载整理后的输入产物，而不是依赖
    # 每次运行输出文件夹中预先存在的文件。
    # 我们从已经结构化的 run 文件夹中派生 <level>/<problem_name>
    # 就像.../level1/<problem_name>/。
    # 在容器中，该文件位于：
    # 容器内源码路径：/kernelblaster/src/kernelblaster/graph/nodes/optimization_rl_ncu.py
    # 所以parents[4] == /kernelblaster (repo root)。
    repo_root = Path(__file__).resolve().parents[4]
    curated_root = Path(state.get("kernelbench_cuda_root", repo_root / "data/kernelbench-cuda"))
    level = base_folder.parent.name
    problem_name = base_folder.name
    curated_dir = curated_root / level / problem_name

    curated_driver_cpp = curated_dir / "driver.cpp"
    curated_init_cu = curated_dir / "init.cu"

    # 后备：允许使用运行文件夹中已存在的文件。
    run_driver_cpp = base_folder / "driver.cpp"
    run_init_cu = base_folder / "init.cu"
    
    # 处理缺失的 cuda_fp - 需要策划或运行文件夹 init.cu
    cuda_fp = state.get("cuda_fp")
    if cuda_fp is None:
        if curated_init_cu.exists():
            cuda_fp = curated_init_cu
            state["logger"].info(f"Using curated init.cu from {curated_dir} as cuda_fp: {cuda_fp}")
        elif run_init_cu.exists():
            cuda_fp = run_init_cu
            state["logger"].info(f"Using run-folder init.cu as cuda_fp: {cuda_fp}")
        else:
            state["logger"].error(
                f"No cuda_fp available. Required files not found:\n"
                f"  - Curated: {curated_init_cu}\n"
                f"  - Run folder: {run_init_cu}\n"
                f"Skipping problem {problem_name} - curated CUDA files are required."
            )
            outcome = RunOutcome(
                status=RunStatus.FAILED,
                reason=f"Missing CUDA source for {problem_name}",
            )
            return {"rl_ncu_cuda_fp": None, "run_outcome": outcome.to_dict()}

    cuda_fp = Path(cuda_fp)
    
    # 处理缺失的 test_code_fp - 需要策划或运行文件夹 driver.cpp
    test_code_fp = state.get("test_code_fp")
    if test_code_fp is None:
        if curated_driver_cpp.exists():
            test_code_fp = curated_driver_cpp
            state["logger"].info(f"Using curated driver.cpp from {curated_dir} as test_code_fp: {test_code_fp}")
        elif run_driver_cpp.exists():
            test_code_fp = run_driver_cpp
            state["logger"].info(f"Using run-folder driver.cpp as test_code_fp: {test_code_fp}")
        else:
            state["logger"].error(
                f"No test_code_fp available. Required files not found:\n"
                f"  - Curated: {curated_driver_cpp}\n"
                f"  - Run folder: {run_driver_cpp}\n"
                f"Skipping problem {problem_name} - curated driver.cpp is required."
            )
            outcome = RunOutcome(
                status=RunStatus.FAILED,
                reason=f"Missing correctness driver for {problem_name}",
            )
            return {"rl_ncu_cuda_fp": None, "run_outcome": outcome.to_dict()}
    
    test_code_fp = Path(test_code_fp)
    
    save_state_to_json(state, base_folder / "state.json")

    # 创建 RL 代理配置
    fb_config = FeedbackConfig(
        agent_name="rl_ncu",
        base_folder=base_folder,
        logger=state["logger"],
        init_user_prompt="",  # 这将在 RLNCUAgent.initialize() 中设置
        model=state["model"],
        gpu=state["gpu"],
        test_code_fp=test_code_fp,
        retry_failed=state["retry_failed"],
        num_pgen=4,  # 强化学习代理使用更少的并行编码器，因为它更具战略性
    )
    
    # 从状态获取 RL 参数（从工作流配置传递）
    database_path = base_folder / "optimization_database.md"
    max_rollout_steps = state.get("rl_rollout_steps", 5)
    replay_buffer_size = state.get("rl_buffer_size", 100)
    update_frequency = state.get("rl_update_frequency", 3)
    rl_iterations = state.get("rl_iterations", 10)
    
    events_backend = EventsProfilerBackend(
        CudaEventsRunner(
            driver_path=test_code_fp,
            gpu=state["gpu"],
            logger=state["logger"],
            work_dir=base_folder / "events",
        )
    )

    agent_rl_ncu = RLNCUAgent(
        fb_config=fb_config,
        code_to_optimize_fp=cuda_fp,
        database_path=database_path,
        max_rollout_steps=max_rollout_steps,
        replay_buffer_size=replay_buffer_size,
        update_frequency=update_frequency,
        database=state.get("shared_optimization_database"),
        profiler_backend=events_backend,
    )
    
    # 初始化并运行 RL 优化
    await agent_rl_ncu.initialize()
    
    # 通过多次迭代运行 RL 优化
    state["logger"].info(f"Starting RL optimization with {rl_iterations} iterations")
    
    # 设置代理中的迭代次数
    agent_rl_ncu.num_rl_iterations = rl_iterations
    
    # 运行 RL 代理（它将在内部处理多次迭代）
    outcome = await agent_rl_ncu.run()
    
    if outcome.success:
        state["logger"].info(
            f"RL optimization completed successfully: {outcome.artifact_path}"
        )
    else:
        state["logger"].warning(
            f"RL optimization ended with {outcome.status.value}: {outcome.reason}"
        )

    # 保存最佳结果
    final_file = None
    if outcome.success:
        final_file = base_folder / "final_rl_cuda_perf.cu"
        final_file.write_text(outcome.artifact_path.read_text(), encoding="utf-8")
        state["logger"].info(f"RL optimization completed. Best result saved to {final_file}")
        outcome = RunOutcome(
            status=outcome.status,
            artifact_path=final_file,
            reason=outcome.reason,
            profiling_mode=outcome.profiling_mode,
            measurement=outcome.measurement,
            execution_status=outcome.execution_status,
            correctness_status=outcome.correctness_status,
            timing_status=outcome.timing_status,
            diagnostic_status=outcome.diagnostic_status,
            reason_code=outcome.reason_code,
            metrics=outcome.metrics,
        )

    save_state_to_json(
        {
            **state,
            "rl_ncu_cuda_fp": final_file,
            "run_outcome": outcome.to_dict(),
        },
        base_folder / "state.json",
    )

    return {"rl_ncu_cuda_fp": final_file, "run_outcome": outcome.to_dict()}
