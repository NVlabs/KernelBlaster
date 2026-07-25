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
"""
用于基于 LLM 的代码优化的强化学习代理。
实现PolicyEvaluation、PerfGapAnalysis 和ParameterUpdate 代理。
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from .database import OptimizationDatabase, OptimizationEntry, CompositeOptimization
from .utils import generate_code_retry, LLMResponse
from ..config import config
from ..measurements import Measurement, MeasurementComparisonError


@dataclass
class TrajectoryStep:
    """代表优化轨迹中的单个步骤。"""
    state: str
    action: str  # 优化技术
    code: str
    measurement: Measurement
    predicted_improvement: float
    actual_improvement: float
    reward: float

    @property
    def cycles(self) -> int | None:
        """Deprecated compatibility accessor; never relabel event time as cycles."""
        if self.measurement.unit.value != "cycles":
            return None
        return int(self.measurement.value)


@dataclass
class Trajectory:
    """代表一个完整的优化轨迹。"""
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    initial_measurement: Measurement | None = None
    final_measurement: Measurement | None = None
    
    def add_step(self, step: TrajectoryStep):
        """
        处理 `add_step` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            step: 调用方提供的 `step` 参数。
        """
        self.steps.append(step)
        self.total_reward += step.reward
        if len(self.steps) == 1:
            self.initial_measurement = step.measurement
        self.final_measurement = step.measurement

    @property
    def initial_cycles(self) -> int | None:
        if self.initial_measurement is None or self.initial_measurement.unit.value != "cycles":
            return None
        return int(self.initial_measurement.value)

    @property
    def final_cycles(self) -> int | None:
        if self.final_measurement is None or self.final_measurement.unit.value != "cycles":
            return None
        return int(self.final_measurement.value)


class ReplayBuffer:
    """存储政策学习的轨迹。"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化 ReplayBuffer 实例，并保存后续流程所需的配置与依赖。

        参数:
            max_size: 调用方提供的 `max_size` 参数。
        """
        self.max_size = max_size
        self.trajectories: List[Trajectory] = []
    
    def add_trajectory(self, trajectory: Trajectory):
        """
        将轨迹添加到缓冲区。

        参数:
            trajectory: 调用方提供的 `trajectory` 参数。
        """
        self.trajectories.append(trajectory)
        if len(self.trajectories) > self.max_size:
            # 删除最旧的轨迹
            self.trajectories.pop(0)
    
    def get_recent_trajectories(self, n: int = None) -> List[Trajectory]:
        """
        获取最近的n条轨迹。

        参数:
            n: 调用方提供的 `n` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        if n is None:
            return self.trajectories
        return self.trajectories[-n:]
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取有关缓冲区中轨迹的统计数据。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        if not self.trajectories:
            return {}
        
        rewards = [t.total_reward for t in self.trajectories]
        improvements = []
        for trajectory in self.trajectories:
            if trajectory.initial_measurement is None or trajectory.final_measurement is None:
                continue
            try:
                improvements.append((trajectory.final_measurement.speedup_over(
                    trajectory.initial_measurement
                ) - 1.0) * 100.0)
            except MeasurementComparisonError:
                continue
        
        return {
            'num_trajectories': len(self.trajectories),
            'avg_reward': statistics.mean(rewards),
            'std_reward': statistics.stdev(rewards) if len(rewards) > 1 else 0,
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'avg_improvement': statistics.mean(improvements) if improvements else 0,
            'success_rate': sum(1 for r in rewards if r > 0) / len(rewards)
        }


class PolicyEvaluationAgent:
    """通过比较预测结果与实际结果来评估策略绩效的代理。"""
    
    def __init__(self, model: str = None):
        """
        初始化 PolicyEvaluationAgent 实例，并保存后续流程所需的配置与依赖。

        参数:
            model: 生成候选时使用的模型标识。
        """
        self.model = model or config.MODEL
        self.system_prompt = """You are a performance analysis expert specializing in CUDA optimization evaluation.

Your task is to analyze the performance discrepancies between predicted and actual optimization results.

Given a set of optimization attempts with their predicted improvements, actual results, and profiling data, 
you should:

1. Identify patterns in prediction accuracy
2. Summarize key performance discrepancies 
3. Highlight successful optimization strategies
4. Note any systematic biases in predictions

Focus on actionable insights that can improve future optimization predictions."""

    async def evaluate_policy(self, replay_buffer: ReplayBuffer, database: OptimizationDatabase) -> str:
        """
        用自然语言评估政策绩效和回报分析。

        参数:
            replay_buffer: 调用方提供的 `replay_buffer` 参数。
            database: 保存历史状态与优化经验的共享数据库。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        
        recent_trajectories = replay_buffer.get_recent_trajectories(10)  # 分析最近 10 条轨迹
        if not recent_trajectories:
            return "No trajectories available for evaluation."
        
        # 收集性能数据
        performance_data = []
        for traj in recent_trajectories:
            for step in traj.steps:
                performance_data.append({
                    'state': step.state,
                    'technique': step.action,
                    'predicted_improvement': step.predicted_improvement,
                    'actual_improvement': step.actual_improvement,
                    'measurement': step.measurement.to_dict(),
                    'reward': step.reward
                })
        
        # 创建评估提示
        prompt = f"""Analyze the following optimization performance data:

RECENT OPTIMIZATION ATTEMPTS:
{json.dumps(performance_data, indent=2)}

BUFFER STATISTICS:
{json.dumps(replay_buffer.get_statistics(), indent=2)}

DATABASE STATISTICS:
{json.dumps(database.get_database_stats(), indent=2)}

Please provide a concise analysis focusing on:
1. Overall prediction accuracy trends
2. Which optimization techniques are over/under-performing
3. Patterns in successful vs failed optimizations
4. Recommendations for improving the optimization strategy database

Keep your response focused and actionable."""

        try:
            # 导入记录器以进行正确的记录
            from loguru import logger
            response = await generate_code_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                logger=logger,
                max_retries=3
            )
            return response.generations[0]
        except Exception as e:
            return f"Error in policy evaluation: {str(e)}"


class PerfGapAnalysisAgent:
    """分析性能差距并确定预测与现实不同的原因的代理。"""
    
    def __init__(self, model: str = None):
        """
        初始化 PerfGapAnalysisAgent 实例，并保存后续流程所需的配置与依赖。

        参数:
            model: 生成候选时使用的模型标识。
        """
        self.model = model or config.MODEL
        self.system_prompt = """You are a CUDA performance analysis expert specializing in understanding optimization failures and successes.

Your task is to analyze why optimization predictions differed from actual results and identify the root causes.

When analyzing performance gaps, consider:
1. Hardware-specific factors (memory bandwidth, compute units, cache behavior)
2. Code characteristics (memory access patterns, control flow, data dependencies)
3. Optimization technique assumptions vs reality
4. Profiling metric interpretation accuracy

Provide specific, technical insights about why certain optimizations succeeded or failed."""

    async def analyze_performance_gaps(self, evaluation_result: str, recent_failures: List[TrajectoryStep]) -> str:
        """
        分析绩效差距并提供有关预测错误的见解。

        参数:
            evaluation_result: 调用方提供的 `evaluation_result` 参数。
            recent_failures: 调用方提供的 `recent_failures` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        
        failure_analysis = []
        for step in recent_failures:
            gap = step.predicted_improvement - step.actual_improvement
            failure_analysis.append({
                'state': step.state,
                'technique': step.action,
                'predicted_improvement': step.predicted_improvement,
                'actual_improvement': step.actual_improvement,
                'performance_gap': gap,
                'measurement': step.measurement.to_dict(),
            })
        
        prompt = f"""POLICY EVALUATION RESULTS:
{evaluation_result}

DETAILED FAILURE ANALYSIS:
{json.dumps(failure_analysis, indent=2)}

Based on the policy evaluation and specific failure cases, analyze:

1. ROOT CAUSES: Why did these optimizations fail to meet predictions?
   - Were the assumptions about bottlenecks incorrect?
   - Did hardware characteristics differ from expectations?
   - Were there unexpected interactions between optimizations?

2. SYSTEMATIC ISSUES: Are there patterns in the prediction errors?
   - Which types of optimizations consistently over/under-perform?
   - Which code states are hardest to analyze correctly?

3. SPECIFIC CORRECTIONS: What specific changes should be made to improve predictions?
   - Adjustment factors for certain optimization types
   - New metrics to consider for state classification
   - Refined prediction models for specific scenarios

Provide concrete, actionable recommendations for database improvements."""

        try:
            # 导入记录器以进行正确的记录
            from loguru import logger
            response = await generate_code_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                logger=logger,
                max_retries=3
            )
            return response.generations[0]
        except Exception as e:
            return f"Error in performance gap analysis: {str(e)}"


class ParameterUpdateAgent:
    """根据分析结果更新优化数据库的代理。"""
    
    def __init__(self, model: str = None):
        """
        初始化 ParameterUpdateAgent 实例，并保存后续流程所需的配置与依赖。

        参数:
            model: 生成候选时使用的模型标识。
        """
        self.model = model or config.MODEL
        self.system_prompt = """You are a creative database management expert for CUDA optimization strategies.

Your task is to update the optimization strategy database based on performance analysis results, and creatively discover new optimization approaches.

You should:
1. Adjust predicted performance values based on actual results
2. Update confidence scores for existing optimizations
3. Add new optimization strategies discovered through analysis
4. Create composite optimizations that combine multiple techniques in specific orders
5. Design parameter-tuned versions of existing optimizations
6. Identify new performance states that don't fit existing categories
7. Remove or deprecate consistently poor-performing strategies

CREATIVE CAPABILITIES:
- Combine 2-3 optimization techniques in specific orders
- Fine-tune parameters like unrolling factors, tile sizes, block dimensions
- Discover new performance states from unusual metric patterns
- Create adaptive optimizations that change based on problem characteristics
- Identify side effects and trade-offs between optimization techniques

Output your recommendations as structured JSON updates that can be applied to the database."""

    async def update_parameters(self, gap_analysis: str, database: OptimizationDatabase) -> Dict[str, Any]:
        """
        根据差距分析更新数据库参数。

        参数:
            gap_analysis: 调用方提供的 `gap_analysis` 参数。
            database: 保存历史状态与优化经验的共享数据库。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        
        current_stats = database.get_database_stats()
        
        # 优化效果不佳
        poor_performers = []
        for state, optimizations in database.optimization_strategies.items():
            for opt in optimizations:
                if (opt.actual_improvement is not None and 
                    opt.predicted_improvement > 0 and
                    opt.actual_improvement < opt.predicted_improvement * 0.5):
                    poor_performers.append({
                        'state': state,
                        'technique': opt.technique,
                        'predicted': opt.predicted_improvement,
                        'actual': opt.actual_improvement,
                        'confidence': opt.confidence_score
                    })
        
        prompt = f"""PERFORMANCE GAP ANALYSIS:
{gap_analysis}

CURRENT DATABASE STATISTICS:
{json.dumps(current_stats, indent=2)}

POOR PERFORMING OPTIMIZATIONS:
{json.dumps(poor_performers, indent=2)}

Based on this analysis, provide comprehensive database update recommendations in the following JSON format:

{{
  "prediction_adjustments": [
    {{
      "state": "state_name",
      "technique1": "technique_name_1",
      "technique2": "technique_name_2", 
      "technique3": "technique_name_3",
      "order_of_techniques": ["1. technique_name_1", "2. technique_name_2", "3. technique_name_3"],
      "parameters_to_fine_tune": {{
        "unrolling_factor": 4,
        "tile_size": 32,
        "block_size": 256
      }},
      "new_predicted_improvement": 35.0,
      "reason": "explanation for this composite optimization",
      "side_effects": "potential negative effects or trade-offs to be aware of"
    }}
  ],
  "confidence_updates": [
    {{
      "state": "state_name",
      "technique": "technique_name",
      "new_confidence": 0.8,
      "reason": "explanation for confidence change"
    }}
  ],
  "new_optimizations": [
    {{
      "state": "state_name",
      "technique": "new_technique_name",
      "predicted_improvement": 20.0,
      "reason": "why this optimization should be added"
    }}
  ],
  "parameter_tuned_optimizations": [
    {{
      "base_technique": "loop_unrolling",
      "parameters": {{
        "unrolling_factor": 8,
        "vectorization": true
      }},
      "predicted_improvement": 25.0,
      "reason": "explanation for parameter choice",
      "applicable_states": ["compute_bound", "latency_bound"]
    }}
  ],
  "discovered_states": [
    {{
      "state_name": "new_state_pattern",
      "description": "description of performance characteristics",
      "characteristics": "metric ranges that define this state",
      "initial_optimizations": [
        {{
          "technique": "suggested_technique",
          "predicted_improvement": 30.0
        }}
      ]
    }}
  ],
  "deprecated_optimizations": [
    {{
      "state": "state_name", 
      "technique": "technique_name",
      "reason": "why this should be deprecated"
    }}
  ]
}}

CREATIVE THINKING GUIDELINES:
1. Look for patterns where combining techniques might yield better results
2. Consider parameter fine-tuning for techniques that partially worked
3. Identify new performance states from unusual metric combinations
4. Think about side effects and trade-offs between optimizations
5. Suggest adaptive approaches that change based on problem characteristics

Focus on innovative, data-driven updates that will improve future optimization performance."""

        try:
            # 导入记录器以进行正确的记录
            from loguru import logger
            response = await generate_code_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                logger=logger,
                max_retries=3
            )
            
            # 解析 JSON 响应
            import re
            json_match = re.search(r'\{.*\}', response.generations[0], re.DOTALL)
            if json_match:
                updates = json.loads(json_match.group())
                
                # 将更新应用到数据库
                self._apply_database_updates(database, updates)
                return updates
            else:
                return {"error": "Could not parse update recommendations"}
                
        except Exception as e:
            return {"error": f"Error in parameter update: {str(e)}"}
    
    def _apply_database_updates(self, database: OptimizationDatabase, updates: Dict[str, Any]):
        """
        将建议的更新应用到数据库。

        参数:
            database: 保存历史状态与优化经验的共享数据库。
            updates: 调用方提供的 `updates` 参数。
        """
        
        # 应用复合优化预测（创建新的复合优化）
        for adj in updates.get("prediction_adjustments", []):
            if "technique1" in adj:  # 这是一个复合优化
                composite = CompositeOptimization(
                    state=adj["state"],
                    technique1=adj["technique1"],
                    technique2=adj.get("technique2"),
                    technique3=adj.get("technique3"),
                    order_of_techniques=adj.get("order_of_techniques", []),
                    parameters_to_fine_tune=adj.get("parameters_to_fine_tune", {}),
                    predicted_improvement=adj["new_predicted_improvement"],
                    reason=adj.get("reason", ""),
                    side_effects=adj.get("side_effects", "")
                )
                database.add_composite_optimization(composite)
            else:  # 传统单一技术调整
                state = adj["state"]
                technique = adj.get("technique", "")
                new_prediction = adj["new_predicted_improvement"]
                
                optimizations = database.get_optimizations_for_state(state)
                for opt in optimizations:
                    if opt.technique == technique:
                        opt.predicted_improvement = new_prediction
                        opt.last_updated = datetime.now().isoformat()
                        break
        
        # 应用置信度更新
        for conf in updates.get("confidence_updates", []):
            state = conf["state"]
            technique = conf["technique"]
            new_confidence = conf["new_confidence"]
            
            optimizations = database.get_optimizations_for_state(state)
            for opt in optimizations:
                if opt.technique == technique:
                    opt.confidence_score = new_confidence
                    opt.last_updated = datetime.now().isoformat()
                    break
        
        # 添加新的优化
        for new_opt in updates.get("new_optimizations", []):
            database.add_new_optimization(
                new_opt["state"],
                new_opt["technique"], 
                new_opt["predicted_improvement"]
            )
        
        # 添加参数调整优化
        for param_opt in updates.get("parameter_tuned_optimizations", []):
            new_technique = database.create_parameter_tuned_optimization(
                param_opt["base_technique"],
                param_opt["parameters"],
                param_opt["predicted_improvement"],
                param_opt.get("reason", "")
            )
            
            # 添加到所有适用的状态
            for state in param_opt.get("applicable_states", []):
                database.add_new_optimization(
                    state,
                    new_technique,
                    param_opt["predicted_improvement"]
                )
        
        # 添加发现的状态
        for discovered in updates.get("discovered_states", []):
            state_name = discovered["state_name"]
            database.discovered_states[state_name] = {
                "description": discovered.get("description", ""),
                "characteristics": discovered.get("characteristics", ""),
                "discovery_context": "AI-discovered state"
            }
            
            # 为新状态添加初始优化
            for opt in discovered.get("initial_optimizations", []):
                database.add_new_optimization(
                    state_name,
                    opt["technique"],
                    opt["predicted_improvement"]
                )
        
        # 标记已弃用的优化（显着降低置信度）
        for dep in updates.get("deprecated_optimizations", []):
            state = dep["state"]
            technique = dep["technique"]
            
            optimizations = database.get_optimizations_for_state(state)
            for opt in optimizations:
                if opt.technique == technique:
                    opt.confidence_score = 0.1  # 信心极低
                    opt.last_updated = datetime.now().isoformat()
                    break
