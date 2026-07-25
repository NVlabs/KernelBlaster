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
增强的数据库实用程序，用于通过 LLM 支持的定性状态分析进行 GPU 优化。

该模块实现了两个LLM代理系统：
1. State Summarizer Agent：定性分析 NCU 报告
2. 状态匹配器代理：将当前状态与已知优化模式进行匹配
"""
from __future__ import annotations
from pathlib import Path
import os
import shutil
import re
import json
import itertools  # 添加以支持使用 itertools.chain 的回退逻辑
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading

from ..measurements import Measurement

def get_elapsed_cycles_v2(text: str) -> int:
    """
    获取 `get_elapsed_cycles_v2` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        text: 调用方提供的 `text` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    groups = re.search(r"Elapsed Cycles: (\d+)", text)
    if groups is None:
        raise ValueError("No elapsed cycles found in text")
    return int(groups.group(1))

def get_speedup_from_files(soln_file: Path) -> Tuple[int, int, float]:
    """
    获取 `get_speedup_from_files` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        soln_file: 调用方提供的 `soln_file` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    final_text = soln_file.read_text()
    if (soln_file.parent /"ncu/0_init_ncu_log.txt").exists():
        initial_text = (soln_file.parent /"ncu/0_init_ncu_log.txt").read_text()
    else:
        initial_text = (soln_file.parent /"ncu_annot/init.cu").read_text()
    final_elapsed_cycles = get_elapsed_cycles_v2(final_text)
    initial_elapsed_cycles = get_elapsed_cycles_v2(initial_text)
    speedup_ratio = initial_elapsed_cycles / final_elapsed_cycles
    return initial_elapsed_cycles, final_elapsed_cycles, speedup_ratio

class LLMInterface:
    """状态分析中使用的 LLM 查询接口。"""
    
    def __init__(self, model_name: str = None, logger = None):
        """
        初始化 LLMInterface 实例，并保存后续流程所需的配置与依赖。

        参数:
            model_name: 调用方提供的 `model_name` 参数。
            logger: 记录诊断信息和任务进度的日志器。
        """
        from ..config import config
        self.model_name = model_name or config.MODEL
        self.logger = logger
    
    async def query(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """
        向 LLM 发送查询并返回响应。

        参数:
            prompt: 调用方提供的 `prompt` 参数。
            max_tokens: 调用方提供的 `max_tokens` 参数。
            temperature: 调用方提供的 `temperature` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        try:
            from .utils import generate_code_retry
        except ImportError:
            if self.logger:
                self.logger.error("Failed to import generate_code_retry from utils")
            return "Error: Could not import LLM utilities"
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await generate_code_retry(
                messages, 
                self.model_name, 
                self.logger, 
                n_tasks=1,
                max_retries=3
            )
            return response.generations[0] if response.generations else ""
        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM query failed: {e}")
            return f"Error: {str(e)}"
    
    def query_sync(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """
        LLM 查询的同步包装器。

        参数:
            prompt: 调用方提供的 `prompt` 参数。
            max_tokens: 调用方提供的 `max_tokens` 参数。
            temperature: 调用方提供的 `temperature` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._mock_response()
            else:
                return loop.run_until_complete(self.query(prompt, max_tokens, temperature))
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Async query failed, using mock response: {e}")
            return self._mock_response()
    
    def _mock_response(self) -> str:
        """
        当 LLM 不可用时的后备模拟响应。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return """
        PRIMARY_BOTTLENECK: memory_bound
        SECONDARY_CHARACTERISTICS:
        - Inefficient memory access patterns
        - Low cache utilization
        - Moderate occupancy
        PERFORMANCE_SIGNATURE: Memory-intensive workload with room for optimization
        """
    
    def is_available(self) -> bool:
        """
        检查LLM服务是否可用。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 这应该是一个轻量级的可用性检查。我们刻意回避
        # 在这里进行网络调用；我们只检查配置的凭据。
        try:
            from ..config import config
        except Exception:
            config = None  # type: ignore

        import os

        # 首选显式配置值（如果存在）。
        if config is not None and bool(getattr(config, "API_KEY", None)):
            return True

        # 在我们支持的 LLM 后端中使用的常见环境变量。
        # 注意：使其与 utils/query.py 中的客户端选择逻辑保持同步。
        if os.getenv("OAI_ATLAS_KEY") or os.getenv("OPENAI_API_KEY"):
            return True
        if os.getenv("NIM_KEY") or os.getenv("CHIPNEMO_KEY") or os.getenv("NGC_KEY"):
            return True
        if os.getenv("AZURE_ENDPOINT") and os.getenv("AZURE_KEY"):
            return True
        if os.getenv("EOS_BASE_URL"):
            return True

        # LLM 网关式凭证（如果使用）。
        if os.getenv("LLM_GATEWAY_URL") and (os.getenv("LLM_GATEWAY_KEY") or os.getenv("LLM_GATEWAY_TOKEN")):
            return True

        return False


@dataclass
class StateProfile:
    """具有主要和次要特征的定性状态概况。"""
    state_name: str
    primary_bottleneck: str  # 支持的状态标签：memory_bound、compute_bound、latency_bound、hybrid_bound。
    secondary_characteristics: List[str]
    performance_signature: str
    context_description: str
    relative_patterns: Dict[str, str]  # 定性模式而不是数值

@dataclass
class OptimizationEntry:
    """封装 `OptimizationEntry` 对应的领域状态与操作。"""
    technique: str
    predicted_improvement: Optional[float] = None
    description: str = ""
    category: str = ""  # 内存、计算、延迟等
    actual_improvement: Optional[float] = None
    confidence_score: float = 0.5
    last_updated: Optional[str] = None
    usage_count: int = 0
    # 加速跟踪字段
    predicted_speedup: float = 1.0  # 预期加速（比率）
    actual_speedup: Optional[float] = None  # 最近的加速测量
    initial_elapsed_cycles: Optional[int] = None  # 基线经过周期
    initial_measurement: Optional[dict[str, Any]] = None
    last_measurement: Optional[dict[str, Any]] = None


@dataclass
class CompositeOptimization:
    """代表多种技术的复合优化。"""
    state: str
    technique1: str
    technique2: Optional[str] = None
    technique3: Optional[str] = None
    order_of_techniques: List[str] = field(default_factory=list)
    parameters_to_fine_tune: Dict[str, Any] = field(default_factory=dict)
    predicted_improvement: float = 0.0
    actual_improvement: Optional[float] = None
    reason: str = ""
    side_effects: str = ""
    confidence_score: float = 0.5
    last_updated: Optional[str] = None
    usage_count: int = 0
    
    def get_composite_id(self) -> str:
        """
        为此复合优化生成唯一的 ID。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        techniques = [t for t in [self.technique1, self.technique2, self.technique3] if t]
        params_str = "_".join(f"{k}_{v}" for k, v in self.parameters_to_fine_tune.items())
        return f"composite_{'+'.join(techniques)}_{params_str}"


class GPUOptimizationDatabase:
    """维护 GPU 性能状态与历史优化经验，并使用 LLM 辅助状态分析和策略检索。"""
    
    def __init__(
        self,
        optimization_db_path: Path,
        gpu_report_path: Path | None,
        llm_interface=None,
    ):
        """
        初始化 GPUOptimizationDatabase 实例，并保存后续流程所需的配置与依赖。

        参数:
            optimization_db_path: 调用方提供的 `optimization_db_path` 参数。
            gpu_report_path: 调用方提供的 `gpu_report_path` 参数。
            llm_interface: 调用方提供的 `llm_interface` 参数。
        """
        import os
        self.optimization_db_path = optimization_db_path
        self.optimization_db_header_path = optimization_db_path.with_name(f"{optimization_db_path.stem}_header{optimization_db_path.suffix}") 
        self.optimization_db_footer_path = optimization_db_path.with_name(f"{optimization_db_path.stem}_footer{optimization_db_path.suffix}") 
        self.gpu_report_path = gpu_report_path
        self.llm_interface = llm_interface or LLMInterface()

        # 在启动时记录一次环境驱动的行为，以便从 run.log 轻松审核运行。
        # 注意：此标志仅影响数据库*后备*选择器，而不影响LLM计划选择。
        try:
            raw_val = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", None)
            parsed_val = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in (
                "1",
                "true",
                "True",
                "yes",
                "YES",
                "y",
                "on",
                "ON",
            )
            msg = f"KERNELAGENT_DB_FALLBACK_TOP1={raw_val!r} (parsed={parsed_val})"
            if hasattr(self.llm_interface, "logger") and self.llm_interface.logger:
                self.llm_interface.logger.info(msg)
            else:
                print(msg)
        except Exception:
            # 切勿让环境日志记录破坏数据库初始化。
            pass

        # ---------- LLM交互记录----------
        # 提示和输出将附加到该文件中，以便我们可以
        # 检查数据库级代理的推理。
        self._llm_log_fp: Path = self.optimization_db_path.parent / "database_llm_log.txt"
        # 捕获对优化数据库所做的每个更改的日志文件
        # （e.g。更新测量的改进、新添加的技术等）。
        self._db_change_log_fp: Path = self.optimization_db_path.parent / "database_change_log.txt"
        # 存储数据库实时 JSON 快照的路径。
        self._persist_json_fp: Path = self.optimization_db_path.with_suffix(".json")

        # 确保 LLM 日志文件存在，以便用户可以可靠地找到它，即使
        # 运行最终采取确定性后备路径。
        try:
            self._llm_log_fp.parent.mkdir(parents=True, exist_ok=True)
            with open(self._llm_log_fp, "a", encoding="utf-8"):
                pass
        except Exception:
            pass

        # 跨共享此实例的任务序列化并发写入（嵌套写入的可重入）
        self._io_lock: threading.RLock = threading.RLock()

        # 数据库结构
        self.known_states: Dict[str, StateProfile] = {}
        self.optimization_strategies: Dict[str, Dict[str, Any]] = {} # 更改为 Dict[str, Dict[str, Any]]
        self.composite_optimizations: Dict[str, List[CompositeOptimization]] = {}
        self.discovered_states: Dict[str, Dict[str, Any]] = {}  # 跟踪人工智能发现的状态
        # 按状态名称键入 LLM 建议的优化的缓存，以便调用者可以
        # 直接检索它们，无需运行额外的选择逻辑。
        # self._llm_recommended_optimizations: 字典[str, OptimizationEntry |复合优化] = {}
        self._llm_recommended_optimizations: Dict[str, OptimizationEntry] = {}
        
        # 加载综合优化知识
        self.gpu_optimization_knowledge = ""
        self.load_databases()

        # 使用 _persist_database 从加载的 llm 数据库创建初始 json
        self._persist_database()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 打印(self.optimization_strategies)
        print(f"Persisted database to {self._persist_json_fp}")

        # 退出(0)

    # ------------------------------------------------------------------
    # Helper：保留 LLM 提示/响应对以进行调试
    # ------------------------------------------------------------------
    def _log_llm_interaction(self, label: str, prompt: str, response: str):
        """
        将带标签的提示/响应对附加到共享日志文件。

        参数:
            label: 调用方提供的 `label` 参数。
            prompt: 调用方提供的 `prompt` 参数。
            response: 需要解析或规范化的服务响应。
        """
        try:
            with self._io_lock:
                with open(self._llm_log_fp, "a", encoding="utf-8") as f:
                    f.write(f"=== {label} | {datetime.now().isoformat()} ===\n")
                    f.write("--- PROMPT ---\n")
                    f.write(prompt.strip() + "\n")
                    f.write("--- RESPONSE ---\n")
                    f.write((response or "<empty response>").strip() + "\n\n")
        except Exception as e:
            # 日志记录失败决不应该使优化过程崩溃
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.warning(f"Failed to write LLM log: {e}")

    # ------------------------------------------------------------------
    # Helper：将结构更改持久保存到优化数据库
    # ------------------------------------------------------------------
    def _log_db_change(self, action: str, details: Any):
        """
        将*操作* 的记录与*详细信息* 一起写入更改日志。

        参数:
            action: 调用方提供的 `action` 参数。
            details: 调用方提供的 `details` 参数。
        """
        # 记录文件写入，包含文件路径
        # 写入记录器
        if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
            self.llm_interface.logger.info(f"=== {action} | {datetime.now().isoformat()} ===\n")
        try:
            if not isinstance(details, str):
                import json as _json
                try:
                    details = _json.dumps(details, indent=2)
                except Exception:
                    details = str(details)

            with self._io_lock:
                with open(self._db_change_log_fp, "a", encoding="utf-8") as f:
                    f.write(f"=== {action} | {datetime.now().isoformat()} ===\n")
                    f.write(details.strip() + "\n\n")

            # 日志记录后，保留完整的数据库快照，以便我们
            # 始终拥有最新的机器可读版本。
            self._persist_database()
        except Exception as e:
            # 永远不要在日志记录方面失败——如果可能的话，只发出警告。
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.warning(f"Failed to write DB change log: {e}")

    # ------------------------------------------------------------------
    # Helper：将整个优化数据库持久保存为 JSON
    # ------------------------------------------------------------------
    def _persist_database(self):
        """将当前内存数据库状态转储到*self._persist_json_fp*。"""

        print(f"_persist_database: Persisting database to {self._persist_json_fp}")
        database_logger = getattr(self.llm_interface, "logger", None)
        if database_logger is not None:
            database_logger.info(
                f"_persist_database: Persisting database to {self._persist_json_fp}"
            )
        try:
            import json as _json
            from dataclasses import asdict as _asdict

            # 将优化策略转换为字典列表
            optimization_strategies = {}
            for k, v in self.optimization_strategies.items():
                # 规范化 secondary_characteristics - 它可能是来自 markdown 解析的字符串
                secondary_chars = v.get("secondary_characteristics", [])
                if isinstance(secondary_chars, str):
                    # 如果是字符串，请尝试将其解析为逗号分隔的列表
                    secondary_chars = [s.strip() for s in secondary_chars.split(",") if s.strip()]
                
                optimization_strategies[k] = {
                    "optimizations": [_asdict(o) for o in v.get("optimizations", [])],
                    "primary_bottleneck": v.get("primary_bottleneck", ""),
                    "secondary_characteristics": secondary_chars if isinstance(secondary_chars, list) else [],
                    # 可选兼容字段示例："performance_signature": v["performance_signature"],
                    # 可选兼容字段示例："context_description": v["context_description"]
                }

            data = {
                "schema_version": "3.0",
                "known_states": {k: _asdict(v) for k, v in self.known_states.items()},
                "optimization_strategies": optimization_strategies,
                "composite_optimizations": {
                    k: [_asdict(o) for o in v] for k, v in self.composite_optimizations.items()
                },
                "discovered_states": self.discovered_states,
            }

            # 保护快照写入
            if hasattr(self, "_io_lock"):
                with self._io_lock:
                    temporary = self._persist_json_fp.with_suffix(
                        self._persist_json_fp.suffix + ".tmp"
                    )
                    with open(temporary, "w", encoding="utf-8") as fp:
                        _json.dump(data, fp, indent=2)
                        fp.flush()
                        os.fsync(fp.fileno())
                    os.replace(temporary, self._persist_json_fp)
            else:
                temporary = self._persist_json_fp.with_suffix(
                    self._persist_json_fp.suffix + ".tmp"
                )
                with open(temporary, "w", encoding="utf-8") as fp:
                    _json.dump(data, fp, indent=2)
                    fp.flush()
                    os.fsync(fp.fileno())
                os.replace(temporary, self._persist_json_fp)
        except Exception as e:
            # 软失败 – 我们仅在持久性失败时发出警告。
            print(f"_persist_database: Failed to persist database JSON: {e}")
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.warning(f"Failed to persist database JSON: {e}")
    
    def load_databases(self):
        """加载优化数据库和GPU优化报告。"""
        print(f"Loading databases from {self.optimization_db_path} and {self.gpu_report_path}")
        # 加载 GPU 优化报告作为综合知识库
        if self.gpu_report_path is not None and self.gpu_report_path.exists():
            self.gpu_optimization_knowledge = self.gpu_report_path.read_text()
            print(f"Loaded GPU optimization report: {len(self.gpu_optimization_knowledge)} characters")
        elif self.gpu_report_path is not None:
            print(f"Warning: GPU optimization report not found at {self.gpu_report_path}")
        
        # 获取 data/kernelblaster 中的默认位置以进行后备
        # Repo 根目录是项目根目录 (e.g./path/to/KernelBlaster)
        repo_root = Path(__file__).resolve().parents[3]
        default_json_path = repo_root / "data" / "kernelblaster" / "optimization_database.json"
        # 默认页眉/页脚与 JSON 模板一起存在
        default_header_path = repo_root / "data" / "kernelblaster" / "optimization_database_header.md"
        default_footer_path = repo_root / "data" / "kernelblaster" / "optimization_database_footer.md"
        
        # 加载当前优化数据库
        # 优先级：1）在输出目录中保留 JSON，2）在输出目录中进行降价，
        # 3) 默认 JSON（并初始化持久副本），4) 默认 markdown
        loaded = False
        if self._persist_json_fp.exists():
            print(f"Loading database from persisted JSON: {self._persist_json_fp}")
            self._regenerate_database_from_json()
            loaded = True
        elif self.optimization_db_path.exists():
            print(f"Loading database from markdown: {self.optimization_db_path}")
            self._parse_optimization_database()
            loaded = True
        elif default_json_path.exists():
            # 通过从默认模板复制来初始化持久化 JSON
            print(f"Initializing database JSON from default location: {default_json_path}")
            try:
                self._persist_json_fp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(default_json_path, self._persist_json_fp)
                print(f"Copied default database JSON to {self._persist_json_fp}")
                # 如果页眉/页脚还不存在，还初始化它们
                if default_header_path.exists() and not self.optimization_db_header_path.exists():
                    shutil.copy2(default_header_path, self.optimization_db_header_path)
                    print(f"Copied default header markdown to {self.optimization_db_header_path}")
                if default_footer_path.exists() and not self.optimization_db_footer_path.exists():
                    shutil.copy2(default_footer_path, self.optimization_db_footer_path)
                    print(f"Copied default footer markdown to {self.optimization_db_footer_path}")
            except Exception as e:
                print(f"Failed to copy default optimization database assets: {e}")
            # 现在从（新创建的）持久化 JSON 加载
            if self._persist_json_fp.exists():
                print(f"Loading database from persisted JSON: {self._persist_json_fp}")
                self._regenerate_database_from_json()
                loaded = True
        
        if not loaded:
            print(f"Warning: No optimization database found. Starting with empty database.")
            print(f"  Searched paths:")
            print(f"    - {self._persist_json_fp}")
            print(f"    - {self.optimization_db_path}")
            print(f"    - {default_json_path}")
            print(f"    - {default_header_path}")
            print(f"    - {default_footer_path}")
        else:
            # 日志数据库统计
            num_states = len(self.optimization_strategies)
            total_optimizations = sum(
                len(state_data.get("optimizations", []))
                for state_data in self.optimization_strategies.values()
            )
            print(f"Database loaded: {num_states} states, {total_optimizations} optimizations")
        
        # 从 GPU 优化报告中提取已知状态
        self._extract_states_from_gpu_report()
    
    def _regenerate_database_from_json(self):
        """从 JSON 文件重新生成数据库。"""
        try:
            import json as _json
            # 1) 加载持久化的 JSON 快照
            if not self._persist_json_fp.exists():
                print(f"_regenerate_database_from_json: JSON snapshot not found at {self._persist_json_fp}")
                return

            data = _json.loads(self._persist_json_fp.read_text(encoding="utf-8"))

            # 2) 从 JSON 重建内存结构
            self.known_states = {}
            for k, v in data.get("known_states", {}).items():
                try:
                    self.known_states[k] = StateProfile(**v)
                except Exception:
                    # 对架构漂移具有鲁棒性
                    self.known_states[k] = StateProfile(
                        state_name=v.get("state_name", k),
                        primary_bottleneck=v.get("primary_bottleneck", "unknown_bound"),
                        secondary_characteristics=v.get("secondary_characteristics", []),
                        performance_signature=v.get("performance_signature", ""),
                        context_description=v.get("context_description", ""),
                        relative_patterns=v.get("relative_patterns", {}),
                    )

            # 优化策略
            self.optimization_strategies = {}
            for state_name, state_data in data.get("optimization_strategies", {}).items():
                optim_dicts = state_data.get("optimizations", [])
                optim_entries: List[OptimizationEntry] = []
                for od in optim_dicts:
                    try:
                        optim_entries.append(OptimizationEntry(**od))
                    except Exception:
                        # 最小兼容结构
                        optim_entries.append(
                            OptimizationEntry(
                                technique=od.get("technique", "unknown"),
                                predicted_improvement=(
                                    float(od.get("predicted_improvement"))
                                    if od.get("predicted_improvement") not in (None, "")
                                    else None
                                ),
                                description=od.get("description", ""),
                                category=od.get("category", "general"),
                                actual_improvement=od.get("actual_improvement"),
                                confidence_score=float(od.get("confidence_score", 0.5) or 0.5),
                                last_updated=od.get("last_updated"),
                                usage_count=int(od.get("usage_count", 0) or 0),
                            )
                        )

                self.optimization_strategies[state_name] = {
                    "optimizations": optim_entries,
                    "primary_bottleneck": state_data.get("primary_bottleneck", ""),
                    "secondary_characteristics": state_data.get("secondary_characteristics", []),
                }

            # 复合优化
            self.composite_optimizations = {}
            for state_name, comp_list in data.get("composite_optimizations", {}).items():
                comps: List[CompositeOptimization] = []
                for cd in comp_list:
                    try:
                        comps.append(CompositeOptimization(**cd))
                    except Exception:
                        comps.append(
                            CompositeOptimization(
                                state=cd.get("state", state_name),
                                technique1=cd.get("technique1", ""),
                                technique2=cd.get("technique2"),
                                technique3=cd.get("technique3"),
                                order_of_techniques=cd.get("order_of_techniques", []),
                                parameters_to_fine_tune=cd.get("parameters_to_fine_tune", {}),
                                predicted_improvement=float(cd.get("predicted_improvement", 0.0) or 0.0),
                                actual_improvement=cd.get("actual_improvement"),
                                reason=cd.get("reason", ""),
                                side_effects=cd.get("side_effects", ""),
                                confidence_score=float(cd.get("confidence_score", 0.5)),
                                last_updated=cd.get("last_updated"),
                                usage_count=int(cd.get("usage_count", 0)),
                            )
                        )
                self.composite_optimizations[state_name] = comps

            # 发现的状态元数据
            self.discovered_states = data.get("discovered_states", {})

            # 3）根据当前内存数据编写完整的markdown并将其写入
            final_markdown = self.get_database_md_text(include_header_footer=True)
            self.optimization_db_path.write_text(final_markdown, encoding="utf-8")
            print(f"_regenerate_database_from_json: Regenerated markdown at {self.optimization_db_path}")
        except Exception as e:
            print(f"_regenerate_database_from_json: Failed to regenerate from JSON: {e}")

    def _build_states_markdown(self) -> str:
        """
        仅从内存结构构建 markdown 中的状态部分。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        def _fmt_chars(chars: Any) -> str:
            """
            处理 `fmt_chars` 所表示的内部步骤；该函数不属于稳定的公开接口。

            参数:
                chars: 调用方提供的 `chars` 参数。

            返回:
                当前操作产生的结果；具体类型由返回注解和调用约定确定。
            """
            if isinstance(chars, list):
                return ", ".join(str(c) for c in chars)
            return str(chars) if chars is not None else ""

        def _fmt_impr(val: Any) -> str:
            """
            处理 `fmt_impr` 所表示的内部步骤；该函数不属于稳定的公开接口。

            参数:
                val: 调用方提供的 `val` 参数。

            返回:
                当前操作产生的结果；具体类型由返回注解和调用约定确定。
            """
            if val is None:
                return "0"
            try:
                return f"{float(val):g}"
            except Exception:
                return "0"

        state_sections: List[str] = []
        for state_name, state_data in self.optimization_strategies.items():
            primary_bottleneck = state_data.get("primary_bottleneck", "")
            secondary_chars = _fmt_chars(state_data.get("secondary_characteristics", []))
            lines: List[str] = []
            lines.append(f"#### State: {state_name}")
            if secondary_chars:
                lines.append(f"**Characteristics**: {secondary_chars}")
            if primary_bottleneck:
                lines.append(f"**Primary Bottleneck**: {primary_bottleneck}")
            lines.append("**Optimizations**:")

            opts: List[OptimizationEntry] = state_data.get("optimizations", [])
            if opts:
                for opt in opts:
                    desc = f" - {opt.description}" if getattr(opt, "description", "") else ""
                    predicted_speedup = getattr(opt, "predicted_speedup", None)
                    if predicted_speedup in (None, 0.0):
                        # 从 predicted_improvement 百分比得出加速作为后备
                        pred_impr = (getattr(opt, "predicted_improvement", 0.0) or 0.0)
                        predicted_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
                    lines.append(
                        f"- **{opt.technique}**: {predicted_speedup:.2f}x predicted speedup{desc}"
                    )
            else:
                lines.append("- (no optimizations available)")

            state_sections.append("\n".join(lines))

        return ("\n\n".join(state_sections).rstrip() + "\n") if state_sections else ""

    def get_database_footer_text(self) -> str:
        """
        返回数据库的页脚文本。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return self.optimization_db_footer_path.read_text(encoding="utf-8") if self.optimization_db_footer_path.exists() else ""

    def get_database_md_text(self, include_header_footer: bool = True) -> str:
        """
        返回完整的数据库 Markdown 文本而不写入磁盘。

        当 include_header_footer 为 True 时，原始页眉和页脚文件
        包含在再生状态降价周围（如果存在）。

        参数:
            include_header_footer: 调用方提供的 `include_header_footer` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        states_md = self._build_states_markdown()
        if not include_header_footer:
            return states_md

        header_text = (
            self.optimization_db_header_path.read_text(encoding="utf-8")
            if self.optimization_db_header_path.exists()
            else ""
        )
        footer_text = (
            self.optimization_db_footer_path.read_text(encoding="utf-8")
            if self.optimization_db_footer_path.exists()
            else ""
        )

        parts: List[str] = []
        if header_text.strip():
            parts.append(header_text.rstrip())
        parts.append(states_md)
        if footer_text.strip():
            parts.append(footer_text.rstrip())

        return "\n\n".join(parts).rstrip() + "\n"
    
    def _parse_optimization_database(self):
        """解析现有的优化数据库。"""
        content = self.optimization_db_path.read_text()
        current_state = None
        print(f"Parsing optimization database from {self.optimization_db_path}")
        
        # 解析 JSON 部分以进行复合优化
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
                self._load_composite_optimizations(json_data)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON section: {e}")
        
        # 解析基本优化
        for line in content.split('\n'):
            line = line.strip()
            
            state_match = re.match(r'#### State: (.+)', line)
            if state_match:
                current_state = state_match.group(1).strip()
                if current_state not in self.optimization_strategies:
                    self.optimization_strategies[current_state] = {"optimizations": []}
                continue
            
            # 扩展正则表达式：捕获改进图后的可选描述。
            # 示例行：
            # - **memory_compute_overlap**：0% 性能提升 - 管道内存和计算操作
            opt_match = re.match(
                r'- \*\*(.+?)\*\*: (\d+(?:\.\d+)?)% performance improvement(?:\s*-\s*(.+))?',
                line,
            )

            # 如果行以 **Characteristics** 开头：
            if line.startswith("**Characteristics**:"):
                current_state_characteristics = line.split(":", 1)[1].strip()
                self.optimization_strategies[current_state]["secondary_characteristics"] = current_state_characteristics
            # 如果行以 **Primary Bottleneck** 开头：
            if line.startswith("**Primary Bottleneck**:"):
                current_state_primary_bottleneck = line.split(":", 1)[1].strip()
                self.optimization_strategies[current_state]["primary_bottleneck"] = current_state_primary_bottleneck


            if opt_match and current_state:
                technique = opt_match.group(1).strip()
                improvement = float(opt_match.group(2))
                description = opt_match.group(3).strip() if opt_match.group(3) else ""
                
                entry = OptimizationEntry(
                    technique=technique,
                    predicted_improvement=improvement,
                    description=description,
                    category=self._categorize_technique(technique),
                )
                self.optimization_strategies[current_state]["optimizations"].append(entry)
                # print(f"为 {current_state} 添加了优化策略：{technique} 并改进了 {improvement}")
    
    def _extract_states_from_gpu_report(self):
        """从综合 GPU 优化报告中提取状态模式。"""
        # 这会根据 GPU 优化报告创建定性状态配置文件
        # 基于决策树结构
        
        memory_bound_profile = StateProfile(
            state_name="memory_bandwidth_limited",
            primary_bottleneck="memory_bound",
            secondary_characteristics=[
                "High memory throughput utilization",
                "Bandwidth saturation",
                "Potential coalescing issues",
                "Cache inefficiencies"
            ],
            performance_signature="Memory bandwidth is the primary limiting factor with potential for access pattern optimization",
            context_description="Workload is limited by memory bandwidth, showing signs of inefficient access patterns or cache behavior",
            relative_patterns={
                "memory_pressure": "high",
                "compute_utilization": "underutilized", 
                "access_patterns": "potentially_uncoalesced",
                "cache_behavior": "suboptimal"
            }
        )
        
        compute_bound_profile = StateProfile(
            state_name="compute_throughput_limited",
            primary_bottleneck="compute_bound",
            secondary_characteristics=[
                "High compute unit utilization",
                "Instruction throughput bottleneck",
                "Potential for specialized units",
                "Arithmetic intensity"
            ],
            performance_signature="Compute units are saturated, indicating opportunity for algorithmic or instruction-level optimization",
            context_description="Workload is compute-intensive with potential for specialized hardware utilization or algorithmic improvements",
            relative_patterns={
                "compute_pressure": "high",
                "memory_utilization": "adequate",
                "instruction_mix": "potentially_suboptimal",
                "parallelism": "high"
            }
        )
        
        latency_bound_profile = StateProfile(
            state_name="latency_occupancy_limited", 
            primary_bottleneck="latency_bound",
            secondary_characteristics=[
                "Low occupancy",
                "Insufficient parallelism",
                "Resource underutilization",
                "Synchronization overhead"
            ],
            performance_signature="Neither memory nor compute are saturated, indicating latency hiding or occupancy issues",
            context_description="Workload has insufficient parallelism or resource conflicts limiting occupancy and latency hiding",
            relative_patterns={
                "occupancy": "low",
                "parallelism": "insufficient",
                "resource_conflicts": "present",
                "latency_hiding": "poor"
            }
        )
        
        # 存储已知的状态配置文件
        self.known_states = {
            "memory_bandwidth_limited": memory_bound_profile,
            "compute_throughput_limited": compute_bound_profile,
            "latency_occupancy_limited": latency_bound_profile
        }
    
    async def analyze_performance_state(self, ncu_report: str, metrics: dict, code_implementation: str, elapsed_cycles: Optional[int] = None) -> StateProfile:
        """
        LLM Agent 1：状态总结器
        分析 NCU 报告并提取定性性能特征。

        参数:
            ncu_report: 调用方提供的 `ncu_report` 参数。
            metrics: 性能分析或正确性检查产生的指标集合。
            code_implementation: 调用方提供的 `code_implementation` 参数。
            elapsed_cycles: 调用方提供的 `elapsed_cycles` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        
        # 如果 ncu_report 为空但我们有循环，则构建一个最小报告
        # 仅当循环 > 0 时才显示循环（0 通常表示解析失败）
        if not ncu_report.strip() and elapsed_cycles is not None and elapsed_cycles > 0:
            ncu_report = f"""Elapsed Cycles: {elapsed_cycles:,}

Note: This is a cycles-only profiling mode. Detailed NCU metrics are not available.
Use the code implementation and elapsed cycles to infer performance characteristics."""
        elif not ncu_report.strip():
            # 如果周期为 0 或 None，表示分析数据不可用
            ncu_report = """Note: Cycles-only profiling mode is enabled, but elapsed cycles were not successfully parsed from the program output.
Detailed NCU metrics are not available. Please analyze based on the code implementation alone."""

        state_analysis_prompt = f"""
You are a GPU performance analysis expert. Analyze this NVIDIA NSight Compute (NCU) profiling report and provide a qualitative summary of the kernel's performance state.

CODE IMPLEMENTATION:
```cpp
{code_implementation}
```

NCU REPORT:
{ncu_report}  

Provide your analysis in this EXACT format:

PERFORMANCE_SIGNATURE: [2-3 sentence summary of what is limiting performance and the overall execution pattern]

RELATIVE_PATTERNS:
- memory_pressure: [very_low|low|moderate|high|very_high]
- compute_utilization: [very_low|low|moderate|high|very_high] 
- access_patterns: [excellent|good|moderate|poor|very_poor]
- cache_efficiency: [excellent|good|moderate|poor|very_poor]
- occupancy_level: [very_low|low|moderate|high|very_high]
- parallelism_utilization: [very_low|low|moderate|high|very_high]
- specialied_hw_usage: [very_low|low|moderate|high|very_high]
- [List 3-5 key secondary performance characteristics]
- [Focus on patterns you observe in the data]
- [Include cache behavior, memory access patterns, occupancy]
- [Note any resource conflicts or inefficiencies]

PRIMARY_BOTTLENECK: [memory_bound|compute_bound|latency_bound|hybrid_bound]

//code signiture: loop pattern /branches(in summary/generic)


CONTEXT_DESCRIPTION: [Brief description of the workload characteristics and optimization opportunities]

Focus on qualitative patterns and relationships rather than specific numbers. Look for the underlying performance characteristics that drive behavior.
"""
        
        if self.llm_interface and self.llm_interface.is_available():
            try:
                analysis = await self.llm_interface.query(state_analysis_prompt, max_tokens=800, temperature=0.1)
                # 记录提示/响应以提高透明度
                self._log_llm_interaction("StateAnalysis", state_analysis_prompt, analysis)
                return self._parse_state_analysis(analysis)
            except Exception as e:
                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                    self.llm_interface.logger.warning(f"State analysis failed: {e}")
                return self._fallback_state_analysis(ncu_report, metrics)
        else:
            return self._fallback_state_analysis(ncu_report, metrics)
    
    def _parse_state_analysis(self, llm_response: str) -> StateProfile:
        """
        将 LLM 状态分析响应解析为 StateProfile。

        参数:
            llm_response: 调用方提供的 `llm_response` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        lines = llm_response.split('\n')
        
        primary_bottleneck = "unknown_bound"
        secondary_characteristics = []
        performance_signature = ""
        context_description = ""
        relative_patterns = {}
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("PRIMARY_BOTTLENECK:"):
                primary_bottleneck = line.split(":", 1)[1].strip()
            elif line.startswith("SECONDARY_CHARACTERISTICS:"):
                current_section = "secondary"
            elif line.startswith("PERFORMANCE_SIGNATURE:"):
                current_section = "signature"
                performance_signature = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.startswith("RELATIVE_PATTERNS:"):
                current_section = "patterns"
            elif line.startswith("CONTEXT_DESCRIPTION:"):
                current_section = "context"
                context_description = line.split(":", 1)[1].strip() if ":" in line else ""
            elif current_section == "secondary" and line.startswith("-"):
                secondary_characteristics.append(line[1:].strip())
            elif current_section == "signature" and line and not line.startswith(("RELATIVE", "CONTEXT")):
                performance_signature += " " + line
            elif current_section == "patterns" and ":" in line:
                key, value = line.split(":", 1)
                relative_patterns[key.strip().replace("- ", "")] = value.strip()
            elif current_section == "context" and line and not line.startswith(("PRIMARY", "SECONDARY")):
                context_description += " " + line
        
        return StateProfile(
            state_name="current_analysis",
            primary_bottleneck=primary_bottleneck,
            secondary_characteristics=secondary_characteristics,
            performance_signature=performance_signature.strip(),
            context_description=context_description.strip(),
            relative_patterns=relative_patterns
        )
    
    async def match_state_against_database(self, current_state: StateProfile) -> str:
        """
        LLM Agent 2：状态匹配器
        将当前状态与已知优化模式进行定性比较。

        参数:
            current_state: 调用方提供的 `current_state` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        
        # 准备已知状态进行比较
        known_states_text = ""
        # 从 optimisation_strategies 构建文本（已删除已弃用的 known_states）
        for state_name, state_data in self.optimization_strategies.items():
            primary_bottleneck = state_data.get("primary_bottleneck", "")
            secondary_chars = state_data.get("secondary_characteristics", "")
            optimization_list = ""
            opts = state_data.get("optimizations", [])[:3]
            if opts:
                optimization_list = "\n".join(
                    [
                        f"  - {opt.technique}: {((getattr(opt, 'predicted_speedup', None) if getattr(opt, 'predicted_speedup', None) not in (None, 0.0) else (1.0 / max(1e-6, 1.0 - ((getattr(opt, 'predicted_improvement', 0.0) or 0.0)/100.0))))):.2f}x speedup"
                        for opt in opts
                    ]
                )

            known_states_text += f"""
STATE: {state_name}
Primary Bottleneck: {primary_bottleneck}
Secondary Characteristics: {secondary_chars}
Available Optimizations:
{optimization_list}

"""
        
        matching_prompt = f"""
You are a GPU optimization expert. Compare the current kernel performance state against known optimization states and find the best match.

CURRENT STATE TO MATCH:
Primary Bottleneck: {current_state.primary_bottleneck}
Secondary Characteristics: {', '.join(current_state.secondary_characteristics)}
Performance Signature: {current_state.performance_signature}
Relative Patterns: {json.dumps(current_state.relative_patterns, indent=2)}
Context: {current_state.context_description}

KNOWN OPTIMIZATION STATES:
{known_states_text}

MATCHING INSTRUCTIONS:
1. Primary bottleneck must align (memory_bound with memory_bound, etc.)
2. Look for similar secondary characteristics and patterns
3. Consider the performance signature and context similarity
4. Focus on qualitative patterns rather than exact matches

Provide your analysis in this EXACT format:

BEST_MATCH: [state_name from database or "NEW_STATE_NEEDED"]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Explain why this state matches, focusing on bottleneck alignment and similar characteristics]

If confidence < 0.6, respond with BEST_MATCH: NEW_STATE_NEEDED
"""
        
        if self.llm_interface and self.llm_interface.is_available():
            try:
                matching_result = await self.llm_interface.query(matching_prompt, max_tokens=500, temperature=0.1)
                # 记录交互
                self._log_llm_interaction("StateMatching", matching_prompt, matching_result)
                return self._parse_matching_result(matching_result)
            except Exception as e:
                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                    self.llm_interface.logger.warning(f"State matching failed: {e}")
                return self._fallback_state_matching(current_state)
        else:
            return self._fallback_state_matching(current_state)
    
    def _parse_matching_result(self, llm_response: str) -> str:
        """
        解析 LLM 匹配响应以提取最佳匹配。

        参数:
            llm_response: 调用方提供的 `llm_response` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        lines = llm_response.split('\n')
        
        for line in lines:
            if line.startswith("BEST_MATCH:"):
                match = line.split(":", 1)[1].strip()
                return match
        
        return "unknown_state"
    
    async def get_state_from_ncu_report(
        self, ncu_report: str, metrics: dict, code_implementation: str = "", elapsed_cycles: Optional[int] = None
    ) -> str:
        """
        主界面：两个LLM代理系统，用于状态识别。

        返回优化选择的匹配状态名称。

        参数:
            ncu_report: 调用方提供的 `ncu_report` 参数。
            metrics: 性能分析或正确性检查产生的指标集合。
            code_implementation: 调用方提供的 `code_implementation` 参数。
            elapsed_cycles: 调用方提供的 `elapsed_cycles` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """

        # 代理 1：定性分析当前状态
        current_state = await self.analyze_performance_state(
            ncu_report, metrics, code_implementation, elapsed_cycles=elapsed_cycles
        )
        
        # 代理 2：与已知优化模式匹配
        matched_state = await self.match_state_against_database(current_state)
        
        # 处理新状态发现 - 混合方法：创建新状态 + 继承策略
        if matched_state == "NEW_STATE_NEEDED" or matched_state == "unknown_state":
            # 创建新状态以保留独特特征
            new_state_name = f"discovered_{current_state.primary_bottleneck}_{len(self.optimization_strategies)}"
            
            # 找到最好的现有状态来继承优化策略
            source_state = self._map_to_existing_state_with_strategies(current_state)
            
            if source_state and source_state in self.optimization_strategies:
                # 复制优化策略，但对新状态的信心降低
                inherited_strategies = []
                for original_strategy in self.optimization_strategies[source_state].get("optimizations", []):
                    inherited_strategy = OptimizationEntry(
                        technique=original_strategy.technique,
                        predicted_improvement=original_strategy.predicted_improvement * 0.8 if original_strategy.predicted_improvement is not None else None,  # 降低信心
                        description=f"Inherited from {source_state}: {original_strategy.description}",
                        category=original_strategy.category,
                        confidence_score=original_strategy.confidence_score * 0.8,  # 对继承的信心降低
                        last_updated=datetime.now().isoformat(),
                        usage_count=original_strategy.usage_count  # 保留累计使用次数而不是重置
                    )
                    inherited_strategies.append(inherited_strategy)
                
                # 将继承的策略分配给新状态（用元数据包装）
                self.optimization_strategies[new_state_name] = {
                    "optimizations": inherited_strategies,
                    "primary_bottleneck": current_state.primary_bottleneck,
                    "secondary_characteristics": current_state.secondary_characteristics,
                }
                
                # 存储详细的发现元数据
                self.discovered_states[new_state_name] = {
                    "original_state": current_state.__dict__,
                    "inherited_from": source_state,
                    "inherited_strategies_count": len(inherited_strategies),
                    "discovery_timestamp": datetime.now().isoformat(),
                    "approach": "hybrid_create_and_inherit"
                }
                
                # 记录混合创建
                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                    self.llm_interface.logger.info(
                        f"Created new state '{new_state_name}' with {len(inherited_strategies)} "
                        f"strategies inherited from '{source_state}' (bottleneck: {current_state.primary_bottleneck})"
                    )
                
                # 预选择并缓存新发现状态的优化
                try:
                    best_opt = await self._select_best_optimization_llm(new_state_name, current_state)
                    if best_opt:
                        self._llm_recommended_optimizations[new_state_name] = best_opt
                except Exception as e:
                    if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                        self.llm_interface.logger.warning(f"LLM optimisation selection failed: {e}")

                return new_state_name
            
            else:
                # 回退：根据瓶颈类型使用默认策略创建状态
                default_strategies = self._create_default_strategies_for_bottleneck(current_state.primary_bottleneck)
                if default_strategies:
                    self.optimization_strategies[new_state_name] = {
                        "optimizations": default_strategies,
                        "primary_bottleneck": current_state.primary_bottleneck,
                        "secondary_characteristics": current_state.secondary_characteristics,
                    }
                    
                    # 存储用于创建默认策略的元数据
                    self.discovered_states[new_state_name] = {
                        "original_state": current_state.__dict__,
                        "strategy_source": "default_for_bottleneck",
                        "default_strategies_count": len(default_strategies),
                        "discovery_timestamp": datetime.now().isoformat(),
                        "approach": "create_with_default_strategies"
                    }
                    
                    if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                        self.llm_interface.logger.info(
                            f"Created new state '{new_state_name}' with {len(default_strategies)} "
                            f"default strategies for bottleneck: {current_state.primary_bottleneck}"
                        )
                    
                    # 预选择并缓存默认策略状态的优化
                    try:
                        best_opt = await self._select_best_optimization_llm(new_state_name, current_state)
                        if best_opt:
                            self._llm_recommended_optimizations[new_state_name] = best_opt
                    except Exception as e:
                        if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                            self.llm_interface.logger.warning(f"LLM optimisation selection failed: {e}")

                return new_state_name
        
        # 缓存匹配现有状态的优化，以便调用者
        # 可以通过select_best_optimization立即检索它。
        try:
            best_opt = await self._select_best_optimization_llm(matched_state, current_state)
            if best_opt:
                self._llm_recommended_optimizations[matched_state] = best_opt
        except Exception as e:
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.warning(f"LLM optimisation selection failed: {e}")

        return matched_state
    
    def _map_to_existing_state_with_strategies(self, current_state: StateProfile) -> Optional[str]:
        """
        将发现的状态映射到具有优化策略的现有状态。

        这可以防止发现的状态没有优化策略的问题。

        参数:
            current_state: 调用方提供的 `current_state` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 主要瓶颈映射到数据库状态名称
        bottleneck_to_state_mapping = {
            "memory_bound": [
                "memory_bandwidth_saturated",
                "memory_latency_bound", 
                "memory_bank_conflicts",
                "cache_inefficient"
            ],
            "compute_bound": [
                "compute_throughput_saturated",
                "instruction_mix_suboptimal",
                "thread_divergence_high"
            ],
            "latency_bound": [
                "low_occupancy_register_pressure",
                "low_occupancy_shared_memory",
                "insufficient_parallelism"
            ],
            "hybrid_bound": [
                "memory_compute_balanced",
                "latency_memory_bound"
            ]
        }
        
        # 根据主要瓶颈获取候选状态
        candidates = bottleneck_to_state_mapping.get(current_state.primary_bottleneck, [])
        
        # 过滤到仅具有优化策略的州
        candidates_with_strategies = [
            state for state in candidates 
            if state in self.optimization_strategies and len(self.optimization_strategies[state].get("optimizations", [])) > 0
        ]
        
        if not candidates_with_strategies:
            # 尝试找到任何具有优化策略的状态作为最后的手段
            candidates_with_strategies = [
                state for state in self.optimization_strategies.keys()
                if len(self.optimization_strategies[state].get("optimizations", [])) > 0
            ]
            
            # 记录调试信息
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.debug(
                    f"No direct candidates found for {current_state.primary_bottleneck}. "
                    f"Available states with strategies: {list(self.optimization_strategies.keys())}"
                )
        
        if candidates_with_strategies:
            # 现在，返回第一个候选者。可以通过相似性评分来改进
            selected_state = candidates_with_strategies[0]
            
            # 记录选择
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.info(
                    f"Selected state '{selected_state}' for bottleneck '{current_state.primary_bottleneck}' "
                    f"with {len(self.optimization_strategies[selected_state].get('optimizations', []))} optimization strategies"
                )
            
            return selected_state
        
        return None
    
    def _fallback_state_analysis(self, ncu_report: str, metrics: dict) -> StateProfile:
        """
        LLM 不可用时的后备分析。

        参数:
            ncu_report: 调用方提供的 `ncu_report` 参数。
            metrics: 性能分析或正确性检查产生的指标集合。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        memory_throughput = metrics.get('memory_throughput', 0)
        compute_throughput = metrics.get('compute_throughput', 0)
        
        if memory_throughput > compute_throughput * 1.2:
            primary_bottleneck = "memory_bound"
        elif compute_throughput > memory_throughput * 1.2:
            primary_bottleneck = "compute_bound"  
        else:
            primary_bottleneck = "hybrid_bound"
        
        return StateProfile(
            state_name="fallback_analysis",
            primary_bottleneck=primary_bottleneck,
            secondary_characteristics=["Fallback analysis - limited detail"],
            performance_signature=f"Fallback analysis indicates {primary_bottleneck} workload",
            context_description="Basic analysis due to LLM unavailability",
            relative_patterns={"analysis_quality": "basic"}
        )
    
    def _fallback_state_matching(self, current_state: StateProfile) -> str:
        """
        LLM 不可用时的后备匹配。

        参数:
            current_state: 调用方提供的 `current_state` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 基于主要瓶颈的简单匹配
        for state_name, state_data in self.optimization_strategies.items():
            if state_data.get("primary_bottleneck") == current_state.primary_bottleneck:
                return state_name
        return "unknown_state"

    async def _select_best_optimization_llm(
        self,
        state: str,
        current_state_profile: StateProfile,
        include_composite: bool = True,
    ) -> Optional[OptimizationEntry | CompositeOptimization]:
        """
        让 LLM 在全局候选中选择最匹配当前状态的优化方案。

        我们现在不再将选择限制为*状态*特定技术
        公开数据库中发现的**所有**优化（跨每个
        状态）。  这给了选择器完全的自由并消除了
        对显式状态匹配阶段的依赖。

        帮助器仍然在提供的 *state* 键下缓存选择，因此
        外部调用者可以通过透明地访问它
        ``select_best_optimization``。

        参数:
            state: 工作流节点读取并按约定更新的共享状态。
            current_state_profile: 调用方提供的 `current_state_profile` 参数。
            include_composite: 调用方提供的 `include_composite` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """

        def _collect_all_opts(include_composite_flag: bool):
            """
            收集 `collect_all_opts` 所表示的内部步骤；该函数不属于稳定的公开接口。

            参数:
                include_composite_flag: 调用方提供的 `include_composite_flag` 参数。

            返回:
                当前操作产生的结果；具体类型由返回注解和调用约定确定。
            """
            opts: List[OptimizationEntry | CompositeOptimization] = []
            for state_data in self.optimization_strategies.values():
                opts.extend(state_data.get("optimizations", []))
            if include_composite_flag:
                for lst in self.composite_optimizations.values():
                    opts.extend(lst)
            return opts

        all_opts: List[OptimizationEntry | CompositeOptimization] = _collect_all_opts(include_composite)
        
        if not all_opts:
            return None

        # --------------- 尝试LLM驱动的选择---------------
        chosen_name: Optional[str] = None
        if self.llm_interface and self.llm_interface.is_available():
            # 采样最多 15 个代表性选项以保持提示大小
            # 可管理的——选择那些预测改进最高的。
            top_opts = sorted(
                all_opts,
                key=lambda o: (
                    getattr(o, "predicted_speedup", None)
                    if getattr(o, "predicted_speedup", None) not in (None, 0.0)
                    else 1.0 / max(1e-6, 1.0 - ((getattr(o, "predicted_improvement", 0.0) or 0.0) / 100.0))
                ),
                reverse=True,
            )

            options_text = "\n".join(
                [
                    f"- {opt.technique}: {((getattr(opt, 'predicted_speedup', None) if getattr(opt, 'predicted_speedup', None) not in (None, 0.0) else (1.0 / max(1e-6, 1.0 - ((getattr(opt, 'predicted_improvement', 0.0) or 0.0)/100.0))))):.2f}x (confidence {getattr(opt,'confidence_score',0.5)})"
                    for opt in top_opts
                ]
            )

            prompt = f"""
You are a GPU optimisation expert. A kernel has been analysed and its qualitative
performance characteristics are shown below. From the list of available
optimisation techniques pick the ONE technique that you judge will yield the
largest performance gain. Respond STRICTLY in the format:

BEST_OPTIMIZATION: <technique name>
REASONING: <brief rationale>

CURRENT STATE SUMMARY
Primary Bottleneck: {current_state_profile.primary_bottleneck}
Secondary Characteristics: {', '.join(current_state_profile.secondary_characteristics)}
Performance Signature: {current_state_profile.performance_signature}

AVAILABLE OPTIMISATIONS:
{self._build_available_optimisations_summary()}
            """

            try:
                response = await self.llm_interface.query(prompt, max_tokens=400, temperature=0.1)
                self._log_llm_interaction("OptSelection", prompt, response)

                for line in response.split("\n"):
                    if line.strip().startswith("BEST_OPTIMIZATION"):
                        chosen_name = line.split(":", 1)[1].strip()
                        break
            except Exception as exc:
                if hasattr(self.llm_interface, "logger") and self.llm_interface.logger:
                    self.llm_interface.logger.warning(f"LLM optimisation selection failed: {exc}")

        # --------------- 回退确定性选择 ---------------
        if not chosen_name:
            def _score(o):
                """
                处理 `score` 所表示的内部步骤；该函数不属于稳定的公开接口。

                参数:
                    o: 调用方提供的 `o` 参数。

                返回:
                    当前操作产生的结果；具体类型由返回注解和调用约定确定。
                """
                pred_speedup = getattr(o, "predicted_speedup", None)
                if pred_speedup in (None, 0.0):
                    pred_impr = (getattr(o, "predicted_improvement", 0.0) or 0.0)
                    pred_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
                return pred_speedup * getattr(o, "confidence_score", 0.5)

            best_opt = max(all_opts, key=_score)
            return best_opt

        # 将 LLM 选择的名称映射回对象。
        for opt in all_opts:
            if opt.technique == chosen_name:
                return opt
            if isinstance(opt, CompositeOptimization) and opt.get_composite_id() == chosen_name:
                return opt

        # LLM 若返回未知名称，则退回确定性评分，保证调用方始终得到可用结果。
        def _score(o):
            """
            处理 `score` 所表示的内部步骤；该函数不属于稳定的公开接口。

            参数:
            o: 调用方提供的 `o` 参数。

            返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
            """
            pred_speedup = getattr(o, "predicted_speedup", None)
            if pred_speedup in (None, 0.0):
                pred_impr = (getattr(o, "predicted_improvement", 0.0) or 0.0)
                pred_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
            return pred_speedup * getattr(o, "confidence_score", 0.5)

        return max(all_opts, key=_score)
    
    async def generate_optimization_plan(
        self,
        state_analysis_response: str,
        code_implementation: str,
        top_n: int = 5, # 生成前 5 个优化选项
    ) -> List[Dict[str, Any]]:
        """
        要求 LLM 选择 *top_n* 个与当前状态最相关的优化技术。

        参数
        ----------
        state_analysis_response：
        :py:meth:`analyze_performance_state` 返回的原始文本（或
        等效手动分析）。  它提供了质量瓶颈
        当前内核的性能特征。
        code_implementation：
        待优化的内核的 CUDA/C++ 实现 - 将是
        嵌入在`````cpp`````块中，因此语法突出显示是
        为 LLM 保留语法高亮和清晰的代码边界。
        top_n：
        向 LLM 请求的优化候选数（默认 3）。

        退货
        -------
        列表[字典[str, 任意]]
        长度为 *top_n* 的列表，其中每个元素都是一个字典，其中
        键“`technique``, ``relevance_score`` and ``reasoning`”。

        参数:
            state_analysis_response: 调用方提供的 `state_analysis_response` 参数。
            code_implementation: 调用方提供的 `code_implementation` 参数。
            top_n: 调用方提供的 `top_n` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """

        # ------------------------- LLM提示-------------------------
        prompt = f"""
You are a world-class GPU optimisation expert.  Based on the kernel implementation
and the qualitative state analysis below, choose the **{top_n}** optimisation
techniques that are most likely to improve performance.  From the list of
AVAILABLE OPTIMISATIONS pick only those with the highest relevance to the
observed performance characteristics **and** the specific code patterns you see.

For *each* chosen technique provide a concise explanation *why* it is relevant
and *how* it should be applied to the given code.  Also assign a numerical
RELEVANCE_SCORE between 0.0 (not relevant) and 1.0 (perfect match).

Return your answer as **valid JSON** in the EXACT format (no extra keys, no
comments, do not wrap the JSON in markdown fences):

[
  {{
    "technique": "<name>",
    "relevance_score": <float 0-1>,
    "description": "<explain what the optimisation does and *why* it applies to the current code>"
  }},
  ... (exactly {top_n} entries)
]


Besides the available optimisation techniques in Performance State Categories, you should always consider using the following techniques: 

- For memory bandwidth bound, or compute bandwidth bound kernels: prioritize using **SIMD_operations**: Use packed SIMD datatypes such as half2
- For compute bandwidth bound and compute throughput bound kernels: prioritize using **tensor_core_utilization**: Use tensor core library such as wmma when there exist a tensor cores (sm_70+) in the target GPU archecture. 
----------------------- CURRENT KERNEL CONTEXT -----------------------
STATE ANALYSIS RESPONSE:
{state_analysis_response}

CODE IMPLEMENTATION:
```cpp
{code_implementation}
```

----------------------- AVAILABLE OPTIMISATIONS ----------------------
{self._build_available_optimisations_summary()}
"""

        # -------------------- 尝试 LLM 推理 --------------------
        if self.llm_interface and self.llm_interface.is_available():
            try:
                llm_resp = await self.llm_interface.query(prompt, max_tokens=800, temperature=0.1)
                self._log_llm_interaction("OptPlan", prompt, llm_resp)
                plan = self._parse_optimization_plan(llm_resp, top_n)
                if plan:
                    return plan
            except Exception as exc:
                if hasattr(self.llm_interface, "logger") and self.llm_interface.logger:
                    self.llm_interface.logger.warning(f"LLM optimisation-plan generation failed: {exc}")

        # ------------------ 回退确定性路径 ------------------
        def _score(opt):
            """
            处理 `score` 所表示的内部步骤；该函数不属于稳定的公开接口。

            参数:
                opt: 调用方提供的 `opt` 参数。

            返回:
                当前操作产生的结果；具体类型由返回注解和调用约定确定。
            """
            pred_speedup = getattr(opt, "predicted_speedup", None)
            if pred_speedup in (None, 0.0):
                pred_impr = (getattr(opt, "predicted_improvement", 0.0) or 0.0)
                pred_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
            return pred_speedup * getattr(opt, "confidence_score", 0.5)

        # 展平所有优化对象
        all_opts: List[OptimizationEntry | CompositeOptimization] = list(
            itertools.chain.from_iterable(
                state_data.get("optimizations", []) for state_data in self.optimization_strategies.values()
            )
        )
        all_opts.extend(itertools.chain.from_iterable(self.composite_optimizations.values()))
        if not all_opts:
            return []

        best_opts = sorted(all_opts, key=_score, reverse=True)[:top_n]
        fallback_plan: List[Dict[str, Any]] = []
        for opt in best_opts:
            fallback_plan.append(
                {
                    "technique": opt.technique if isinstance(opt, OptimizationEntry) else opt.get_composite_id(),
                    "relevance_score": min(1.0, _score(opt) / 100.0),  # 粗标准化
                    "description": "Selected via deterministic fallback based on predicted speedup.",
                }
            )
        return fallback_plan

    # ------------------------------------------------------------------
    # Helper：解析LLM返回的优化计划JSON
    # ------------------------------------------------------------------
    def _parse_optimization_plan(self, llm_response: str, expected_n: int) -> List[Dict[str, Any]]:
        """
        尝试 JSON 解码 *llm_response* 并验证结构。

        参数:
            llm_response: 调用方提供的 `llm_response` 参数。
            expected_n: 调用方提供的 `expected_n` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        import json as _json

        try:
            plan = _json.loads(llm_response)
            if (
                isinstance(plan, list)
                and len(plan) == expected_n
                and all(isinstance(p, dict) for p in plan)
            ):
                return plan  # type: ignore[return-value]
        except Exception:
            pass  # Fallthrough——调用者将触发确定性回退

        return []
    
    # ------------------------------------------------------------------
    # Helper：构建所有优化技术的人类可读的摘要
    # ------------------------------------------------------------------
    def _build_available_optimisations_summary(self) -> str:
        """
        返回一个多行字符串，枚举所有优化技术。

        格式：
        状态：<state_name>
        - <技术> (pred <x>% | conf <y>): <描述>

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """

        lines: List[str] = []
        for state, state_data in self.optimization_strategies.items():
            opts = state_data.get("optimizations", [])
            if not opts:
                continue
            lines.append(f"STATE: {state}")
            for opt in opts:
                desc = opt.description or "(no description)"
                predicted_speedup = getattr(opt, 'predicted_speedup', None)
                if predicted_speedup in (None, 0.0):
                    pred_impr = (getattr(opt, 'predicted_improvement', 0.0) or 0.0)
                    predicted_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
                lines.append(
                    f"  - {opt.technique} (pred {predicted_speedup:.2f}x | conf {opt.confidence_score}): {desc}"
                )
            lines.append("")  # 状态之间的空行

        # 包括复合优化
        for state, comps in self.composite_optimizations.items():
            if not comps:
                continue
            lines.append(f"STATE (composite): {state}")
            for comp in comps:
                desc = comp.reason or "(no description)"
                predicted_speedup = getattr(comp, 'predicted_speedup', None)
                if predicted_speedup in (None, 0.0):
                    pred_impr = (getattr(comp, 'predicted_improvement', 0.0) or 0.0)
                    predicted_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
                lines.append(
                    f"  - {comp.get_composite_id()} (pred {predicted_speedup:.2f}x | conf {comp.confidence_score}): {desc}"
                )
            lines.append("")

        return "\n".join(lines)
    
    def get_optimizations_for_state(self, state: str) -> List[OptimizationEntry]:
        """
        获取给定状态的优化策略。

        参数:
            state: 工作流节点读取并按约定更新的共享状态。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return self.optimization_strategies.get(state, {}).get("optimizations", [])
    
    def get_composite_optimizations_for_state(self, state: str) -> List[CompositeOptimization]:
        """
        获得给定状态的复合优化。

        参数:
            state: 工作流节点读取并按约定更新的共享状态。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return self.composite_optimizations.get(state, [])
    
    def select_best_optimization(self, state: str, exclude_used: bool = False, 
                                include_composite: bool = True) -> Optional[OptimizationEntry | CompositeOptimization]:
        """
        返回 LLM 之前为 *state* 选择的优化。

        此方法现在是一个轻量级访问器，以便现有的外部
        代码可以保持不变。  如果用户已经请求排除
        我们遵守该合同所使用的技术；否则缓存的
        推荐直接返回。  如果由于任何原因没有缓存
        存在建议，我们回到传统的随机选择器
        以保留行为。

        参数:
            state: 工作流节点读取并按约定更新的共享状态。
            exclude_used: 调用方提供的 `exclude_used` 参数。
            include_composite: 调用方提供的 `include_composite` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """

        # ----------------------------------------------------------
        # 1）快速路径——LLM已经提出了建议。
        # ----------------------------------------------------------
        if state in self._llm_recommended_optimizations:
            recommended = self._llm_recommended_optimizations[state]
            if exclude_used and getattr(recommended, "usage_count", 0) > 0:
                return None
            try:
                tech = (
                    recommended.get_composite_id()
                    if isinstance(recommended, CompositeOptimization)
                    else getattr(recommended, "technique", str(recommended))
                )
                self.llm_interface.logger.info(
                    f"[select_best_optimization] Using cached LLM recommendation for state='{state}': {tech}"
                )
            except Exception:
                # 仅尽力记录
                pass
            return recommended

        # ----------------------------------------------------------
        # 2) Fallback——使用旧的概率评分机制。
        # ----------------------------------------------------------
        try:
            self.llm_interface.logger.info(
                f"[select_best_optimization] No cached LLM recommendation for state='{state}'. Falling back to global chooser."
            )
        except Exception:
            pass

        import math, random, itertools, os

        # 记录环境驱动的行为一次，以便从 run.log 轻松审核运行。
        if not getattr(self, "_logged_fallback_top1_env", False):
            self._logged_fallback_top1_env = True
            raw_val = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", None)
            parsed_val = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in (
                "1",
                "true",
                "True",
                "yes",
                "YES",
                "y",
                "on",
                "ON",
            )
            msg = f"KERNELAGENT_DB_FALLBACK_TOP1={raw_val!r} (parsed={parsed_val})"
            if hasattr(self.llm_interface, "logger") and self.llm_interface.logger:
                self.llm_interface.logger.info(msg)
            else:
                print(msg)

        # 也将全局优化池用于后备路径。
        optimizations = [
            opt
            for state_data in self.optimization_strategies.values()
            for opt in state_data.get("optimizations", [])
        ]
        composite_opts = (
            list(itertools.chain.from_iterable(self.composite_optimizations.values()))
            if include_composite
            else []
        )

        if exclude_used:
            optimizations = [opt for opt in optimizations if getattr(opt, "usage_count", 0) == 0]
            composite_opts = [opt for opt in composite_opts if getattr(opt, "usage_count", 0) == 0]

        if not optimizations and not composite_opts:
            return None

        def score_optimization(opt) -> float:
            """
            处理 `score_optimization` 对应的领域操作，并返回调用方所需的标准化结果。

            参数:
                opt: 调用方提供的 `opt` 参数。

            返回:
                当前操作产生的结果；具体类型由返回注解和调用约定确定。
            """
            pred_speedup = getattr(opt, "predicted_speedup", None)
            if pred_speedup in (None, 0.0):
                pred_impr = (getattr(opt, "predicted_improvement", 0.0) or 0.0)
                pred_speedup = 1.0 / max(1e-6, 1.0 - (pred_impr / 100.0))
            base_score = pred_speedup * getattr(opt, "confidence_score", 0.5)
            usage_penalty = min(getattr(opt, "usage_count", 0) * 0.1, 0.5)
            composite_bonus = 0.1 if isinstance(opt, CompositeOptimization) else 0
            return base_score - usage_penalty + composite_bonus

        all_opts = optimizations + composite_opts

        scores = [score_optimization(o) for o in all_opts]

        if all(abs(s) < 1e-6 for s in scores):
            scores = [1.0 for _ in scores]

        # 探索温度（较小=>贪婪）
        tau = 0.5  # 稍后可以作为参数公开

        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / max(tau, 1e-6)) for s in scores]
        total = sum(exp_scores)
        probs = [es / total for es in exp_scores]

        # 用于调试/重现的可选确定性回退。
        # 如果设置，我们将选择单个最佳得分优化而不是采样。
        # 环境变量有意限定在该后备路径范围内（LLM 建议不变）。
        force_top1 = os.getenv("KERNELAGENT_DB_FALLBACK_TOP1", "0") in (
            "1",
            "true",
            "True",
            "yes",
            "YES",
            "y",
            "on",
            "ON",
        )
        if force_top1:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return all_opts[best_idx]

        # 根据概率随机选择 – 消除顺序偏差
        chosen_opt = random.choices(all_opts, weights=probs, k=1)[0]
        return chosen_opt
        
    def _categorize_technique(self, technique: str) -> str:
        """
        按类型对优化技术进行分类。

        参数:
            technique: 调用方提供的 `technique` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        technique_lower = technique.lower()
        if any(term in technique_lower for term in ['memory', 'coalesced', 'cache', 'tiling']):
            return 'memory'
        elif any(term in technique_lower for term in ['compute', 'tensor', 'instruction']):
            return 'compute'
        elif any(term in technique_lower for term in ['occupancy', 'thread', 'block']):
            return 'latency'
        else:
            return 'general'
        
    def _create_default_strategies_for_bottleneck(self, bottleneck_type: str) -> List[OptimizationEntry]:
        """
        为给定的瓶颈类型创建默认优化策略。

        参数：
        bottleneck_type：瓶颈类型（memory_bound、compute_bound、latency_bound、hybrid_bound）

        返回：
        具有默认策略的 OptimizationEntry 对象列表

        参数:
            bottleneck_type: 调用方提供的 `bottleneck_type` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        
        default_strategies = []
        
        if bottleneck_type == "memory_bound":
            default_strategies = [
                OptimizationEntry(
                    technique="memory_coalescing_optimization",
                    predicted_improvement=20.0,
                    description="Optimize memory access patterns for coalesced reads/writes",
                    category="memory",
                    confidence_score=0.8,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="shared_memory_tiling",
                    predicted_improvement=25.0,
                    description="Use shared memory tiling to reduce global memory accesses",
                    category="memory",
                    confidence_score=0.7,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="vectorized_memory_access",
                    predicted_improvement=15.0,
                    description="Use vectorized loads/stores to improve memory bandwidth utilization",
                    category="memory",
                    confidence_score=0.6,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        elif bottleneck_type == "compute_bound":
            default_strategies = [
                OptimizationEntry(
                    technique="instruction_level_parallelism",
                    predicted_improvement=30.0,
                    description="Optimize instruction scheduling and parallelism",
                    category="compute",
                    confidence_score=0.8,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="fast_math_optimization",
                    predicted_improvement=20.0,
                    description="Use fast math operations where precision allows",
                    category="compute",
                    confidence_score=0.7,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="tensor_core_utilization",
                    predicted_improvement=40.0,
                    description="Utilize tensor cores for supported operations",
                    category="compute",
                    confidence_score=0.6,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        elif bottleneck_type == "latency_bound":
            default_strategies = [
                OptimizationEntry(
                    technique="occupancy_optimization",
                    predicted_improvement=35.0,
                    description="Optimize thread block size and resource usage for higher occupancy",
                    category="latency",
                    confidence_score=0.8,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="register_pressure_reduction",
                    predicted_improvement=30.0,
                    description="Reduce register usage to improve occupancy",
                    category="latency",
                    confidence_score=0.7,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="work_per_thread_increase",
                    predicted_improvement=25.0,
                    description="Increase work per thread to hide latency",
                    category="latency",
                    confidence_score=0.6,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        elif bottleneck_type == "hybrid_bound":
            default_strategies = [
                OptimizationEntry(
                    technique="memory_compute_overlap",
                    predicted_improvement=40.0,
                    description="Overlap memory operations with compute to hide latency",
                    category="general",
                    confidence_score=0.8,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="algorithmic_optimization",
                    predicted_improvement=35.0,
                    description="Optimize algorithm structure for better resource utilization",
                    category="general",
                    confidence_score=0.7,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="adaptive_block_sizing",
                    predicted_improvement=30.0,
                    description="Dynamically adjust block size based on workload characteristics",
                    category="general",
                    confidence_score=0.6,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        else:
            # 未知瓶颈类型的后备策略
            default_strategies = [
                OptimizationEntry(
                    technique="general_optimization",
                    predicted_improvement=20.0,
                    description="General optimization techniques",
                    category="general",
                    confidence_score=0.5,
                    last_updated=datetime.now().isoformat()
                ),
                OptimizationEntry(
                    technique="performance_tuning",
                    predicted_improvement=15.0,
                    description="Basic performance tuning",
                    category="general",
                    confidence_score=0.4,
                    last_updated=datetime.now().isoformat()
                )
            ]
        
        return default_strategies
    
    def _load_composite_optimizations(self, json_data: dict):
        """
        从 JSON 数据加载复合优化。

        参数:
            json_data: 调用方提供的 `json_data` 参数。
        """
        for adjustment in json_data.get("prediction_adjustments", []):
            state = adjustment["state"]
            if state not in self.composite_optimizations:
                self.composite_optimizations[state] = []
            
            composite = CompositeOptimization(
                state=state,
                technique1=adjustment["technique1"],
                technique2=adjustment.get("technique2"),
                technique3=adjustment.get("technique3"),
                order_of_techniques=adjustment.get("order_of_techniques", []),
                parameters_to_fine_tune=adjustment.get("parameters_to_fine_tune", {}),
                predicted_improvement=adjustment.get("new_predicted_improvement", 0.0),
                reason=adjustment.get("reason", ""),
                side_effects=adjustment.get("side_effects", "")
            )
            self.composite_optimizations[state].append(composite)

    def update_optimization_result(self, state: str, technique: str, actual_improvement: float,
                                    current_file_path: Optional[Path] = None,
                                    measurement: Measurement | None = None):
            # 记录更新尝试
            # 以实际结果更新优化条目
            """
            更新 `update_optimization_result` 对应的领域操作，并返回调用方所需的标准化结果。

            参数:
                state: 工作流节点读取并按约定更新的共享状态。
                technique: 调用方提供的 `technique` 参数。
                actual_improvement: 调用方提供的 `actual_improvement` 参数。
                current_file_path: 调用方提供的 `current_file_path` 参数。

            异常:
                ValueError: 输入、外部调用或状态不满足执行要求时抛出。
            """
            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                self.llm_interface.logger.info(f"Attempting to update optimization result for {technique} in state {state} with actual improvement {actual_improvement}")
            if state in self.optimization_strategies:
                for opt in self.optimization_strategies[state].get("optimizations", []):
                    if opt.technique == technique:
                        prev_usage = opt.usage_count
                        new_usage = prev_usage + 1
                        # 存储最近的测量值
                        opt.actual_improvement = actual_improvement
                        opt.usage_count = new_usage
                        opt.last_updated = datetime.now().isoformat()

                        self.llm_interface.logger.info(f"Updating database entry for {technique} in state {state}")
                        # ----------------- 计算加速比 -----------------
                        speedup_of_cur_optimization = 1.0  # 默认不加速
                        if current_file_path:
                            try:
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: current_file_path={current_file_path}"
                                    )
                                # 对于第一次迭代，从文件中获取初始基线
                                if opt.initial_elapsed_cycles is None:
                                    baseline_ncu = current_file_path.parent / "ncu/0_init_ncu_log.txt"
                                    init_cu = current_file_path.parent / "ncu_annot/init.cu"
                                    if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                        self.llm_interface.logger.info(
                                            f"Speedup calc: baseline paths exist? ncu={baseline_ncu.exists()} init_cu={init_cu.exists()}"
                                        )
                                    if baseline_ncu.exists():
                                        initial_text = baseline_ncu.read_text()
                                        used_path = baseline_ncu
                                    else:
                                        initial_text = init_cu.read_text()
                                        used_path = init_cu
                                    opt.initial_elapsed_cycles = get_elapsed_cycles_v2(initial_text)
                                    if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                        self.llm_interface.logger.info(
                                            f"Speedup calc: parsed initial_elapsed_cycles={opt.initial_elapsed_cycles} from {used_path}"
                                        )
                                # 使用传入的 actual_improvement 作为当前经过的周期来计算加速比
                                # 和存储的 initial_elapsed_cycles 作为基线
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: actual_improvement arg value={actual_improvement} (type={type(actual_improvement)})"
                                    )
                                current_elapsed_cycles = int(actual_improvement)
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: current_elapsed_cycles={current_elapsed_cycles}, baseline={opt.initial_elapsed_cycles}"
                                    )
                                if current_elapsed_cycles <= 0:
                                    raise ValueError(f"Non-positive current_elapsed_cycles={current_elapsed_cycles}")
                                speedup_of_cur_optimization = abs(float(opt.initial_elapsed_cycles) / float(current_elapsed_cycles))
                                opt.actual_speedup = speedup_of_cur_optimization
                            except (ValueError, FileNotFoundError, AttributeError) as e:
                                # 如果无法读取文件，则退回到无加速计算
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.warning(f"Could not calculate speedup (baseline flow): {e}")
                                speedup_of_cur_optimization = 1.0
                        else:
                            # 未提供文件路径，无法读取基线。如果适用，尝试从改进百分比进行推断。
                            if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                self.llm_interface.logger.info("Speedup calc: no current_file_path provided; attempting percent-based inference.")
                            try:
                                denom = 1.0 - (float(actual_improvement) / 100.0)
                                if abs(denom) < 1e-6:
                                    denom = 1e-6 if denom >= 0 else -1e-6
                                inferred_speedup = abs(1.0 / denom)
                                # 当基线文件路径不可用时，使用推断的加速比作为测量值
                                speedup_of_cur_optimization = inferred_speedup
                                opt.actual_speedup = inferred_speedup
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: inferred speedup from percent improvement={inferred_speedup:.4f}x (actual_improvement={actual_improvement})"
                                    )
                            except Exception as e:
                                if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                                    self.llm_interface.logger.info(
                                        f"Speedup calc: percent-based inference failed: {e} (actual_improvement={actual_improvement})"
                                    )
                        # 记录测量的加速
                        if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                            self.llm_interface.logger.info(
                                f"Measured speedup for {technique} in state {state}: {speedup_of_cur_optimization:.4f}x"
                            )
                        # ----------------- 使用滚动平均值 ------------- 更新 predicted_speedup
                        # 使用运行加权平均值进行加速跟踪（与 predicted_improvement 分开）
                        if opt.predicted_speedup is None:
                            opt.predicted_speedup = 1.0
                        # 逆权重：给后面的迭代更多的权重
                        # cur_num_iter 是当前使用次数（从 1 开始）
                        cur_num_iter = new_usage
                        # inverse_weight = 1.0 / 最大值(1, 100 - cur_num_iter)
                        try:
                        # 分子 = opt.predicted_speedup * 浮点数(prev_usage) + speedup_of_cur_optimization * inverse_weight
                        # 分母 = max(浮点(prev_usage) + inverse_weight, 1e-6)
                            numerator = opt.predicted_speedup * float(prev_usage) + speedup_of_cur_optimization 
                            denom = max(float(new_usage), 1e-6)

                            opt.predicted_speedup = numerator / denom
                        except ZeroDivisionError:
                            # 不应该发生，但要警惕以防万一。
                            opt.predicted_speedup = speedup_of_cur_optimization
                        # 记录预测与实际加速和权重
                        if hasattr(self.llm_interface, 'logger') and self.llm_interface.logger:
                            self.llm_interface.logger.info(
                                f"Predicted speedup now {opt.predicted_speedup:.4f}x; actual speedup {getattr(opt, 'actual_speedup', None)} | prev_usage={prev_usage}"
                            )
                        # ----------------- 更新 predicted_improvement -------------
                        # 使用运行加权平均值，其中现有
                        # 预测值被视为 *prev_usage* 的平均值
                        # 历史数据点。  这使价值保持稳定
                        # 但随着更多真实数据的到来，它可以收敛。
                        if opt.predicted_improvement is None:
                            opt.predicted_improvement = 0.0
                        try:
                            opt.predicted_improvement = (
                                opt.predicted_improvement * prev_usage + actual_improvement
                            ) / max(new_usage, 1)
                        except ZeroDivisionError:
                            # 不应该发生，但要警惕以防万一。
                            opt.predicted_improvement = actual_improvement
                        # ----------------- 调整置信度分数 -----------------
                        # 使用*先前*预测计算准确性，因此
                        # 调整反映了先前估计的质量。
                        if prev_usage > 0 and opt.predicted_improvement > 0:
                            # 使用*旧*预测值（更新前）
                            # 是 (new_pred * new_usage - 实际) / prev_usage
                            prev_pred = (
                                opt.predicted_improvement * new_usage - actual_improvement
                            ) / max(prev_usage, 1)
                            accuracy = actual_improvement / prev_pred if prev_pred else 0.0
                            if 0.8 <= accuracy <= 1.2:  # 良好的预测（±20%）
                                opt.confidence_score = min(1.0, opt.confidence_score + 0.1)
                            else:  # 预测不佳
                                opt.confidence_score = max(0.1, opt.confidence_score - 0.1)
                        # -- 日志更改 --------------------------------------------------
                        if measurement is not None:
                            opt.last_measurement = measurement.to_dict()
                        self._log_db_change(
                            "update_optimization_result",
                            {
                                "state": state,
                                "technique": technique,
                                "actual_improvement": actual_improvement,
                                "predicted_improvement": opt.predicted_improvement,
                                "confidence_score": opt.confidence_score,
                                "usage_count": opt.usage_count,
                                "speedup_of_cur_optimization": speedup_of_cur_optimization,
                                "predicted_speedup": opt.predicted_speedup,
                                "actual_speedup": opt.actual_speedup,
                                "initial_elapsed_cycles": opt.initial_elapsed_cycles,
                                "last_measurement": opt.last_measurement,
                            },
                        )
                        # 对于首次使用的条目，我们保留原样的信心。
                        break

    def update_composite_optimization_result(
        self,
        state: str,
        composite_id: str,
        actual_improvement: float,
        measurement: Measurement | None = None,
    ):
        """
        更新复合优化结果以进行跟踪和学习。

        参数:
            state: 工作流节点读取并按约定更新的共享状态。
            composite_id: 调用方提供的 `composite_id` 参数。
            actual_improvement: 调用方提供的 `actual_improvement` 参数。
        """
        if state in self.composite_optimizations:
            for comp_opt in self.composite_optimizations[state]:
                if comp_opt.get_composite_id() == composite_id:
                    comp_opt.actual_improvement = actual_improvement
                    comp_opt.usage_count += 1
                    comp_opt.last_updated = datetime.now().isoformat()
                    # 更新置信度得分
                    if comp_opt.predicted_improvement > 0:
                        accuracy = actual_improvement / comp_opt.predicted_improvement
                        if 0.8 <= accuracy <= 1.2:
                            comp_opt.confidence_score = min(1.0, comp_opt.confidence_score + 0.1)
                        else:
                            comp_opt.confidence_score = max(0.1, comp_opt.confidence_score - 0.1)
                    # -- 日志更改 --------------------------------------------------
                    self._log_db_change(
                        "update_composite_optimization_result",
                        {
                            "state": state,
                            "composite_id": composite_id,
                            "actual_improvement": actual_improvement,
                            "predicted_improvement": comp_opt.predicted_improvement,
                            "confidence_score": comp_opt.confidence_score,
                            "usage_count": comp_opt.usage_count,
                            "measurement": measurement.to_dict() if measurement is not None else None,
                        },
                    )
                    break

    def add_composite_optimization(self, composite: CompositeOptimization):
        """
        向数据库添加复合优化。

        参数:
            composite: 调用方提供的 `composite` 参数。
        """
        state = composite.state
        if state not in self.composite_optimizations:
            self.composite_optimizations[state] = []
        
        # 检查该组合是否已存在
        for existing_comp in self.composite_optimizations[state]:
            if existing_comp.get_composite_id() == composite.get_composite_id():
                # 更新现有复合材料
                existing_comp.predicted_improvement = composite.predicted_improvement
                existing_comp.reason = composite.reason
                existing_comp.side_effects = composite.side_effects
                existing_comp.last_updated = datetime.now().isoformat()

                # 记录更新事件
                self._log_db_change(
                    "update_composite_optimization",
                    {
                        "state": state,
                        "composite_id": existing_comp.get_composite_id(),
                        "predicted_improvement": existing_comp.predicted_improvement,
                    },
                )
                return
        
        # 添加新的复合优化
        composite.last_updated = datetime.now().isoformat()
        self.composite_optimizations[state].append(composite)

        # 日志创建事件
        self._log_db_change(
            "add_composite_optimization",
            {
                "state": state,
                "composite_id": composite.get_composite_id(),
                "predicted_improvement": composite.predicted_improvement,
            },
        )

    def add_new_optimization(self, state: str, technique: str, predicted_improvement: float):
        """
        向数据库添加新的优化技术。

        参数:
            state: 工作流节点读取并按约定更新的共享状态。
            technique: 调用方提供的 `technique` 参数。
            predicted_improvement: 调用方提供的 `predicted_improvement` 参数。
        """
        if state not in self.optimization_strategies:
            self.optimization_strategies[state] = {"optimizations": []}
        
        # 检查此状态是否已存在此技术
        for existing_opt in self.optimization_strategies[state].get("optimizations", []):
            if existing_opt.technique == technique:
                # 更新现有优化
                existing_opt.predicted_improvement = predicted_improvement
                existing_opt.last_updated = datetime.now().isoformat()

                self._log_db_change(
                    "update_optimization",
                    {
                        "state": state,
                        "technique": technique,
                        "predicted_improvement": predicted_improvement,
                    },
                )
                return
        
        # 添加新的优化
        new_opt = OptimizationEntry(
            technique=technique,
            predicted_improvement=predicted_improvement,
            category=self._categorize_technique(technique),
            last_updated=datetime.now().isoformat()
        )
        self.optimization_strategies[state]["optimizations"].append(new_opt)

        self._log_db_change(
            "add_new_optimization",
            {
                "state": state,
                "technique": technique,
                "predicted_improvement": predicted_improvement,
            },
        )

    def create_parameter_tuned_optimization(self, base_technique: str, parameters: Dict[str, Any], 
                                          predicted_improvement: float, reason: str = "") -> str:
        """
        创建参数调整的优化技术名称。

        参数:
            base_technique: 调用方提供的 `base_technique` 参数。
            parameters: 调用方提供的 `parameters` 参数。
            predicted_improvement: 调用方提供的 `predicted_improvement` 参数。
            reason: 调用方提供的 `reason` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        # 创建包含参数的唯一技术名称
        param_str = "_".join(f"{k}_{v}" for k, v in parameters.items())
        tuned_technique = f"{base_technique}_tuned_{param_str}"
        
        # 存储参数信息以供以后使用
        if not hasattr(self, 'parameter_tuned_techniques'):
            self.parameter_tuned_techniques = {}
        
        self.parameter_tuned_techniques[tuned_technique] = {
            "base_technique": base_technique,
            "parameters": parameters,
            "predicted_improvement": predicted_improvement,
            "reason": reason,
            "created_at": datetime.now().isoformat()
        }

        # 对数调整技术创建
        self._log_db_change(
            "create_parameter_tuned_optimization",
            {
                "tuned_technique": tuned_technique,
                "base_technique": base_technique,
                "parameters": parameters,
                "predicted_improvement": predicted_improvement,
            },
        )
        
        return tuned_technique

    # 旧版兼容性
    def get_state_from_metrics(self, metrics: dict, performance_pattern: str = "") -> str:
        """
        传统方法 - 使用 get_state_from_ncu_report 代替。

        参数:
            metrics: 性能分析或正确性检查产生的指标集合。
            performance_pattern: 调用方提供的 `performance_pattern` 参数。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        print("Warning: get_state_from_metrics is deprecated. Use get_state_from_ncu_report instead.")
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_state_from_ncu_report(performance_pattern, metrics))
        except:
            return self._fallback_state_matching(self._fallback_state_analysis(performance_pattern, metrics))

    def get_database_stats(self) -> dict:
        """
        获取有关优化数据库的统计信息。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        
        # 计算总优化次数
        total_optimizations = 0
        total_composite_optimizations = 0
        state_counts = {}
        
        for state, state_data in self.optimization_strategies.items():
            opts = state_data.get("optimizations", [])
            total_optimizations += len(opts)
            state_counts[state] = len(opts)
        
        for state, composite_opts in self.composite_optimizations.items():
            total_composite_optimizations += len(composite_opts)
            if state in state_counts:
                state_counts[state] += len(composite_opts)
            else:
                state_counts[state] = len(composite_opts)
        
        # 计算平均改进
        improvements = []
        for state_data in self.optimization_strategies.values():
            for opt in state_data.get("optimizations", []):
                if opt.actual_improvement is not None:
                    improvements.append(opt.actual_improvement)
        
        for state_opts in self.composite_optimizations.values():
            for opt in state_opts:
                if opt.actual_improvement is not None:
                    improvements.append(opt.actual_improvement)
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        
        return {
            "total_states": len(self.optimization_strategies),
            "total_optimizations": total_optimizations,
            "total_composite_optimizations": total_composite_optimizations,
            "state_counts": state_counts,
            "average_improvement": avg_improvement,
            "total_measured_optimizations": len(improvements),
            "last_updated": datetime.now().isoformat(),
            "database_health": "healthy" if total_optimizations > 0 else "empty"
        }


# 保持向后兼容性
OptimizationDatabase = GPUOptimizationDatabase 


def print_database_summary(db: GPUOptimizationDatabase):
    """
    打印优化数据库的人类可读摘要。

    参数:
        db: 调用方提供的 `db` 参数。
    """
    import json as _json

    print("\n=== Optimisation Database Summary ===\n")
    if not db.optimization_strategies:
        print("No optimisation strategies loaded.")
        return

    for state, state_data in db.optimization_strategies.items():
        print(f"State: {state}")
        opts = state_data.get("optimizations", [])
        if opts:
            for opt in opts:
                imp = getattr(opt, "predicted_improvement", 0.0) or 0.0
                print(f"  - {opt.technique}: {imp}% predicted improvement")
        else:
            print("  (no optimisation strategies)")
        print()

    # 可选：打印简单的数据库统计信息
    try:
        stats = db.get_database_stats()
        print("Database statistics:\n" + _json.dumps(stats, indent=2))
    except Exception as exc:
        print(f"Could not compute database stats: {exc}")



def _main():
    """
    通过命令行进行快速手动测试的入口点。

    异常:
        FileNotFoundError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Load an optimisation database markdown file and list states/optimisations."
    )
    parser.add_argument(
        "--optimization_db",
        help="Path to the optimisation database markdown file (.md)",
    )
    parser.add_argument(
        "--gpu_report",
        default="",
        help="Optional path to a GPU optimisation knowledge report (text/markdown)",
    )
    parser.add_argument(
        "--prompt_file",
        default="",
        help="Optional path to a text file containing a prompt to send to the LLM for quick testing",
    )
    # ------------ 分析+计划工作流程的开发测试 ------------
    parser.add_argument(
        "--ncu_report_file",
        default="",
        help="Path to an NSight Compute (NCU) report text file used for test analysis",
    )
    parser.add_argument(
        "--code_impl_file",
        default="",
        help="Path to a CUDA/C++ kernel implementation file for optimisation planning",
    )
    parser.add_argument(
        "--metrics_json",
        default="",
        help="Optional path to a JSON file providing numeric metrics for the NCU report test",
    )
    parser.add_argument(
        "--regenerate_from_json",
        action="store_true",
        help="Regenerate the optimisation database markdown using the persisted JSON snapshot and optional header/footer",
    )
    args = parser.parse_args()

    optimisation_db_path = Path(args.optimization_db).expanduser().resolve()
    gpu_report_path = (
        Path(args.gpu_report).expanduser().resolve() if args.gpu_report else Path("/dev/null")
    )

    if not optimisation_db_path.exists():
        raise FileNotFoundError(f"Optimisation database file not found: {optimisation_db_path}")

    db = GPUOptimizationDatabase(
        optimization_db_path=optimisation_db_path,
        gpu_report_path=gpu_report_path,
        llm_interface=LLMInterface(),
    )

    print_database_summary(db)

    # ------------------------------------------------------------
    # 测试助手：从 JSON 重新生成 Markdown 并显示预览
    # ------------------------------------------------------------
    if getattr(args, "regenerate_from_json", False):
        print("\n>>> Testing markdown regeneration from JSON snapshot...\n")
        db._regenerate_database_from_json()
        try:
            md_text = db.optimization_db_path.read_text(encoding="utf-8")
            print(f"Regenerated markdown written to: {db.optimization_db_path}")
            preview_lines = md_text.splitlines()
            preview = "\n".join(preview_lines[: min(25, len(preview_lines))])
            print("\n--- Preview (first 25 lines) ---\n" + preview + "\n--- End preview ---\n")
        except Exception as exc:
            print(f"Could not read regenerated markdown: {exc}")

    # ------------------------------------------------------------
    # 快速测试：打印可用优化摘要
    # 数据库已加载。  这有助于开发人员验证
    # 技术被正确解析并且帮助器呈现
    # 它们以预期的格式。
    # ------------------------------------------------------------
    optim_summary = db._build_available_optimisations_summary()
    print("\n=== Available Optimisations Summary ===\n")
    if optim_summary.strip():
        print(optim_summary)
    else:
        print("(no optimisation techniques found)")

    # ==============================================================
    # 开发者测试：端到端analyse_performance_state +
    # generate_optimization_plan 流量。
    # ==============================================================
    if args.ncu_report_file and args.code_impl_file:
        ncu_path = Path(args.ncu_report_file).expanduser().resolve()
        code_path = Path(args.code_impl_file).expanduser().resolve()

        if not ncu_path.exists():
            print(f"NCU report file not found: {ncu_path}")
        elif not code_path.exists():
            print(f"Code implementation file not found: {code_path}")
        else:
            ncu_report_text = ncu_path.read_text(encoding="utf-8")
            code_impl_text = code_path.read_text(encoding="utf-8")

            # 加载可选指标 JSON
            metrics: dict = {}
            if args.metrics_json:
                metrics_path = Path(args.metrics_json).expanduser().resolve()
                if metrics_path.exists():
                    import json as _json
                    try:
                        metrics = _json.loads(metrics_path.read_text())
                    except Exception as exc:
                        print(f"Could not parse metrics JSON ({metrics_path}): {exc}")

            import asyncio, json as _json

            async def _run_flow():
                """运行 `run_flow` 所表示的内部步骤；该函数不属于稳定的公开接口。"""
                print("\n>>> Running analyse_performance_state...")
                profile = await db.analyze_performance_state(
                    ncu_report_text, metrics, code_impl_text
                )

                # 将计划生成器的分析表示为 JSON
                analysis_json_str = _json.dumps(asdict(profile), indent=2)

                print("Analysis result:\n" + analysis_json_str + "\n")

                print(">>> Generating optimisation plan...\n")
                plan = await db.generate_optimization_plan(
                    analysis_json_str, code_impl_text
                )

                print("Optimisation plan (top suggestions):\n" + _json.dumps(plan, indent=2))

            try:
                asyncio.run(_run_flow())
            except RuntimeError:
                # 如果我们已经处于 asyncio 循环 (e.g.Jupyter) 中，则回退
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_run_flow())

    # --------------------------------------------------------------
    # 可选的快速测试：将用户提供的提示发送给 LLM 并
    # 打印原始响应。  这**不是**正常的一部分
    # 优化工作流程——它只是一个方便的帮手
    # 想要对 LLM 连接进行健全性检查的开发人员。
    # --------------------------------------------------------------
    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        if not prompt_path.exists():
            print(f"Prompt file not found: {prompt_path}")
        elif not db.llm_interface.is_available():
            print("LLM interface is not available – skipping test query.")
        else:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            print("\n=== Prompt to LLM ===\n" + prompt_text.strip() + "\n")
            print("Querying LLM... (this may take a moment)\n")
            response = db.llm_interface.query_sync(prompt_text, max_tokens=800, temperature=0.1)
            print("=== LLM response ===\n" + response.strip() + "\n")


if __name__ == "__main__":
    _main()
