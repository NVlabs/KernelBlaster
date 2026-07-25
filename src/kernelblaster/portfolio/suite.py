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

"""解析 Portfolio Suite 文件，并将任务条目转换为强类型对象。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PortfolioTask:
    """封装 `PortfolioTask` 对应的领域状态与操作。"""
    number: int
    task_id: str
    name: str
    path: str
    category: str


@dataclass(frozen=True)
class PortfolioSuite:
    """表示一组可复现执行的 Benchmark 任务。"""
    source_path: Path
    raw: dict[str, Any]
    name: str
    subset: str
    precision: str
    rollouts: int
    steps: int
    tasks: tuple[PortfolioTask, ...]

    @property
    def problem_numbers(self) -> str:
        """
        处理 `problem_numbers` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return ",".join(str(task.number) for task in self.tasks)


def resolve_suite_path(value: str, repo_root: Path) -> Path:
    """
    解析 `resolve_suite_path` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        value: 需要转换、保存或校验的值。
        repo_root: 调用方提供的 `repo_root` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    alias = value.removesuffix(".json")
    candidate = repo_root / "portfolio" / "suites" / f"{alias}.json"
    if candidate.is_file():
        return candidate.resolve()
    raise ValueError(
        f"Unknown suite '{value}'. Use a JSON file or an alias from portfolio/suites/."
    )


def load_suite(value: str, repo_root: Path) -> PortfolioSuite:
    """
    加载 `load_suite` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        value: 需要转换、保存或校验的值。
        repo_root: 调用方提供的 `repo_root` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    source_path = resolve_suite_path(value, repo_root)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Suite root must be a JSON object.")

    required = ("name", "subset", "precision", "defaults", "tasks")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Suite is missing required fields: {', '.join(missing)}")
    if raw.get("schema_version") != "1.0":
        raise ValueError("Suite schema_version must be '1.0'.")
    if raw["subset"] not in {"level1", "level2", "level3"}:
        raise ValueError("Suite subset must be level1, level2, or level3.")
    if raw["precision"] not in {"fp16", "fp32", "bf16"}:
        raise ValueError("Suite precision must be fp16, fp32, or bf16.")

    defaults = raw["defaults"]
    rollouts = int(defaults.get("rollouts", 0))
    steps = int(defaults.get("steps", 0))
    if rollouts < 1 or steps < 1:
        raise ValueError("Suite rollouts and steps must both be positive.")

    if not isinstance(raw["tasks"], list):
        raise ValueError("Suite tasks must be a JSON array.")
    task_fields = {"number", "id", "name", "path", "category"}
    tasks_list = []
    for index, item in enumerate(raw["tasks"]):
        if not isinstance(item, dict):
            raise ValueError(f"Suite task at index {index} must be an object.")
        missing_task_fields = task_fields - set(item)
        if missing_task_fields:
            raise ValueError(
                f"Suite task at index {index} is missing: "
                f"{', '.join(sorted(missing_task_fields))}"
            )
        tasks_list.append(
            PortfolioTask(
                number=int(item["number"]),
                task_id=str(item["id"]),
                name=str(item["name"]),
                path=str(item["path"]),
                category=str(item["category"]),
            )
        )
    tasks = tuple(tasks_list)
    if not tasks:
        raise ValueError("Suite must contain at least one task.")
    if len({task.number for task in tasks}) != len(tasks):
        raise ValueError("Suite task numbers must be unique.")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Suite task IDs must be unique.")

    for task in tasks:
        task_dir = (repo_root / task.path).resolve()
        if not task_dir.is_relative_to(repo_root.resolve()):
            raise ValueError(f"Task path escapes the repository: {task.path}")
        for artifact in ("init.cu", "driver.cpp"):
            artifact_path = task_dir / artifact
            if not artifact_path.is_file():
                raise ValueError(f"Task {task.task_id} is missing {artifact}.")
            resolved_artifact = artifact_path.resolve()
            if not resolved_artifact.is_relative_to(repo_root.resolve()):
                raise ValueError(
                    f"Task {task.task_id} artifact escapes the repository via "
                    f"a symbolic link: {artifact}."
                )

    return PortfolioSuite(
        source_path=source_path,
        raw=raw,
        name=str(raw["name"]),
        subset=str(raw["subset"]),
        precision=str(raw["precision"]),
        rollouts=rollouts,
        steps=steps,
        tasks=tasks,
    )
