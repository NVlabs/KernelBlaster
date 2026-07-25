# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
用于研究 CUDA 候选人的机器可读执行合同。

原始的“`launch_gpu_implementation`”ABI 故意保持很小并且不能
检查张量元数据。调用者必须在创建之前在此处验证请求
产物、编译代码或初始化 CUDA。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CAPABILITY_SCHEMA_VERSION = "2.0"
CAPABILITY_MARKER = "KERNELBLASTER_CAPABILITY_JSON "
HARDENED_TASK_IDS = frozenset({"004", "007", "036", "040", "095"})


@dataclass(frozen=True)
class CapabilityResult:
    """验证一个候选调用的结果。"""

    supported: bool
    task_id: str | None
    request: dict[str, Any]
    reason_code: str | None = None
    supported_values: dict[str, Any] | None = None

    @property
    def exit_code(self) -> int:
        """
        处理 `exit_code` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        if self.supported:
            return 0
        if self.reason_code in {"invalid_request", "unknown_task"}:
            return 2
        return 5

    def to_dict(self) -> dict[str, Any]:
        """
        处理 `to_dict` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        payload: dict[str, Any] = {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "supported": self.supported,
            "task_id": self.task_id,
            "request": self.request,
            "reason_code": self.reason_code,
        }
        if self.supported_values is not None:
            payload["supported_values"] = self.supported_values
        return payload


def load_capability_manifest(path: Path) -> dict[str, Any]:
    """
    加载并从结构上验证候选人能力清单。

    参数:
        path: 待读取、写入或校验的文件系统路径。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CAPABILITY_SCHEMA_VERSION:
        raise ValueError(
            f"Candidate manifest must use schema_version {CAPABILITY_SCHEMA_VERSION}."
        )
    contract = payload.get("runtime_contract")
    if not isinstance(contract, dict):
        raise ValueError("Candidate manifest is missing runtime_contract.")
    required_contract = {
        "device",
        "gpu_architectures",
        "input_dtype",
        "accumulation_dtype",
        "layout",
        "stream_mode",
        "max_streams",
        "graph_capture",
        "directions",
        "backward",
        "fallback",
        "production_ready",
    }
    missing_contract = sorted(required_contract - set(contract))
    if missing_contract:
        raise ValueError(
            f"Candidate runtime_contract is missing fields: {missing_contract}"
        )
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Candidate manifest tasks must be a list.")
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Candidate manifest task entries must be objects.")
        task_id = str(task.get("id", ""))
        if not task_id or task_id in seen:
            raise ValueError(f"Missing or duplicate candidate task ID: {task_id!r}")
        seen.add(task_id)
        cases = task.get("supported_cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"Candidate {task_id} has no supported_cases.")
        case_ids: set[str] = set()
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("shape"), dict):
                raise ValueError(f"Candidate {task_id} has an invalid supported case.")
            case_id = str(case.get("case_id", ""))
            if not case_id or case_id in case_ids:
                raise ValueError(
                    f"Candidate {task_id} has a missing or duplicate case_id."
                )
            if not all(
                isinstance(name, str)
                and name
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for name, value in case["shape"].items()
            ):
                raise ValueError(f"Candidate {task_id} has an invalid case shape.")
            case_ids.add(case_id)
        if task_id in HARDENED_TASK_IDS:
            for field in (
                "numerics_profile",
                "resource_policy",
                "reentrant_under_contract",
                "requires_prewarm",
            ):
                if field not in task:
                    raise ValueError(f"Candidate {task_id} is missing {field}.")
    return payload


def task_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """
    处理 `task_map` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        manifest: 调用方提供的 `manifest` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return {str(task["id"]): task for task in manifest["tasks"]}


def hardened_task_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """
    仅返回 schema-v2 执行合约涵盖的候选者。

    参数:
        manifest: 调用方提供的 `manifest` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """

    return {
        task_id: task
        for task_id, task in task_map(manifest).items()
        if task.get("capability_status") == "hardened"
    }


def supported_values(
    manifest: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    """
    处理 `supported_values` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        manifest: 调用方提供的 `manifest` 参数。
        task: 调用方提供的 `task` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    contract = manifest["runtime_contract"]
    values: dict[str, Any] = {
        "gpu_architectures": list(contract["gpu_architectures"]),
        "input_dtypes": [contract["input_dtype"]],
        "layouts": [contract["layout"]],
        "stream_modes": [contract["stream_mode"]],
        "max_streams": contract["max_streams"],
        "graph_capture": contract["graph_capture"],
        "directions": list(contract["directions"]),
        "shapes": [case["shape"] for case in task["supported_cases"]],
        "case_ids": [case["case_id"] for case in task["supported_cases"]],
    }
    if "target_dtype" in task:
        values["target_dtypes"] = [task["target_dtype"]]
    return values


def validate_candidate_request(
    manifest: Mapping[str, Any],
    task_id: str | None,
    request: Mapping[str, Any],
) -> CapabilityResult:
    """
    按照记录的原因代码优先顺序进行验证。

    参数:
        manifest: 调用方提供的 `manifest` 参数。
        task_id: 调用方分配的任务唯一标识。
        request: 经过类型约束的服务请求对象。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """

    normalized = dict(request)
    tasks = hardened_task_map(manifest)
    required_types = {
        "arch": str,
        "dtype": str,
        "layout": str,
        "stream_mode": str,
        "stream_count": int,
        "graph_capture": bool,
        "backward": bool,
        "shape": dict,
    }
    malformed = task_id is None or any(
        key not in normalized
        or not isinstance(normalized[key], expected)
        or (expected is int and isinstance(normalized[key], bool))
        for key, expected in required_types.items()
    )
    shape = normalized.get("shape")
    malformed = malformed or not isinstance(shape, dict) or not shape or any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for name, value in (shape.items() if isinstance(shape, dict) else ())
    )
    if isinstance(normalized.get("stream_count"), int):
        malformed = malformed or normalized["stream_count"] < 1
    if malformed:
        return CapabilityResult(False, task_id, normalized, "invalid_request")
    if task_id not in tasks:
        return CapabilityResult(False, task_id, normalized, "unknown_task")

    task = tasks[task_id]
    if "target_dtype" in task and (
        "target_dtype" not in normalized
        or not isinstance(normalized["target_dtype"], str)
        or not normalized["target_dtype"].strip()
    ):
        return CapabilityResult(False, task_id, normalized, "invalid_request")
    values = supported_values(manifest, task)
    contract = manifest["runtime_contract"]
    if normalized["arch"] not in contract["gpu_architectures"]:
        reason = "unsupported_arch"
    elif normalized["backward"] or "inference_forward" not in contract["directions"]:
        reason = "unsupported_backward"
    elif normalized["dtype"] != contract["input_dtype"]:
        reason = "unsupported_dtype"
    elif "target_dtype" in task and normalized.get("target_dtype") != task["target_dtype"]:
        reason = "unsupported_target_dtype"
    elif normalized["layout"] != contract["layout"]:
        reason = "unsupported_layout"
    elif (
        normalized["stream_mode"] != contract["stream_mode"]
        or normalized["stream_count"] > contract["max_streams"]
    ):
        reason = "unsupported_stream"
    elif normalized["graph_capture"] is not contract["graph_capture"]:
        reason = "unsupported_graph_capture"
    elif normalized["shape"] not in values["shapes"]:
        reason = "unsupported_shape"
    else:
        return CapabilityResult(True, task_id, normalized, supported_values=values)
    return CapabilityResult(False, task_id, normalized, reason, values)


def canonical_shape(task: Mapping[str, Any]) -> dict[str, int]:
    """
    处理 `canonical_shape` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        task: 调用方提供的 `task` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    for case in task["supported_cases"]:
        if case["case_id"] == "canonical":
            return dict(case["shape"])
    raise ValueError(f"Candidate {task.get('id')} has no canonical case.")


def parse_shape(value: str, task: Mapping[str, Any]) -> dict[str, int]:
    """
    解析“`canonical``, a case id, JSON, or ``name=value`”尺寸。

    参数:
        value: 需要转换、保存或校验的值。
        task: 调用方提供的 `task` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """

    for case in task["supported_cases"]:
        if value == case["case_id"]:
            return dict(case["shape"])
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    dimensions: dict[str, int] = {}
    try:
        for item in value.split(","):
            name, raw = item.split("=", 1)
            dimensions[name.strip()] = int(raw)
    except (TypeError, ValueError):
        return {}
    return dimensions


def describe_capabilities(
    manifest: Mapping[str, Any], task_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """
    描述 `describe_capabilities` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        manifest: 调用方提供的 `manifest` 参数。
        task_ids: 调用方提供的 `task_ids` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    tasks = hardened_task_map(manifest)
    selected = list(task_ids) if task_ids else list(tasks)
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "runtime_contract": manifest["runtime_contract"],
        "numerics_profiles": manifest.get("numerics_profiles", {}),
        "tasks": [tasks[task_id] for task_id in selected if task_id in tasks],
        "unknown_tasks": [task_id for task_id in selected if task_id not in tasks],
    }
