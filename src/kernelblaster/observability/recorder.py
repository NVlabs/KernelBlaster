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

"""以可复现和可脱敏的方式记录配置、Prompt、事件、产物及源码指纹。"""

from __future__ import annotations

import atexit
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import threading
from typing import Any
import uuid

from ..measurements import Measurement
from ..storage import StateStore


SCHEMA_VERSION = "3.0"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_SAFE_USAGE_KEYS = {
    "api_key_configured",
    "cached_tokens",
    "cache_write_tokens",
    "completion_tokens",
    "input_tokens",
    "max_completion_tokens",
    "max_total_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_tokens",
    "total_tokens",
}
_BEARER_PATTERN = re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]+")
_AUTH_PATTERN = re.compile(
    r"(?i)(\bauthorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;}]+"
)
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&#\s]+"
)
_URL_USERINFO_PATTERN = re.compile(r"(?i)(https?://)[^/@\s]+@")


def utc_now() -> str:
    """
    处理 `utc_now` 对应的领域操作，并返回调用方所需的标准化结果。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_secrets(value: Any, key: str = "") -> Any:
    """
    递归地编辑字段名称通常包含秘密的值。

    参数:
        value: 需要转换、保存或校验的值。
        key: 调用方提供的 `key` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    normalized_key = key.lower()
    if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
        if normalized_key in _SAFE_USAGE_KEYS:
            return value
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_secrets(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item, key) for item in value]
    if isinstance(value, str):
        redacted = _BEARER_PATTERN.sub(r"\1[REDACTED]", value)
        redacted = _AUTH_PATTERN.sub(r"\1[REDACTED]", redacted)
        redacted = _QUERY_SECRET_PATTERN.sub(r"\1[REDACTED]", redacted)
        return _URL_USERINFO_PATTERN.sub(r"\1[REDACTED]@", redacted)
    return value


def _serialize_measurements(value: Any) -> Any:
    if isinstance(value, Measurement):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _serialize_measurements(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_measurements(item) for item in value]
    return value


def prompt_metadata(messages: list[dict], include_content: bool = False) -> dict[str, Any]:
    """
    处理 `prompt_metadata` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        messages: 按对话顺序排列的 LLM 消息。
        include_content: 调用方提供的 `include_content` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    canonical = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    metadata: dict[str, Any] = {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "character_count": sum(
            len(str(message.get("content", ""))) for message in messages
        ),
        "message_count": len(messages),
    }
    if include_content:
        metadata["messages"] = redact_secrets(deepcopy(messages))
    return metadata


def _git_commit(repo_root: Path) -> str | None:
    """
    处理 `git_commit` 所表示的内部步骤；该函数不属于稳定的公开接口。

    参数:
        repo_root: 调用方提供的 `repo_root` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _source_tree_sha256(repo_root: Path) -> str | None:
    """
    哈希跟踪工作树内容，包括本地修改。

    参数:
        repo_root: 调用方提供的 `repo_root` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        digest = hashlib.sha256()
        for raw_path in completed.stdout.split(b"\0"):
            if not raw_path:
                continue
            relative = raw_path.decode("utf-8", errors="surrogateescape")
            path = repo_root / relative
            if not path.is_file():
                continue
            digest.update(raw_path)
            digest.update(b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """
    处理 `atomic_json_write` 所表示的内部步骤；该函数不属于稳定的公开接口。

    参数:
        path: 待读取、写入或校验的文件系统路径。
        payload: 跨接口传递的序列化载荷。
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RunRecorder:
    """具有清单和摘要快照的仅附加事件记录器。"""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model: str,
        provider_config: dict[str, Any],
        suite: dict[str, Any] | None = None,
        gpu_target: str | None = None,
        run_id: str | None = None,
        dry_run: bool = False,
        repo_root: str | Path | None = None,
        state_store: StateStore | None = None,
    ) -> None:
        """
        初始化 RunRecorder 实例，并保存后续流程所需的配置与依赖。

        参数:
            output_dir: 调用方提供的 `output_dir` 参数。
            model: 生成候选时使用的模型标识。
            provider_config: 调用方提供的 `provider_config` 参数。
            suite: 调用方提供的 `suite` 参数。
            gpu_target: 调用方提供的 `gpu_target` 参数。
            run_id: 调用方提供的 `run_id` 参数。
            dry_run: 调用方提供的 `dry_run` 参数。
            repo_root: 调用方提供的 `repo_root` 参数。
        """
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.events_path = self.output_dir / "events.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.run_id = run_id or uuid.uuid4().hex
        self.state_store = state_store
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._started_at = utc_now()
        self._summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "dry_run" if dry_run else "running",
            "started_at": self._started_at,
            "finished_at": None,
            "llm": {
                "requests_started": 0,
                "requests_completed": 0,
                "requests_failed": 0,
                "retries": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
            },
            "cuda": {
                "compilations": 0,
                "correctness_checks": 0,
                "profiles": 0,
                "validation_status": "NOT RUN",
                "performance_results": "pending",
            },
            "tasks": {
                "total": 0,
                "by_outcome": {},
                "profiling_modes": {},
                "results": [],
            },
            "errors": 0,
        }

        root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": self._started_at,
            "status": "dry_run" if dry_run else "running",
            "source": {
                "git_commit": _git_commit(root),
                "tree_sha256": _source_tree_sha256(root),
            },
            "model": model,
            "provider": redact_secrets(provider_config),
            "suite": redact_secrets(suite or {}),
            "target": {"gpu": gpu_target},
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "machine": platform.machine(),
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            },
            "validation": {
                "cuda": "NOT RUN",
                "llm_smoke_test": "NOT RUN",
                "performance_results": "pending",
                "gates": {
                    "environment": "NOT RUN",
                    "compile": "NOT RUN",
                    "correctness": "NOT RUN",
                    "events_stability": "NOT RUN",
                    "ncu_permission": "NOT RUN",
                    "api_smoke": "NOT RUN",
                },
            },
            "budget": {
                "limits": {
                    key: provider_config.get(key)
                    for key in (
                        "max_requests",
                        "max_total_tokens",
                        "max_completion_tokens",
                        "max_concurrency",
                    )
                },
                "consumed": {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
            "outcomes": [],
            "profiling_modes": [],
            "failure_classification": {},
        }
        _atomic_json_write(self.manifest_path, manifest)
        self.events_path.touch(exist_ok=True)
        _atomic_json_write(self.summary_path, self._summary)
        if self.state_store is not None:
            self.state_store.repository.create_run(
                self.run_id,
                status=self._summary["status"],
                metadata={
                    "model": model,
                    "suite": redact_secrets(suite or {}),
                    "target": {"gpu": gpu_target},
                    "manifest_path": self.manifest_path.name,
                },
            )
        atexit.register(self.close)

    def record_event(
        self,
        event_type: str,
        *,
        status: str = "ok",
        task_id: str | int | None = None,
        rollout_id: str | int | None = None,
        stage: str | None = None,
        candidate_id: str | None = None,
        attempt: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        记录 `record_event` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            event_type: 调用方提供的 `event_type` 参数。
            status: 调用方提供的 `status` 参数。
            task_id: 调用方分配的任务唯一标识。
            rollout_id: 调用方提供的 `rollout_id` 参数。
            stage: 调用方提供的 `stage` 参数。
            candidate_id: 调用方提供的 `candidate_id` 参数。
            attempt: 调用方提供的 `attempt` 参数。
            data: 待处理的结构化数据。
        """
        with self._lock:
            if self._closed:
                return
            self._sequence += 1
            event = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "timestamp": utc_now(),
                "event_type": event_type,
                "status": status,
                "task_id": task_id,
                "rollout_id": rollout_id,
                "stage": stage,
                "candidate_id": candidate_id,
                "attempt": attempt,
                "data": redact_secrets(_serialize_measurements(data or {})),
            }
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
            self._update_summary(event_type, status, event["data"])
            _atomic_json_write(self.summary_path, self._summary)

    def close(self, status: str | None = None) -> None:
        """
        处理 `close` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            status: 调用方提供的 `status` 参数。
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if status:
                self._summary["status"] = status
            elif self._summary["status"] == "running":
                self._summary["status"] = (
                    "completed_with_errors" if self._summary["errors"] else "completed"
                )
            self._summary["finished_at"] = utc_now()
            cuda_summary = self._summary["cuda"]
            if cuda_summary["correctness_checks"]:
                cuda_summary["validation_status"] = "RUN"
            if cuda_summary["profiles"]:
                cuda_summary["performance_results"] = "collected_pending_analysis"
            _atomic_json_write(self.summary_path, self._summary)

            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = self._summary["status"]
            manifest["finished_at"] = self._summary["finished_at"]
            if cuda_summary["correctness_checks"]:
                manifest["validation"]["cuda"] = "RUN"
            llm_summary = self._summary["llm"]
            if llm_summary["requests_failed"]:
                manifest["validation"]["llm_smoke_test"] = "FAILED"
            elif llm_summary["requests_completed"]:
                manifest["validation"]["llm_smoke_test"] = "RUN"
            if cuda_summary["profiles"]:
                manifest["validation"][
                    "performance_results"
                ] = "collected_pending_analysis"
            tasks = self._summary["tasks"]
            manifest["outcomes"] = tasks["results"]
            manifest["profiling_modes"] = sorted(tasks["profiling_modes"])
            manifest["failure_classification"] = {
                status: count
                for status, count in tasks["by_outcome"].items()
                if status in {"failed", "timeout", "blocked"}
            }
            manifest["budget"]["consumed"] = {
                # 失败的 HTTP 尝试仍然会消耗请求预算。这
                # 提供者每次尝试都会发出一个 request_started 事件，
                # 包括重试，因此这是可审计的硬上限计数。
                "requests": llm_summary["requests_started"],
                "prompt_tokens": llm_summary["prompt_tokens"],
                "completion_tokens": llm_summary["completion_tokens"],
                "total_tokens": llm_summary["total_tokens"],
            }
            gates = manifest["validation"]["gates"]
            if cuda_summary["compilations"]:
                gates["compile"] = "RUN"
            if cuda_summary["correctness_checks"]:
                gates["correctness"] = "RUN"
            if "events_only" in tasks["profiling_modes"]:
                gates["events_stability"] = "RUN"
            if "ncu" in tasks["profiling_modes"]:
                gates["ncu_permission"] = "RUN"
            if llm_summary["requests_failed"]:
                gates["api_smoke"] = "FAILED"
            elif llm_summary["requests_completed"]:
                gates["api_smoke"] = "RUN"
            _atomic_json_write(self.manifest_path, manifest)
            if self.state_store is not None:
                for role, path, media_type in (
                    ("manifest", self.manifest_path, "application/json"),
                    ("events", self.events_path, "application/x-ndjson"),
                    ("summary", self.summary_path, "application/json"),
                ):
                    artifact = self.state_store.cas.put_file(
                        path,
                        media_type=media_type,
                        producer="run_recorder",
                        schema=SCHEMA_VERSION,
                    )
                    self.state_store.repository.register_artifact(artifact)
                    self.state_store.repository.link_run_artifact(
                        run_id=self.run_id,
                        digest=artifact.digest,
                        role=role,
                    )
                self.state_store.repository.finish_run(self.run_id, self._summary["status"])

    def _update_summary(
        self,
        event_type: str,
        status: str,
        data: dict[str, Any],
    ) -> None:
        """
        更新 `update_summary` 所表示的内部步骤；该函数不属于稳定的公开接口。

        参数:
            event_type: 调用方提供的 `event_type` 参数。
            status: 调用方提供的 `status` 参数。
            data: 待处理的结构化数据。
        """
        llm = self._summary["llm"]
        if event_type == "llm_request_started":
            llm["requests_started"] += 1
        elif event_type == "llm_request_completed":
            llm["requests_completed"] += 1
            usage = data.get("usage", {})
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                llm[key] += int(usage.get(key, 0) or 0)
            llm["latency_seconds"] += float(data.get("latency_seconds", 0) or 0)
        elif event_type == "llm_request_failed":
            llm["requests_failed"] += 1
        elif event_type == "llm_retry_scheduled":
            llm["retries"] += 1
        elif event_type == "cuda_compile_completed":
            self._summary["cuda"]["compilations"] += 1
        elif event_type == "cuda_correctness_completed":
            self._summary["cuda"]["correctness_checks"] += 1
        elif event_type == "cuda_profile_completed":
            self._summary["cuda"]["profiles"] += 1
        elif event_type == "task_outcome":
            tasks = self._summary["tasks"]
            outcome = str(data.get("outcome", "failed"))
            profiling_mode = str(data.get("profiling_mode", "unknown"))
            tasks["total"] += 1
            tasks["by_outcome"][outcome] = tasks["by_outcome"].get(outcome, 0) + 1
            tasks["profiling_modes"][profiling_mode] = (
                tasks["profiling_modes"].get(profiling_mode, 0) + 1
            )
            tasks["results"].append(
                {
                    "task_id": data.get("task_id"),
                    "outcome": outcome,
                    "profiling_mode": profiling_mode,
                    "reason": data.get("reason"),
                    "reason_code": data.get("reason_code", "none"),
                    "execution_status": data.get("execution_status", "not_run"),
                    "correctness_status": data.get("correctness_status", "not_run"),
                    "timing_status": data.get("timing_status", "not_run"),
                    "diagnostic_status": data.get("diagnostic_status", "not_requested"),
                    "measurement": data.get("measurement"),
                    "metrics": data.get("metrics", {}),
                }
            )

        if status == "error":
            self._summary["errors"] += 1
