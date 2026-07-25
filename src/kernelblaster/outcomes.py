
"""定义工作流终态及跨模块传递的标准运行结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .measurements import Measurement


class RunStatus(str, Enum):
    """一次 KernelBlaster 工作流调用的标准终态。"""

    IMPROVED = "improved"
    NO_IMPROVEMENT = "no_improvement"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class ExecutionStatus(str, Enum):
    NOT_RUN = "not_run"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BLOCKED = "blocked"


class CorrectnessStatus(str, Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class TimingStatus(str, Enum):
    NOT_RUN = "not_run"
    MEASURED = "measured"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class DiagnosticStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"


class ReasonCode(str, Enum):
    NONE = "none"
    COMPILE_FAILED = "compile_failed"
    EXECUTION_FAILED = "execution_failed"
    CORRECTNESS_FAILED = "correctness_failed"
    PROFILER_UNAVAILABLE = "profiler_unavailable"
    NCU_PERMISSION_DENIED = "ncu_permission_denied"
    TIMING_INVALID = "timing_invalid"
    MEASUREMENT_INCOMPARABLE = "measurement_incomparable"
    PERFORMANCE_GATE_FAILED = "performance_gate_failed"
    LEGACY_INFERRED_UNIT = "legacy_inferred_unit"


@dataclass(frozen=True)
class RunOutcome:
    """保存工作流终态、诊断原因、性能指标和可选成功产物。"""

    status: RunStatus
    artifact_path: Path | None = None
    reason: str | None = None
    profiling_mode: str | None = None
    measurement: Measurement | None = None
    execution_status: ExecutionStatus = ExecutionStatus.NOT_RUN
    correctness_status: CorrectnessStatus = CorrectnessStatus.NOT_RUN
    timing_status: TimingStatus = TimingStatus.NOT_RUN
    diagnostic_status: DiagnosticStatus = DiagnosticStatus.NOT_REQUESTED
    reason_code: ReasonCode = ReasonCode.NONE
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """
        处理 `success` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return bool(
            self.status is RunStatus.IMPROVED
            and self.artifact_path is not None
            and self.artifact_path.is_file()
        )

    def to_dict(self) -> dict[str, Any]:
        """
        处理 `to_dict` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["execution_status"] = self.execution_status.value
        payload["correctness_status"] = self.correctness_status.value
        payload["timing_status"] = self.timing_status.value
        payload["diagnostic_status"] = self.diagnostic_status.value
        payload["reason_code"] = self.reason_code.value
        payload["measurement"] = (
            self.measurement.to_dict() if self.measurement is not None else None
        )
        payload["artifact_path"] = (
            str(self.artifact_path) if self.artifact_path is not None else None
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunOutcome":
        """
        处理 `from_dict` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            payload: 跨接口传递的序列化载荷。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        artifact = payload.get("artifact_path")
        measurement_payload = payload.get("measurement")
        measurement = (
            Measurement.from_dict(measurement_payload)
            if isinstance(measurement_payload, dict)
            else None
        )
        return cls(
            status=RunStatus(payload["status"]),
            artifact_path=Path(artifact) if artifact else None,
            reason=payload.get("reason"),
            profiling_mode=payload.get("profiling_mode"),
            measurement=measurement,
            execution_status=ExecutionStatus(
                payload.get("execution_status", ExecutionStatus.NOT_RUN.value)
            ),
            correctness_status=CorrectnessStatus(
                payload.get("correctness_status", CorrectnessStatus.NOT_RUN.value)
            ),
            timing_status=TimingStatus(
                payload.get("timing_status", TimingStatus.NOT_RUN.value)
            ),
            diagnostic_status=DiagnosticStatus(
                payload.get("diagnostic_status", DiagnosticStatus.NOT_REQUESTED.value)
            ),
            reason_code=ReasonCode(payload.get("reason_code", ReasonCode.NONE.value)),
            metrics=dict(payload.get("metrics") or {}),
        )


__all__ = [
    "CorrectnessStatus",
    "DiagnosticStatus",
    "ExecutionStatus",
    "ReasonCode",
    "RunOutcome",
    "RunStatus",
    "TimingStatus",
]
