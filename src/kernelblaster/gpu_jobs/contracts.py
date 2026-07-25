"""Strict schemas shared by Control and the trusted GPU Supervisor."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARCH_PATTERN = re.compile(r"^sm_[0-9]{2,3}$")


class GpuJobStage(str, Enum):
    COMPILE = "compile"
    CORRECTNESS = "correctness"
    EVENTS = "events"


class GpuJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class GpuReasonCode(str, Enum):
    NONE = "none"
    TRUSTED_EXECUTOR_NOT_CONFIGURED = "trusted_executor_not_configured"
    COMPILE_FAILED = "compile_failed"
    CORRECTNESS_FAILED = "correctness_failed"
    EVENTS_FAILED = "events_failed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"
    CONTROL_CALLBACK_FAILED = "control_callback_failed"
    STAGE_TIMEOUT = "stage_timeout"
    GPU_OOM = "gpu_oom"
    SANDBOX_VIOLATION = "sandbox_violation"
    GPU_RECOVERY_FAILED = "gpu_recovery_failed"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"


class ResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_seconds: int = Field(default=300, ge=1, le=3600)
    stdout_bytes: int = Field(default=1024 * 1024, ge=1024, le=16 * 1024 * 1024)
    stderr_bytes: int = Field(default=1024 * 1024, ge=1024, le=16 * 1024 * 1024)
    temporary_bytes: int = Field(default=512 * 1024 * 1024, ge=1024 * 1024, le=4 * 1024**3)


class GpuJobManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["gpu-job/v1"] = "gpu-job/v1"
    job_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=256)
    stage: GpuJobStage
    source_bundle_digest: str | None = None
    driver_digest: str | None = None
    executable_digest: str | None = None
    private_evaluation_profile_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    target_arch: str
    benchmark_protocol_id: str = Field(min_length=1, max_length=128)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    deadline: datetime
    trusted_bundle_kind: Literal["trusted_smoke_v1", "generated_v1"] = "trusted_smoke_v1"

    @model_validator(mode="after")
    def validate_stage_inputs(self) -> "GpuJobManifest":
        for name in ("source_bundle_digest", "driver_digest", "executable_digest"):
            value = getattr(self, name)
            if value is not None and not DIGEST_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not ARCH_PATTERN.fullmatch(self.target_arch):
            raise ValueError("target_arch must use sm_XX or sm_XXX notation")
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        if self.deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc):
            raise ValueError("deadline_exceeded")
        if self.trusted_bundle_kind == "generated_v1":
            if self.private_evaluation_profile_id is None:
                raise ValueError("generated jobs require private_evaluation_profile_id")
            if self.driver_digest is not None:
                raise ValueError("generated jobs may not accept driver_digest")
        elif self.private_evaluation_profile_id is not None:
            raise ValueError("trusted jobs may not accept private_evaluation_profile_id")

        if self.stage is GpuJobStage.COMPILE:
            if self.source_bundle_digest is None:
                raise ValueError("compile requires source_bundle_digest")
            if (
                self.trusted_bundle_kind == "trusted_smoke_v1"
                and self.driver_digest is None
            ):
                raise ValueError("trusted compile requires driver_digest")
            if self.executable_digest is not None:
                raise ValueError("compile may not accept executable_digest")
        elif self.stage is GpuJobStage.CORRECTNESS:
            if self.executable_digest is None:
                raise ValueError("correctness requires executable_digest")
            if (
                self.trusted_bundle_kind == "trusted_smoke_v1"
                and self.driver_digest is None
            ):
                raise ValueError("trusted correctness requires driver_digest")
        elif self.stage is GpuJobStage.EVENTS and self.executable_digest is None:
            raise ValueError("events requires executable_digest")
        return self

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def input_digests(self) -> tuple[str, ...]:
        return tuple(
            digest
            for digest in (
                self.source_bundle_digest,
                self.driver_digest if self.trusted_bundle_kind == "trusted_smoke_v1" else None,
                self.executable_digest,
            )
            if digest is not None
        )


class GpuDeviceCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logical_id: str
    name: str
    compute_capability: str
    target_arch: str
    total_memory_bytes: int = Field(ge=1)


class GpuRuntimeCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cuda_version: str
    driver_version: str
    image_digest: str | None = None


class GpuCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["gpu-capabilities/v1"] = "gpu-capabilities/v1"
    supervisor_id: str
    device: GpuDeviceCapability
    runtime: GpuRuntimeCapability
    supported_stages: tuple[GpuJobStage, ...] = (
        GpuJobStage.COMPILE,
        GpuJobStage.CORRECTNESS,
        GpuJobStage.EVENTS,
    )
    max_concurrent_jobs: Literal[1] = 1
    generated_jobs_enabled: bool = False


class GpuJobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["gpu-result/v1"] = "gpu-result/v1"
    job_id: str
    run_id: str
    stage: GpuJobStage
    status: GpuJobStatus
    reason_code: GpuReasonCode = GpuReasonCode.NONE
    artifact_roles: dict[str, str] = Field(default_factory=dict)
    measurement: dict[str, Any] | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
