"""Versioned, hardware-aware GPU Job contracts."""

from .bundles import build_deterministic_bundle, validate_bundle
from .capabilities import detect_gpu_capabilities
from .contracts import (
    GpuCapabilities,
    GpuJobManifest,
    GpuJobResult,
    GpuReasonCode,
    GpuJobStage,
    GpuJobStatus,
    ResourceLimits,
)
from .sandbox import (
    DockerSandboxRuntime,
    PrivateEvaluationProfile,
    PrivateEvaluationProfileManifest,
    SandboxConfiguration,
    SandboxPolicy,
    SandboxStageExecutor,
    public_generated_feedback,
)

__all__ = [
    "GpuCapabilities",
    "GpuJobManifest",
    "GpuJobResult",
    "GpuReasonCode",
    "GpuJobStage",
    "GpuJobStatus",
    "ResourceLimits",
    "DockerSandboxRuntime",
    "PrivateEvaluationProfile",
    "PrivateEvaluationProfileManifest",
    "SandboxConfiguration",
    "SandboxPolicy",
    "SandboxStageExecutor",
    "build_deterministic_bundle",
    "detect_gpu_capabilities",
    "validate_bundle",
    "public_generated_feedback",
]
