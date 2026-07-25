# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ephemeral Docker sandbox for untrusted generated GPU candidates.

The Supervisor is the sole trusted component that talks to Docker and Control.
Untrusted containers receive immutable input files and a deliberately tiny
environment; they never receive a credential, host path, or Docker socket.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import PurePosixPath
import subprocess
import tarfile
import threading
import time
from typing import Any, Protocol
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .bundles import validate_bundle
from .contracts import (
    DIGEST_PATTERN,
    GpuCapabilities,
    GpuJobManifest,
    GpuJobResult,
    GpuJobStage,
    GpuJobStatus,
    GpuReasonCode,
)


_JOB_IMAGE_DIGEST = __import__("re").compile(
    r"^(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})$"
)
_SAFE_PATH = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")
_LABEL_PREFIX = "kernelblaster.sandbox"


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not _SAFE_PATH.fullmatch(value)
        or path.is_absolute()
        or ".." in path.parts
        or not path.parts
    ):
        raise ValueError("path must be a safe relative POSIX path")
    return path.as_posix()


class PrivateEvaluationProfile(BaseModel):
    """Supervisor-only mapping from a public profile ID to private test input."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    bundle_digest: str
    driver_path: str

    @field_validator("bundle_digest")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        if not DIGEST_PATTERN.fullmatch(value):
            raise ValueError("bundle_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator("driver_path")
    @classmethod
    def _safe_driver_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class PrivateEvaluationProfileManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "gpu-private-evaluation-profiles/v1"
    profiles: tuple[PrivateEvaluationProfile, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> "PrivateEvaluationProfileManifest":
        identifiers = [profile.id for profile in self.profiles]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("private evaluation profile IDs must be unique")
        return self

    @classmethod
    def load(cls, path: str) -> "PrivateEvaluationProfileManifest":
        payload = json.loads(open(path, encoding="utf-8").read())
        return cls.model_validate(payload)

    def get(self, identifier: str) -> PrivateEvaluationProfile:
        for profile in self.profiles:
            if profile.id == identifier:
                return profile
        raise KeyError("private_evaluation_profile_unknown")

    @property
    def ids(self) -> set[str]:
        return {profile.id for profile in self.profiles}


@dataclass(frozen=True)
class SandboxPolicy:
    """Fixed deployment policy; manifests cannot expand these limits."""

    cpu_nano: int = 2_000_000_000
    memory_bytes: int = 8 * 1024**3
    pids_limit: int = 64
    temporary_bytes: int = 512 * 1024**2
    stdout_bytes: int = 1024 * 1024
    stderr_bytes: int = 1024 * 1024
    executable_bytes: int = 256 * 1024**2
    compile_seconds: int = 180
    correctness_seconds: int = 60
    events_seconds: int = 90
    recovery_seconds: int = 60

    def timeout_for(self, stage: GpuJobStage) -> int:
        return {
            GpuJobStage.COMPILE: self.compile_seconds,
            GpuJobStage.CORRECTNESS: self.correctness_seconds,
            GpuJobStage.EVENTS: self.events_seconds,
        }[stage]

    def environment(self, target_arch: str) -> dict[str, str]:
        digits = target_arch.removeprefix("sm_")
        compute = digits[:-1] + "." + digits[-1]
        return {
            "HOME": "/work",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "TMPDIR": "/work/tmp",
            "CUDA_VISIBLE_DEVICES": "0",
            "TORCH_CUDA_ARCH_LIST": compute,
            "CMAKE_CUDA_ARCHITECTURES": digits,
            "CUDAARCHS": digits,
        }


@dataclass(frozen=True)
class SandboxConfiguration:
    image: str
    gpu_device: str
    profiles: PrivateEvaluationProfileManifest
    policy: SandboxPolicy = SandboxPolicy()

    @classmethod
    def from_environment(
        cls, environment: dict[str, str] | None = None
    ) -> "SandboxConfiguration":
        environment = environment or os.environ
        image = environment.get("KERNELBLASTER_GPU_JOB_IMAGE", "").strip()
        if not _JOB_IMAGE_DIGEST.fullmatch(image):
            raise RuntimeError("KERNELBLASTER_GPU_JOB_IMAGE must be an immutable sha256 digest")
        profile_path = environment.get(
            "KERNELBLASTER_PRIVATE_EVALUATION_PROFILES",
            "/run/kernelblaster/private-evaluation-profiles.json",
        ).strip()
        if not profile_path:
            raise RuntimeError("KERNELBLASTER_PRIVATE_EVALUATION_PROFILES is required")
        profiles = PrivateEvaluationProfileManifest.load(profile_path)
        if not profiles.profiles:
            raise RuntimeError("generated jobs require at least one private evaluation profile")
        return cls(
            image=image,
            gpu_device=environment.get("KERNELBLASTER_GPU_DEVICE", "0").strip(),
            profiles=profiles,
        )


@dataclass
class SandboxExecution:
    reason: GpuReasonCode
    outputs: dict[str, bytes]
    measurement: dict[str, object] | None = None


class SandboxRuntime(Protocol):
    def validate(self) -> None: ...

    def execute(
        self,
        *,
        input_archive: bytes,
        manifest: GpuJobManifest,
    ) -> SandboxExecution: ...

    def cancel_active(self) -> None: ...


def _tar_files(
    files: dict[str, bytes], *, mode: int = 0o444, executable: set[str] | None = None
) -> bytes:
    """Build a regular-file-only archive for Docker's put_archive API."""
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in sorted(files.items()):
            safe = _safe_relative_path(name)
            info = tarfile.TarInfo(safe)
            info.size = len(payload)
            info.mode = 0o555 if executable and safe in executable else mode
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            archive.addfile(info, BytesIO(payload))
    return output.getvalue()


def _bundle_files(payload: bytes, prefix: str) -> dict[str, bytes]:
    """Copy a validated source archive without ever extracting on the host."""
    names = validate_bundle(payload)
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
        for name in names:
            source = archive.extractfile(name)
            assert source is not None
            files[f"{prefix}/{name}"] = source.read()
    return files


def _output_names(stage: GpuJobStage) -> set[str]:
    names = {"stdout.log", "stderr.log", "result.json"}
    if stage is GpuJobStage.COMPILE:
        names.add("candidate")
    if stage is GpuJobStage.EVENTS:
        names.add("measurement.json")
    return names


def _read_output_archive(
    payload: bytes, *, stage: GpuJobStage, policy: SandboxPolicy
) -> dict[str, bytes]:
    """Reject output traversal, links, devices, unknown names, and oversize files."""
    allowed = _output_names(stage)
    limits = {
        "stdout.log": policy.stdout_bytes,
        "stderr.log": policy.stderr_bytes,
        "result.json": 64 * 1024,
        "measurement.json": policy.stdout_bytes,
        "candidate": policy.executable_bytes,
    }
    outputs: dict[str, bytes] = {}
    with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            parts = path.parts
            if parts and parts[0] == "out":
                parts = parts[1:]
            if not parts:
                continue
            name = PurePosixPath(*parts).as_posix()
            if path.is_absolute() or ".." in parts or name not in allowed:
                raise ValueError("sandbox output contains an unapproved path")
            if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                raise ValueError("sandbox output must contain regular files only")
            if name in outputs or member.size > limits[name]:
                raise ValueError("sandbox output is duplicate or exceeds its limit")
            source = archive.extractfile(member)
            assert source is not None
            outputs[name] = source.read()
    return outputs


def _reason_from_runner(payload: dict[str, object], stage: GpuJobStage) -> GpuReasonCode:
    raw = str(payload.get("reason", ""))
    try:
        return GpuReasonCode(raw)
    except ValueError:
        return {
            GpuJobStage.COMPILE: GpuReasonCode.COMPILE_FAILED,
            GpuJobStage.CORRECTNESS: GpuReasonCode.CORRECTNESS_FAILED,
            GpuJobStage.EVENTS: GpuReasonCode.EVENTS_FAILED,
        }[stage]


class DockerSandboxRuntime:
    """Synchronous Docker SDK adapter, isolated behind an injectable runtime."""

    def __init__(self, client: Any, configuration: SandboxConfiguration) -> None:
        self.client = client
        self.configuration = configuration
        self._active_lock = threading.Lock()
        self._active_job: Any | None = None

    @classmethod
    def from_environment(
        cls, environment: dict[str, str] | None = None
    ) -> "DockerSandboxRuntime":
        configuration = SandboxConfiguration.from_environment(environment)
        try:
            import docker
        except ImportError as error:  # pragma: no cover - deployment-only guard
            raise RuntimeError("Docker SDK is required for generated GPU jobs") from error
        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        return cls(client, configuration)

    def validate(self) -> None:
        self.client.ping()
        image = self.client.images.get(self.configuration.image)
        if self.configuration.image.startswith("sha256:"):
            if getattr(image, "id", "") != self.configuration.image:
                raise RuntimeError("configured job image ID does not match its digest")
        elif self.configuration.image not in set(image.attrs.get("RepoDigests") or []):
            raise RuntimeError("configured job image is not available at its pinned digest")

    def _labels(self, job_id: str) -> dict[str, str]:
        return {
            _LABEL_PREFIX: "true",
            f"{_LABEL_PREFIX}.job_id": job_id,
        }

    def _remove(self, resource: Any, *, force: bool = True) -> None:
        if resource is None:
            return
        try:
            resource.remove(force=force)
        except Exception:
            pass

    def _kill_and_wait(self, container: Any) -> None:
        try:
            container.kill(signal="SIGKILL")
        except Exception:
            pass
        try:
            container.wait(timeout=10)
        except Exception:
            pass

    def cancel_active(self) -> None:
        """Synchronously kill the running Job when its Supervisor task is cancelled."""
        with self._active_lock:
            active = self._active_job
        if active is not None:
            self._kill_and_wait(active)

    def _gpu_recovered(self) -> bool:
        deadline = time.monotonic() + self.configuration.policy.recovery_seconds
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.configuration.gpu_device}",
                        "--query-gpu=uuid",
                        "--format=csv,noheader",
                    ],
                    check=False,
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except (OSError, subprocess.SubprocessError):
                pass
            time.sleep(1)
        return False

    def execute(
        self, *, input_archive: bytes, manifest: GpuJobManifest
    ) -> SandboxExecution:
        """Run exactly one stage in a fresh container and remove all Docker state."""
        suffix = uuid.uuid4().hex
        labels = self._labels(manifest.job_id)
        volume = None
        stager = None
        job = None
        timed_out = False
        outputs: dict[str, bytes] = {}
        try:
            volume = self.client.volumes.create(
                name=f"kernelblaster-input-{suffix}", labels=labels
            )
            stager = self.client.containers.create(
                self.configuration.image,
                command=["/bin/true"],
                network_mode="none",
                read_only=True,
                user="0:0",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                volumes={volume.name: {"bind": "/input", "mode": "rw"}},
                labels=labels,
            )
            if not stager.put_archive("/input", input_archive):
                raise RuntimeError("Docker refused sandbox input archive")
            self._remove(stager)
            stager = None
            job = self.client.containers.create(
                self.configuration.image,
                detach=True,
                network_mode="none",
                ipc_mode="private",
                read_only=True,
                user="65532:65532",
                working_dir="/work",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                pids_limit=self.configuration.policy.pids_limit,
                mem_limit=self.configuration.policy.memory_bytes,
                nano_cpus=self.configuration.policy.cpu_nano,
                tmpfs={
                    "/work": f"rw,exec,nosuid,nodev,mode=1777,size={self.configuration.policy.temporary_bytes}",
                    "/tmp": "rw,nosuid,nodev,mode=1777,size=67108864",
                },
                volumes={volume.name: {"bind": "/input", "mode": "ro"}},
                environment=self.configuration.policy.environment(manifest.target_arch),
                device_requests=[
                    {"Driver": "", "Count": 0, "DeviceIDs": [self.configuration.gpu_device], "Capabilities": [["gpu"]]}
                ],
                log_config={"type": "none"},
                labels=labels,
            )
            with self._active_lock:
                self._active_job = job
            job.start()
            try:
                job.wait(timeout=self.configuration.policy.timeout_for(manifest.stage))
            except Exception as error:
                if isinstance(error, TimeoutError) or "timed out" in str(error).lower():
                    timed_out = True
                    self._kill_and_wait(job)
                else:
                    raise
            try:
                stream, _stat = job.get_archive("/work/out")
                output_archive = b"".join(stream)
                outputs = _read_output_archive(
                    output_archive, stage=manifest.stage, policy=self.configuration.policy
                )
            except Exception:
                outputs = {}

            if timed_out:
                reason = (
                    GpuReasonCode.STAGE_TIMEOUT
                    if self._gpu_recovered()
                    else GpuReasonCode.GPU_RECOVERY_FAILED
                )
                return SandboxExecution(reason=reason, outputs=outputs)

            try:
                job.reload()
                state = job.attrs.get("State", {})
            except Exception:
                state = {}
            if state.get("OOMKilled"):
                return SandboxExecution(reason=GpuReasonCode.GPU_OOM, outputs=outputs)
            result_raw = outputs.pop("result.json", None)
            if result_raw is None:
                return SandboxExecution(
                    reason=GpuReasonCode.SANDBOX_VIOLATION, outputs=outputs
                )
            try:
                result = json.loads(result_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return SandboxExecution(
                    reason=GpuReasonCode.SANDBOX_VIOLATION, outputs=outputs
                )
            reason = _reason_from_runner(result, manifest.stage)
            measurement = None
            if reason is GpuReasonCode.NONE and manifest.stage is GpuJobStage.EVENTS:
                raw_measurement = outputs.get("measurement.json")
                try:
                    candidate = json.loads((raw_measurement or b"").decode("utf-8"))
                    if not isinstance(candidate, dict):
                        raise ValueError
                    measurement = candidate
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    reason = GpuReasonCode.EVENTS_FAILED
            return SandboxExecution(reason=reason, outputs=outputs, measurement=measurement)
        finally:
            with self._active_lock:
                if self._active_job is job:
                    self._active_job = None
            self._remove(job)
            self._remove(stager)
            self._remove(volume)


class SandboxStageExecutor:
    """Materialize private inputs in Supervisor and import only approved outputs."""

    def __init__(
        self,
        control: Any,
        runtime: SandboxRuntime,
        profiles: PrivateEvaluationProfileManifest,
    ) -> None:
        self.control = control
        self.runtime = runtime
        self.profiles = profiles

    async def _download(self, digest: str) -> bytes:
        payload = await self.control.download(digest)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("artifact_digest_mismatch")
        return payload

    async def _input_archive(self, manifest: GpuJobManifest) -> bytes:
        assert manifest.private_evaluation_profile_id is not None
        profile = self.profiles.get(manifest.private_evaluation_profile_id)
        files: dict[str, bytes] = {}
        if manifest.stage is GpuJobStage.COMPILE:
            assert manifest.source_bundle_digest is not None
            source, private = await asyncio.gather(
                self._download(manifest.source_bundle_digest),
                self._download(profile.bundle_digest),
            )
            files.update(_bundle_files(source, "candidate"))
            private_files = _bundle_files(private, "private")
            if f"private/{profile.driver_path}" not in private_files:
                raise ValueError("private profile does not contain its declared driver")
            files.update(private_files)
        else:
            assert manifest.executable_digest is not None
            files["candidate/candidate"] = await self._download(manifest.executable_digest)
        files["request.json"] = json.dumps(
            {
                "stage": manifest.stage.value,
                "target_arch": manifest.target_arch,
                "benchmark_protocol_id": manifest.benchmark_protocol_id,
                "driver_path": f"private/{profile.driver_path}",
                "stdout_bytes": 1024 * 1024,
                "stderr_bytes": 1024 * 1024,
                "wall_seconds": {
                    "compile": 180,
                    "correctness": 60,
                    "events": 90,
                }[manifest.stage.value],
            },
            sort_keys=True,
        ).encode("utf-8")
        executable = {"candidate/candidate"} if manifest.stage is not GpuJobStage.COMPILE else set()
        return _tar_files(files, executable=executable)

    async def _upload(
        self, payload: bytes, *, media_type: str, schema: str
    ) -> str:
        expected = hashlib.sha256(payload).hexdigest()
        uploaded = await self.control.upload(payload, media_type=media_type, schema=schema)
        if uploaded.get("digest") != expected:
            raise ValueError("control artifact digest mismatch")
        return expected

    async def __call__(
        self,
        manifest: GpuJobManifest,
        capabilities: GpuCapabilities,
        cancelled: asyncio.Event,
    ) -> GpuJobResult:
        started = datetime.now(timezone.utc)
        artifacts: dict[str, str] = {}
        try:
            if cancelled.is_set():
                raise asyncio.CancelledError
            archive = await self._input_archive(manifest)
            try:
                execution = await asyncio.to_thread(
                    self.runtime.execute, input_archive=archive, manifest=manifest
                )
            except asyncio.CancelledError:
                cancel_active = getattr(self.runtime, "cancel_active", None)
                if callable(cancel_active):
                    await asyncio.shield(asyncio.to_thread(cancel_active))
                raise
            outputs = execution.outputs
            if manifest.stage is GpuJobStage.COMPILE and "candidate" in outputs:
                digest = await self._upload(
                    outputs["candidate"],
                    media_type="application/x-executable",
                    schema="gpu-executable/v1",
                )
                artifacts[digest] = "executable"
            if manifest.stage is GpuJobStage.COMPILE:
                logs = outputs.get("stdout.log", b"") + b"\n" + outputs.get("stderr.log", b"")
                if logs:
                    digest = await self._upload(
                        logs, media_type="text/plain", schema="gpu-compile-log/v1"
                    )
                    artifacts[digest] = "compile_log"
            else:
                for filename, role, media_type in (
                    ("stdout.log", f"{manifest.stage.value}_stdout", "application/json"),
                    ("stderr.log", f"{manifest.stage.value}_stderr", "text/plain"),
                ):
                    if filename in outputs:
                        digest = await self._upload(
                            outputs[filename],
                            media_type=media_type,
                            schema=f"gpu-{manifest.stage.value}-{filename}/v1",
                        )
                        artifacts[digest] = role
            status = (
                GpuJobStatus.SUCCEEDED
                if execution.reason is GpuReasonCode.NONE
                else (
                    GpuJobStatus.TIMED_OUT
                    if execution.reason is GpuReasonCode.STAGE_TIMEOUT
                    else GpuJobStatus.FAILED
                )
            )
            return GpuJobResult(
                job_id=manifest.job_id,
                run_id=manifest.run_id,
                stage=manifest.stage,
                status=status,
                reason_code=execution.reason,
                artifact_roles=artifacts,
                measurement=execution.measurement,
                hardware=capabilities.model_dump(mode="json"),
                started_at=started,
                finished_at=datetime.now(timezone.utc),
            )
        except asyncio.CancelledError:
            raise
        except KeyError:
            reason = GpuReasonCode.SANDBOX_VIOLATION
        except Exception:
            reason = GpuReasonCode.SANDBOX_UNAVAILABLE
        return GpuJobResult(
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            stage=manifest.stage,
            status=GpuJobStatus.FAILED,
            reason_code=reason,
            artifact_roles=artifacts,
            hardware=capabilities.model_dump(mode="json"),
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )


def public_generated_feedback(result: GpuJobResult) -> dict[str, object]:
    """The only generated-job payload suitable for an LLM feedback prompt."""
    return {
        "stage": result.stage.value,
        "status": result.status.value,
        "reason_code": result.reason_code.value,
        "measurement": result.measurement,
    }


__all__ = [
    "DockerSandboxRuntime",
    "PrivateEvaluationProfile",
    "PrivateEvaluationProfileManifest",
    "SandboxConfiguration",
    "SandboxExecution",
    "SandboxPolicy",
    "SandboxStageExecutor",
    "public_generated_feedback",
]
