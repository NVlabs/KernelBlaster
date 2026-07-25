"""Trusted, single-GPU Supervisor exposing the versioned Job API."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI, Header, HTTPException
import uvicorn

from .capabilities import detect_gpu_capabilities
from .contracts import (
    GpuCapabilities,
    GpuJobManifest,
    GpuJobResult,
    GpuJobStatus,
    GpuReasonCode,
)


Executor = Callable[[GpuJobManifest, GpuCapabilities, asyncio.Event], Awaitable[GpuJobResult]]


async def _safe_default_executor(
    manifest: GpuJobManifest,
    capabilities: GpuCapabilities,
    cancelled: asyncio.Event,
) -> GpuJobResult:
    """Fail closed until a trusted stage executor is explicitly installed."""
    del cancelled
    return GpuJobResult(
        job_id=manifest.job_id,
        run_id=manifest.run_id,
        stage=manifest.stage,
        status=GpuJobStatus.BLOCKED,
        reason_code=GpuReasonCode.TRUSTED_EXECUTOR_NOT_CONFIGURED,
        hardware=capabilities.model_dump(mode="json"),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )


@dataclass
class SupervisorJob:
    manifest: GpuJobManifest
    manifest_digest: str
    status: GpuJobStatus = GpuJobStatus.QUEUED
    result: GpuJobResult | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    lease_id: str | None = None

    def response(self) -> dict[str, object]:
        return {
            "job_id": self.manifest.job_id,
            "run_id": self.manifest.run_id,
            "stage": self.manifest.stage.value,
            "status": self.status.value,
            "deadline": self.manifest.deadline.isoformat(),
            "manifest_digest": self.manifest_digest,
            "result": self.result.model_dump(mode="json") if self.result is not None else None,
        }


class InMemoryGpuSupervisor:
    """Bounded active/recent view; Control SQLite remains authoritative."""

    def __init__(
        self,
        capabilities: GpuCapabilities,
        *,
        executor: Executor = _safe_default_executor,
        trusted_source_digests: set[str] | None = None,
        benchmark_protocol_ids: set[str] | None = None,
        generated_profile_ids: set[str] | None = None,
        recent_limit: int = 256,
        reporter: Any | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.executor = executor
        self.trusted_source_digests = trusted_source_digests or set()
        self.benchmark_protocol_ids = benchmark_protocol_ids or {"trusted-smoke-v1"}
        self.generated_profile_ids = generated_profile_ids or set()
        self.recent_limit = recent_limit
        self.reporter = reporter
        self._jobs: dict[str, SupervisorJob] = {}
        self._idempotency: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()
        self._gpu = asyncio.Semaphore(1)

    def _validate_manifest(self, manifest: GpuJobManifest) -> None:
        if manifest.target_arch != self.capabilities.device.target_arch:
            raise ValueError("target_arch_mismatch")
        if manifest.stage not in self.capabilities.supported_stages:
            raise ValueError("stage_not_supported")
        if manifest.benchmark_protocol_id not in self.benchmark_protocol_ids:
            raise ValueError("benchmark_protocol_unknown")
        if (
            manifest.trusted_bundle_kind == "generated_v1"
            and not self.capabilities.generated_jobs_enabled
        ):
            raise ValueError("generated_jobs_disabled")
        if (
            manifest.trusted_bundle_kind == "generated_v1"
            and manifest.private_evaluation_profile_id not in self.generated_profile_ids
        ):
            raise ValueError("private_evaluation_profile_unknown")
        if manifest.trusted_bundle_kind == "trusted_smoke_v1":
            source = manifest.source_bundle_digest
            if source not in self.trusted_source_digests:
                raise ValueError("trusted_bundle_not_allowed")

    async def submit(self, manifest: GpuJobManifest) -> tuple[SupervisorJob, bool]:
        self._validate_manifest(manifest)
        digest = manifest.canonical_sha256()
        key = (manifest.run_id, manifest.idempotency_key)
        async with self._lock:
            existing_id = self._idempotency.get(key)
            if existing_id is not None:
                existing = self._jobs[existing_id]
                if existing.manifest_digest != digest:
                    raise RuntimeError("idempotency_conflict")
                return existing, True
            if manifest.job_id in self._jobs:
                raise RuntimeError("job_id_conflict")
            job = SupervisorJob(manifest=manifest, manifest_digest=digest)
            self._jobs[manifest.job_id] = job
            self._idempotency[key] = manifest.job_id
            job.task = asyncio.create_task(self._run(job))
            self._trim_recent()
            return job, False

    def _trim_recent(self) -> None:
        terminal = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status
            in {
                GpuJobStatus.SUCCEEDED,
                GpuJobStatus.FAILED,
                GpuJobStatus.BLOCKED,
                GpuJobStatus.TIMED_OUT,
                GpuJobStatus.CANCELLED,
            }
        ]
        for job_id in terminal[: max(0, len(self._jobs) - self.recent_limit)]:
            job = self._jobs.pop(job_id)
            self._idempotency.pop((job.manifest.run_id, job.manifest.idempotency_key), None)

    async def _run(self, job: SupervisorJob) -> None:
        lease_id: str | None = None
        try:
            async with self._gpu:
                if job.cancel_event.is_set():
                    raise asyncio.CancelledError
                job.status = GpuJobStatus.RUNNING
                if self.reporter is not None:
                    lease = await self.reporter.lease(
                        job.manifest.job_id,
                        ttl_seconds=min(
                            3600, job.manifest.resource_limits.wall_seconds + 30
                        ),
                    )
                    lease_id = str(lease["lease_id"])
                    job.lease_id = lease_id
                now = datetime.now(timezone.utc)
                deadline_seconds = (job.manifest.deadline.astimezone(timezone.utc) - now).total_seconds()
                timeout = deadline_seconds
                if job.manifest.trusted_bundle_kind == "trusted_smoke_v1":
                    timeout = min(float(job.manifest.resource_limits.wall_seconds), timeout)
                if timeout <= 0:
                    job.status = GpuJobStatus.TIMED_OUT
                    job.result = self._terminal_result(
                        job, GpuJobStatus.TIMED_OUT, GpuReasonCode.DEADLINE_EXCEEDED
                    )
                else:
                    try:
                        result = await asyncio.wait_for(
                            self.executor(job.manifest, self.capabilities, job.cancel_event),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        job.status = GpuJobStatus.TIMED_OUT
                        job.result = self._terminal_result(
                            job, GpuJobStatus.TIMED_OUT, GpuReasonCode.DEADLINE_EXCEEDED
                        )
                    else:
                        job.result = result
                        job.status = result.status
        except asyncio.CancelledError:
            job.status = GpuJobStatus.CANCELLED
            job.result = self._terminal_result(
                job, GpuJobStatus.CANCELLED, GpuReasonCode.CANCELLED
            )
        except Exception as error:
            job.status = GpuJobStatus.FAILED
            job.result = self._terminal_result(
                job, GpuJobStatus.FAILED, GpuReasonCode.INTERNAL_ERROR
            )
        if self.reporter is not None and lease_id is not None and job.result is not None:
            try:
                await self.reporter.complete(
                    job.manifest,
                    lease_id,
                    job.result.model_dump(mode="json"),
                )
            except Exception as error:
                job.status = GpuJobStatus.FAILED
                job.result = self._terminal_result(
                    job, GpuJobStatus.FAILED, GpuReasonCode.CONTROL_CALLBACK_FAILED
                )

    def _terminal_result(
        self, job: SupervisorJob, status: GpuJobStatus, reason_code: GpuReasonCode
    ) -> GpuJobResult:
        return GpuJobResult(
            job_id=job.manifest.job_id,
            run_id=job.manifest.run_id,
            stage=job.manifest.stage,
            status=status,
            reason_code=reason_code,
            hardware=self.capabilities.model_dump(mode="json"),
            finished_at=datetime.now(timezone.utc),
        )

    async def get(self, job_id: str) -> SupervisorJob:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    async def cancel(self, job_id: str) -> SupervisorJob:
        job = await self.get(job_id)
        if job.status in {GpuJobStatus.QUEUED, GpuJobStatus.RUNNING}:
            job.cancel_event.set()
            if job.task is not None:
                job.task.cancel()
                try:
                    await job.task
                except asyncio.CancelledError:
                    pass
            job.status = GpuJobStatus.CANCELLED
            if job.result is None:
                job.result = self._terminal_result(
                    job, GpuJobStatus.CANCELLED, GpuReasonCode.CANCELLED
                )
            if self.reporter is not None and job.lease_id is None:
                await self.reporter.cancel(job_id)
        return job


def _load_trusted_manifest(path: str | None) -> tuple[set[str], set[str]]:
    if not path:
        return set(), {"trusted-smoke-v1"}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        {str(value) for value in payload.get("source_bundle_digests", [])},
        {str(value) for value in payload.get("benchmark_protocol_ids", ["trusted-smoke-v1"])},
    )


APP = FastAPI(title="KernelBlaster GPU Supervisor")


async def _require_supervisor_token(
    authorization: str | None = Header(default=None),
) -> None:
    # Keep the queue/contracts importable in minimal CPU test environments;
    # service configuration is loaded only when the HTTP boundary is exercised.
    from ..servers.auth import require_supervisor_token

    await require_supervisor_token(authorization)


def _supervisor() -> InMemoryGpuSupervisor:
    supervisor = getattr(APP.state, "supervisor", None)
    if supervisor is None:
        trusted, protocols = _load_trusted_manifest(
            os.getenv("KERNELBLASTER_TRUSTED_BUNDLE_MANIFEST")
        )
        capabilities = detect_gpu_capabilities()
        control_url = os.getenv("KERNELBLASTER_CONTROL_URL", "").strip()
        worker_token = os.getenv("KERNELBLASTER_WORKER_TOKEN", "").strip()
        reporter = None
        if control_url and worker_token:
            from .client import ControlWorkerClient

            reporter = ControlWorkerClient(
                control_url, worker_token, capabilities.supervisor_id
            )
        executor = _safe_default_executor
        generated_profile_ids: set[str] = set()
        if reporter is not None:
            from .executor import TrustedStageExecutor

            executor = TrustedStageExecutor(reporter)
        if capabilities.generated_jobs_enabled:
            if reporter is None:
                raise RuntimeError("generated GPU jobs require a Control worker client")
            from .sandbox import DockerSandboxRuntime, SandboxStageExecutor

            runtime = DockerSandboxRuntime.from_environment()
            runtime.validate()
            sandbox_executor = SandboxStageExecutor(
                reporter, runtime, runtime.configuration.profiles
            )
            generated_profile_ids = runtime.configuration.profiles.ids
            trusted_executor = executor

            async def executor(
                manifest: GpuJobManifest,
                supervisor_capabilities: GpuCapabilities,
                cancelled: asyncio.Event,
            ) -> GpuJobResult:
                if manifest.trusted_bundle_kind == "generated_v1":
                    return await sandbox_executor(manifest, supervisor_capabilities, cancelled)
                return await trusted_executor(manifest, supervisor_capabilities, cancelled)
        supervisor = InMemoryGpuSupervisor(
            capabilities,
            executor=executor,
            reporter=reporter,
            trusted_source_digests=trusted,
            benchmark_protocol_ids=protocols,
            generated_profile_ids=generated_profile_ids,
        )
        APP.state.supervisor = supervisor
    return supervisor


@APP.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "gpu-supervisor"}


@APP.get("/ready")
async def ready(
    _authorized: None = Depends(_require_supervisor_token),
) -> dict[str, str]:
    return {"status": "ready", "service": "gpu-supervisor"}


@APP.get("/v1/capabilities")
async def capabilities(
    _authorized: None = Depends(_require_supervisor_token),
) -> dict[str, object]:
    return _supervisor().capabilities.model_dump(mode="json")


@APP.post("/v1/jobs", status_code=202)
async def submit_job(
    manifest: GpuJobManifest,
    _authorized: None = Depends(_require_supervisor_token),
) -> dict[str, object]:
    try:
        job, idempotent = await _supervisor().submit(manifest)
        return {**job.response(), "idempotent": idempotent}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@APP.get("/v1/jobs/{job_id}")
async def get_job(
    job_id: str,
    _authorized: None = Depends(_require_supervisor_token),
) -> dict[str, object]:
    try:
        return (await _supervisor().get(job_id)).response()
    except KeyError as error:
        raise HTTPException(status_code=404, detail="job_not_found") from error


@APP.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    _authorized: None = Depends(_require_supervisor_token),
) -> dict[str, object]:
    try:
        return (await _supervisor().cancel(job_id)).response()
    except KeyError as error:
        raise HTTPException(status_code=404, detail="job_not_found") from error


def main() -> None:
    from ..servers.auth import validate_supervisor_token, validate_worker_token

    parser = argparse.ArgumentParser(description="Run the trusted single-GPU Supervisor")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2002)
    args = parser.parse_args()
    validate_worker_token()
    validate_supervisor_token()
    _supervisor()
    uvicorn.run(APP, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
