from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.kernelblaster.gpu_jobs.contracts import (
    GpuCapabilities,
    GpuDeviceCapability,
    GpuJobManifest,
    GpuJobResult,
    GpuJobStatus,
    GpuRuntimeCapability,
)
from src.kernelblaster.gpu_jobs.supervisor import InMemoryGpuSupervisor


SOURCE = "a" * 64
DRIVER = "b" * 64


def _capability(arch: str = "sm_86") -> GpuCapabilities:
    return GpuCapabilities(
        supervisor_id="test",
        device=GpuDeviceCapability(
            logical_id="0",
            name="test-gpu",
            compute_capability=arch.removeprefix("sm_")[:-1] + "." + arch[-1],
            target_arch=arch,
            total_memory_bytes=1024,
        ),
        runtime=GpuRuntimeCapability(cuda_version="12.8", driver_version="test"),
    )


def _manifest(job_id: str, key: str, **updates) -> GpuJobManifest:
    payload = {
        "job_id": job_id,
        "run_id": "run-1",
        "idempotency_key": key,
        "stage": "compile",
        "source_bundle_digest": SOURCE,
        "driver_digest": DRIVER,
        "target_arch": "sm_86",
        "benchmark_protocol_id": "trusted-smoke-v1",
        "deadline": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    payload.update(updates)
    if payload.get("trusted_bundle_kind") == "generated_v1":
        payload.pop("driver_digest", None)
        payload.setdefault("private_evaluation_profile_id", "private-v1")
    return GpuJobManifest.model_validate(payload)


def test_submit_is_idempotent_and_conflicting_manifest_is_rejected():
    async def scenario():
        supervisor = InMemoryGpuSupervisor(_capability(), trusted_source_digests={SOURCE})
        manifest = _manifest("job-1", "same")
        first, first_idempotent = await supervisor.submit(manifest)
        second, second_idempotent = await supervisor.submit(manifest)
        assert first is second
        assert first_idempotent is False and second_idempotent is True
        conflicting = _manifest("job-2", "same", resource_limits={"wall_seconds": 10})
        with pytest.raises(RuntimeError, match="idempotency_conflict"):
            await supervisor.submit(conflicting)
        await supervisor.cancel("job-1")

    asyncio.run(scenario())


def test_arch_generated_and_trusted_allowlist_are_enforced():
    async def scenario():
        supervisor = InMemoryGpuSupervisor(_capability(), trusted_source_digests=set())
        with pytest.raises(ValueError, match="trusted_bundle_not_allowed"):
            await supervisor.submit(_manifest("job-1", "a"))
        with pytest.raises(ValueError, match="target_arch_mismatch"):
            await supervisor.submit(_manifest("job-2", "b", target_arch="sm_89"))
        with pytest.raises(ValueError, match="generated_jobs_disabled"):
            await supervisor.submit(
                _manifest("job-3", "c", trusted_bundle_kind="generated_v1")
            )

    asyncio.run(scenario())


def test_single_gpu_semaphore_cancel_and_terminal_status_are_stable():
    async def scenario():
        active = 0
        maximum = 0
        release = asyncio.Event()

        async def executor(manifest, capabilities, cancelled):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await release.wait()
            finally:
                active -= 1
            return GpuJobResult(
                job_id=manifest.job_id,
                run_id=manifest.run_id,
                stage=manifest.stage,
                status=GpuJobStatus.SUCCEEDED,
                hardware=capabilities.model_dump(mode="json"),
            )

        supervisor = InMemoryGpuSupervisor(
            _capability(), executor=executor, trusted_source_digests={SOURCE}
        )
        first, _ = await supervisor.submit(_manifest("job-1", "a"))
        second, _ = await supervisor.submit(_manifest("job-2", "b"))
        await asyncio.sleep(0)
        assert first.status is GpuJobStatus.RUNNING
        assert second.status is GpuJobStatus.QUEUED
        await supervisor.cancel("job-2")
        assert (await supervisor.get("job-2")).status is GpuJobStatus.CANCELLED
        release.set()
        assert first.task is not None
        await first.task
        assert first.status is GpuJobStatus.SUCCEEDED
        assert maximum == 1

    asyncio.run(scenario())


def test_running_result_and_queued_cancel_are_reported_to_control():
    async def scenario():
        release = asyncio.Event()

        class Reporter:
            def __init__(self):
                self.completed = []
                self.cancelled = []

            async def lease(self, job_id, *, ttl_seconds):
                return {"lease_id": f"lease-{job_id}", "ttl_seconds": ttl_seconds}

            async def complete(self, manifest, lease_id, result):
                self.completed.append((manifest.job_id, lease_id, result["status"]))

            async def cancel(self, job_id):
                self.cancelled.append(job_id)

        async def executor(manifest, capabilities, cancelled):
            await release.wait()
            return GpuJobResult(
                job_id=manifest.job_id,
                run_id=manifest.run_id,
                stage=manifest.stage,
                status=GpuJobStatus.SUCCEEDED,
                hardware=capabilities.model_dump(mode="json"),
            )

        reporter = Reporter()
        supervisor = InMemoryGpuSupervisor(
            _capability(),
            executor=executor,
            reporter=reporter,
            trusted_source_digests={SOURCE},
        )
        running, _ = await supervisor.submit(_manifest("job-running", "running"))
        queued, _ = await supervisor.submit(_manifest("job-queued", "queued"))
        await asyncio.sleep(0)
        await supervisor.cancel(queued.manifest.job_id)
        release.set()
        assert running.task is not None
        await running.task
        assert reporter.cancelled == ["job-queued"]
        assert reporter.completed == [
            ("job-running", "lease-job-running", "succeeded")
        ]

    asyncio.run(scenario())


def test_deadline_enters_stable_timed_out_terminal_state():
    async def scenario():
        async def executor(manifest, capabilities, cancelled):
            await asyncio.sleep(10)
            raise AssertionError("deadline should cancel the executor")

        supervisor = InMemoryGpuSupervisor(
            _capability(), executor=executor, trusted_source_digests={SOURCE}
        )
        job, _ = await supervisor.submit(
            _manifest(
                "job-timeout",
                "timeout",
                deadline=datetime.now(timezone.utc) + timedelta(milliseconds=200),
            )
        )
        assert job.task is not None
        await job.task
        assert job.status is GpuJobStatus.TIMED_OUT
        assert job.result is not None
        assert job.result.reason_code.value == "deadline_exceeded"

    asyncio.run(scenario())
