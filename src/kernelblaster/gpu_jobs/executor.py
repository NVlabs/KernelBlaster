"""Fixed-command executor for allowlisted smoke bundles only.

This is not the PR 05 untrusted-code sandbox.  It deliberately accepts no
caller argv, paths, environment, profiler, mount, or container options.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from ..servers.security import sanitized_worker_environment
from .bundles import extract_bundle
from .contracts import (
    GpuCapabilities,
    GpuJobManifest,
    GpuJobResult,
    GpuReasonCode,
    GpuJobStage,
    GpuJobStatus,
)


class TrustedStageExecutor:
    def __init__(self, control: Any) -> None:
        self.control = control

    async def __call__(
        self,
        manifest: GpuJobManifest,
        capabilities: GpuCapabilities,
        cancelled: asyncio.Event,
    ) -> GpuJobResult:
        started = datetime.now(timezone.utc)
        try:
            if manifest.stage is GpuJobStage.COMPILE:
                artifacts, measurement = await self._compile(manifest, cancelled)
            elif manifest.stage is GpuJobStage.CORRECTNESS:
                artifacts, measurement = await self._run_binary(
                    manifest, cancelled, mode="correctness"
                )
            else:
                artifacts, measurement = await self._run_binary(
                    manifest, cancelled, mode="events"
                )
            status = GpuJobStatus.SUCCEEDED
            reason = GpuReasonCode.NONE
        except asyncio.CancelledError:
            raise
        except Exception as error:
            artifacts = {}
            measurement = None
            status = GpuJobStatus.FAILED
            reason = {
                GpuJobStage.COMPILE: GpuReasonCode.COMPILE_FAILED,
                GpuJobStage.CORRECTNESS: GpuReasonCode.CORRECTNESS_FAILED,
                GpuJobStage.EVENTS: GpuReasonCode.EVENTS_FAILED,
            }[manifest.stage]
            error_payload = (type(error).__name__ + ": " + str(error)).encode("utf-8")[:65536]
            uploaded = await self.control.upload(
                error_payload, media_type="text/plain", schema="gpu-stage-error/v1"
            )
            artifacts = {uploaded["digest"]: "error_log"}
        return GpuJobResult(
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            stage=manifest.stage,
            status=status,
            reason_code=reason,
            artifact_roles=artifacts,
            measurement=measurement,
            hardware=capabilities.model_dump(mode="json"),
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )

    async def _compile(
        self, manifest: GpuJobManifest, cancelled: asyncio.Event
    ) -> tuple[dict[str, str], None]:
        assert manifest.source_bundle_digest is not None
        assert manifest.driver_digest is not None
        source_bundle, driver = await asyncio.gather(
            self._verified_download(manifest.source_bundle_digest),
            self._verified_download(manifest.driver_digest),
        )
        if len(source_bundle) + len(driver) > manifest.resource_limits.temporary_bytes:
            raise ValueError("temporary_limit_exceeded")
        with tempfile.TemporaryDirectory(prefix="kernelblaster-compile-") as temporary:
            root = Path(temporary)
            sources = extract_bundle(source_bundle, root / "input")
            cuda_sources = sorted(path for path in sources if path.suffix == ".cu")
            if not cuda_sources:
                raise ValueError("trusted bundle contains no .cu source")
            driver_path = root / "driver.cpp"
            driver_path.write_bytes(driver)
            output = root / "candidate"
            command = [
                "nvcc",
                "-O3",
                "-std=c++17",
                f"-arch={manifest.target_arch}",
                *(str(path) for path in cuda_sources),
                str(driver_path),
                "-o",
                str(output),
            ]
            stdout, stderr = await self._command(
                command, root=root, cancelled=cancelled, manifest=manifest
            )
            executable = await self.control.upload(
                output.read_bytes(),
                media_type="application/x-executable",
                schema="gpu-executable/v1",
            )
            log = await self.control.upload(
                stdout + b"\n" + stderr,
                media_type="text/plain",
                schema="gpu-compile-log/v1",
            )
            return {executable["digest"]: "executable", log["digest"]: "compile_log"}, None

    async def _run_binary(
        self,
        manifest: GpuJobManifest,
        cancelled: asyncio.Event,
        *,
        mode: str,
    ) -> tuple[dict[str, str], dict[str, object] | None]:
        assert manifest.executable_digest is not None
        executable = await self._verified_download(manifest.executable_digest)
        if len(executable) > manifest.resource_limits.temporary_bytes:
            raise ValueError("temporary_limit_exceeded")
        with tempfile.TemporaryDirectory(prefix=f"kernelblaster-{mode}-") as temporary:
            root = Path(temporary)
            binary = root / "candidate"
            binary.write_bytes(executable)
            binary.chmod(0o500)
            command = [str(binary), "--mode", mode]
            if mode == "events":
                command.extend(("--protocol", manifest.benchmark_protocol_id))
            stdout, stderr = await self._command(
                command, root=root, cancelled=cancelled, manifest=manifest
            )
            stdout_artifact = await self.control.upload(
                stdout, media_type="application/json", schema=f"gpu-{mode}-stdout/v1"
            )
            stderr_artifact = await self.control.upload(
                stderr, media_type="text/plain", schema=f"gpu-{mode}-stderr/v1"
            )
            measurement = None
            if mode == "events":
                decoded = json.loads(stdout.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ValueError("Events output must be a JSON object")
                measurement = decoded
            return {
                stdout_artifact["digest"]: f"{mode}_stdout",
                stderr_artifact["digest"]: f"{mode}_stderr",
            }, measurement

    async def _command(
        self,
        command: list[str],
        *,
        root: Path,
        cancelled: asyncio.Event,
        manifest: GpuJobManifest,
    ) -> tuple[bytes, bytes]:
        if cancelled.is_set():
            raise asyncio.CancelledError
        environment = sanitized_worker_environment()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        digits = manifest.target_arch.removeprefix("sm_")
        compute = digits[:-1] + "." + digits[-1]
        environment["TORCH_CUDA_ARCH_LIST"] = compute
        environment["CMAKE_CUDA_ARCHITECTURES"] = digits
        environment["CUDAARCHS"] = digits
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=root,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        if len(stdout) > manifest.resource_limits.stdout_bytes:
            raise ValueError("stdout_limit_exceeded")
        if len(stderr) > manifest.resource_limits.stderr_bytes:
            raise ValueError("stderr_limit_exceeded")
        if process.returncode != 0:
            raise RuntimeError(f"fixed stage command exited with {process.returncode}")
        return stdout, stderr

    async def _verified_download(self, digest: str) -> bytes:
        payload = await self.control.download(digest)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("artifact_digest_mismatch")
        return payload
