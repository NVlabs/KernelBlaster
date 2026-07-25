"""Control-side client for the authenticated GPU Supervisor API."""

from __future__ import annotations

from typing import Any

import aiohttp

from .contracts import GpuCapabilities, GpuJobManifest


class SupervisorClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    async def capabilities(self) -> GpuCapabilities:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v1/capabilities", headers=self.headers
            ) as response:
                response.raise_for_status()
                return GpuCapabilities.model_validate(await response.json())

    async def submit(self, manifest: GpuJobManifest) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/jobs",
                json=manifest.model_dump(mode="json"),
                headers=self.headers,
            ) as response:
                if response.status not in {200, 202}:
                    raise RuntimeError(
                        f"Supervisor rejected job with HTTP {response.status}: {await response.text()}"
                    )
                return await response.json()

    async def get(self, job_id: str) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v1/jobs/{job_id}", headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def cancel(self, job_id: str) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/jobs/{job_id}/cancel", headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()


class ControlWorkerClient:
    """Supervisor-only digest transport and atomic completion client."""

    def __init__(self, base_url: str, token: str, worker_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.worker_id = worker_id

    async def download(self, digest: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v1/worker/artifacts/{digest}", headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.read()

    async def upload(
        self, payload: bytes, *, media_type: str, schema: str | None = None
    ) -> dict[str, Any]:
        headers = {**self.headers, "content-type": media_type}
        if schema:
            headers["x-kernelblaster-schema"] = schema
        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{self.base_url}/v1/artifacts", data=payload, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def lease(self, job_id: str, *, ttl_seconds: int) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/jobs/{job_id}/lease",
                json={"worker_id": self.worker_id, "ttl_seconds": ttl_seconds},
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def complete(
        self, manifest: GpuJobManifest, lease_id: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        status = str(result["status"])
        if status == "queued" or status == "running":
            raise ValueError("Control completion requires a terminal status")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/jobs/{manifest.job_id}/complete",
                json={
                    "lease_id": lease_id,
                    "worker_id": self.worker_id,
                    "status": status,
                    "result": result,
                    "reason": result.get("reason_code"),
                    "artifact_roles": result.get("artifact_roles", {}),
                },
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def cancel(self, job_id: str) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/jobs/{job_id}/cancel", headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()
