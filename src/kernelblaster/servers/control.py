# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only control service owning durable local job state and artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
import uuid

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import config
from ..gpu_jobs.client import SupervisorClient
from ..gpu_jobs.contracts import GpuJobManifest
from ..storage import StateStore
from .auth import require_control_token, require_worker_token, validate_token_boundaries


app = FastAPI(title="KernelBlaster Control")


class RunSubmission(BaseModel):
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobSubmission(BaseModel):
    run_id: str
    idempotency_key: str
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    job_id: str | None = None


class LeaseRequest(BaseModel):
    worker_id: str
    ttl_seconds: int = Field(default=60, ge=1, le=3600)


class CompletionRequest(BaseModel):
    lease_id: str
    worker_id: str
    status: str
    result: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    artifact_roles: dict[str, str] = Field(default_factory=dict)


def _state_store() -> StateStore:
    store = getattr(app.state, "state_store", None)
    if store is None:
        store = StateStore(
            state_dir=getattr(app.state, "state_dir", None),
            sqlite_path=getattr(app.state, "sqlite_path", None),
            cas_dir=getattr(app.state, "cas_dir", None),
        )
        app.state.state_store = store
    return store


def _not_found(error: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error))


@app.get("/health")
async def health(_authorized: None = Depends(require_control_token)) -> dict[str, str]:
    return {"status": "ok", "service": "control"}


@app.get("/ready")
async def ready(_authorized: None = Depends(require_control_token)) -> dict[str, str]:
    """Report the authenticated readiness boundary without starting a workflow."""
    return {"status": "ready", "service": "control"}


@app.post("/v1/runs")
async def create_run(
    submission: RunSubmission, _authorized: None = Depends(require_control_token)
) -> dict[str, Any]:
    run_id = submission.run_id or uuid.uuid4().hex
    return _state_store().repository.create_run(run_id, metadata=submission.metadata)


@app.get("/v1/runs/{run_id}")
async def get_run(run_id: str, _authorized: None = Depends(require_control_token)) -> dict[str, Any]:
    try:
        return _state_store().repository.get_run(run_id)
    except KeyError as error:
        raise _not_found(error) from error


@app.post("/v1/jobs")
async def submit_job(
    submission: JobSubmission, _authorized: None = Depends(require_control_token)
) -> dict[str, Any]:
    try:
        return _state_store().repository.submit_job(**submission.model_dump())
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/v1/jobs/lease")
async def lease_job(
    request: LeaseRequest, _authorized: None = Depends(require_worker_token)
) -> dict[str, Any] | None:
    return _state_store().repository.acquire_lease(**request.model_dump())


@app.post("/v1/jobs/{job_id}/lease")
async def lease_specific_job(
    job_id: str,
    request: LeaseRequest,
    _authorized: None = Depends(require_worker_token),
) -> dict[str, Any]:
    try:
        return _state_store().repository.acquire_job_lease(
            job_id=job_id, **request.model_dump()
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_pending_job(
    job_id: str, _authorized: None = Depends(require_worker_token)
) -> dict[str, Any]:
    try:
        return _state_store().repository.cancel_pending_job(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/leases/{lease_id}/heartbeat")
async def heartbeat_lease(
    lease_id: str, request: LeaseRequest, _authorized: None = Depends(require_worker_token)
) -> dict[str, Any]:
    try:
        return _state_store().repository.heartbeat_lease(lease_id=lease_id, **request.model_dump())
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/jobs/{job_id}/complete")
async def complete_job(
    job_id: str, request: CompletionRequest, _authorized: None = Depends(require_worker_token)
) -> dict[str, Any]:
    store = _state_store()
    try:
        outcome = store.repository.complete_job(
            job_id=job_id,
            lease_id=request.lease_id,
            worker_id=request.worker_id,
            status=request.status,
            result=request.result,
            reason=request.reason,
            artifact_roles=request.artifact_roles,
        )
        return outcome
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


async def _store_request_artifact(request: Request, *, producer: str | None = None) -> dict[str, Any]:
    maximum = int(os.getenv("KERNELBLASTER_MAX_ARTIFACT_BYTES", str(256 * 1024 * 1024)))
    payload = await request.body()
    if len(payload) > maximum:
        raise HTTPException(status_code=413, detail="artifact exceeds configured byte limit")
    store = _state_store()
    artifact = store.cas.put_bytes(
        payload,
        media_type=request.headers.get("content-type", "application/octet-stream"),
        producer=producer or request.headers.get("x-kernelblaster-producer"),
        source_digest=request.headers.get("x-kernelblaster-source-digest"),
        schema=request.headers.get("x-kernelblaster-schema"),
    )
    return store.repository.register_artifact(artifact)


@app.put("/v1/artifacts")
async def upload_artifact(
    request: Request, _authorized: None = Depends(require_worker_token)
) -> dict[str, Any]:
    return await _store_request_artifact(request, producer="gpu-supervisor")


@app.put("/v1/control/artifacts")
async def upload_control_artifact(
    request: Request, _authorized: None = Depends(require_control_token)
) -> dict[str, Any]:
    return await _store_request_artifact(request, producer="control")


@app.get("/v1/worker/artifacts/{digest}")
async def download_worker_artifact(
    digest: str, _authorized: None = Depends(require_worker_token)
) -> FileResponse:
    try:
        path = _state_store().cas.get_path(digest)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="application/octet-stream", filename=digest)


@app.get("/v1/artifacts/{digest}")
async def download_artifact(
    digest: str, _authorized: None = Depends(require_control_token)
) -> FileResponse:
    try:
        path = _state_store().cas.get_path(digest)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="application/octet-stream", filename=digest)


@app.post("/v1/gpu/jobs", status_code=202)
async def submit_gpu_job(
    manifest: GpuJobManifest,
    _authorized: None = Depends(require_control_token),
) -> dict[str, Any]:
    store = _state_store()
    try:
        for digest in manifest.input_digests():
            store.cas.verify(digest)
        client = SupervisorClient(config.GPU_SUPERVISOR_URL, config.SUPERVISOR_TOKEN)
        capabilities = await client.capabilities()
        if capabilities.device.target_arch != manifest.target_arch:
            raise ValueError("target_arch_mismatch")
        payload = manifest.model_dump(mode="json")
        job = store.repository.submit_job(
            run_id=manifest.run_id,
            idempotency_key=manifest.idempotency_key,
            kind=f"gpu:{manifest.stage.value}",
            payload=payload,
            job_id=manifest.job_id,
        )
        if job["payload"] != payload:
            raise RuntimeError("idempotency_conflict")
        submitted_manifest = manifest.model_copy(update={"job_id": job["id"]})
        supervisor = await client.submit(submitted_manifest)
        return {"job": job, "supervisor": supervisor, "capabilities": capabilities.model_dump(mode="json")}
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail="artifact_not_found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/v1/gpu/capabilities")
async def get_gpu_capabilities(
    _authorized: None = Depends(require_control_token),
) -> dict[str, Any]:
    client = SupervisorClient(config.GPU_SUPERVISOR_URL, config.SUPERVISOR_TOKEN)
    return (await client.capabilities()).model_dump(mode="json")


@app.get("/v1/gpu/jobs/{job_id}")
async def get_gpu_job(
    job_id: str, _authorized: None = Depends(require_control_token)
) -> dict[str, Any]:
    client = SupervisorClient(config.GPU_SUPERVISOR_URL, config.SUPERVISOR_TOKEN)
    return await client.get(job_id)


@app.post("/v1/gpu/jobs/{job_id}/cancel")
async def cancel_gpu_job(
    job_id: str, _authorized: None = Depends(require_control_token)
) -> dict[str, Any]:
    client = SupervisorClient(config.GPU_SUPERVISOR_URL, config.SUPERVISOR_TOKEN)
    return await client.cancel(job_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPU-only control boundary")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=None)
    parser.add_argument("--cas-dir", type=Path, default=None)
    args = parser.parse_args()
    validate_token_boundaries()
    app.state.state_dir = args.state_dir
    app.state.sqlite_path = args.sqlite_path
    app.state.cas_dir = args.cas_dir
    app.state.state_store = None
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
