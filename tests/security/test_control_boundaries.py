# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.kernelblaster.config import config
from src.kernelblaster.servers import control
from src.kernelblaster.servers.auth import (
    require_control_token,
    require_worker_token,
    validate_token_boundaries,
)
from src.kernelblaster.servers.serve_api import app as legacy_workflow_app
from src.kernelblaster.storage import StateStore


def _set_distinct_tokens(monkeypatch) -> None:
    monkeypatch.setattr(config, "CONTROL_TOKEN", "control-token")
    monkeypatch.setattr(config, "WORKER_TOKEN", "worker-token")


def test_token_configuration_requires_both_distinct_audiences(monkeypatch):
    _set_distinct_tokens(monkeypatch)
    assert validate_token_boundaries() is None

    monkeypatch.setattr(config, "CONTROL_TOKEN", "")
    with pytest.raises(RuntimeError, match="KERNELBLASTER_CONTROL_TOKEN"):
        validate_token_boundaries()

    monkeypatch.setattr(config, "CONTROL_TOKEN", "same-token")
    monkeypatch.setattr(config, "WORKER_TOKEN", "same-token")
    with pytest.raises(RuntimeError, match="different values"):
        validate_token_boundaries()


@pytest.mark.asyncio
async def test_token_audiences_are_not_interchangeable(monkeypatch):
    _set_distinct_tokens(monkeypatch)
    assert await require_control_token("Bearer control-token") is None
    assert await require_worker_token("Bearer worker-token") is None
    with pytest.raises(HTTPException) as control_rejection:
        await require_control_token("Bearer worker-token")
    with pytest.raises(HTTPException) as worker_rejection:
        await require_worker_token("Bearer control-token")
    assert control_rejection.value.status_code == 401
    assert worker_rejection.value.status_code == 401


def test_control_exposes_authenticated_health_and_job_boundaries(monkeypatch):
    _set_distinct_tokens(monkeypatch)
    client = TestClient(control.app)
    assert client.get("/health").status_code == 401
    assert client.get("/ready", headers={"Authorization": "Bearer worker-token"}).status_code == 401
    response = client.get("/ready", headers={"Authorization": "Bearer control-token"})
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "service": "control"}
    assert {route.path for route in control.app.routes} >= {"/health", "/ready", "/v1/jobs"}
    assert "/submit" not in {route.path for route in control.app.routes}


def test_control_and_worker_job_api_audiences_are_isolated(monkeypatch, tmp_path):
    _set_distinct_tokens(monkeypatch)
    monkeypatch.setattr(
        control.app.state, "state_store", StateStore(state_dir=tmp_path / "state"), raising=False
    )
    client = TestClient(control.app)
    control_headers = {"Authorization": "Bearer control-token"}
    worker_headers = {"Authorization": "Bearer worker-token"}

    created = client.post("/v1/runs", json={"run_id": "run-1"}, headers=control_headers)
    assert created.status_code == 200
    assert client.post(
        "/v1/jobs",
        json={"run_id": "run-1", "idempotency_key": "candidate-1", "kind": "compile"},
        headers=worker_headers,
    ).status_code == 401
    submitted = client.post(
        "/v1/jobs",
        json={"run_id": "run-1", "idempotency_key": "candidate-1", "kind": "compile"},
        headers=control_headers,
    )
    assert submitted.status_code == 200
    job_id = submitted.json()["id"]
    lease = client.post(
        "/v1/jobs/lease", json={"worker_id": "worker-1"}, headers=worker_headers
    )
    assert lease.status_code == 200
    lease_id = lease.json()["lease_id"]
    assert client.post(
        f"/v1/leases/{lease_id}/heartbeat",
        json={"worker_id": "worker-1"},
        headers=worker_headers,
    ).status_code == 200
    uploaded = client.put(
        "/v1/artifacts",
        content=b"profile output",
        headers={**worker_headers, "content-type": "text/plain"},
    )
    assert uploaded.status_code == 200
    completed = client.post(
        f"/v1/jobs/{job_id}/complete",
        json={
            "lease_id": lease_id,
            "worker_id": "worker-1",
            "status": "succeeded",
            "artifact_roles": {uploaded.json()["digest"]: "profile"},
        },
        headers=worker_headers,
    )
    assert completed.status_code == 200
    assert completed.json()["job"]["status"] == "succeeded"


def test_legacy_workflow_routes_use_the_control_audience():
    protected = {"/submit", "/status/{task_id}", "/cancel/{task_id}"}
    for route in legacy_workflow_app.routes:
        if route.path in protected:
            assert route.dependant.dependencies[0].call is require_control_token
