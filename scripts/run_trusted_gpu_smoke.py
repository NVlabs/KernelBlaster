#!/usr/bin/env python3
"""Run the reviewed vector-add compile/correctness/Events Job chain."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kernelblaster.gpu_jobs import build_deterministic_bundle  # noqa: E402


TERMINAL = {"succeeded", "failed", "blocked", "timed_out", "cancelled"}


class ControlClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.authorization = f"Bearer {token}"

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict | None = None,
        body: bytes | None = None,
        media_type: str = "application/json",
    ) -> dict:
        data = body
        if json_payload is not None:
            data = json.dumps(json_payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Authorization": self.authorization, "Content-Type": media_type},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    def wait(self, job_id: str, timeout_seconds: int = 600) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            job = self.request("GET", f"/v1/gpu/jobs/{job_id}")
            if job["status"] in TERMINAL:
                return job
            time.sleep(0.25)
        self.request("POST", f"/v1/gpu/jobs/{job_id}/cancel")
        raise TimeoutError(f"trusted smoke job {job_id} did not finish")


def _artifact_for_role(job: dict, role: str) -> str:
    result = job.get("result") or {}
    for digest, actual_role in result.get("artifact_roles", {}).items():
        if actual_role == role:
            return digest
    raise RuntimeError(f"job did not produce artifact role {role}: {job}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.getenv("KERNELBLASTER_CONTROL_TOKEN"))
    args = parser.parse_args()
    if not args.token:
        parser.error("--token or KERNELBLASTER_CONTROL_TOKEN is required")
    client = ControlClient(args.control_url, args.token)
    smoke = ROOT / "portfolio" / "trusted_gpu_smoke"
    source_bundle = build_deterministic_bundle(
        {"vector_add.cu": (smoke / "vector_add.cu").read_bytes()}
    )
    driver = (smoke / "driver.cpp").read_bytes()
    manifest = json.loads(
        (ROOT / "portfolio" / "trusted-gpu-bundles.json").read_text(encoding="utf-8")
    )["bundles"][0]
    if hashlib.sha256(source_bundle).hexdigest() != manifest["source_bundle_digest"]:
        raise RuntimeError("trusted source bundle digest mismatch")
    if hashlib.sha256(driver).hexdigest() != manifest["driver_digest"]:
        raise RuntimeError("trusted driver digest mismatch")
    source = client.request(
        "PUT", "/v1/control/artifacts", body=source_bundle, media_type="application/x-tar"
    )
    driver_artifact = client.request(
        "PUT", "/v1/control/artifacts", body=driver, media_type="text/x-c++src"
    )
    capabilities = client.request("GET", "/v1/gpu/capabilities")
    target_arch = capabilities["device"]["target_arch"]
    run_id = "trusted-smoke-" + uuid.uuid4().hex
    client.request("POST", "/v1/runs", json_payload={"run_id": run_id, "metadata": {"kind": "trusted-smoke-v1"}})

    common = {
        "schema_version": "gpu-job/v1",
        "run_id": run_id,
        "source_bundle_digest": source["digest"],
        "driver_digest": driver_artifact["digest"],
        "target_arch": target_arch,
        "benchmark_protocol_id": "trusted-smoke-v1",
        "resource_limits": {"wall_seconds": 120},
        "deadline": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        "trusted_bundle_kind": "trusted_smoke_v1",
    }
    compile_id = uuid.uuid4().hex
    client.request(
        "POST",
        "/v1/gpu/jobs",
        json_payload={**common, "job_id": compile_id, "idempotency_key": "compile", "stage": "compile"},
    )
    compiled = client.wait(compile_id)
    if compiled["status"] != "succeeded":
        raise RuntimeError(f"trusted compile failed: {compiled}")
    executable = _artifact_for_role(compiled, "executable")

    correctness_id = uuid.uuid4().hex
    client.request(
        "POST",
        "/v1/gpu/jobs",
        json_payload={
            **common,
            "job_id": correctness_id,
            "idempotency_key": "correctness",
            "stage": "correctness",
            "executable_digest": executable,
        },
    )
    correctness = client.wait(correctness_id)
    if correctness["status"] != "succeeded":
        raise RuntimeError(f"trusted correctness failed: {correctness}")

    events_id = uuid.uuid4().hex
    client.request(
        "POST",
        "/v1/gpu/jobs",
        json_payload={
            **common,
            "job_id": events_id,
            "idempotency_key": "events",
            "stage": "events",
            "executable_digest": executable,
        },
    )
    events = client.wait(events_id)
    if events["status"] != "succeeded":
        raise RuntimeError(f"trusted Events failed: {events}")
    print(json.dumps({"run_id": run_id, "target_arch": target_arch, "measurement": events["result"]["measurement"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
