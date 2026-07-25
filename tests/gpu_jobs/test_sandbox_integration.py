# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Docker/GPU probes for the ephemeral generated-candidate sandbox.

Run on AutoDL/self-hosted GPU only, after pinning KERNELBLASTER_GPU_JOB_IMAGE:
``KERNELBLASTER_RUN_GPU_SANDBOX_TESTS=1 pytest -m gpu_sandbox``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import pytest

from src.kernelblaster.gpu_jobs.capabilities import detect_gpu_capabilities
from src.kernelblaster.gpu_jobs.contracts import GpuJobManifest, GpuJobStage, GpuReasonCode
from src.kernelblaster.gpu_jobs.sandbox import (
    DockerSandboxRuntime,
    SandboxConfiguration,
    _tar_files,
)


pytestmark = [
    pytest.mark.gpu_sandbox,
    pytest.mark.skipif(
        os.getenv("KERNELBLASTER_RUN_GPU_SANDBOX_TESTS") != "1",
        reason="set KERNELBLASTER_RUN_GPU_SANDBOX_TESTS=1 on an AutoDL/self-hosted GPU host",
    ),
]

ROOT = Path(__file__).resolve().parents[2]


def _manifest(stage: GpuJobStage, target_arch: str, profile_id: str) -> GpuJobManifest:
    return GpuJobManifest.model_validate(
        {
            "job_id": f"gpu-sandbox-{stage.value}",
            "run_id": "gpu-sandbox-integration",
            "idempotency_key": stage.value,
            "stage": stage.value,
            "source_bundle_digest": "a" * 64 if stage is GpuJobStage.COMPILE else None,
            "executable_digest": "b" * 64 if stage is not GpuJobStage.COMPILE else None,
            "private_evaluation_profile_id": profile_id,
            "target_arch": target_arch,
            "benchmark_protocol_id": "trusted-smoke-v1",
            "deadline": datetime.now(timezone.utc) + timedelta(minutes=10),
            "trusted_bundle_kind": "generated_v1",
        }
    )


def _request(stage: GpuJobStage, target_arch: str) -> bytes:
    return json.dumps(
        {
            "stage": stage.value,
            "target_arch": target_arch,
            "benchmark_protocol_id": "trusted-smoke-v1",
            "driver_path": "private/driver.cpp",
            "stdout_bytes": 1024 * 1024,
            "stderr_bytes": 1024 * 1024,
            "wall_seconds": {"compile": 180, "correctness": 60, "events": 90}[stage.value],
        }
    ).encode("utf-8")


def _compile(
    runtime: DockerSandboxRuntime,
    *,
    target_arch: str,
    profile_id: str,
    source: bytes,
    driver: bytes,
) -> bytes:
    execution = runtime.execute(
        input_archive=_tar_files(
            {
                "candidate/candidate.cu": source,
                "private/driver.cpp": driver,
                "request.json": _request(GpuJobStage.COMPILE, target_arch),
            }
        ),
        manifest=_manifest(GpuJobStage.COMPILE, target_arch, profile_id),
    )
    assert execution.reason is GpuReasonCode.NONE, execution.outputs.get("stderr.log", b"")
    return execution.outputs["candidate"]


def _run_events(
    runtime: DockerSandboxRuntime, *, target_arch: str, profile_id: str, executable: bytes
):
    return runtime.execute(
        input_archive=_tar_files(
            {
                "candidate/candidate": executable,
                "request.json": _request(GpuJobStage.EVENTS, target_arch),
            },
            executable={"candidate/candidate"},
        ),
        manifest=_manifest(GpuJobStage.EVENTS, target_arch, profile_id),
    )


def test_ephemeral_job_hides_env_blocks_network_and_recovers_with_vector_add():
    """Probe `/proc`, Docker socket, DNS/TCP, and read-only write boundaries.

    The immediate vector-add run is the post-attack smoke proving that a Job's
    private volume/processes did not poison the next GPU Job.
    """
    configuration = SandboxConfiguration.from_environment()
    runtime = DockerSandboxRuntime.from_environment()
    runtime.validate()
    capabilities = detect_gpu_capabilities()
    profile_id = next(iter(configuration.profiles.ids))
    attack_source = br'''
#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
extern "C" int probe() {
  char env[8192] = {};
  int proc = open("/proc/self/environ", O_RDONLY);
  int count = proc < 0 ? 0 : (int)read(proc, env, sizeof(env));
  if (proc >= 0) close(proc);
  bool secret = count > 0 && std::strstr(env, "KERNELBLASTER_TEST_SECRET");
  bool socket_present = access("/var/run/docker.sock", F_OK) == 0;
  int root_write = open("/root/kernelblaster-escape", O_WRONLY | O_CREAT, 0600);
  if (root_write >= 0) close(root_write);
  int input_write = open("/input/escape", O_WRONLY | O_CREAT, 0600);
  if (input_write >= 0) close(input_write);
  addrinfo* info = nullptr;
  int dns = getaddrinfo("example.com", "80", nullptr, &info);
  if (info) freeaddrinfo(info);
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in peer{}; peer.sin_family = AF_INET; peer.sin_port = htons(80);
  inet_pton(AF_INET, "1.1.1.1", &peer.sin_addr);
  int tcp = fd < 0 ? -1 : connect(fd, (sockaddr*)&peer, sizeof(peer));
  if (fd >= 0) close(fd);
  return (secret || socket_present || root_write >= 0 || input_write >= 0 || dns == 0 || tcp == 0) ? 1 : 0;
}
'''
    attack_driver = br'''
#include <cstdio>
extern "C" int probe();
int main() { std::printf("{\"probe\":%d}\n", probe()); return 0; }
'''
    attack_executable = _compile(
        runtime,
        target_arch=capabilities.device.target_arch,
        profile_id=profile_id,
        source=attack_source,
        driver=attack_driver,
    )
    attack = _run_events(
        runtime,
        target_arch=capabilities.device.target_arch,
        profile_id=profile_id,
        executable=attack_executable,
    )
    assert attack.reason is GpuReasonCode.NONE
    assert attack.measurement == {"probe": 0}

    smoke = ROOT / "portfolio" / "trusted_gpu_smoke"
    smoke_executable = _compile(
        runtime,
        target_arch=capabilities.device.target_arch,
        profile_id=profile_id,
        source=(smoke / "vector_add.cu").read_bytes(),
        driver=(smoke / "driver.cpp").read_bytes(),
    )
    recovered = _run_events(
        runtime,
        target_arch=capabilities.device.target_arch,
        profile_id=profile_id,
        executable=smoke_executable,
    )
    assert recovered.reason is GpuReasonCode.NONE
    assert isinstance(recovered.measurement, dict)
