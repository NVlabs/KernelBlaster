from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import HTTPException
import pytest

from src.kernelblaster.config import config
from src.kernelblaster.servers.auth import require_worker_token
from src.kernelblaster.servers.gpu import (
    GpuCommandError,
    build_execution_argv,
    exec_command,
    read_upload_with_limit,
    sanitized_worker_environment,
)
from src.kernelblaster.servers.security import allowed_source_path, validate_worker_environment
from src.kernelblaster.servers.management import _worker_environment


@pytest.mark.asyncio
async def test_worker_endpoint_rejects_missing_or_wrong_token(monkeypatch):
    monkeypatch.setattr(config, "WORKER_TOKEN", "unit-token")
    with pytest.raises(HTTPException) as missing:
        await require_worker_token(None)
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as wrong:
        await require_worker_token("Bearer wrong")
    assert wrong.value.status_code == 401
    assert await require_worker_token("Bearer unit-token") is None


def test_profiler_and_shell_injection_are_not_executed_by_a_shell():
    argv, environment = build_execution_argv(
        "/tmp/program", prefix_command=["ncu", "--section", "SpeedOfLight"]
    )
    assert argv == ["ncu", "--section", "SpeedOfLight", "/tmp/program"]
    assert environment == {}
    with pytest.raises(ValueError, match="argument is not allowed"):
        build_execution_argv(
            "/tmp/program",
            prefix_command=["ncu", ";", "touch", "/tmp/kernelblaster-pwned"],
        )
    with pytest.raises(ValueError, match="Profiler is not allowed"):
        build_execution_argv("/tmp/program", prefix_command="sh -c id")


def test_worker_environment_does_not_inherit_llm_credentials():
    sanitized = sanitized_worker_environment(
        {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "secret",
            "KERNELBLASTER_LLM_API_KEY": "secret-2",
            "KERNELBLASTER_WORKER_TOKEN": "worker-secret",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    assert sanitized == {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0"}


def test_worker_environment_allows_only_its_dedicated_token():
    validate_worker_environment(
        {
            "PATH": "/usr/bin",
            "KERNELBLASTER_WORKER_TOKEN": "worker-secret",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    with pytest.raises(RuntimeError, match="KERNELBLASTER_CONTROL_TOKEN"):
        validate_worker_environment(
            {
                "KERNELBLASTER_WORKER_TOKEN": "worker-secret",
                "KERNELBLASTER_CONTROL_TOKEN": "control-secret",
                "KERNELBLASTER_LLM_API_KEY": "provider-secret",
            }
        )


def test_managed_worker_environment_keeps_only_the_worker_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "control-secret")
    monkeypatch.setenv("KERNELBLASTER_LLM_API_KEY", "provider-secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(config, "WORKER_TOKEN", "unit-worker-token")
    environment = _worker_environment()
    assert environment["PATH"] == "/usr/bin"
    assert environment["KERNELBLASTER_WORKER_TOKEN"] == "unit-worker-token"
    assert "OPENAI_API_KEY" not in environment
    assert "KERNELBLASTER_LLM_API_KEY" not in environment


def test_source_path_escape_is_rejected(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "driver.cpp"
    source.write_text("// driver\n", encoding="utf-8")
    monkeypatch.setenv("KERNELBLASTER_ALLOWED_SOURCE_ROOTS", str(allowed))
    assert allowed_source_path(str(source)) == source.resolve()
    with pytest.raises(HTTPException) as escaped:
        allowed_source_path(str(tmp_path / "outside.cpp"))
    assert escaped.value.status_code == 400


@pytest.mark.asyncio
async def test_oversized_upload_stops_at_limit():
    class Upload:
        def __init__(self):
            self._chunks = [b"1234", b"5678"]

        async def read(self, _size):
            return self._chunks.pop(0) if self._chunks else b""

    with pytest.raises(HTTPException) as oversized:
        await read_upload_with_limit(Upload(), 6)
    assert oversized.value.status_code == 413


@pytest.mark.asyncio
async def test_timeout_kills_worker_process(monkeypatch):
    class Process:
        returncode = None
        killed = False

        async def communicate(self):
            await asyncio.sleep(60)

    process = Process()

    async def create_process(*_args, **_kwargs):
        return process

    async def kill_process(target, _logger):
        target.killed = True

    import src.kernelblaster.servers.gpu as gpu_module

    monkeypatch.setattr(gpu_module.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(gpu_module, "safe_kill_process", kill_process)
    monkeypatch.setattr(gpu_module, "env", {})
    with pytest.raises(GpuCommandError, match="timed out"):
        await exec_command(["fake-binary"], timeout=0.001)
    assert process.killed is True
