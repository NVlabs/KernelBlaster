from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_trusted_pilot", ROOT / "scripts" / "run_trusted_pilot.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SMOKE_SPEC = importlib.util.spec_from_file_location(
    "smoke_llm", ROOT / "scripts" / "smoke_llm.py"
)
assert SMOKE_SPEC and SMOKE_SPEC.loader
SMOKE_MODULE = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(SMOKE_MODULE)


def test_missing_optional_command_is_recorded_instead_of_crashing(tmp_path):
    log_path = tmp_path / "missing.log"

    completed = MODULE._run(
        ["kernelblaster-command-that-does-not-exist"],
        log_path=log_path,
        timeout=1,
    )

    assert completed.returncode == 127
    assert "FileNotFoundError" in completed.stderr
    assert "kernelblaster-command-that-does-not-exist" in log_path.read_text(
        encoding="utf-8"
    )


def test_ncu_probe_classification_is_strict():
    assert MODULE._classify_ncu_probe(
        subprocess.CompletedProcess([], 0, "", "ERR_NVGPUCTRPERM")
    ) == ("available", "ncu")
    assert MODULE._classify_ncu_probe(
        subprocess.CompletedProcess([], 1, "", "ERR_NVGPUCTRPERM")
    ) == ("events_only", "events_only")
    assert MODULE._classify_ncu_probe(
        subprocess.CompletedProcess([], 1, "", "target executable missing")
    ) == ("failed", None)
    assert MODULE._classify_ncu_probe(
        subprocess.CompletedProcess([], 124, "ERR_NVGPUCTRPERM", "timeout")
    ) == ("failed", None)
    assert MODULE._classify_ncu_probe(
        subprocess.CompletedProcess([], 127, "ERR_NVGPUCTRPERM", "ncu missing")
    ) == ("failed", None)


def test_trusted_pilot_uses_bounded_smoke_and_rmsnorm_suite(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, *, log_path, env=None, timeout):
        calls.append({"command": command, "env": env, "timeout": timeout})
        return subprocess.CompletedProcess(command, 0, "", "")

    output_dir = tmp_path / "trusted-pilot"
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("KERNELBLASTER_LLM_API_KEY", "kernelblaster-test-secret")
    monkeypatch.setenv("MODEL", "gpt-5.6-terra")
    monkeypatch.setattr(MODULE, "_run", fake_run)
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        ["run_trusted_pilot.py", "--output-dir", str(output_dir)],
    )

    assert MODULE.main() == 0
    assert [
        call["command"][1] if call["command"][0] != "ncu" else "ncu"
        for call in calls
    ] == [
        "scripts/check_runtime_versions.py",
        "scripts/benchmark_candidates.py",
        "ncu",
        "scripts/smoke_llm.py",
        "scripts/run_RL.py",
    ]

    smoke = calls[3]["command"]
    assert smoke[smoke.index("--model") + 1] == "gpt-5.6-sol"
    assert smoke[smoke.index("--base-url") + 1] == "https://api.openai.com/v1"
    assert smoke[smoke.index("--max-completion-tokens") + 1] == "64"
    assert smoke[smoke.index("--reasoning-effort") + 1] == "none"
    for call in calls[:3]:
        assert "OPENAI_API_KEY" not in call["env"]
        assert "KERNELBLASTER_LLM_API_KEY" not in call["env"]
    assert calls[3]["env"] is None

    pilot = calls[4]
    command = pilot["command"]
    assert command[command.index("--problem-numbers") + 1] == "36"
    assert command[command.index("--rl-iterations") + 1] == "2"
    assert command[command.index("--rl-rollout-steps") + 1] == "2"
    assert command[command.index("--portfolio-suite") + 1] == (
        "portfolio/suites/rmsnorm.json"
    )
    assert pilot["env"]["LLM_MAX_REQUESTS"] == "32"
    assert pilot["env"]["LLM_MAX_TOTAL_TOKENS"] == "250000"
    assert pilot["env"]["LLM_MAX_CONCURRENCY"] == "2"
    assert pilot["env"]["LLM_MAX_RETRIES"] == "2"
    assert pilot["env"]["LLM_REASONING_EFFORT"] == "low"
    assert pilot["env"]["KERNELBLASTER_LLM_PROVIDER"] == "openai_compatible"
    assert pilot["env"]["KERNELBLASTER_LLM_BASE_URL"] == (
        "https://api.openai.com/v1"
    )
    assert pilot["env"]["MODEL"] == "gpt-5.6-sol"

    preflight = json.loads((output_dir / "preflight.json").read_text())
    assert preflight["profiling_mode"] == "ncu"


def test_trusted_pilot_rejects_model_override(monkeypatch, tmp_path):
    output_dir = tmp_path / "trusted-pilot"
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        [
            "run_trusted_pilot.py",
            "--model",
            "gpt-5.6-terra",
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit) as error:
        MODULE.main()

    assert error.value.code == 2
    assert not output_dir.exists()


def test_unexpected_ncu_failure_stops_before_api_request(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, *, log_path, env=None, timeout):
        calls.append(command)
        if command[0] == "ncu":
            return subprocess.CompletedProcess(command, 1, "", "target missing")
        return subprocess.CompletedProcess(command, 0, "", "")

    output_dir = tmp_path / "trusted-pilot"
    monkeypatch.setattr(MODULE, "_run", fake_run)
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        ["run_trusted_pilot.py", "--output-dir", str(output_dir)],
    )

    assert MODULE.main() == 2
    assert len(calls) == 3
    assert not any("scripts/smoke_llm.py" in command for command in calls)
    preflight = json.loads((output_dir / "preflight.json").read_text())
    assert preflight["profiling_mode"] is None
    assert preflight["stages"][-1]["status"] == "failed"


def test_permission_only_ncu_failure_allows_events_fallback(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, *, log_path, env=None, timeout):
        calls.append(command)
        if command[0] == "ncu":
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                "==ERROR== ERR_NVGPUCTRPERM",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    output_dir = tmp_path / "trusted-pilot"
    monkeypatch.setattr(MODULE, "_run", fake_run)
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        ["run_trusted_pilot.py", "--output-dir", str(output_dir)],
    )

    assert MODULE.main() == 0
    assert any("scripts/smoke_llm.py" in command for command in calls)
    preflight = json.loads((output_dir / "preflight.json").read_text())
    assert preflight["profiling_mode"] == "events_only"
    assert preflight["stages"][2]["status"] == "events_only"


def test_smoke_is_one_non_retrying_bounded_request(monkeypatch, tmp_path):
    providers = []

    class FakeProvider:
        def __init__(self, settings):
            self.settings = settings
            self.calls = []
            providers.append(self)

        def public_config(self):
            return {"api_key_configured": True}

        async def generate(self, messages, *, model, n):
            self.calls.append({"messages": messages, "model": model, "n": n})
            return SimpleNamespace(
                response="KERNELBLASTER_OK",
                response_models=[model],
                provider="openai_compatible",
                request_ids=["req-test"],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                usage_source="provider",
                attempts=1,
                elapsed_time=0.01,
            )

    secret = "temporary-unit-test-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(SMOKE_MODULE, "OpenAICompatibleProvider", FakeProvider)
    args = SimpleNamespace(
        api_key_env="OPENAI_API_KEY",
        output_dir=tmp_path / "smoke",
        base_url="https://api.openai.com/v1",
        timeout_seconds=180.0,
        max_total_tokens=10_000,
        max_completion_tokens=64,
        reasoning_effort="none",
        model="gpt-5.6-sol",
    )

    assert asyncio.run(SMOKE_MODULE._run(args)) == 0
    assert len(providers) == 1
    provider = providers[0]
    assert len(provider.calls) == 1
    assert provider.calls[0]["n"] == 1
    assert provider.calls[0]["model"] == "gpt-5.6-sol"
    assert provider.settings.max_concurrency == 1
    assert provider.settings.max_retries == 0
    assert provider.settings.max_requests == 1
    assert provider.settings.max_total_tokens == 10_000
    assert provider.settings.max_completion_tokens == 64
    assert provider.settings.reasoning_effort == "none"

    artifacts = [
        args.output_dir / "run_manifest.json",
        args.output_dir / "events.jsonl",
        args.output_dir / "summary.json",
        args.output_dir / "smoke_result.json",
    ]
    assert all(path.is_file() for path in artifacts)
    assert secret not in "".join(path.read_text(encoding="utf-8") for path in artifacts)


def test_smoke_model_environment_cannot_override_fixed_model(monkeypatch):
    captured = []

    async def fake_run(args):
        captured.append(args)
        return 0

    monkeypatch.setenv("MODEL", "gpt-5.6-terra")
    monkeypatch.setattr(SMOKE_MODULE, "_run", fake_run)
    monkeypatch.setattr(SMOKE_MODULE.sys, "argv", ["smoke_llm.py"])

    assert SMOKE_MODULE.main() == 0
    assert captured[0].model == "gpt-5.6-sol"


def test_smoke_rejects_cli_model_override(monkeypatch):
    monkeypatch.setattr(
        SMOKE_MODULE.sys,
        "argv",
        ["smoke_llm.py", "--model", "gpt-5.6-terra"],
    )

    with pytest.raises(SystemExit) as error:
        SMOKE_MODULE.main()

    assert error.value.code == 2
