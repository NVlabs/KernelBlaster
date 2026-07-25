from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.run_RL as run_rl
from scripts.run_RL import resolve_portfolio_suite, resolve_target_gpu
from src.kernelblaster.config import GPUType


ROOT = Path(__file__).resolve().parents[2]
RMSNORM_SUITE = ROOT / "portfolio" / "suites" / "rmsnorm.json"


def test_explicit_remote_gpu_does_not_probe_local_hardware():
    with patch.object(GPUType, "current", side_effect=AssertionError("unexpected probe")):
        assert resolve_target_gpu("l40s") is GPUType.L40S


def test_unspecified_remote_gpu_uses_local_hardware():
    with patch.object(GPUType, "current", return_value=GPUType.RTX3080):
        assert resolve_target_gpu(None) is GPUType.RTX3080


def test_rmsnorm_portfolio_suite_resolves_exact_pilot_contract():
    suite = resolve_portfolio_suite(
        RMSNORM_SUITE,
        problem_numbers="36",
        rollouts=2,
        steps=2,
        trusted_pilot=True,
    )

    assert suite["source"] == "portfolio/suites/rmsnorm.json"
    assert suite["resolved"] == {
        "task_ids": ["036"],
        "rollouts": 2,
        "steps": 2,
    }


def test_non_pilot_rmsnorm_suite_keeps_generic_rollout_defaults():
    suite = resolve_portfolio_suite(
        RMSNORM_SUITE,
        problem_numbers="36",
        rollouts=8,
        steps=5,
    )

    assert suite["resolved"]["rollouts"] == 8
    assert suite["resolved"]["steps"] == 5


def test_portfolio_suite_rejects_task_mismatch(tmp_path):
    suite_path = tmp_path / "wrong-suite.json"
    suite_path.write_text(
        json.dumps({"tasks": [{"id": "040", "number": 40}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match"):
        resolve_portfolio_suite(
            suite_path,
            problem_numbers="36",
            rollouts=2,
            steps=2,
        )


def test_trusted_pilot_rejects_noncanonical_suite_copy(tmp_path):
    suite_path = tmp_path / "rmsnorm-copy.json"
    suite_path.write_text(
        json.dumps({"tasks": [{"id": "036", "number": 36}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="portfolio/suites/rmsnorm.json"):
        resolve_portfolio_suite(
            suite_path,
            problem_numbers="36",
            rollouts=2,
            steps=2,
            trusted_pilot=True,
        )


def test_run_rl_rejects_wrong_trusted_suite_before_dataset_or_provider(
    tmp_path, monkeypatch
):
    suite_path = tmp_path / "rmsnorm-copy.json"
    suite_path.write_text(
        json.dumps({"tasks": [{"id": "036", "number": 36}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_rl.sys,
        "argv",
        [
            "run_RL.py",
            "--experiment-name",
            "trusted-rmsnorm-pilot",
            "--problem-numbers",
            "36",
            "--portfolio-suite",
            str(suite_path),
            "--rl-iterations",
            "2",
            "--rl-rollout-steps",
            "2",
        ],
    )
    monkeypatch.setattr(GPUType, "current", lambda: GPUType.RTX3080)
    monkeypatch.setattr(
        run_rl,
        "get_dataset",
        lambda *_args, **_kwargs: pytest.fail("dataset must not be loaded"),
    )
    monkeypatch.setattr(
        run_rl,
        "get_llm_provider",
        lambda *_args, **_kwargs: pytest.fail("provider must not be initialized"),
    )

    with pytest.raises(SystemExit) as error:
        asyncio.run(run_rl.async_main())

    assert error.value.code == 2


@pytest.mark.parametrize(("rollouts", "steps"), [(3, 2), (2, 3)])
def test_rmsnorm_portfolio_suite_rejects_wrong_pilot_shape(rollouts, steps):
    with pytest.raises(ValueError, match="2 rollouts x 2 steps"):
        resolve_portfolio_suite(
            RMSNORM_SUITE,
            problem_numbers="36",
            rollouts=rollouts,
            steps=steps,
            trusted_pilot=True,
        )
