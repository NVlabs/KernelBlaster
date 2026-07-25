from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import AsyncMock

from loguru import logger
import pytest

from src.kernelblaster.agents.database import GPUOptimizationDatabase
from src.kernelblaster.agents.opt_ncu_rl import RLNCUAgent
from src.kernelblaster.agents.rl_agents import ReplayBuffer, Trajectory
from src.kernelblaster.config import GPUType, WorkflowConfig
from src.kernelblaster.outcomes import (
    CorrectnessStatus,
    DiagnosticStatus,
    ExecutionStatus,
    ReasonCode,
    RunOutcome,
    RunStatus,
    TimingStatus,
)
from src.kernelblaster.profiling import (
    EventsProfilerBackend,
    NCUFallbackProfilerBackend,
    ProfilingMode,
    ProfilingResult,
    evaluate_performance_gate,
)
from src.kernelblaster.workflow import workflow as workflow_module


def _workflow_config(*, retry_failed: bool = False) -> WorkflowConfig:
    return WorkflowConfig(
        model="unit-model",
        run_cuda=True,
        run_cuda_perf=True,
        run_cuda_bench=False,
        run_cuda_perf_bench=False,
        retry_failed=retry_failed,
        gpu=GPUType.RTX3080,
    )


def test_workflow_config_does_not_copy_or_serialize_shared_database():
    config = _workflow_config()
    config.shared_optimization_database = SimpleNamespace(lock=threading.RLock())

    payload = config.dict()

    assert "shared_optimization_database" not in payload
    assert payload["gpu"] is GPUType.RTX3080


def test_run_outcome_requires_both_improved_status_and_artifact(tmp_path):
    artifact = tmp_path / "candidate.cu"
    artifact.write_text("// candidate\n", encoding="utf-8")
    assert RunOutcome(RunStatus.IMPROVED, artifact).success is True
    assert RunOutcome(RunStatus.IMPROVED).success is False
    assert RunOutcome(RunStatus.IMPROVED, tmp_path / "missing.cu").success is False
    assert RunOutcome(RunStatus.NO_IMPROVEMENT, artifact).success is False
    assert RunOutcome(RunStatus.FAILED, artifact).success is False


@pytest.mark.asyncio
async def test_workflow_does_not_promote_no_improvement_and_forwards_shared_db(
    tmp_path, monkeypatch
):
    captured = {}

    class FakeGraph:
        async def ainvoke(self, state):
            captured.update(state)
            return {
                **state,
                "run_outcome": RunOutcome(
                    status=RunStatus.NO_IMPROVEMENT,
                    reason="formal gate failed",
                    profiling_mode="events_only",
                ).to_dict(),
            }

    monkeypatch.setattr(workflow_module, "build_graph", lambda: FakeGraph())
    shared_db = object()
    result = await workflow_module.run_workflow(
        "036",
        "optimize",
        "",
        tmp_path / "task",
        _workflow_config(),
        logger.bind(problem_id="036"),
        timeout_seconds=10,
        shared_database=shared_db,
    )
    assert result.success is False
    assert result.outcome.status is RunStatus.NO_IMPROVEMENT
    assert result.rl_cuda_perf_filepath is None
    assert captured["shared_optimization_database"] is shared_db
    assert (tmp_path / "task" / "failed_rl_cuda_perf").is_file()


def test_resume_skips_success_and_respects_retry_failed(tmp_path):
    config = _workflow_config(retry_failed=False)
    success = tmp_path / "success"
    success.mkdir()
    (success / "final_rl_cuda_perf.cu").write_text("// ok\n", encoding="utf-8")
    assert config.should_skip_folder(success) is True

    failed = tmp_path / "failed"
    (failed / "rl_ncu").mkdir(parents=True)
    (failed / "failed_rl_cuda_perf").write_text("failed\n", encoding="utf-8")
    (failed / "rl_ncu" / ".finished").write_text("failed\n", encoding="utf-8")
    assert config.should_skip_folder(failed) is True
    assert _workflow_config(retry_failed=True).should_skip_folder(failed) is False


@pytest.mark.asyncio
async def test_trajectory_count_and_policy_update_frequency():
    agent = RLNCUAgent.__new__(RLNCUAgent)
    agent.replay_buffer = ReplayBuffer(max_size=10)
    agent._trajectory_lock = asyncio.Lock()
    agent._policy_lock = asyncio.Lock()
    agent.total_trajectories = 0
    agent.iteration_count = 0
    agent.update_frequency = 2
    agent.policy_update_cycle = AsyncMock()

    await agent._record_completed_trajectory(Trajectory())
    await agent._record_completed_trajectory(Trajectory())
    await agent._record_completed_trajectory(Trajectory())
    assert agent.total_trajectories == 3
    assert len(agent.replay_buffer.trajectories) == 3
    assert agent.iteration_count == 1
    agent.policy_update_cycle.assert_awaited_once()


def test_state_transition_never_writes_none():
    assert RLNCUAgent.next_performance_state("compute_bound", None) == "compute_bound"
    assert RLNCUAgent.next_performance_state(None, None) == "events_only/unknown"
    assert RLNCUAgent.next_performance_state("old", "new") == "new"


@pytest.mark.asyncio
async def test_events_only_profile_never_invents_an_ncu_state():
    agent = RLNCUAgent.__new__(RLNCUAgent)
    agent.profiling_mode = "events_only"
    agent.database = SimpleNamespace(get_state_from_ncu_report=AsyncMock())

    state = await agent._classify_profile_state("", {}, "// kernel", 123)

    assert state == "events_only/unknown"
    agent.database.get_state_from_ncu_report.assert_not_awaited()


def test_database_persistence_is_atomic_schema_v2(tmp_path):
    db = GPUOptimizationDatabase.__new__(GPUOptimizationDatabase)
    db._persist_json_fp = tmp_path / "optimization_database.json"
    db.optimization_strategies = {}
    db.known_states = {}
    db.composite_optimizations = {}
    db.discovered_states = {}
    db._io_lock = threading.RLock()
    db.llm_interface = SimpleNamespace(logger=None)
    db._persist_database()
    payload = json.loads(db._persist_json_fp.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "3.0"
    assert not list(tmp_path.glob("*.tmp"))


def test_five_session_bootstrap_performance_gate():
    passed = evaluate_performance_gate([10.0] * 5, [9.0] * 5)
    assert passed.passed is True
    assert passed.median_speedup == pytest.approx(10.0 / 9.0)
    assert passed.bootstrap_95_lower > 1.0

    too_small = evaluate_performance_gate([10.0] * 5, [9.95] * 5)
    assert too_small.passed is False
    too_few = evaluate_performance_gate([10.0] * 3, [9.0] * 3)
    assert too_few.passed is False


@pytest.mark.asyncio
async def test_ncu_permission_failure_downgrades_to_events():
    class NCUBackend:
        mode = ProfilingMode.NCU

        async def profile(self, _filepath):
            return ProfilingResult(
                mode=ProfilingMode.NCU,
                stderr="ERR_NVGPUCTRPERM",
                error="NCU failed",
            )

    async def events_runner(_filepath: Path, *, sessions: int):
        return [12.5] * sessions

    events = EventsProfilerBackend(events_runner)
    backend = NCUFallbackProfilerBackend(NCUBackend(), events)
    result = await backend.profile(Path("candidate.cu"))
    assert result.available is True
    assert result.mode is ProfilingMode.EVENTS_ONLY
    assert result.elapsed_us == 12.5


def test_run_outcome_v3_preserves_independent_statuses():
    outcome = RunOutcome(
        status=RunStatus.NO_IMPROVEMENT,
        execution_status=ExecutionStatus.SUCCEEDED,
        correctness_status=CorrectnessStatus.PASSED,
        timing_status=TimingStatus.MEASURED,
        diagnostic_status=DiagnosticStatus.UNAVAILABLE,
        reason_code=ReasonCode.PERFORMANCE_GATE_FAILED,
    )
    payload = outcome.to_dict()
    assert payload["profiling_mode"] is None
    assert payload["timing_status"] == "measured"
    assert RunOutcome.from_dict(payload).reason_code is ReasonCode.PERFORMANCE_GATE_FAILED
