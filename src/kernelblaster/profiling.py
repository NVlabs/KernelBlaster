
"""抽象 CUDA 性能分析后端，并实现统计置信区间和正确性优先的性能门控。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Protocol

from .measurements import (
    Measurement,
    MeasurementSource,
    MeasurementUnit,
    hardware_fingerprint,
)


class ProfilingMode(str, Enum):
    """封装 `ProfilingMode` 对应的领域状态与操作。"""
    NCU = "ncu"
    EVENTS_ONLY = "events_only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ProfilingResult:
    """保存一次操作的标准化结果及其诊断信息。"""
    mode: ProfilingMode
    measurement: Measurement | None = None
    # Temporary read compatibility for third-party profiler backends. New
    # backends must populate ``measurement`` instead.
    elapsed_cycles: int | None = None
    elapsed_us: float | None = None
    annotated_source: str = ""
    raw_output: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.measurement is not None:
            return
        if self.elapsed_cycles is not None:
            object.__setattr__(
                self,
                "measurement",
                Measurement(
                    value=self.elapsed_cycles,
                    unit=MeasurementUnit.CYCLES,
                    source=MeasurementSource.NCU,
                    protocol_id="legacy-profiler-result",
                    hardware_fingerprint="legacy-unknown",
                    legacy_inferred_unit=True,
                ),
            )
        elif self.elapsed_us is not None:
            object.__setattr__(
                self,
                "measurement",
                Measurement(
                    value=self.elapsed_us,
                    unit=MeasurementUnit.MICROSECONDS,
                    source=MeasurementSource.CUDA_EVENTS,
                    protocol_id="legacy-profiler-result",
                    hardware_fingerprint="legacy-unknown",
                    legacy_inferred_unit=True,
                ),
            )

    @property
    def available(self) -> bool:
        """
        处理 `available` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return self.error is None and self.measurement is not None


class ProfilerBackend(Protocol):
    """实现统一接口背后的具体执行与结果转换逻辑。"""
    mode: ProfilingMode

    async def profile(self, filepath: Path) -> ProfilingResult:
        """
        分析指定 CUDA 候选并返回统一的性能结果。

        参数:
        filepath: 待分析的 CUDA 源文件路径。

        返回:
        包含分析模式、耗时指标和诊断信息的性能结果。
        """
        ...


@dataclass(frozen=True)
class PerformanceGateResult:
    """保存一次操作的标准化结果及其诊断信息。"""
    passed: bool
    median_speedup: float | None
    bootstrap_95_lower: float | None
    bootstrap_95_upper: float | None
    session_speedups: tuple[float, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        处理 `to_dict` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return {
            "passed": self.passed,
            "median_speedup": self.median_speedup,
            "bootstrap_95_lower": self.bootstrap_95_lower,
            "bootstrap_95_upper": self.bootstrap_95_upper,
            "session_speedups": list(self.session_speedups),
            "reason": self.reason,
        }


def paired_bootstrap_interval(
    speedups: list[float],
    *,
    seed: int = 20260719,
    resamples: int = 10_000,
) -> tuple[float, float] | None:
    """
    对配对基线与候选样本执行 Bootstrap，估计加速比的置信区间。

    参数:
    speedups: 调用方提供的 `speedups` 参数。
    seed: 调用方提供的 `seed` 参数。
    resamples: 调用方提供的 `resamples` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    if len(speedups) < 2:
        return None
    rng = random.Random(seed)
    estimates = [
        statistics.median(rng.choices(speedups, k=len(speedups)))
        for _ in range(resamples)
    ]
    estimates.sort()
    return (
        estimates[int(0.025 * (len(estimates) - 1))],
        estimates[int(0.975 * (len(estimates) - 1))],
    )


def evaluate_performance_gate(
    baseline_session_us: list[float],
    candidate_session_us: list[float],
    *,
    minimum_sessions: int = 5,
    minimum_speedup: float = 1.01,
) -> PerformanceGateResult:
    """
    结合基线配对样本、置信区间和正确性状态判断候选能否通过性能门控。

    参数:
        baseline_session_us: 各独立会话测得的基线延迟，单位为微秒。
        candidate_session_us: 与基线按会话配对的候选延迟，单位为微秒。
        minimum_sessions: 允许执行统计判断所需的最少配对会话数。
        minimum_speedup: 候选中位加速比必须达到的下限。

    返回:
        包含是否通过、中位加速比、Bootstrap 区间和失败原因的门控结果。
    """
    if len(baseline_session_us) != len(candidate_session_us):
        return PerformanceGateResult(
            passed=False,
            median_speedup=None,
            bootstrap_95_lower=None,
            bootstrap_95_upper=None,
            reason="Baseline and candidate confirmation sessions are not paired.",
        )
    if len(baseline_session_us) < minimum_sessions:
        return PerformanceGateResult(
            passed=False,
            median_speedup=None,
            bootstrap_95_lower=None,
            bootstrap_95_upper=None,
            reason=f"At least {minimum_sessions} confirmation sessions are required.",
        )
    if any(value <= 0 or not math.isfinite(value) for value in baseline_session_us):
        return PerformanceGateResult(
            passed=False,
            median_speedup=None,
            bootstrap_95_lower=None,
            bootstrap_95_upper=None,
            reason="Baseline sessions contain a non-positive or non-finite latency.",
        )
    if any(value <= 0 or not math.isfinite(value) for value in candidate_session_us):
        return PerformanceGateResult(
            passed=False,
            median_speedup=None,
            bootstrap_95_lower=None,
            bootstrap_95_upper=None,
            reason="Candidate sessions contain a non-positive or non-finite latency.",
        )

    speedups = [
        baseline / candidate
        for baseline, candidate in zip(
            baseline_session_us, candidate_session_us, strict=True
        )
    ]
    interval = paired_bootstrap_interval(speedups)
    median_speedup = statistics.median(speedups)
    lower, upper = interval if interval is not None else (None, None)
    passed = bool(
        median_speedup >= minimum_speedup
        and lower is not None
        and lower > 1.0
    )
    reason = None
    if not passed:
        reason = (
            f"Performance gate failed: median speedup={median_speedup:.6f}, "
            f"bootstrap_95_lower={lower}."
        )
    return PerformanceGateResult(
        passed=passed,
        median_speedup=median_speedup,
        bootstrap_95_lower=lower,
        bootstrap_95_upper=upper,
        session_speedups=tuple(speedups),
        reason=reason,
    )


class EventsRunner(Protocol):
    """封装外部命令或测量过程，并返回可测试的结构化结果。"""
    async def __call__(
        self,
        filepath: Path,
        *,
        sessions: int,
    ) -> list[float]:
        """
        运行若干独立 CUDA Events 测量会话。

        参数:
        filepath: 待分析的 CUDA 源文件路径。
        sessions: 需要采集的独立会话数量。

        返回:
        每个会话得到的代表性延迟样本。
        """
        ...


class EventsProfilerBackend:
    """由 CUDA 事件会话中位数驱动的搜索和确认后端。"""

    mode = ProfilingMode.EVENTS_ONLY

    def __init__(
        self,
        runner: EventsRunner,
        *,
        discovery_sessions: int = 3,
        confirmation_sessions: int = 5,
        gpu: Any = None,
        protocol_id: str | None = None,
    ) -> None:
        """
        初始化 EventsProfilerBackend 实例，并保存后续流程所需的配置与依赖。

        参数:
        runner: 调用方提供的 `runner` 参数。
        discovery_sessions: 调用方提供的 `discovery_sessions` 参数。
        confirmation_sessions: 调用方提供的 `confirmation_sessions` 参数。
        """
        self.runner = runner
        self.discovery_sessions = discovery_sessions
        self.confirmation_sessions = confirmation_sessions
        self._hardware_fingerprint = hardware_fingerprint(gpu)
        self.protocol_id = protocol_id or (
            f"cuda-events:warmup={getattr(runner, 'warmup', 'unknown')}:"
            f"repetitions={getattr(runner, 'repetitions', 'unknown')}:"
            f"discovery_sessions={discovery_sessions}"
        )

    async def profile(self, filepath: Path) -> ProfilingResult:
        """
        处理 `profile` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        filepath: 目标文件路径。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        samples = await self.runner(filepath, sessions=self.discovery_sessions)
        if len(samples) != self.discovery_sessions:
            return ProfilingResult(
                mode=self.mode,
                error="CUDA Events runner returned an unexpected session count.",
            )
        median_us = statistics.median(samples)
        return ProfilingResult(
            mode=self.mode,
            measurement=Measurement(
                value=median_us,
                unit=MeasurementUnit.MICROSECONDS,
                source=MeasurementSource.CUDA_EVENTS,
                samples=tuple(samples),
                protocol_id=self.protocol_id,
                hardware_fingerprint=self._hardware_fingerprint,
            ),
            elapsed_us=median_us,
            metrics={"session_medians_us": samples, "phase": "discovery"},
        )

    async def confirm_pair(
        self,
        baseline: Path,
        candidate: Path,
    ) -> PerformanceGateResult:
        """
        处理 `confirm_pair` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        baseline: 作为正确性或性能比较基准的数据。
        candidate: 当前正在验证或评估的候选实现。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
        ProfilerUnavailable: 输入、外部调用或状态不满足执行要求时抛出。
        """
        baseline_samples: list[float] = []
        candidate_samples: list[float] = []
        for session in range(self.confirmation_sessions):
            order = (
                ((baseline, baseline_samples), (candidate, candidate_samples))
                if session % 2 == 0
                else ((candidate, candidate_samples), (baseline, baseline_samples))
            )
            for filepath, destination in order:
                values = await self.runner(filepath, sessions=1)
                if len(values) != 1:
                    raise ProfilerUnavailable(
                        "CUDA Events confirmation runner did not return one session."
                    )
                destination.append(values[0])
        return evaluate_performance_gate(
            baseline_samples,
            candidate_samples,
            minimum_sessions=self.confirmation_sessions,
        )


class CudaEventsRunner:
    """编译检测正确性驱动程序并执行单独的会话。"""

    def __init__(
        self,
        *,
        driver_path: Path,
        gpu: Any,
        logger: Any,
        work_dir: Path,
        warmup: int = 20,
        repetitions: int = 100,
        seed: int = 20260719,
        timeout: float = 1200,
    ) -> None:
        """
        初始化 CudaEventsRunner 实例，并保存后续流程所需的配置与依赖。

        参数:
        driver_path: 调用方提供的 `driver_path` 参数。
        gpu: 执行或分析任务使用的 GPU 配置。
        logger: 记录诊断信息和任务进度的日志器。
        work_dir: 调用方提供的 `work_dir` 参数。
        warmup: 调用方提供的 `warmup` 参数。
        repetitions: 调用方提供的 `repetitions` 参数。
        seed: 调用方提供的 `seed` 参数。
        timeout: 允许操作等待的最长秒数。
        """
        self.driver_path = driver_path
        self.gpu = gpu
        self.logger = logger
        self.work_dir = work_dir
        self.warmup = warmup
        self.repetitions = repetitions
        self.seed = seed
        self.timeout = timeout
        self._counter = 0
        self._compiled: dict[tuple[str, str], Path] = {}

    async def __call__(self, filepath: Path, *, sessions: int) -> list[float]:
        """
        处理 `__call__` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        filepath: 目标文件路径。
        sessions: 调用方提供的 `sessions` 参数。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

        异常:
        ProfilerUnavailable: 输入、外部调用或状态不满足执行要求时抛出。
        """
        from .agents.utils import (
            NamedTimer,
            compile_and_run_cu_file,
            run_gpu_executable,
        )
        from .benchmarking import BENCHMARK_MARKER, instrument_driver

        source_digest = hashlib.sha256(filepath.read_bytes()).hexdigest()
        cache_key = (str(filepath.resolve()), source_digest)
        compiled = self._compiled.get(cache_key)
        if compiled is None:
            self._counter += 1
            driver = instrument_driver(
                self.driver_path.read_text(encoding="utf-8"),
                seed=self.seed,
                warmup=self.warmup,
                repetitions=self.repetitions,
                inner_loops=0,
            )
            instrumented_path = self.work_dir / (
                f"events_driver_{self._counter}_{filepath.stem}.cpp"
            )
            instrumented_path.parent.mkdir(parents=True, exist_ok=True)
            instrumented_path.write_text(driver, encoding="utf-8")
            stdout, _stderr, binary, success = await compile_and_run_cu_file(
                instrumented_path,
                filepath,
                self.gpu,
                NamedTimer(),
                self.logger,
                persistent_artifacts=True,
                timeout=self.timeout,
                num_runs=sessions,
                passed_keyword="passed",
            )
            if not success:
                raise ProfilerUnavailable("CUDA Events correctness execution failed.")
            compiled = Path(binary)
            self._compiled[cache_key] = compiled
        else:
            raw_stdout, _stderr = await run_gpu_executable(
                compiled,
                self.gpu,
                self.timeout,
                job_name=f"{filepath} (CUDA Events confirmation)",
                n_runs=sessions,
            )
            stdout = [raw_stdout] if isinstance(raw_stdout, str) else raw_stdout
        values: list[float] = []
        for session_stdout in stdout:
            markers = [
                line
                for line in session_stdout.splitlines()
                if line.startswith(BENCHMARK_MARKER)
            ]
            if len(markers) != 1:
                raise ProfilerUnavailable(
                    "CUDA Events output did not contain exactly one benchmark marker."
                )
            payload = json.loads(markers[0][len(BENCHMARK_MARKER) :])
            samples = [float(value) for value in payload["samples_us"]]
            if not samples:
                raise ProfilerUnavailable("CUDA Events session contained no samples.")
            values.append(statistics.median(samples))
        return values


class NCUFallbackProfilerBackend:
    """在可用时使用 NCU，并仅将权限失败降级为事件。"""

    def __init__(
        self,
        ncu_backend: ProfilerBackend,
        events_backend: EventsProfilerBackend,
    ) -> None:
        """
        初始化 NCUFallbackProfilerBackend 实例，并保存后续流程所需的配置与依赖。

        参数:
        ncu_backend: 调用方提供的 `ncu_backend` 参数。
        events_backend: 调用方提供的 `events_backend` 参数。
        """
        self.ncu_backend = ncu_backend
        self.events_backend = events_backend
        self.mode = ncu_backend.mode

    async def profile(self, filepath: Path) -> ProfilingResult:
        """
        处理 `profile` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        filepath: 目标文件路径。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        result = await self.ncu_backend.profile(filepath)
        combined_error = "\n".join(
            value for value in (result.error, result.stderr, result.raw_output) if value
        )
        if result.available or not ncu_permission_blocked(combined_error):
            self.mode = result.mode
            return result
        self.mode = ProfilingMode.EVENTS_ONLY
        return await self.events_backend.profile(filepath)

    async def confirm_pair(
        self,
        baseline: Path,
        candidate: Path,
    ) -> PerformanceGateResult:
        """
        处理 `confirm_pair` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
        baseline: 作为正确性或性能比较基准的数据。
        candidate: 当前正在验证或评估的候选实现。

        返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return await self.events_backend.confirm_pair(baseline, candidate)


class ProfilerUnavailable(RuntimeError):
    """封装 `ProfilerUnavailable` 对应的领域状态与操作。"""
    pass


def ncu_permission_blocked(output: str) -> bool:
    """
    处理 `ncu_permission_blocked` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
    output: 调用方提供的 `output` 参数。

    返回:
    当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return "ERR_NVGPUCTRPERM" in output


__all__ = [
    "CudaEventsRunner",
    "EventsProfilerBackend",
    "NCUFallbackProfilerBackend",
    "PerformanceGateResult",
    "ProfilerBackend",
    "ProfilerUnavailable",
    "ProfilingMode",
    "ProfilingResult",
    "evaluate_performance_gate",
    "ncu_permission_blocked",
    "paired_bootstrap_interval",
]
