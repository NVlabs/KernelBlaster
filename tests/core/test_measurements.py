from __future__ import annotations

from pathlib import Path

import pytest

from src.kernelblaster.measurements import (
    Measurement,
    MeasurementComparisonError,
    MeasurementSource,
    MeasurementUnit,
)
from src.kernelblaster.profiling import EventsProfilerBackend


def _measurement(**overrides) -> Measurement:
    values = {
        "value": 12.5,
        "unit": MeasurementUnit.MICROSECONDS,
        "source": MeasurementSource.CUDA_EVENTS,
        "samples": (12.4, 12.5, 12.6),
        "protocol_id": "events-v1",
        "hardware_fingerprint": "gpu-a",
    }
    values.update(overrides)
    return Measurement(**values)


@pytest.mark.asyncio
async def test_events_measurement_keeps_microseconds_and_source():
    async def runner(_filepath: Path, *, sessions: int):
        return [12.5] * sessions

    result = await EventsProfilerBackend(runner, protocol_id="events-v1").profile(
        Path("candidate.cu")
    )

    assert result.measurement is not None
    assert result.measurement.unit is MeasurementUnit.MICROSECONDS
    assert result.measurement.source is MeasurementSource.CUDA_EVENTS
    assert result.measurement.value == 12.5


def test_incompatible_measurements_fail_with_clear_error():
    events = _measurement()
    cycles = _measurement(
        value=12345,
        unit=MeasurementUnit.CYCLES,
        source=MeasurementSource.NCU,
    )
    with pytest.raises(MeasurementComparisonError, match="unit, source"):
        events.is_faster_than(cycles)
    with pytest.raises(MeasurementComparisonError):
        events.is_faster_than(_measurement(hardware_fingerprint="gpu-b"))


def test_legacy_measurement_warns_and_is_not_ranked():
    with pytest.warns(UserWarning, match="legacy measurement"):
        legacy = Measurement.from_dict({"elapsed_cycles": 12345})
    assert legacy.legacy_inferred_unit is True
    with pytest.raises(MeasurementComparisonError, match="Legacy-inferred"):
        legacy.is_faster_than(legacy)


def test_ncu_measurement_requires_integral_cycles():
    measurement = Measurement(
        value=12345,
        unit=MeasurementUnit.CYCLES,
        source=MeasurementSource.NCU,
        protocol_id="ncu-v1",
        hardware_fingerprint="gpu-a",
    )
    assert measurement.to_dict()["value"] == 12345
    with pytest.raises(TypeError):
        Measurement(
            value=12.5,
            unit=MeasurementUnit.CYCLES,
            source=MeasurementSource.NCU,
            protocol_id="ncu-v1",
            hardware_fingerprint="gpu-a",
        )
