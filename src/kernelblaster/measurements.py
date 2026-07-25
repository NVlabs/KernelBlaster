"""Typed performance measurements and their compatibility contract.

Performance values are only meaningful in the context of a unit, collector,
hardware, and measurement protocol.  This module keeps that context attached
to every value written by new code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import subprocess
from typing import Any, Mapping
import warnings


MEASUREMENT_SCHEMA_VERSION = "3.0"


class MeasurementUnit(str, Enum):
    CYCLES = "cycles"
    MICROSECONDS = "us"


class MeasurementSource(str, Enum):
    NCU = "ncu"
    CUDA_EVENTS = "cuda_events"
    LEGACY_INFERRED = "legacy_inferred"


class MeasurementComparisonError(ValueError):
    """Raised when measurements do not share a comparison contract."""

    reason_code = "measurement_incomparable"


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def hardware_fingerprint(gpu: Any = None) -> str:
    """Return a stable, explicit fingerprint for the device used to measure.

    ``nvidia-smi`` is intentionally optional: CPU-only tests and managed remote
    workers still receive a deterministic configuration fingerprint rather than
    an unlabelled value.
    """
    configured_gpu = getattr(gpu, "value", gpu)
    payload: dict[str, Any] = {"configured_gpu": str(configured_gpu or "unknown")}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        payload["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        payload["nvidia_smi"] = "unavailable"
    return _canonical_fingerprint(payload)


@dataclass(frozen=True)
class Measurement:
    value: int | float
    unit: MeasurementUnit
    source: MeasurementSource
    samples: tuple[int | float, ...] = ()
    protocol_id: str = "unspecified"
    hardware_fingerprint: str = "unknown"
    legacy_inferred_unit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise TypeError("Measurement value must be numeric")
        if self.value <= 0:
            raise ValueError("Measurement value must be positive")
        if self.unit is MeasurementUnit.CYCLES and not isinstance(self.value, int):
            raise TypeError("NCU cycle measurements must be integral")
        if not self.protocol_id:
            raise ValueError("Measurement protocol_id is required")
        if not self.hardware_fingerprint:
            raise ValueError("Measurement hardware_fingerprint is required")

    @property
    def comparison_key(self) -> tuple[str, str, str, str]:
        return (
            self.unit.value,
            self.source.value,
            self.hardware_fingerprint,
            self.protocol_id,
        )

    def assert_comparable(self, other: "Measurement") -> None:
        if self.legacy_inferred_unit or other.legacy_inferred_unit:
            raise MeasurementComparisonError(
                "Legacy-inferred measurements cannot be ranked without explicit migration."
            )
        if self.comparison_key != other.comparison_key:
            raise MeasurementComparisonError(
                "Measurements differ in unit, source, hardware fingerprint, or protocol."
            )

    def is_faster_than(self, other: "Measurement") -> bool:
        self.assert_comparable(other)
        return self.value < other.value

    def speedup_over(self, baseline: "Measurement") -> float:
        self.assert_comparable(baseline)
        return float(baseline.value) / float(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MEASUREMENT_SCHEMA_VERSION,
            "value": self.value,
            "unit": self.unit.value,
            "source": self.source.value,
            "samples": list(self.samples),
            "protocol_id": self.protocol_id,
            "hardware_fingerprint": self.hardware_fingerprint,
            "legacy_inferred_unit": self.legacy_inferred_unit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Measurement":
        if payload.get("schema_version") == MEASUREMENT_SCHEMA_VERSION or {
            "value", "unit", "source", "protocol_id", "hardware_fingerprint"
        }.issubset(payload):
            return cls(
                value=payload["value"],
                unit=MeasurementUnit(payload["unit"]),
                source=MeasurementSource(payload["source"]),
                samples=tuple(payload.get("samples") or ()),
                protocol_id=str(payload["protocol_id"]),
                hardware_fingerprint=str(payload["hardware_fingerprint"]),
                legacy_inferred_unit=bool(payload.get("legacy_inferred_unit", False)),
            )
        return cls.from_legacy_dict(payload)

    @classmethod
    def from_legacy_dict(cls, payload: Mapping[str, Any]) -> "Measurement":
        if "elapsed_us" in payload:
            value, unit = payload["elapsed_us"], MeasurementUnit.MICROSECONDS
        elif "elapsed_cycles" in payload:
            value, unit = payload["elapsed_cycles"], MeasurementUnit.CYCLES
        elif "cycles" in payload:
            value, unit = payload["cycles"], MeasurementUnit.CYCLES
        else:
            raise ValueError("Legacy artifact contains no recognizable measurement value")
        warnings.warn(
            "Loading a legacy measurement with an inferred unit; it cannot be ranked automatically.",
            UserWarning,
            stacklevel=2,
        )
        return cls(
            value=int(value) if unit is MeasurementUnit.CYCLES else float(value),
            unit=unit,
            source=MeasurementSource.LEGACY_INFERRED,
            samples=(),
            protocol_id="legacy-unversioned",
            hardware_fingerprint="legacy-unknown",
            legacy_inferred_unit=True,
        )


def format_measurement(measurement: Measurement | None) -> str:
    if measurement is None:
        return "unavailable"
    return f"{measurement.value} {measurement.unit.value} ({measurement.source.value})"


__all__ = [
    "MEASUREMENT_SCHEMA_VERSION",
    "Measurement",
    "MeasurementComparisonError",
    "MeasurementSource",
    "MeasurementUnit",
    "format_measurement",
    "hardware_fingerprint",
]
