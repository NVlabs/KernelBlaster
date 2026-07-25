# Measurement and status contract (v3)

Chinese version: [measurement-status-contract.zh-CN.md](measurement-status-contract.zh-CN.md).

New performance data is a `Measurement`, never a bare number. Its JSON form is:

```json
{
  "schema_version": "3.0",
  "value": 12.5,
  "unit": "us",
  "source": "cuda_events",
  "samples": [12.4, 12.5, 12.7],
  "protocol_id": "cuda-events:warmup=20:repetitions=100:discovery_sessions=3",
  "hardware_fingerprint": "sha256:...",
  "legacy_inferred_unit": false
}
```

`cycles/ncu` values are integral; `us/cuda_events` values retain their native
microsecond value. Measurements may be ranked only when their unit, source,
hardware fingerprint, and protocol ID match. A legacy-inferred measurement is
never ranked automatically.

`RunOutcome` records independent execution, correctness, timing, and diagnostic
statuses plus a stable reason code. `profiling_mode` is optional and no longer
defaults to NCU. Legacy v2 or unversioned fields (`elapsed_cycles`, `elapsed_us`,
or `cycles`) can be read with a warning and `legacy_inferred_unit=true`; readers
must not rewrite the historical artifact as v3.
