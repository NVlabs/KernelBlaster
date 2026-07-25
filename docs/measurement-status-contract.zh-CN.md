# 测量与状态契约（v3）

English version: [measurement-status-contract.md](measurement-status-contract.md).

新的性能数据必须是 `Measurement`，不能是没有单位的裸数。其 JSON 形式如下：

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

`cycles/ncu` 的值必须是整数；`us/cuda_events` 保留原生微秒值。只有 unit、source、硬件指纹和 protocol ID 都相同的测量才能排名。带有 `legacy_inferred_unit` 的旧测量不得自动排名。

`RunOutcome` 分别记录 execution、correctness、timing 和 diagnostic 状态，并保存稳定的 reason code。`profiling_mode` 现在是可选字段，不再默认等同于 NCU。旧版 v2 或无版本字段（`elapsed_cycles`、`elapsed_us`、`cycles`）仍可读取，但会给出 warning 并标记 `legacy_inferred_unit=true`；读取方不得将历史 artifact 重写为 v3。
