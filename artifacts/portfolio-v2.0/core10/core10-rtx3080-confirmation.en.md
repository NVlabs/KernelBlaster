# RTX 3080 Core 10 schema-v2 full confirmation

**English** | [简体中文](core10-rtx3080-confirmation.zh-CN.md)

This is a full Core 10 confirmation of manual candidates, not an Agent-generated search result. A formal improvement requires correctness, five independent process Sessions, at most 5% spread on both sides, at least 1.01× median speedup, and a paired-bootstrap 95% lower bound above 1.0.

| Task | Candidate outcome | Candidate speedup | Bootstrap 95% lower | Baseline/candidate spread | Stable PyTorch baseline | Strict selected / PyTorch |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 004 | improved | 182.275× | 180.537× | 1.375% / 0.192% | pytorch_eager | 1.032× |
| 007 | improved | 11.829× | 11.777× | 3.736% / 0.247% | pytorch_preallocated_out | 1.006× |
| 019 | inconclusive | 1.000× | 0.974× | 5.607% / 2.750% | pytorch_preallocated_out | 2.013× |
| 023 | inconclusive | 1.013× | 1.004× | 20.895% / 0.798% | pytorch_preallocated_out | 0.798× |
| 026 | inconclusive | 0.993× | 0.974× | 7.536% / 6.947% | — | — |
| 036 | improved | 54.228× | 54.068× | 1.893% / 0.344% | pytorch_eager | 1.764× |
| 040 | improved | 22.254× | 22.113× | 1.470% / 0.827% | pytorch_eager | 9.228× |
| 047 | inconclusive | 1.436× | 1.387× | 24.271% / 5.560% | pytorch_eager | 0.576× |
| 088 | no improvement | 1.009× | 1.000× | 1.101% / 0.171% | pytorch_fused_gelu_tanh | 0.990× |
| 095 | inconclusive | 28.566× | 26.681× | 13.876% / 4.838% | pytorch_eager | 0.103× |

The strict result contains 4 improvements, 1 no-improvement result, and 5 inconclusive results. The strict Core 10 geometric mean versus upstream is 4.381×. A correct and stable PyTorch method exists for 9/10 tasks; on only those comparable tasks, the strict geometric mean versus the fastest stable PyTorch method is 1.053×.

Task 026 has no stable PyTorch method and is excluded from the PyTorch geometric mean. Tasks 019, 023, 026, 047, and 095 still fail the CUDA Session-stability gate after automatic retesting, so their diagnostic speedups are not formal claims. NCU counters remain unavailable and the local profiling mode is `events_only`.

Canonical machine-readable evidence: [JSON](core10_rtx3080_comparison.json).
