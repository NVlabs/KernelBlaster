# RTX 3080 Core 10 schema-v2 完整确认

**简体中文** | [English](core10-rtx3080-confirmation.en.md)

本报告是手工候选的完整 Core 10 确认，不是 Agent 自动搜索结果。正式提升要求正确、五个独立进程 Session、双方 spread 不超过 5%、中位加速至少 1.01×，且配对 bootstrap 95% 下界大于 1.0。

| 任务 | 候选结论 | 候选加速 | Bootstrap 95% 下界 | 基线/候选 spread | 稳定 PyTorch 基线 | 严格选中结果 / PyTorch |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 004 | 正式提升 | 182.275× | 180.537× | 1.375% / 0.192% | pytorch_eager | 1.032× |
| 007 | 正式提升 | 11.829× | 11.777× | 3.736% / 0.247% | pytorch_preallocated_out | 1.006× |
| 019 | 无法定论 | 1.000× | 0.974× | 5.607% / 2.750% | pytorch_preallocated_out | 2.013× |
| 023 | 无法定论 | 1.013× | 1.004× | 20.895% / 0.798% | pytorch_preallocated_out | 0.798× |
| 026 | 无法定论 | 0.993× | 0.974× | 7.536% / 6.947% | — | — |
| 036 | 正式提升 | 54.228× | 54.068× | 1.893% / 0.344% | pytorch_eager | 1.764× |
| 040 | 正式提升 | 22.254× | 22.113× | 1.470% / 0.827% | pytorch_eager | 9.228× |
| 047 | 无法定论 | 1.436× | 1.387× | 24.271% / 5.560% | pytorch_eager | 0.576× |
| 088 | 无提升 | 1.009× | 1.000× | 1.101% / 0.171% | pytorch_fused_gelu_tanh | 0.990× |
| 095 | 无法定论 | 28.566× | 26.681× | 13.876% / 4.838% | pytorch_eager | 0.103× |

严格结果为 4 项提升、1 项无提升、5 项无法定论。相对上游的严格 Core 10 几何平均为 4.381×。PyTorch 有 9/10 题存在正确且稳定的方法；仅在这些可比题上，严格结果相对最快稳定 PyTorch 的几何平均为 1.053×。

026 没有稳定 PyTorch 方法，因此不进入 PyTorch 几何平均。019、023、026、047、095 在自动重测后仍未满足 CUDA Session 稳定性门槛；其诊断 speedup 不作为正式声明。NCU 计数器仍不可用，本地模式为 `events_only`。

Canonical machine-readable evidence: [JSON](core10_rtx3080_comparison.json).
