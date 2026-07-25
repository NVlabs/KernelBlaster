# Portfolio 验证与开发进度

**简体中文** | [English](README.md)

<!-- PORTFOLIO_PROGRESS:START -->
**更新日期：2026-07-23**

- Day 1–2：Provider、Recorder、Suite、dry-run 与 CPU 测试已完成。
- Day 3–7：WSL2/RTX 3080、容器、编译、正确性和 CUDA Events 基准设施已完成；API 冒烟状态：失败：当前 HTTP 401（1 次请求、0 次重试、0 tokens；2026-07-22）。
- Day 8–10：RMSNorm V0–V3c 已完成，独立深度结果 49.348×；统一 Core 10 复测 52.772×。
- 后续 Core 10：schema v2 完整手工确认通过 10/10 正确性，确认 4 项提升、1 项无提升、5 项无法定论；9/10 题有稳定 PyTorch 方法。
- Schema v2 PyTorch 对照：仅在 9/10 个存在正确且稳定方法的可比任务上，严格结果相对最快稳定方法的几何平均为 1.053×；026 不进入该几何平均。

[Schema v2 完整 Core 10 验证](../../artifacts/portfolio-v2.0/core10/core10-rtx3080-confirmation.zh-CN.md) · [Schema v2 完整结果 JSON](../../artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json) · [Schema v2 定向验证](../../artifacts/portfolio-v2.0/reports/rtx3080-targeted-validation.zh-CN.md) · [Schema v2 结果 JSON](../../artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json) · [中文完整报告](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md) · [英文摘要](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md) · [逐题 JSON](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json) · [对比图](../../artifacts/portfolio-v1.0/figures/core10_rtx3080_comparison.svg) · [原始文件哈希](../../artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv) · [候选清单](../../portfolio/case_studies/core10/candidates.json)
<!-- PORTFOLIO_PROGRESS:END -->

## 文档导航

- 架构、Provider、Recorder、Benchmark 与文档同步边界：[简体中文](architecture.zh-CN.md) · [English](architecture.md)
- 验证门槛、已完成工作、证据与剩余阻塞项：[简体中文](validation.zh-CN.md) · [English](validation.md)
- RMSNorm V0–V3c 实测案例：[简体中文](rmsnorm-case-study.zh-CN.md) · [English](rmsnorm-case-study.md)
- [RTX 3080 已发布 artifact 包](../../artifacts/portfolio-v1.0/README.md)
- [Core 10 与 PyTorch 中文完整分析](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md)
- [英文独立摘要](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md)

## 更新流程

`portfolio/status.json` 记录叙述状态并指向权威证据；性能数字直接从已提交结果 JSON 推导，不在状态清单中重复手填。任何 Benchmark、候选、分析或发布变更后执行：

```bash
python scripts/sync_portfolio_docs.py --write
python scripts/sync_portfolio_docs.py --check
```

GitHub 文档同步 workflow 会重复检查；相关源码或结果变更没有同步 README/docs/status 时，PR 将失败。
