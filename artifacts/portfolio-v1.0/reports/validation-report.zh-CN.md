# KernelBlaster Day 1–10 RTX 3080 验证报告

> **历史快照：** 本报告记录 2026-07-19 的 API 受阻 Agent 流程与首个 RMSNorm 深度案例，不应再被解读为当前 Core 10 覆盖状态。九个后续候选、统一十题复测和同卡 PyTorch 对比见 [`core10-rtx3080-comparison.zh-CN.md`](core10-rtx3080-comparison.zh-CN.md)。API 401 与 NCU 权限结论仍保持有效。

生成时间：2026-07-19T07:53:02.260926+00:00

## 结论

- 正式 CUDA Events 覆盖：1/10。
- Baseline 覆盖：10/10。
- Baseline 已尝试：10/10；稳定通过：7/10。
- 已验证提升任务：1/10。
- Agent 已验证提升任务：0/10；手工深度案例：1。
- 已测正确候选几何平均（1 个手工候选）：49.34767666277695。
- 全十题组合分数（手工案例计入、未优化按 1.0）：1.4768169594231786。
- Agent-only 全十题组合分数：1.0。
- API 估算成本：$0.0000；未包含 cache write 与区域加价。
- 深度案例去留门槛：PASS。

唯一一次 API smoke 返回 401 `invalid_api_key`，未重试、无成功响应、无已记录 token；因此 Pilot 与 Core 10 Agent 搜索未执行。`provider_api` 失败分类为同一次请求在 request/fan-out/smoke 三层产生的 3 条错误事件，不代表 3 次计费请求。

NCU 2025.1 权限探针返回 `ERR_NVGPUCTRPERM`。Windows 注册表开关已设为允许非管理员访问，但当前驱动尚未重新加载；本报告不发布任何 NCU 硬件计数器归因。CUDA Events、正确性和源码层面的映射分析单独成立。

未完成或不稳定的候选保持 NOT RUN/明确失败状态，不用于性能声明。

## Core 10

| Task | Kernel | Provenance | Status | Speedup |
| --- | --- | --- | --- | --- |
| 004 | Matrix-vector multiplication | - | baseline_only | NOT RUN |
| 007 | Small-K matrix multiplication | - | baseline_only | NOT RUN |
| 019 | ReLU | - | baseline_unstable | NOT RUN |
| 023 | Softmax | - | baseline_unstable | NOT RUN |
| 026 | GELU | - | baseline_only | NOT RUN |
| 036 | RMSNorm | manual_case_study | verified_improved | 49.348x |
| 040 | LayerNorm | - | baseline_only | NOT RUN |
| 047 | Sum reduction | - | baseline_unstable | NOT RUN |
| 088 | MinGPT GELU | - | baseline_only | NOT RUN |
| 095 | Cross entropy loss | - | baseline_only | NOT RUN |

## RMSNorm 深度案例

| Candidate | Speedup | Scope | Stable | All sessions not slower | Provenance |
| --- | ---: | --- | --- | --- | --- |
| v3a | 47.7782x | upstream_baseline | True | True | manual_case_study |
| v3c | 49.3477x | upstream_baseline | True | True | manual_case_study |
| v1 | 47.6681x | upstream_baseline | True | True | manual_case_study |
| v3b | 46.7420x | upstream_baseline | True | True | manual_case_study |
| v2 | 48.0017x | upstream_baseline | True | True | manual_case_study |
| v3c | 1.0069x | variant_head_to_head | True | True | manual_case_study |
| self | 1.0000x | variant_head_to_head | True | True | runner_validation |

最佳 V3c 相对同次 V0 为 49.3477x。V3c 对 V1 的直接 head-to-head 为 1.0069x；V3c 对自身为 1.0000x。主收益来自把线程映射到连续空间位置并移除跨线程 reduction；`half2`、128-thread block 和两对 work/thread 都没有超过最终最佳版本。

## 失败分类

```json
{
  "provider_api": 3,
  "baseline_stability": 3
}
```

## English summary

Formal CUDA Events coverage is 1/10. Results without a correct, reproducible candidate remain pending and are excluded from measured-candidate claims. The all-task portfolio score assigns 1.0 only as an explicit conservative policy value for unoptimized tasks.

## 复现命令

```bash
python -m pytest -q
python scripts/run_portfolio.py --suite core10 --model gpt-5.6-terra --gpu rtx3080 --dry-run
python scripts/benchmark_suite.py --suite core10 --warmup 20 --repetitions 100 --sessions 3
python scripts/benchmark_cuda.py --task-dir data/kernelbench-cuda/level1/036_RMSNorm --task-id 036 --kernel RMSNorm --candidate portfolio/case_studies/rmsnorm/best_rmsnorm_sm86.cu --candidate-name v3c --extra-correctness-driver portfolio/case_studies/rmsnorm/edge_driver.cpp
```
