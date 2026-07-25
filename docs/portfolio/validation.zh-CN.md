# 验证状态与基准协议

**简体中文** | [English](validation.md)

这是 Portfolio fork 的持续验证记录。历史原始运行仍以追加方式保存在被忽略的 `out/portfolio/` 目录；审阅后的证据发布在 `artifacts/portfolio-v1.0/`。

<!-- VALIDATION_STATUS:START -->
| 门禁 | 当前状态 | 规范证据 |
| --- | --- | --- |
| Provider/Recorder/Suite CPU 测试 | 通过 — 177 项通过 | `tests/` |
| 真实网关冒烟 | 失败：当前 HTTP 401（1 次请求、0 次重试、0 tokens；2026-07-22） | `artifacts/portfolio-v2.1/issue-7/rtx3080/trusted-pilot-summary.json` |
| RTX 3080 容器与 `sm_86` 构建 | 通过 | `artifacts/portfolio-v1.0/environment/environment.json` |
| 官方候选正确性 | 历史 v1 通过 — 10/10；schema v2 完整 10/10 通过 | `artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json` |
| RMSNorm 边界正确性 | 通过 | 已提交的 `edge_driver.cpp` 与深度案例 artifacts |
| CUDA Events 计时 | schema v2 完整确认：4 项提升；1 项无提升；5 项无法定论 | `artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json` |
| 同卡 PyTorch 对比 | schema v2 完整确认；9/10 题有稳定方法 | `artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json` |
| Issue #10 能力与资源加固 | 4 项正式提升；095 因 upstream baseline spread 24.37% 仍无法定论，Issue 保持开启 | `artifacts/portfolio-v2.1/issue-10/rtx3080/correctness-summary.json` |
| Portfolio v2.1 证据完整性 | 精确 SHA256 清单已发布 | `artifacts/portfolio-v2.1/SHA256SUMS.json` |
| NCU 硬件计数器 | 阻塞 — `ERR_NVGPUCTRPERM (non-root Docker/WSL; one no-network SYS_ADMIN retry also blocked; Windows native control passed)` | `artifacts/portfolio-v2.1/issue-8/rtx3080/ncu-preflight-summary.json` |
| 跨 GPU 对比 | 阻塞 — `requires authorized A100/L40S rental` | 未发布跨卡聚合性能声明 |

历史手工跟进验证了全部十个候选，并在旧门槛下改进 4/10。相关声明作为不可变历史证据保留。schema v2 完整结果仍是手工候选确认，不能外推为 Agent 搜索结论。
<!-- VALIDATION_STATUS:END -->

## 已完成的开发与实验时间线

1. **Day 1–2 — Provider 与可观测性：**完成 OpenAI 兼容 Provider 边界、客户端 fan-out、重试分类、原子请求/Token 预留、用量回退、Recorder 序号/原子性/脱敏、Suite 校验、离线 dry-run artifact。
2. **Day 3–7 — 本地 GPU 验证：**固定 NGC PyTorch 25.01 容器，将 RTX 3080 映射到 `sm_86`，验证编译/GPU Server，加入独立 CUDA Events Runner，并记录 Core 10 基线。受限 API 冒烟返回 HTTP 401，因此没有运行 Agent Pilot/Core 10 rollout。
3. **Day 8–10 — RMSNorm 深度案例：**实现 V1–V3c，加入奇数/极小/63–65 通道正确性输入，保留失败候选，并发布相对 V0 的 49.348× 配对 V3c 结果。NCU 计数器访问仍阻塞，没有发布硬件计数器归因。
4. **Core 10 跟进：**新增九个手工候选，使用 20 次预热、100 次采样、三个独立进程 Session、AB/BA 顺序重跑十个任务，并与同卡 PyTorch eager/out/fused 方法比较。
5. **发布：**提交脱敏 JSON/CSV/SVG 报告和 SHA256 链接，覆盖 362 个被忽略的原始 JSON/JSONL/CSV/SVG/log 文件。已合并的 PR #4 和 #5 包含 artifact 发布与持续文档跟进。
6. **Schema-v2 完整确认：**用五个独立进程 Session 重跑全部十个手工候选和同卡 PyTorch 方法。正确性 10/10 通过；004/007/036/040 正式提升，088 报告无提升，019/023/026/047/095 在自动重测后仍无法定论。

## 结果解释

- 历史 v1 的诊断候选中位数包含不稳定任务，适合确定后续方向，不代表发布声明。
- 当前 schema-v2 门槛要求正确、跨 Session spread 不超过 5%、五个配对 Session、中位加速至少 1.01×，且配对 bootstrap 95% 下界大于 1.0。
- Schema v2 严格 Core 10 相对上游的几何平均为 4.381×。9/10 题存在稳定 PyTorch 方法；仅在这些可比任务上，严格结果相对最快稳定 PyTorch 的比值为 1.053×。
- 九个新候选相对上游的诊断/严格几何平均为 5.020×/3.302×；004、007、040 通过严格门槛。
- 完整 Core 10（含现有 RMSNorm）相对上游的诊断/严格几何平均为 6.351×/4.356×。
- 相对测得最快的 PyTorch 方法，九题诊断/严格比值为 1.415×/0.931×；完整十题为 1.447×/0.992×。
- 007 调用 cuBLAS，体现的是成熟库的正确集成，而不是自定义 GEMM 击败 cuBLAS。

规范证据：

- [Schema-v2 完整 Core 10 确认](../../artifacts/portfolio-v2.0/core10/core10-rtx3080-confirmation.zh-CN.md)
- [Schema-v2 完整对比 JSON](../../artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json)
- [Schema-v2 定向验证](../../artifacts/portfolio-v2.0/reports/rtx3080-targeted-validation.zh-CN.md)
- [Schema-v2 结果 JSON](../../artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json)
- [逐题对比 JSON](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json)
- [中文完整分析](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md)
- [英文摘要](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md)
- [环境清单](../../artifacts/portfolio-v1.0/environment/environment.json)
- [原始 artifact SHA256 清单](../../artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv)

## 验收规则

- 任一官方或额外正确性输入失败都拒绝候选，无论计时结果如何。
- 每个原始清单保留原始源码和计时规范化源码的 SHA256。
- 报告 GPU、SM 目标、精度、形状、seed、预热、重复次数、inner loops、Session/顺序、中位数、p10/p90、遥测、驱动/CUDA/容器和源码出处。
- 上游 CUDA、候选 CUDA、PyTorch eager/out/fused 以及论文报告结果必须作为不同基线处理。
- 自动冷却/重测后仍超过 5% Session 门槛时，停止正式性能声明。
- 至少要求五个配对确认 Session、中位加速不低于 1.01×，且配对 bootstrap 95% 下界大于 1.0。
- 不得从 CUDA Events 或代码检查推断 NCU 瓶颈；`ERR_NVGPUCTRPERM` 仍是明确的归因阻塞项。

## 剩余阻塞与延后工作

- 提供有效外部 API 凭据后再运行 Agent Pilot/Core 10 搜索；单次 401 请求未作为成功 completion 重试或计费。
- 诊断 019/023/026/047/095 可复现的 Session 不稳定，不得把其诊断 speedup 升级为正式声明。
- 重新运行 PyTorch 026，直到获得正确且稳定的框架基线后才能将其纳入 PyTorch 几何平均。
- 在明确授权的 Profiler Worker 上采集所需 NCU sections；本地 Docker/WSL 保持 `events_only`。
- 分别运行并发布 L40S/A100 匹配对比。
- 在生产库声明前，将固定形状候选推广到不同 shape、dtype、layout、stream、图捕获、反向路径，并采用更严格数值容差。

## 复现命令

```bash
python scripts/benchmark_candidates.py \
  --phase confirmation \
  --warmup 20 --repetitions 100 --sessions 5 \
  --cooldown-seconds 60 \
  --output-dir out/portfolio/candidates/<run-id>

python scripts/benchmark_pytorch.py \
  --warmup 20 --repetitions 100 --sessions 5 \
  --output-dir out/portfolio/pytorch/<run-id>

python scripts/analyze_core10_comparison.py \
  --candidate-summary out/portfolio/candidates/<run-id>/suite_summary.json \
  --pytorch-summary out/portfolio/pytorch/<run-id>/pytorch_summary.json \
  --output-dir out/portfolio/analysis/<run-id>

python scripts/sync_portfolio_docs.py --write
python scripts/sync_portfolio_docs.py --check
```

GPU/profiler worker 使用根目录权威 `compose.yaml` 中的 `gpu-supervisor` target，并以非 root 用户运行，固定
使用内部 worker 网络、`--cap-drop ALL --security-opt no-new-privileges`；不会向其中
传入任何 API 凭据。只有已经明确返回 `ERR_NVGPUCTRPERM` 的一次性 NCU 容器才可
增加 `--cap-add SYS_ADMIN`，永远不接受 `--privileged` 作为规避方案。WSL 下的
Nsys smoke 还会采用 [NVIDIA 官方 WSL 时间戳回退配置](https://archive.docs.nvidia.com/nsight-systems/2025.2/ReleaseNotes/index.html)
（`CuptiUseRawGpuTimestamps=false`）、预热 RMSNorm，并且只有 GPU
kernel 表中出现 `rmsnorm_half2_rsqrt` 才算通过。

语言模型仍是外部推理服务。Rollout 会更新轨迹和优化数据库，但不会训练或微调模型权重。
