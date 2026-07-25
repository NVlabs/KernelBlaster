# RMSNorm 优化案例

**简体中文** | [English](rmsnorm-case-study.md)

<!-- RMSNORM_STATUS:START -->
当前状态：**已在 NVIDIA GeForce RTX 3080（sm_86）上验证**。

- V0–V3c 及基准规范化后的源码均通过官方 Driver。
- V1–V3c 也通过 `edge_driver.cpp` 中的小尺寸、奇数空间尺寸、63/64/65 通道和边界用例。
- 独立 V3c 深度测试相对配对 V0 测得 49.348×。
- 后续统一 Core 10 复测测得 52.772×；该差异作为跨运行证据保留，不做静默平均。
- Schema v2 的五 Session 定向确认测得配对中位加速 56.332×，并通过 bootstrap 与稳定性门槛。
- Schema v2 完整 Core 10 复现测得 54.228×，同样通过正式门槛；两次结果分别保留。
- NCU 硬件计数器归因仍受 `ERR_NVGPUCTRPERM (non-root Docker/WSL; one no-network SYS_ADMIN retry also blocked; Windows native control passed)` 阻塞；CUDA Events 与源码推导的映射证据分开报告。
<!-- RMSNORM_STATUS:END -->

## 问题与结论

本案例研究一个正确性优先、由 profile 证据引导的重设计，能否相对未修改的 channel-first FP16 RMSNorm kernel 将独立测得延迟至少降低 5%。该门槛在 RTX 3080 上明显通过。主要收益来自线程/数据映射变化，而不是后续微调变体。

官方 tensor 为 `[B=16, C=64, D1=256, D2=256]`，沿 `C` 维归一化。V0 为每个空间位置分配一个 block，相邻 warp lane 读取 channel-strided 地址，并执行跨线程归约。V1 为每个空间位置分配一个线程；每个 channel 的 lane 读取相邻地址，并独立累加 64 个 channel 值。

## 实测版本

协议：NGC PyTorch 25.01、CUDA 12.8、`sm_86`、FP16、20 次预热、100 次 CUDA Events 采样、自动 inner loops、三个独立进程 Session、固定 seed 和 AB/BA 顺序。每一行在计时前均通过官方及边界 Driver。

| 版本 | 独立变化 | 配对 V0 中位数（μs） | 候选中位数（μs） | 加速比 | 候选 Session spread |
| --- | --- | ---: | ---: | ---: | ---: |
| V1 | 连续空间线程映射 | 28603.904 | 600.064 | 47.668× | 1.025% |
| V2 | `half2` 加奇数步长回退 | 29123.584 | 606.720 | 48.002× | 0.338% |
| V3a | 128-thread block | 29122.561 | 609.536 | 47.778× | 0.378% |
| V3b | 每线程两对数据 | 29125.119 | 623.104 | 46.742× | 0.082% |
| V3c | `rsqrtf` 与乘法 | 29359.104 | 594.944 | 49.348× | 0.086% |

V3c 在直接 head-to-head 测试中只比 V1 快 1.0069×。该小幅差异用于候选选择；可辩护的优化结论是相对 V0/V1/V3c 的约 48–49× 映射重设计收益。V2、V3a 和 V3b 作为正确但负收益或中性的实验保留。

后续统一 Core 10 运行独立测得 V0 为 31234.048 μs、V3c 为 591.872 μs，即 52.772×。该结果不与深度案例平均：两次运行都保留，以显示进程/Session 与 GPU 状态变化。

## 正确性与数值边界

原始源码和基准规范化源码均通过：

- 未修改的官方 Driver；
- 极小 tensor；
- 奇数空间步长与标量尾部；
- 63、64、65 个 channel；
- 更大的奇数边界 shape。

官方 FP16 容差仍为 `rtol=0.1, atol=0.1`。这验证了基准契约，但不能替代任意 shape 和 dtype 上的生产级误差分布测试。

## 性能解释与 Profiler 边界

官方 shape 大约移动 384 MiB（两次输入读取和一次输出写入），约有 0.5 FLOP/byte。V0 启动 1,048,576 个 256-thread block；V1/V3c 启动 4,096 个 block，并为每个线程分配一个独立空间位置。这些是从源码推导的流量与启动估算。

NCU 2025.1 可用，但性能计数器采集返回 `ERR_NVGPUCTRPERM`。本案例不声称 occupancy、内存吞吐、调度器或 warp stall 归因。CUDA Events 测量与源码层映射解释明确分开标记。

## 证据与复现

- [已提交源码、边界 Driver 与变体讨论](../../portfolio/case_studies/rmsnorm/README.md)
- [已发布深度案例 CSV](../../artifacts/portfolio-v1.0/results/deep_case_results.csv)
- [Core 10 跟进对比](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json)
- [中文完整报告](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md)

```bash
python scripts/benchmark_cuda.py \
  --task-dir data/kernelbench-cuda/level1/036_RMSNorm \
  --task-id 036 --kernel RMSNorm \
  --candidate portfolio/case_studies/rmsnorm/best_rmsnorm_sm86.cu \
  --candidate-name v3c \
  --extra-correctness-driver portfolio/case_studies/rmsnorm/edge_driver.cpp \
  --phase confirmation \
  --warmup 20 --repetitions 100 --sessions 5 \
  --output-dir out/portfolio/deep-rmsnorm/<run-id>
```
