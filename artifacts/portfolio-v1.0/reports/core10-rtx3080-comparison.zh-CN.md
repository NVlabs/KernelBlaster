# Core 10 手工二次优化与同卡 PyTorch 对比（RTX 3080）

## 结论

这轮二次开发不是“十题都比成熟库快”。更准确的结论是：

- 相对仓库原版，十个候选的观测中位数几何平均为 **6.351×**；其中包含 4 个跨 Session 不稳定结果，不能作为正式组合性能声明。
- 按“正确、Session 波动不超过 5%、每个 Session 不退化、提升至少 1.01×”筛选，并将其余题按 1.0 计入分母后，全十题严格组合几何平均为 **4.356×**，正式通过的是 004、007、036、040 四题。
- 相对同一张 RTX 3080 上每题最快的已测 PyTorch 方法，全部候选的诊断性几何平均为 **1.447×**，7/10 题候选更快；但这个数字包含候选或 PyTorch 自身不稳定的微秒级结果。
- 只采用严格验证候选、其余题回退仓库原版时，对 PyTorch 的十题几何平均为 **0.992×**。换言之，严格组合与 PyTorch 总体基本持平，PyTorch 约快 0.8%。

单独排除既有 036 RMSNorm、只看本轮新增的九个候选时：相对仓库原版的诊断几何平均为 **5.020×**，严格全九题分数为 **3.302×**，其中 004、007、040 三题正式通过；相对每题最快 PyTorch 方法的诊断几何平均为 **1.415×**（6/9 候选更快），严格回退状态为 **0.931×**，即 PyTorch 约快 7.4%。

因此，目前能力可以定位为：**固定形状上的较强 CUDA 专项优化原型，已经能在部分归约/归一化工作负载上超过 PyTorch 通用路径，但尚未达到成熟高性能算子库的通用性、稳定性和工程完整度。**

## 环境与口径

- GPU：NVIDIA GeForce RTX 3080，compute capability 8.6
- 驱动：591.86
- 容器：`kernelblaster:validation-25.01`
- 镜像 ID：`sha256:5a0cc1c81da988e417de729b7bf5c630c5220e78074d184ab86130b9e51f01d1`
- CUDA：12.8；PyTorch：NGC 25.01 / PyTorch 2.6 系列
- dtype：FP16；形状与仓库 Core 10 官方 Driver 完全一致
- 每个 CUDA 变体：原始源码和移除 launcher 主机同步后的源码均先做正确性；20 次 warmup、100 个 CUDA Events 样本、自动 inner-loop、3 个独立进程 Session，Baseline/Candidate 使用 AB/BA 顺序
- PyTorch：同样 20/100/3；常用 eager API 与可用的预分配 `out` 路径都测，088 额外测等价的融合 `gelu(approximate="tanh")`
- 官方 Driver 的 FP16 正确性阈值为 `rtol=0.1, atol=0.1`。036 另外通过小尺寸、奇数空间长度、63/64/65 channels 和边界尺寸 Driver。

正式运行的 `git_commit` 字段记录候选开发前的基线提交 `2891562686b42bc91f13c83cf20a114644f2e5b3`；每个实际候选源码另有 SHA256，最终 PR 包含完全相同的源码。

## 逐题结果

表中“候选/PyTorch”大于 1 表示候选更快。带“波动”的值只用于诊断，不属于稳定性能声明。

| ID | 算子 | 仓库原版 μs | 候选 μs | 原版→候选 | 正式状态 | 同卡 PyTorch 最快 μs | 候选/PyTorch |
|---|---|---:|---:|---:|---|---:|---:|
| 004 | Matrix-vector | 18472.960 | 107.008 | 172.632× | 已验证 | 110.182 eager | 1.030× |
| 007 | Small-K Matmul | 10031.616 | 809.472 | 12.393× | 已验证；候选调用 cuBLAS | 823.808 eager | 1.018× |
| 019 | ReLU | 9.652 | 9.484 | 1.018× | 波动；一 Session 退化 | 9.344 eager | 0.985× |
| 023 | Softmax | 12.800 | 12.695 | 1.008× | 波动且低于实质门槛 | 10.174 `out`（波动） | 0.801× |
| 026 | GELU | 9.252 | 9.211 | 1.004× | 近似持平；一 Session 退化 | 9.710 eager（波动） | 1.054× |
| 036 | RMSNorm | 31234.048 | 591.872 | 52.772× | 已验证，含边界 Driver | 1041.408 eager 公式 | 1.760× |
| 040 | LayerNorm | 15339.008 | 703.744 | 21.796× | 已验证 | 6640.128 eager | 9.435× |
| 047 | Sum reduction | 19.477 | 13.305 | 1.464× | 明显潜力但跨 Session 波动 | 11.136 eager（波动） | 0.837× |
| 088 | MinGPT GELU | 27.745 | 27.703 | 1.001× | 与原版持平 | 27.745 fused GELU | 1.001× |
| 095 | Cross entropy | 560.640 | 19.513 | 28.732× | 25.94–32.38×，但绝对延迟波动 | 64.512 eager（波动） | 3.306× |

## 与仓库原版的差距来自哪里

004、007、036、040、095 的大倍数主要来自算法结构或执行路径变化，不是简单的 block size 微调：

- 004 把一行映射到一个 block，使用 `half2` 连续访问和 warp/block 归约，候选 107.0 μs，已经接近 PyTorch 的 110.2 μs。
- 007 将 row-major 等价映射交给 `cublasGemmEx`，候选 809.5 μs 与 PyTorch 823.8 μs 接近。这题证明的是正确利用成熟库，而不是自研 GEMM 超过 cuBLAS。
- 036 让线程沿连续空间位置工作、每线程独立累积 channel，并使用 `half2`；消除了原版不必要的跨线程归约与低效访问。
- 040 使用每 batch 256 个 tile 的多 block 局部统计、设备端二级汇总和 `half2` apply，把超大 normalized shape 的通用路径特化为固定形状路径。
- 095 用每 warp 一个样本处理 10 类 logits，并在设备端完成二级归约，消除了原版 launcher 中的重复分配/拷贝/释放。其方向性收益非常大，但本轮仍按稳定性规则阻止正式数值声明。

019、023、026、047、088 已经进入约 10–30 μs 区间，kernel launch、GPU P-state、Python/C++ 调度和实现细节都会放大 1–5% 的噪声。这里不能用单次 smoke 结果宣称胜出。

## 相比成熟高性能算子库，能力如何

### 已接近成熟库

- 004 比同卡 PyTorch eager 快约 3.0%，属于基本同一水平。
- 007 比 PyTorch eager 快约 1.8%，而候选本身就是 cuBLAS 路径；这不构成“击败 cuBLAS”的证据。
- 088 与 PyTorch 融合 GELU 几乎完全持平。若错误地拿候选和 PyTorch 的多算子 Driver 公式比较，会看到约 8.8×，但成熟库用户应选融合原语，因此正式分析使用融合 GELU。

### 固定形状特化明显领先

- 036 相对 PyTorch Driver 等价多算子公式快 1.76×。
- 040 相对 PyTorch 通用 LayerNorm 快 9.44×。
- 095 的候选中位数相对 PyTorch 快 3.31×，但两边跨进程波动均超过 5%，只作为下一轮复测依据。

这些结果说明对固定 shape、固定 dtype、固定布局进行融合和专用归约设计是有效的；它们不自动外推到任意 shape、BF16/FP32、非连续张量、动态图、多流并发或训练反向。

### PyTorch 仍领先的例子

- 023 Softmax：PyTorch 最快观测值约快 1.25×。
- 047 Sum：PyTorch eager 最快观测值约快 1.19×。
- 019 ReLU：两者接近，PyTorch 约快 1.5%。

这类基础算子通常已由成熟库针对访存、launch 和多种形状长期调优。当前候选的 pack/vectorize 思路正确，但还没有稳定超过库实现。

## 为什么还不能称为“生产级高性能算子库”

1. 当前候选针对官方单一固定形状；成熟库覆盖广泛 shape、dtype、stride、layout、device 和异常输入。
2. 040/095 使用静态 workspace/handle，尚未完成多流并发、生命周期、图捕获和线程安全设计。
3. 官方 FP16 容差 `0.1/0.1` 较宽；除 036 外，其余候选还缺随机 shape、严格误差分布和边界矩阵。
4. 019/023/047/095 未通过 5% 跨 Session 稳定性门槛。
5. 这轮九题新增候选使用 CUDA Events 做性能比较；Nsight Compute 的完整归因仍以已有 RMSNorm 深度案例为主，尚未给每个新候选建立可发布的 counter 证据。

综合评价：**算法判断和固定形状 CUDA 调优能力已经达到较强的工程原型水平；若以 PyTorch/cuBLAS/CUTLASS/CUB/Triton 这类成熟库为参照，当前短板主要是泛化、稳定性、数值验证和运行时工程，而不是完全缺乏单点性能。**

## 复现

```bash
# NGC 25.01 容器内，RTX 3080 / sm_86
python scripts/benchmark_candidates.py \
  --warmup 20 --repetitions 100 --sessions 3 \
  --cooldown-seconds 60 \
  --output-dir out/portfolio/candidates/<run-id>

python scripts/benchmark_pytorch.py \
  --warmup 20 --repetitions 100 --sessions 3 \
  --output-dir out/portfolio/pytorch/<run-id>

python scripts/analyze_core10_comparison.py \
  --candidate-summary out/portfolio/candidates/<run-id>/suite_summary.json \
  --pytorch-summary out/portfolio/pytorch/<run-id>/pytorch_summary.json \
  --output-dir out/portfolio/analysis/<run-id>
```

本机原始产物保存在被忽略的：

- `out/portfolio/candidates/core10-rtx3080-20260720T092500Z/`
- `out/portfolio/pytorch/core10-rtx3080-20260720T131000Z/`
- `out/portfolio/analysis/core10-rtx3080-20260720/`

PR 只提交脱敏后的 CSV、JSON、SVG、报告、候选源码和 SHA256 清单。
