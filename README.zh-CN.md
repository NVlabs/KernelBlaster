# KernelBlaster

[English](README.md) | **简体中文**

## Portfolio Fork 状态

<!-- PORTFOLIO_STATUS:START -->
当前 Fork 已在 **NVIDIA GeForce RTX 3080（sm_86）** 上完成 Day 1–10 基础设施、RMSNorm 深度案例、Core 10 手工候选和同卡 PyTorch 对比。环境为 WSL2、CUDA 12.8.61、驱动 591.86。

| 验证项目 | 当前状态 |
| --- | --- |
| CPU 测试 | **177 项通过**（当前分支） |
| CUDA 编译与官方正确性 | **历史 10/10；schema v2 完整验证 10/10 通过** |
| CUDA Events 与同卡 PyTorch | **schema v2 完整验证：4 项提升、1 项无提升、5 项无法定论；9/10 题有稳定 PyTorch 方法** |
| 外部 LLM 冒烟测试 | **失败：当前 HTTP 401（1 次请求、0 次重试、0 tokens；2026-07-22）** |
| Nsight Compute 硬件计数器 | **阻塞：ERR_NVGPUCTRPERM (non-root Docker/WSL; one no-network SYS_ADMIN retry also blocked; Windows native control passed)** |
| 跨 GPU 复测 | **阻塞：requires authorized A100/L40S rental** |

| 历史 v1 实测范围 | 相对仓库原版（诊断 / 旧严格口径） | 相对 PyTorch 最快方法（诊断 / 旧严格口径） |
| --- | ---: | ---: |
| 本轮新增九题 | 5.020× / 3.302× | 1.415× / 0.931× |
| 完整 Core 10（含 RMSNorm） | 6.351× / 4.356× | 1.447× / 0.992× |

上述严格值作为不可变的历史 v1 证据保留。独立的 schema v2 完整手工确认验证了 10/10 正确性，正式确认 004/007/036/040，将 088 标为无提升，并把 019/023/026/047/095 保持为无法定论。当前口径下，严格 Core 10 相对上游的几何平均为 4.381×；仅在 9/10 个存在正确且稳定 PyTorch 方法的可比任务上，严格结果相对最快稳定方法的几何平均为 1.053×。它仍不是 Agent 搜索结果。新口径还检查 p99/max 误差回归、NaN/Inf 和五次确定性。当前 Agent Pilot 与 Core 10 Agent 搜索均未运行。

[Schema v2 完整 Core 10 验证](artifacts/portfolio-v2.0/core10/core10-rtx3080-confirmation.zh-CN.md) · [Schema v2 完整结果 JSON](artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json) · [Schema v2 定向验证](artifacts/portfolio-v2.0/reports/rtx3080-targeted-validation.zh-CN.md) · [Schema v2 结果 JSON](artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json) · [中文完整报告](artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md) · [英文摘要](artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md) · [逐题 JSON](artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json) · [对比图](artifacts/portfolio-v1.0/figures/core10_rtx3080_comparison.svg) · [原始文件哈希](artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv) · [候选清单](portfolio/case_studies/core10/candidates.json)
<!-- PORTFOLIO_STATUS:END -->

### 复现 RTX 3080 正式对比

以下命令应在固定的 NGC 25.01 容器和 `sm_86` GPU 中执行。原始输出保存在被忽略的 `out/portfolio/`，审核后的结果单独提交。

```bash
python scripts/benchmark_candidates.py \
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

python -m pytest -q
python scripts/sync_portfolio_docs.py --check
```

这里的优化循环执行的是基于 rollout 的搜索和经验库更新，不会微调或训练底层大语言模型的权重。

### Portfolio v2.1 证据

v2.1 发布内容加固了 Issue #10 涉及的五个 CUDA 候选，但没有扩大其生产可用性声明。稳定能力契约仅接受已经审核的 `sm_86`、FP16、连续 row-major、legacy default stream、单 stream、仅 forward、非 graph capture 和 manifest 白名单场景；不支持的请求会返回明确 reason code，且 `production_ready` 仍为 `false`。

- [证据索引与 SHA-256 清单](artifacts/portfolio-v2.1/SHA256SUMS.json)
- [五任务正确性与资源生命周期汇总](artifacts/portfolio-v2.1/issue-10/rtx3080/correctness-summary.json)
- [Issue #7 API/Pilot 状态](artifacts/portfolio-v2.1/issue-7/rtx3080/trusted-pilot-summary.json)：HTTP 401，Pilot 未运行
- [Issue #8 Profiler 状态](artifacts/portfolio-v2.1/issue-8/rtx3080/ncu-preflight-summary.json)：已发布 Windows 原生 NCU/NSYS 证据；WSL counters 与跨 GPU 复测仍未完成

## 上游项目介绍

<p><strong><span style="color:#0f766e;">KernelBlaster 是一个基于记忆增强上下文强化学习（Memory-Augmented In-context Reinforcement Learning，MAIC-RL）的框架</span></strong></p>

在不同代 GPU 上优化 CUDA 代码并不容易，因为最佳实现取决于庞大且高度硬件相关的搜索空间。一个在某张 GPU 上看起来合理的 Kernel，换到另一张 GPU 上可能仍会浪费大量性能，而简单改写通常不足以得到最佳结果。

传统编译器流水线受限于固定启发式规则；针对每种优化场景完整微调大语言模型又十分昂贵。许多 CUDA Agent 工作流还存在一个更直接的问题：它们无法充分记住此前探索得到的经验，因而容易重复犯错、产生有偏采样，并做出较弱的优化选择。

KernelBlaster 旨在让这一搜索过程更加智能。它不会把每个 Kernel 当作相互孤立的 Prompt，而是结合性能分析反馈、持久化 CUDA 优化知识库，以及类似强化学习的探索过程。Agent 不只是生成代码，还会执行性能分析、反思、检索既有优化知识、探索新候选，并持续更新搜索策略。

最终得到的是一个可复用的开源 CUDA 优化框架，内置正确性验证、性能分析、经验回放和可复现实验评估能力。

根据上游作者报告，与 PyTorch 基线相比，KernelBlaster 在 KernelBench Level 1、Level 2 和 Level 3 上分别取得了 <strong><span style="color:#ef4444;">1.43x</span></strong>、<strong><span style="color:#2563eb;">2.50x</span></strong> 和 <strong><span style="color:#16a34a;">1.50x</span></strong> 的几何平均加速。这些论文全量数据仅作为背景，与上方本 Fork 的 RTX 3080 Core 10 实测严格分开。

## 论文链接

**arXiv：** [**arXiv:2602.14293**](https://arxiv.org/abs/2602.14293) | **PDF：** [**KernelBlaster.pdf**](docs/figures/KernelBlaster.pdf)

## 为什么使用 KernelBlaster

| 其他方法的局限 | KernelBlaster 的设计 |
| --- | --- |
| CUDA 优化往往缺乏硬件针对性，并需要搜索巨大的设计空间。 | KernelBlaster 通过硬件感知、性能分析引导的状态提取与定向优化选择来缩小搜索空间。 |
| 固定编译器启发式难以适应每个 Kernel 和每一代 GPU。 | KernelBlaster 通过检索与迭代搜索，让优化决策适应具体 Kernel 和 GPU 架构。 |
| 为每个优化任务微调 LLM 成本高、迭代慢。 | KernelBlaster 使用上下文记忆与类似 RL 的探索改进优化，不依赖昂贵的任务专用微调。 |
| 朴素 Agent 循环容易遗忘之前任务和 rollout 的经验。 | KernelBlaster 通过持久化优化数据库和经验回放复用既有经验。 |

## 工作原理

KernelBlaster 从 KernelBench-CUDA 的初始输入产物开始工作。每个问题都提供一个起始 CUDA 实现 `init.cu` 和配套的 C++ 测试程序 `driver.cpp`。CUDA 文件是待优化代码，Driver 负责编译、运行，并根据参考行为验证 Kernel。

随后，系统执行 Agent 优化循环：

1. 从 `data/kernelbench-cuda/<level>/<problem>/` 加载输入问题。
2. 使用 `init.cu` 作为起始 CUDA Kernel，使用 `driver.cpp` 作为验证程序。
3. 编译并分析候选 Kernel，以 Nsight Compute 指标和 Elapsed Cycles 作为主要性能信号。
4. 从持久化 CUDA 知识库中检索相关优化思路。
5. 使用性能分析引导、类似文本梯度的 Prompt 生成新候选。
6. 评估候选，对成功轨迹给予奖励，并将其写入经验回放缓冲区。
7. 根据有效方案、失败方案和 Profiler 反馈更新后续决策。
8. 将最佳优化 Kernel 保存为 `final_rl_cuda_perf.cu`。

代码中的默认单次运行链路如下：

- `scripts/run_single_kernelblaster.sh` 启动运行环境并发起 RL 运行。
- `scripts/run_RL.py` 准备数据集、服务和工作流输入。
- `src/kernelblaster/workflow/workflow.py` 调用基于 Graph 的工作流。
- `src/kernelblaster/graph/nodes/optimization_rl_ncu.py` 加载 `init.cu` 和 `driver.cpp`，然后启动 RL 优化 Agent。
- `src/kernelblaster/agents/opt_ncu_rl.py` 执行 rollout、性能分析、经验回放缓冲区和策略更新循环。

<p align="center">
  <img src="docs/figures/flow_chart.png" alt="KernelBlaster 端到端 Agent 优化流程" width="720" />
</p>

上图展示了端到端优化循环。KernelBlaster 从输入 Kernel 和目标 GPU 硬件出发，提取性能状态，在知识库中匹配相应状态，选择有潜力的优化，将其转换成代码，执行正确性测试和性能分析，并持续迭代，直到终止条件判定搜索已收敛。最后阶段使用 LLM 软验证，然后写出优化后的 Kernel。

## 快速开始

### 推荐方式：Docker Desktop + WSL2

仓库源码和活动实验数据保存在 Ubuntu ext4 文件系统中。Windows 负责 NVIDIA 驱动、WSL、Docker Desktop 和编辑器；项目容器负责 CUDA 12.8、`nvcc`、PyTorch 与 Python 依赖。不要在 Ubuntu 中再安装一套 Docker Engine，也不要在 WSL 内安装 Linux NVIDIA 显示驱动。

在 `~/workspace/KernelBlaster` 这样的常规克隆目录中，先准备容器外部的持久化目录和仅供 Control 使用的密钥文件：

```bash
mkdir -p ../../{datasets,checkpoints,runs}/KernelBlaster
mkdir -p ../../runs/KernelBlaster/state
mkdir -p ../../caches/{huggingface,torch,triton} ../../secrets
cp -n .env.example ../../secrets/KernelBlaster.control.env
# 仅在本机编辑 ../../secrets/KernelBlaster.control.env，绝不能提交到 Git。
```

根目录 `compose.yaml` 是唯一的部署规范。将 Compose 指向外部文件，并为两个服务设置不同的 token audience：

```bash
export KERNELBLASTER_CONTROL_ENV_FILE="$HOME/secrets/KernelBlaster.control.env"
export KERNELBLASTER_STATE_HOST_DIR="$HOME/runs/KernelBlaster/state"
# 外部文件同时向 Compose 提供两个变量；它们的值必须不同。
docker compose --env-file "$KERNELBLASTER_CONTROL_ENV_FILE" config
docker compose --env-file "$KERNELBLASTER_CONTROL_ENV_FILE" build control gpu-supervisor
docker compose --env-file "$KERNELBLASTER_CONTROL_ENV_FILE" up --wait

# 验证 health 和不可互换的 token audience 后清理。
docker compose --env-file "$KERNELBLASTER_CONTROL_ENV_FILE" --profile smoke run --rm smoke
docker compose --env-file "$KERNELBLASTER_CONTROL_ENV_FILE" down --remove-orphans

# 可信的交互式 CUDA 开发环境按需显式启动。
docker compose --env-file "$KERNELBLASTER_CONTROL_ENV_FILE" --profile dev run --rm dev \
  bash scripts/run-with-metadata.sh python -m pytest -q
```

| Ubuntu 路径 | 容器路径 | 默认权限 |
| --- | --- | --- |
| 仓库目录 | `/workspace` | 读写 |
| `~/datasets/KernelBlaster` | `/data` | 只读 |
| `~/checkpoints/KernelBlaster` | `/checkpoints` | 读写 |
| `~/runs/KernelBlaster` | `/runs` | 读写 |
| `~/runs/KernelBlaster/state` | 仅映射到 `control` 的 `/state` | SQLite/CAS 读写 |
| `~/caches/{huggingface,torch,triton}` | `/cache/...` | 读写 |
| `~/secrets/KernelBlaster.control.env` | 仅注入 `control` | 永不复制进镜像 |

`control` 是仅使用 CPU 的 Python 镜像，只绑定 `127.0.0.1:8000`，也是唯一接收 LLM Provider 配置的容器。`gpu-supervisor` 使用固定的 CUDA 镜像，以 UID 10001 运行，根文件系统只读，移除 capabilities，启用 `no-new-privileges`，并限制内存、PID 和仅内部可见的 `worker-plane` 网络。它只接收 `KERNELBLASTER_WORKER_TOKEN`，绝不会接收 LLM Key。`dev` profile 保留为 `nvcc`、测试和交互式调试的可信路径。

### 持久化本地状态与实验记忆

Control 服务在 `KERNELBLASTER_STATE_HOST_DIR` 下独占本地 SQLite 任务库和
SHA-256 内容寻址存储（CAS）。该目录必须位于 WSL ext4 或 AutoDL 本地磁盘；服务会拒绝
已识别的 NFS、SMB/CIFS 和 `drvfs` 挂载。GPU supervisor 不挂载此目录，也不直接打开
SQLite，只能通过带鉴权的 Control API 获取 lease 和上报结果。

SQLite 只保存 run/job 状态、lease、attempt 和小型元数据；源码、日志、profile 与报告都
以 CAS digest 的形式引用。这是为后续检索层准备的、可审计的实验长期记忆，并不是
embedding 数据库或 RAG 本身。

已有状态库迁移前请先停止 Control 并备份：

```bash
cp "$KERNELBLASTER_STATE_HOST_DIR/control.sqlite3" \
  "$KERNELBLASTER_STATE_HOST_DIR/control.sqlite3.backup-$(date -u +%Y%m%dT%H%M%SZ)"
```

Control 启动时只执行向前 migration。CLI 中的 `--state-dir`、`--sqlite-path` 与
`--cas-dir` 优先于相应的 `KERNELBLASTER_*` 环境变量，因此无需把机器绝对路径硬编码到代码中。

普通 CUDA Events 路径不需要 host network、`--privileged` 或 `SYS_ADMIN`。
如果本地 NCU 计数器仍不可用，运行会明确记录为 `events_only`；需要硬件计数器时应部署单独授权的 Profiler Worker，而不是提升控制容器权限。

`docker/compose.worker.yml` 仅是包含根 Compose 文件的弃用兼容 wrapper，不要再在其中增加第二套部署定义。

#### 设置 API Key 并运行默认示例

```bash
export OPENAI_API_KEY=<your_api_key>
export MODEL=${MODEL:-gpt-5-mini-2025-08-07}
export GPU_TYPE=${GPU_TYPE:-L40S}
export DATASET=${DATASET:-kernelbench-cuda}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-timing_analysis}
export RL_EXPERIMENT_NAME=${RL_EXPERIMENT_NAME:-kernelblaster}

bash scripts/run_single_kernelblaster.sh
```

如需执行有预算上限的研究验收顺序，请使用
`python scripts/run_trusted_pilot.py`。脚本固定执行“运行环境 → 编译/正确性 →
3 Session Events → NCU 权限探针 → 单次 64-token API 冒烟 → 2×2 RMSNorm
Pilot”，任一必要门禁失败都会立即停止。

默认情况下，`scripts/run_single_kernelblaster.sh` 会启动单个使用 CUDA Events 的 KernelBench-CUDA RL 优化任务；如有需要，它还会启动仅监听回环地址的共享 GPU Server。运行输出保存在 `out/<dataset>/<precision>/<experiment>/` 下。

该示例默认运行 KernelBench-CUDA Level 1 的一个样本。可以通过 `--problem-numbers` 和 `--subset` 参数扩展到更多问题：

```bash
bash scripts/run_single_kernelblaster.sh --problem-numbers 1-10 --subset level2
```

#### 4. 运行产物

- 输入 Kernel 来自 `data/kernelbench-cuda/`。
- 默认脚本运行一个 Level 1 问题，并执行基于 RL 的 CUDA 优化。
- 轨迹产物、Prompt、日志和最佳输出保存在运行目录的 `out` 下。
- 最佳优化 Kernel 写入 `final_rl_cuda_perf.cu`。
- 经 rollout 搜索更新的优化数据库写入运行目录中的 `optimization_database.json`。

#### 5. 复现 PyTorch 基线

如需比较或复现上游 KernelBlaster 的加速效果，可在 Benchmark 问题上运行 `scripts/run_baselines.py`（Torch Eager）和 `scripts/run_baselines_compile.py`（Torch Compile）。

首先将 KernelBench Clone 到 `data/`：

```bash
git clone https://github.com/ScalingIntelligence/KernelBench.git data/KernelBench
```

Baseline Runner 会在根目录下查找 `problem.py`，动态导入问题模块，构建 `Model`，通过 `get_init_inputs()` 和 `get_inputs()` 获取输入，将模型和数据移动到 CPU 或 CUDA，执行预热和计时，并报告延迟统计。在 NCU 模式下，它会启动 Nsight Compute，并报告 Elapsed Cycles 或指定的其他原始指标。

```bash
# Torch Eager 基线
python scripts/run_baselines.py --root data/KernelBench/KernelBench/level1 --device cuda

# torch.compile 基线
python scripts/run_baselines_compile.py --root data/KernelBench/KernelBench/level1 --device cuda

# Nsight Compute（NCU）模式，默认报告 Elapsed Cycles
python scripts/run_baselines.py --root data/KernelBench/KernelBench/level1 --device cuda --ncu
```

## 仓库结构

```text
KernelBlaster/
|-- compose.yaml
|-- data/
|   |-- kernelbench-cuda/
|   |   |-- level1/
|   |   |-- level2/
|   |   `-- level3/
|   `-- kernelblaster/
|       |-- optimization_database.json
|       |-- optimization_database_header.md
|       `-- optimization_database_footer.md
|-- docker/
|   `-- Dockerfile
|-- portfolio/
|   |-- status.json
|   |-- suites/
|   `-- case_studies/
|       |-- core10/
|       `-- rmsnorm/
|-- artifacts/
|   |-- portfolio-v1.0/
|   |-- portfolio-v2.0/
|   `-- portfolio-v2.1/
|-- scripts/
|   |-- container.sh
|   |-- run-with-metadata.sh
|   |-- benchmark_cuda.py
|   |-- benchmark_candidates.py
|   |-- benchmark_pytorch.py
|   |-- analyze_core10_comparison.py
|   |-- sync_portfolio_docs.py
|   |-- run_single_kernelblaster.sh
|   |-- run_RL.py
|   |-- run_baselines.py
|   |-- run_baselines_compile.py
|   |-- run_reprofile.py
|   `-- start_gpu_server.py
|-- src/kernelblaster/
|   |-- agents/
|   |-- config/
|   |-- graph/
|   |-- resources/
|   |-- servers/
|   `-- workflow/
`-- utils/
```

### 关键目录

- `data/kernelbench-cuda/`：整理后的 KernelBench-CUDA 任务，每个任务包含 `init.cu` 和 `driver.cpp`。
- `data/kernelblaster/`：优化数据库和整理后的优化知识。
- `portfolio/`：实时状态清单、可复现 Suite、已提交候选和深度案例。
- `artifacts/portfolio-v1.0/`：不可变的历史环境、结果、报告、图表与 SHA256 发布包。
- `artifacts/portfolio-v2.0/`：schema-v2 Core 10 完整确认与定向验证证据。
- `artifacts/portfolio-v2.1/`：加固后的 Issue 证据、精简 NCU/NSYS 报告与自动生成的 SHA-256 索引。
- `scripts/`：Agent 入口，以及正确性优先的 CUDA、PyTorch、分析和文档同步 Runner。
- `docs/portfolio/`：架构、验证状态、深度案例证据和双语进度导航。
- `src/kernelblaster/agents/`：优化 Agent、经验回放组件、数据库逻辑和性能分析工具。
- `src/kernelblaster/graph/`：工作流 Graph Node 和共享状态定义。
- `src/kernelblaster/servers/`：优化过程中使用的编译和 GPU Server 基础设施。
- `src/kernelblaster/workflow/`：顶层工作流执行逻辑。

### CUDA 知识库数据结构

<p align="center">
  <img src="docs/figures/json.png" alt="知识库中的状态条目示例" width="520" />
</p>

知识库使用以状态为中心的结构存储优化经验。每个状态记录一种瓶颈模式、主要性能问题、识别该状态的次要特征，以及过去对类似 Kernel 有效的优化。借助这一设计，KernelBlaster 可以复用此前的搜索经验，而不是让每个任务都从零开始。

### 状态分组与优化选择

<p align="center">
  <img src="docs/figures/ODEa_small.png" alt="知识库状态分组及其优化效果" width="520" />
</p>

上图展示了知识库如何围绕 Memory-limited、Compute-bound 和 Hybrid 等状态族进行组织。系统会在每个状态中记录不同优化技术过去的效果，从而让未来搜索更倾向于预期收益较高的策略，同时保留探索空间。

### 跨任务与 Rollout 的记忆

<p align="center">
  <img src="docs/figures/KB.png" alt="跨任务和时间的记忆增强搜索" width="720" />
</p>

上图说明了 MAIC-RL 中的记忆增强机制。之前任务的 rollout 会将实际测得的性能结果写入知识库。当 KernelBlaster 在未来 rollout 中遇到新状态时，会使用这些历史结果引导搜索进入价值更高的区域，并避开过去表现较差的路径。

### 不同状态下的优化多样性

<p align="center">
  <img src="docs/figures/opt_pie.png" alt="按状态分组的优化技术应用分布" width="920" />
</p>

上图展示了框架覆盖的优化空间，包括向量化访存、Tensor Core 使用、Work-per-thread 调优、Shared Memory Tiling、Kernel Fusion、Occupancy 调优和多种专用优化。不同状态需要不同技术，不存在一种能够统治所有 CUDA Kernel 的单一优化策略。

知识库位于 `KernelBlaster/data/kernelblaster/optimization_database.json`，既可用于指导通用性能工程 Agent，也可以作为模型训练的标注数据。后者属于上游知识库的潜在用途，不在本 Fork 当前的模型训练范围内。

## 贡献者

[Kris Shengjun Dong](https://people.eecs.berkeley.edu/~chrisdong/)、[Sahil Modi](https://www.linkedin.com/in/sahil-modi)、[Dima Nikiforov](https://www.linkedin.com/in/dima-n/)、[Sana Damani](https://sanadamani.com/)、Edward Lin、[Siva Kumar Sastry Hari](https://sivahari.github.io/)、[Christos Kozyrakis](https://web.stanford.edu/~kozyraki/)

该项目的大部分工作由 Kris Shengjun Dong 在 2025 年 NVIDIA 暑期实习期间完成。

如果使用 KernelBlaster，请引用：

```bibtex
@article{dong2026kernelblaster,
  title={KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning},
  author={Dong, Kris Shengjun and Modi, Sahil and Nikiforov, Dima and Damani, Sana and Lin, Edward and Hari, Siva Kumar Sastry and Kozyrakis, Christos},
  journal={arXiv preprint arXiv:2602.14293},
  year={2026}
}
```
