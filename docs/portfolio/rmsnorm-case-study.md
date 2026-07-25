# RMSNorm optimization case study

**English** | [简体中文](rmsnorm-case-study.zh-CN.md)

<!-- RMSNORM_STATUS:START -->
Current status: **validated on NVIDIA GeForce RTX 3080 (sm_86)**.

- V0–V3c and the benchmark-normalized sources pass the official Driver.
- V1–V3c also pass the small, odd-spatial, 63/64/65-channel, and boundary cases in `edge_driver.cpp`.
- The independent V3c deep run measured 49.348× versus its paired V0.
- The later unified Core 10 rerun measured 52.772×; the difference is retained as run-to-run evidence, not silently averaged.
- The targeted five-Session schema-v2 confirmation measured 56.332× median paired speedup and passed the bootstrap and stability gates.
- The full schema-v2 Core 10 rerun measured 54.228× and also passed the formal gate; both runs remain separate.
- NCU hardware-counter attribution remains blocked by `ERR_NVGPUCTRPERM (non-root Docker/WSL; one no-network SYS_ADMIN retry also blocked; Windows native control passed)`; CUDA Events and code-derived mapping evidence are reported separately.
<!-- RMSNORM_STATUS:END -->

## Question and outcome

The case study asked whether a correctness-first, profile-guided redesign could reduce independently measured latency by at least 5% over the unmodified channel-first FP16 RMSNorm kernel. The gate passed decisively on RTX 3080. The primary gain came from changing the thread/data mapping, not from the later micro-tuning variants.

The official tensor is `[B=16, C=64, D1=256, D2=256]` and normalization is across `C`. V0 assigns a block to each spatial position, makes adjacent warp lanes read channel-strided addresses, and performs a cross-thread reduction. V1 assigns one thread to each spatial position; lanes then read adjacent addresses for every channel and independently accumulate 64 channel values.

## Measured versions

Protocol: NGC PyTorch 25.01, CUDA 12.8, `sm_86`, FP16, 20 warmups, 100 CUDA Events samples, automatic inner loops, three independent process Sessions, fixed seed, and AB/BA order. Every row passed the official and edge Drivers before timing.

| Version | Isolated change | Paired V0 median (μs) | Candidate median (μs) | Speedup | Candidate Session spread |
| --- | --- | ---: | ---: | ---: | ---: |
| V1 | contiguous spatial-thread mapping | 28603.904 | 600.064 | 47.668× | 1.025% |
| V2 | `half2` plus odd-stride fallback | 29123.584 | 606.720 | 48.002× | 0.338% |
| V3a | 128-thread block | 29122.561 | 609.536 | 47.778× | 0.378% |
| V3b | two pairs per thread | 29125.119 | 623.104 | 46.742× | 0.082% |
| V3c | `rsqrtf` and multiply | 29359.104 | 594.944 | 49.348× | 0.086% |

V3c beat V1 by only 1.0069× in the direct head-to-head run. That small delta is used for candidate selection, while the defensible optimization result is the approximately 48–49× mapping redesign from V0 to V1/V3c. V2, V3a, and V3b are preserved as correct negative or neutral experiments.

The later unified Core 10 run independently measured V0 at 31234.048 μs and V3c at 591.872 μs, or 52.772×. It is not averaged with the deep-case value: both runs remain visible to show process/session and GPU-state variation.

## Correctness and numerical boundary

Both original and benchmark-normalized sources pass:

- the unmodified official Driver;
- a tiny tensor;
- an odd spatial stride and scalar tail;
- 63, 64, and 65 channels;
- a larger odd boundary shape.

The official FP16 tolerance remains `rtol=0.1, atol=0.1`. This validates the benchmark contract but does not replace production-grade error-distribution testing over arbitrary shapes and dtypes.

## Performance explanation and profiler boundary

The official shape moves roughly 384 MiB for two input reads and one output write and performs about 0.5 FLOP/byte. V0 launches 1,048,576 blocks of 256 threads; V1/V3c launch 4,096 blocks and give each thread an independent spatial position. These are source-derived traffic and launch estimates.

NCU 2025.1 was available, but performance-counter collection returned `ERR_NVGPUCTRPERM`. No occupancy, memory-throughput, scheduler, or warp-stall attribution is claimed. CUDA Events measurements and the source-level mapping explanation are deliberately labeled separately.

## Evidence and reproduction

- [Committed source, edge Driver, and full variant discussion](../../portfolio/case_studies/rmsnorm/README.md)
- [Published deep-case CSV](../../artifacts/portfolio-v1.0/results/deep_case_results.csv)
- [Core 10 follow-up comparison](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json)
- [Full Chinese report](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md)

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
