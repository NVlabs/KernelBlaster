# Validation status and benchmark protocol

**English** | [简体中文](validation.zh-CN.md)

This document is the living validation record for the portfolio fork. Historical raw runs remain append-only below ignored `out/portfolio/` directories; reviewed evidence is published under `artifacts/portfolio-v1.0/`.

<!-- VALIDATION_STATUS:START -->
| Gate | Current status | Canonical evidence |
| --- | --- | --- |
| Provider/Recorder/Suite CPU tests | PASSED — 177 passed | `tests/` |
| Real gateway smoke | failed: current HTTP 401 (1 request, 0 retries, 0 tokens; 2026-07-22) | `artifacts/portfolio-v2.1/issue-7/rtx3080/trusted-pilot-summary.json` |
| RTX 3080 container and `sm_86` build | PASSED | `artifacts/portfolio-v1.0/environment/environment.json` |
| Official candidate correctness | HISTORICAL V1 PASSED — 10/10; schema-v2 full 10/10 passed | `artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json` |
| RMSNorm edge correctness | PASSED | committed `edge_driver.cpp` and deep-case artifacts |
| CUDA Events timing | schema-v2 full confirmation: 4 improved; 1 no improvement; 5 inconclusive | `artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json` |
| Same-GPU PyTorch comparison | schema-v2 full confirmation; 9/10 tasks have a stable method | `artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json` |
| Issue #10 capability/resource hardening | 4 formal improvements; 095 remains inconclusive because the upstream baseline spread is 24.37%, so the Issue stays open | `artifacts/portfolio-v2.1/issue-10/rtx3080/correctness-summary.json` |
| Portfolio v2.1 evidence integrity | Exact SHA256 index published | `artifacts/portfolio-v2.1/SHA256SUMS.json` |
| NCU hardware counters | BLOCKED — `ERR_NVGPUCTRPERM (non-root Docker/WSL; one no-network SYS_ADMIN retry also blocked; Windows native control passed)` | `artifacts/portfolio-v2.1/issue-8/rtx3080/ncu-preflight-summary.json` |
| Cross-GPU comparison | BLOCKED — `requires authorized A100/L40S rental` | no aggregate cross-GPU performance claim published |

The historical manual follow-up validated all ten candidates and improved 4/10 under the old gate. Those claims remain immutable historical evidence. The full schema-v2 result still confirms manual candidates and must not be generalized to an Agent-search claim.
<!-- VALIDATION_STATUS:END -->

## Completed development and experiment timeline

1. **Days 1–2 — provider and observability:** implemented the OpenAI-compatible provider boundary, client-side fan-out, retry classification, atomic request/token reservation, usage fallback, Recorder sequence/atomicity/redaction, Suite validation, and offline dry-run artifacts.
2. **Days 3–7 — local GPU validation:** pinned the NGC PyTorch 25.01 container, mapped the RTX 3080 to `sm_86`, validated compiler/GPU servers, added the independent CUDA Events runner, and recorded the baseline Core 10 state. The bounded live API smoke returned HTTP 401, so Agent Pilot/Core 10 rollouts did not run.
3. **Days 8–10 — RMSNorm deep case:** implemented V1–V3c, added odd/tiny/63–65-channel correctness inputs, retained failed variants, and published the 49.348× paired V3c result. NCU counter access remained blocked, so no hardware-counter attribution was published.
4. **Core 10 follow-up:** added nine manual candidates, reran all ten tasks with 20 warmups, 100 samples, three independent process Sessions, AB/BA ordering, and compared them with same-GPU PyTorch eager/out/fused methods.
5. **Publication:** checked in redacted JSON/CSV/SVG reports and SHA256 links for 362 ignored raw JSON/JSONL/CSV/SVG/log files. Merged PRs #4 and #5 contain the artifact publication and living-documentation follow-up.
6. **Schema-v2 full confirmation:** reran all ten manual candidates and the same-GPU PyTorch methods with five independent process Sessions. Correctness passed 10/10; 004/007/036/040 improved, 088 reported no improvement, and 019/023/026/047/095 remained inconclusive after automatic retesting.

## Result interpretation

- Historical v1 diagnostic candidate medians include unstable tasks and are useful for prioritizing follow-up work, not release claims.
- The current schema-v2 gate requires correctness, no more than 5% cross-session spread, five paired Sessions, at least 1.01× median speedup, and a paired-bootstrap 95% lower bound above 1.0.
- Under schema v2, the strict Core 10 geometric mean versus upstream is 4.381×. A stable PyTorch method exists for 9/10 tasks; only across those comparable tasks, the strict ratio versus the fastest stable PyTorch method is 1.053×.
- The nine newly developed candidates score 5.020× diagnostic and 3.302× strict versus upstream; 004, 007, and 040 pass the strict gate.
- Full Core 10, including the existing RMSNorm case, scores 6.351× diagnostic and 4.356× strict versus upstream.
- Against the fastest measured PyTorch method, the nine-candidate diagnostic/strict ratios are 1.415×/0.931×; the full-ten ratios are 1.447×/0.992×.
- Task 007 calls cuBLAS and therefore demonstrates correct mature-library integration rather than a custom GEMM beating cuBLAS.

Canonical evidence:

- [Schema-v2 full Core 10 confirmation](../../artifacts/portfolio-v2.0/core10/core10-rtx3080-confirmation.en.md)
- [Schema-v2 full comparison JSON](../../artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json)
- [Schema-v2 targeted validation](../../artifacts/portfolio-v2.0/reports/rtx3080-targeted-validation.en.md)
- [Schema-v2 result JSON](../../artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json)
- [Per-task comparison JSON](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json)
- [Full Chinese analysis](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md)
- [Standalone English summary](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md)
- [Environment manifest](../../artifacts/portfolio-v1.0/environment/environment.json)
- [Raw-artifact SHA256 manifest](../../artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv)

## Acceptance rules

- Reject a candidate when any official or extra correctness input fails, regardless of timing.
- Preserve original and timing-normalized source SHA256 values in each raw manifest.
- Report GPU, SM target, precision, shape, seed, warmup, repetitions, inner loops, Session/order, median, p10/p90, telemetry, driver/CUDA/container, and source provenance.
- Treat upstream CUDA, candidate CUDA, PyTorch eager/out/fused, and paper-reported results as separate baselines.
- Stop formal performance claims when the automatic cooldown/retest still exceeds the 5% Session gate.
- Require at least five paired confirmation Sessions, median speedup of 1.01× or higher, and a paired-bootstrap 95% lower bound above 1.0.
- Do not infer NCU bottlenecks from CUDA Events or code inspection; `ERR_NVGPUCTRPERM` remains an explicit attribution blocker.

## Remaining blockers and deferred work

- Supply a valid external API credential before Agent-driven Pilot/Core 10 search; the single 401 request was not retried or billed as a successful completion.
- Diagnose the repeatable Session instability in 019/023/026/047/095 without promoting their diagnostic speedups.
- Rerun PyTorch 026 until a correct stable framework baseline exists before including it in a PyTorch geometric mean.
- Collect the required NCU sections on an explicitly authorized profiler worker; local Docker/WSL remains `events_only`.
- Run and publish L40S/A100 matching comparisons independently.
- Generalize fixed-shape candidates across shapes, dtypes, layouts, streams, graph capture, backward paths, and stricter numerical tolerances before production-library claims.

## Reproduction commands

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

The GPU/profiler worker uses the `gpu-supervisor` target in the canonical root
`compose.yaml` and runs as a non-root user with an internal worker network,
`--cap-drop ALL`, and `no-new-privileges`; no API credential is passed into it. Only a one-shot NCU
container that has already produced `ERR_NVGPUCTRPERM` may add
`--cap-add SYS_ADMIN`. `--privileged` is never an accepted workaround. On WSL,
the Nsys smoke also applies [NVIDIA's documented WSL timestamp fallback](https://archive.docs.nvidia.com/nsight-systems/2025.2/ReleaseNotes/index.html)
(`CuptiUseRawGpuTimestamps=false`), prewarms RMSNorm, and
accepts the run only when the GPU kernel table contains `rmsnorm_half2_rsqrt`.

The language model remains an external inference service. Rollouts update trajectories and the optimization database; they do not train or fine-tune model weights.
