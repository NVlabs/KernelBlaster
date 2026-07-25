# KernelBlaster Days 1-10 validation summary

> **Historical snapshot:** this file records the 2026-07-19 API-blocked Agent
> workflow and first RMSNorm publication. The later nine-candidate, unified
> Core 10, and same-GPU PyTorch results are in
> [`core10-rtx3080-summary.en.md`](core10-rtx3080-summary.en.md). The API 401
> and NCU permission outcomes remain valid.

- Environment: RTX 3080 10 GiB, `sm_86`, NGC PyTorch 25.01, CUDA 12.8.
- CPU tests: offline Provider, Recorder, Suite, dry-run, benchmark, and analysis coverage passed before publication.
- API: the single bounded smoke request failed with HTTP 401 and was not retried. Pilot and Core 10 Agent search therefore remain blocked; Agent-only portfolio score is 1.0.
- Core baselines: 10/10 attempted and 7/10 passed the 5% session-spread gate. The remaining tasks are explicitly marked unstable.
- Manual RMSNorm case study: V3c passed official and edge-shape correctness and measured 49.3477x versus the paired upstream V0. Every Session was faster; the runner self-check measured 1.0000x.
- NCU: attribution is blocked by `ERR_NVGPUCTRPERM` until the Windows driver reloads the enabled counter setting. No hardware-counter conclusion is published.
- Cost: one failed request, zero recorded tokens, estimated API cost $0.00.

Raw logs and reports remain under ignored `out/` directories. Curated CSV/JSON/SVG files link back to them through SHA256 values without storing credentials or response text.
