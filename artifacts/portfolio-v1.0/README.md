# KernelBlaster RTX 3080 validation artifacts

This directory contains the curated, redacted outputs for Days 1-10. Raw logs,
build trees, benchmark samples, and any `.ncu-rep` files remain under ignored
`out/portfolio/` directories.

The original Days 1–10 validation report is an immutable historical snapshot
of the API-blocked Agent run and the first RMSNorm publication. The later
manual Core 10/PyTorch follow-up supersedes its per-task performance coverage,
but does not change the historical API 401 or NCU permission outcomes. Start
with `reports/core10-rtx3080-summary.en.md` or the full Chinese follow-up report
for the latest measured state.

- `environment/`: sanitized WSL, Docker, CUDA, GPU, Server, API, and NCU status.
- `results/`: machine-readable Core 10, deep-case, failure, usage, and aggregate data.
- `figures/`: SVG generated from the result CSV/JSON files.
- `reports/`: Chinese validation report and standalone English summary.
- `manifests/`: SHA256 links to local raw inputs and generated analysis outputs.

The follow-up manual Core 10 optimization and same-RTX-3080 PyTorch comparison
adds:

- `reports/core10-rtx3080-comparison.zh-CN.md`: full Chinese analysis.
- `reports/core10-rtx3080-summary.en.md`: standalone English summary.
- `results/core10_rtx3080_comparison.csv` and `.json`: per-task medians,
  p10/p90, session medians, stability gates, and PyTorch ratios.
- `results/pytorch_core10_rtx3080.csv`: every measured PyTorch eager,
  preallocated-out, and fused method with session medians.
- `figures/core10_rtx3080_comparison.svg`: log-scale visual comparison.
- `manifests/core10_rtx3080_SHA256SUMS.json`: hashes for the generated
  comparison files.
- `manifests/core10_rtx3080_raw_sha256.csv`: hashes and sizes for the 362
  ignored local JSON/JSONL/CSV/SVG/log files used by the comparison.

Important interpretation boundaries:

- The single live API request failed with HTTP 401 and was not retried. No Core
  10 Agent search result is claimed.
- In the original Days 1–10 baseline snapshot, all ten Core tasks were attempted;
  seven passed the session-stability gate and three remained unstable. The
  follow-up reran candidates and PyTorch with a separate strict gate.
- RMSNorm is a manual Day 8-10 case study. Its CUDA Events result is separate
  from the Agent-only portfolio score.
- Nsight Compute hardware-counter attribution is blocked by
  `ERR_NVGPUCTRPERM` until the Windows driver reloads the enabled setting.
- The manual follow-up candidates all pass correctness, but only four meet the
  strict material/stability/no-regression gate. Unstable values remain visible
  as diagnostic evidence and are excluded from the verified portfolio score.
