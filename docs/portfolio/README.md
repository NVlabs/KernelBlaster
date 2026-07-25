# Portfolio validation and development progress

[简体中文](README.zh-CN.md) | **English**

<!-- PORTFOLIO_PROGRESS:START -->
**Last updated: 2026-07-23**

- Days 1–2: provider, recorder, suite validation, dry-run, and CPU tests are complete.
- Days 3–7: WSL2/RTX 3080, container, compilation, correctness, and CUDA Events infrastructure are complete; API smoke status: failed: current HTTP 401 (1 request, 0 retries, 0 tokens; 2026-07-22).
- Days 8–10: RMSNorm V0–V3c is complete, with 49.348× in the independent deep run and 52.772× in the unified Core 10 rerun.
- Core 10 follow-up: full manual schema v2 passed 10/10 correctness and produced 4 improvements, 1 no-improvement result, and 5 inconclusive results; 9/10 tasks have a stable PyTorch method.
- Schema-v2 PyTorch comparison: across the 9/10 comparable tasks with a correct and stable method, the strict geometric mean versus the fastest stable method is 1.053×; task 026 is excluded from that geometric mean.

[Schema-v2 full Core 10 validation](../../artifacts/portfolio-v2.0/core10/core10-rtx3080-confirmation.en.md) · [Schema-v2 full result JSON](../../artifacts/portfolio-v2.0/core10/core10_rtx3080_comparison.json) · [Schema-v2 targeted validation](../../artifacts/portfolio-v2.0/reports/rtx3080-targeted-validation.en.md) · [Schema-v2 result JSON](../../artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json) · [Full Chinese report](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md) · [English summary](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md) · [Per-task JSON](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json) · [Comparison figure](../../artifacts/portfolio-v1.0/figures/core10_rtx3080_comparison.svg) · [Raw-file hashes](../../artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv) · [Candidate manifest](../../portfolio/case_studies/core10/candidates.json)
<!-- PORTFOLIO_PROGRESS:END -->

## Documentation map

- Architecture, provider, recorder, benchmark, and docs-sync boundaries: [English](architecture.md) · [简体中文](architecture.zh-CN.md)
- Validation gates, completed work, evidence, and remaining blockers: [English](validation.md) · [简体中文](validation.zh-CN.md)
- Measured RMSNorm V0–V3c case study: [English](rmsnorm-case-study.md) · [简体中文](rmsnorm-case-study.zh-CN.md)
- [Published RTX 3080 artifact bundle](../../artifacts/portfolio-v1.0/README.md)
- [Full Chinese Core 10 and PyTorch analysis](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md)
- [Standalone English result summary](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md)

## Update workflow

`portfolio/status.json` records narrative state and points to canonical evidence. Performance values are derived from the checked-in result JSON rather than copied into the manifest. After any benchmark, candidate, analysis, or publication change:

```bash
python scripts/sync_portfolio_docs.py --write
python scripts/sync_portfolio_docs.py --check
```

The GitHub documentation-sync workflow repeats the check and rejects relevant changes that omit README/docs or status updates.
