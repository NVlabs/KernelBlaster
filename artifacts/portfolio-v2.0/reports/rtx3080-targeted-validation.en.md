# RTX 3080 targeted schema-v2 validation

**English** | [简体中文](rtx3080-targeted-validation.zh-CN.md)

This report publishes a targeted manual validation of five Core 10 candidates. It is not an Agent-generated search result and does not replace the immutable v1 reports.

Protocol: FP16, default stream, 20 warmups, 100 CUDA Events samples, five independent process Sessions, alternating AB/BA order, maximum 5% Session spread, median speedup at least 1.01×, and a paired-bootstrap 95% lower bound above 1.0. Official and edge inputs also require finite output, five-run determinism, and p99/max error no worse than `baseline * 1.10 + 1e-4`.

| Task | Result | Median paired speedup | Bootstrap 95% interval | Stability note |
| --- | --- | ---: | ---: | --- |
| 004 | improved | 182.176× | [179.696×, 185.507×] | baseline/candidate spread 2.876%/0.861% |
| 007 | inconclusive | 11.849× diagnostic only | [11.027×, 12.547×] | upstream baseline spread 13.779%; excluded |
| 036 | improved | 56.332× | [56.206×, 56.371×] | baseline/candidate spread 0.506%/0.344% |
| 040 | improved | 23.093× | [23.048×, 23.368×] | baseline/candidate spread 1.686%/1.598% |
| 095 | exploratory | — | — | correctness passed; no five-Session confirmation |

All five targeted candidates passed official and added boundary correctness, finite-output, determinism, and error-regression checks. The raw suite summary was generated before the final `inconclusive` status patch; the canonical [result JSON](../results/rtx3080_targeted_validation.json) records the session medians, gate outputs, and source/Driver/raw-file hashes used for this classification.

NCU remained unavailable with `ERR_NVGPUCTRPERM`, so the local result is explicitly `events_only`. No profile-guided hardware-counter attribution or full Agent-loop completion is claimed.
