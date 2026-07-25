# Core 10 manual optimization vs same-GPU PyTorch

## Result

On one RTX 3080 (sm_86), all ten candidates passed the official FP16 correctness Driver. The raw candidate medians produced a 6.351x geometric mean over the upstream implementations, but four tasks exceeded the 5% cross-session stability limit. Applying the strict gate—correct, stable, no slower session, and at least 1.01x—validated tasks 004, 007, 036, and 040. Counting every other task as 1.0 gives a 4.356x all-ten portfolio score.

Against the fastest same-GPU PyTorch method measured per task, the candidate medians have a diagnostic 1.447x geometric mean and win 7/10 tasks. This includes unstable microbenchmarks and is not a release-grade claim. After falling back to upstream for every unverified candidate, the geometric mean versus PyTorch is 0.992x: effectively parity, with PyTorch about 0.8% faster overall.

Excluding the pre-existing 036 RMSNorm case and considering only the nine new candidates, the diagnostic and strict geometric means versus upstream are 5.020x and 3.302x, respectively; 004, 007, and 040 pass the strict gate. Versus the fastest measured PyTorch method, the nine-candidate diagnostic mean is 1.415x (6/9 wins), while the strict fallback state is 0.931x, meaning PyTorch is about 7.4% faster.

| ID | Upstream μs | Candidate μs | Upstream/candidate | PyTorch best μs | Candidate/PyTorch | Status |
|---|---:|---:|---:|---:|---:|---|
| 004 | 18472.960 | 107.008 | 172.632x | 110.182 | 1.030x | verified |
| 007 | 10031.616 | 809.472 | 12.393x | 823.808 | 1.018x | verified; cuBLAS candidate |
| 019 | 9.652 | 9.484 | 1.018x | 9.344 | 0.985x | unstable |
| 023 | 12.800 | 12.695 | 1.008x | 10.174 | 0.801x | unstable / immaterial |
| 026 | 9.252 | 9.211 | 1.004x | 9.710 | 1.054x | immaterial; one slower session |
| 036 | 31234.048 | 591.872 | 52.772x | 1041.408 | 1.760x | verified, including edge Driver |
| 040 | 15339.008 | 703.744 | 21.796x | 6640.128 | 9.435x | verified |
| 047 | 19.477 | 13.305 | 1.464x | 11.136 | 0.837x | unstable |
| 088 | 27.745 | 27.703 | 1.001x | 27.745 fused GELU | 1.001x | parity |
| 095 | 560.640 | 19.513 | 28.732x | 64.512 | 3.306x | large but unstable |

The strongest specialized results are RMSNorm and the fixed-shape large LayerNorm. Matrix-vector and small-K matmul are already close to PyTorch; the latter explicitly uses cuBLAS, so it demonstrates correct library use rather than a custom GEMM beating cuBLAS. MinGPT GELU matches PyTorch only when compared with the equivalent fused GELU primitive; comparing against the unfused Driver formula would misleadingly show an 8.8x win.

This is best described as a strong fixed-shape CUDA optimization prototype, not yet a production operator library. The candidates do not yet cover arbitrary shapes, dtypes, layouts, streams, graph capture, backward passes, or strict randomized numerical testing. Static workspaces in 040/095 also need production lifecycle and concurrency work.

Protocol: NGC PyTorch 25.01, CUDA 12.8, driver 591.86, FP16, official Core 10 shapes, 20 warmups, 100 CUDA Event samples, automatic inner-loop calibration, three independent processes, and AB/BA baseline/candidate order.
