# RMSNorm Day 8–10 case study

This case study normalizes channel-first FP16 tensors shaped `[B, C, D1, D2]`
across `C`. The upstream V0 source remains unchanged at
`data/kernelbench-cuda/level1/036_RMSNorm/init.cu`.

## Thread mapping

V0 assigns one block to one spatial output position. Threads in a warp walk
channels, so adjacent threads read addresses separated by `D1 * D2`, and the
block performs a reduction even though `C=64` in the official Driver.

V1 assigns one thread to one spatial position:

```text
warp lane:       0        1        2       ...       31
spatial index:   p        p+1      p+2     ...       p+31
channel c load:  x[c,p]   x[c,p+1] x[c,p+2] ...     x[c,p+31]
```

Each thread accumulates all channels for its own position. A warp therefore
loads adjacent addresses for each channel and no cross-thread reduction is
required.

V2 packs two adjacent spatial positions into `half2` when the spatial stride is
even. If `D1 * D2` is odd, alternating channel bases are not four-byte aligned;
the explicit scalar fallback covers the entire odd shape and its tail safely.

V3a, V3b, and V3c independently test a 128-thread block, two `half2` pairs per
thread, and `rsqrtf` multiplication. They are compared with V2 separately so
each performance change has one primary hypothesis.

## Correctness protocol

Every variant must pass both the unmodified official Driver and the additional
`edge_driver.cpp`, before and after benchmark-only host synchronization removal.
The edge cases include a tiny tensor, odd spatial sizes, 63/64/65 channels, and
a larger odd boundary shape.

## RTX 3080 CUDA Events results

Protocol: NGC PyTorch 25.01, CUDA 12.8, `sm_86`, 20 warmups, 100 samples,
automatic inner-loop calibration, three independent process Sessions, fixed
seed, and alternating AB/BA order. All rows passed the official and edge Drivers
before timing. Speedup uses the paired V0 median from the same run.

| Variant | Main hypothesis | Paired V0 median (us) | Candidate median (us) | Speedup | Session spread |
| --- | --- | ---: | ---: | ---: | ---: |
| V1 | Coalesced spatial-thread mapping | 28603.904 | 600.064 | 47.668x | 1.025% |
| V2 | `half2` plus odd-stride fallback | 29123.584 | 606.720 | 48.002x | 0.338% |
| V3a | 128-thread block | 29122.561 | 609.536 | 47.778x | 0.378% |
| V3b | Two pairs per thread | 29125.119 | 623.104 | 46.742x | 0.082% |
| V3c | `rsqrtf` and multiply | 29359.104 | 594.944 | 49.348x | 0.086% |

V3c beat V1 by 1.0069x in a direct head-to-head run, with every paired Session
at least as fast. The difference is small and is used only for candidate
selection. The defensible main result is the mapping change from V0 to V1; V2,
V3a, and V3b are retained as correct negative experiments relative to the best
candidate. A V3c-vs-V3c runner self-check measured exactly 1.0000x and passed
the required 0.95-1.05 interval.

The later unified Core 10 run independently measured V0 at 31234.048 μs and
V3c at 591.872 μs, or 52.772x. The original 49.348x deep-case result remains
the primary ablation record; the unified rerun is reported alongside it rather
than averaged, because independent process Sessions and GPU state changed.
Both runs bind the measured source with SHA256 and pass the same official and
edge correctness gates.

The official shape moves roughly 384 MiB for two input reads and one output
write and performs about 0.5 FLOP/byte. V0 launches 1,048,576 blocks of 256
threads, while V1/V3c launch 4,096 blocks and give each thread an independent
spatial position. These are code-derived estimates, not profiler measurements.
NCU attribution remains blocked by `ERR_NVGPUCTRPERM` until the Windows driver
reloads the enabled performance-counter setting.

## Reproduction

Run from the repository root in WSL. Raw output directories are append-only and
ignored by Git.

```bash
docker run --rm --gpus all --ipc host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/kernelblaster" -w /kernelblaster \
  kernelblaster:validation-25.01 \
  python scripts/benchmark_cuda.py \
    --task-dir data/kernelbench-cuda/level1/036_RMSNorm \
    --task-id 036 --kernel RMSNorm \
    --candidate portfolio/case_studies/rmsnorm/best_rmsnorm_sm86.cu \
    --candidate-name v3c \
    --extra-correctness-driver \
      portfolio/case_studies/rmsnorm/edge_driver.cpp \
    --warmup 20 --repetitions 100 --sessions 3 \
    --output-dir out/portfolio/deep-rmsnorm/v2/<run-id>
```

Nsight Compute attribution additionally requires GPU Performance Counter access
on the Windows host. A failed `ERR_NVGPUCTRPERM` run is not valid profiler
evidence; CUDA Events measurements remain separately labeled.
