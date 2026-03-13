# KernelBench-CUDA

This directory contains CUDA-based KernelBench problems included here for ease of use by KernelBlaster users. These problems were generated via a separate project and are provided as input benchmarks for KernelBlaster's optimization pipeline.

Each problem directory contains:

- `init.cu` — the CUDA kernel implementation to be optimized
- `driver.cpp` — the C++ driver that builds, runs, and validates the kernel against a reference implementation

## Problem Levels

| Level | Problems | Description |
|-------|----------|-------------|
| `level1/` | 94 | Single-operator kernels (matrix multiply, activations, reductions, etc.) |
| `level2/` | 81 | Multi-operator fused kernels (conv + activation + pooling, etc.) |
| `level3/` | 9 | Full model blocks (MLP, LeNet, ResNet block, attention, etc.) |
