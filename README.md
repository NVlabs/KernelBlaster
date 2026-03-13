# KernelBlaster

## Paper

Corresponding paper: [arXiv:2602.14293](http://arxiv.org/abs/2602.14293)

**Authors**  
[Kris Shengjun Dong](https://people.eecs.berkeley.edu/~chrisdong/), [Sahil Modi](https://www.linkedin.com/in/sahil-modi), [Dima Nikiforov](https://www.linkedin.com/in/dima-n/), [Sana Damani](https://sanadamani.com/), Edward Lin, [Siva Kumar Sastry Hari](https://sivahari.github.io/), [Christos Kozyrakis](https://web.stanford.edu/~kozyraki/)

**Affiliation:** NVIDIA, University of California, Berkeley

***Note:** This repository hosts an archival release of KernelBlaster. The initial commit in this repository does not reflect the original authorship; Most of the original work was contributed to by Kris Shengjun Dong during her 2025 summer internship at NVIDIA.

## Contributors

Main code contributors: Kris Shengjun Dong, Sahil Modi, and Dima Nikiforov.


## Project Intro

<p><strong><span style="color:#0f766e;">Introducing KernelBlaster, a Memory-Augmented In-context Reinforcement Learning (MAIC-RL) framework</span></strong></p>

Optimizing CUDA code across multiple GPU generations is difficult because the best implementation depends on a large and hardware-specific search space. A kernel that looks reasonable on one GPU can leave performance on the table on another, and simple rewrites are rarely enough to reach the best result.

Traditional compiler pipelines are limited by fixed heuristics, while fully finetuning large language models for every optimization setting is expensive. Many agentic CUDA workflows also have a simpler problem: they do not remember enough from previous exploration. That leads to repeated mistakes, biased sampling, and weaker optimization choices.

KernelBlaster is built to make that search smarter. Instead of treating each kernel as an isolated prompt, it combines profiling feedback, a persistent CUDA optimization knowledge base, and reinforcement-learning-style exploration. The agent does not just generate code; it profiles, reflects, retrieves prior optimization knowledge, explores new candidates, and updates its strategy over time.

The result is a reusable open-source framework for CUDA optimization with verification, profiling, replay, and reproducible evaluation built in.

Compared to the PyTorch baseline, KernelBlaster achieves geometric mean speedups of <strong><span style="color:#ef4444;">1.43x</span></strong> on KernelBench Level 1, <strong><span style="color:#2563eb;">2.50x</span></strong> on Level 2, and <strong><span style="color:#16a34a;">1.50x</span></strong> on Level 3.

## Why KernelBlaster

| Others | KernelBlaster |
| --- | --- |
| CUDA optimization is hardware-aware and requires searching a large design space. | KernelBlaster narrows that search with profiling-guided state extraction and targeted optimization selection. |
| Fixed compiler heuristics cannot easily adapt to every kernel or GPU generation. | KernelBlaster adapts optimization decisions to each kernel and GPU generation through retrieval and iterative search. |
| Finetuning LLMs for optimization is costly and slow to iterate on. | KernelBlaster improves optimization through in-context memory and RL-style exploration without depending on expensive task-specific finetuning. |
| Naive agent loops forget what they learned from earlier kernels and earlier rollouts. | KernelBlaster keeps memory in the loop through a persistent optimization database and replay-driven exploration. |

## How It Works

KernelBlaster starts from the initial KernelBench-CUDA input artifacts. Each problem provides a starter CUDA implementation in `init.cu` and a matching C++ harness in `driver.cpp`. The CUDA file is the code to optimize; the driver builds, runs, and validates the kernel against the reference behavior.

From there, the pipeline runs an agentic optimization loop:

1. Load the input problem from `data/kernelbench-cuda/<level>/<problem>/`.
2. Use `init.cu` as the starting CUDA kernel and `driver.cpp` as the validation harness.
3. Compile and profile candidate kernels, with Nsight Compute metrics and elapsed cycles as the main performance signal.
4. Retrieve relevant optimization ideas from the persistent CUDA knowledge base.
5. Generate a new candidate using profile-guided, textual-gradient-style prompts.
6. Evaluate the candidate, reward successful trajectories, and store them in the replay buffer.
7. Update future decisions using what worked, what failed, and the feedback from the profiler.
8. Save the best optimized kernel as `final_rl_cuda_perf.cu`.

In code, the default single-run path is:

- `scripts/run_single_kernelblaster.sh` starts the runtime environment and launches the RL run.
- `scripts/run_RL.py` prepares the dataset, servers, and workflow inputs.
- `src/kernelblaster/workflow/workflow.py` invokes the graph-based workflow.
- `src/kernelblaster/graph/nodes/optimization_rl_ncu.py` loads `init.cu` and `driver.cpp`, then launches the RL optimization agent.
- `src/kernelblaster/agents/opt_ncu_rl.py` runs the rollout, profiling, replay-buffer, and strategy-update loop.

<p align="center">
  <img src="docs/figures/flow_chart.png" alt="KernelBlaster end-to-end agentic flow" width="720" />
</p>

This figure shows the end-to-end optimization loop. KernelBlaster starts from the input kernel and the target GPU hardware, extracts a performance state, matches that state against the knowledge base, selects a promising optimization, lowers it into code, tests correctness, profiles the result, and repeats until the termination check decides that the search has converged. The final stage uses LLM soft verification before writing the optimized output kernel.

## Quick Start

### 1. Build the container

```bash
docker build . -t kernelblaster -f docker/Dockerfile
```

### 2. Launch the container

```bash
docker run --rm -it --name=kernelblaster \
    --privileged --gpus all --cap-add=SYS_ADMIN --device /dev/fuse \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --ipc=host --net=host \
    -e USER_NAME=$(whoami) \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    -v $(pwd):/kernelblaster \
    kernelblaster \
    dev
```

### 3. Set your API key and run the default example

```bash
export OPENAI_API_KEY=<your_api_key>
export MODEL=${MODEL:-gpt-5-mini-2025-08-07}
export GPU_TYPE=${GPU_TYPE:-L40S}
export DATASET=${DATASET:-kernelbench-cuda}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-timing_analysis}
export RL_EXPERIMENT_NAME=${RL_EXPERIMENT_NAME:-kernelblaster}

bash scripts/run_single_kernelblaster.sh
```
