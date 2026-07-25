# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Keep the two runtime trust domains explicit.  `control` deliberately starts
# from the official Python image, while `gpu-supervisor` keeps the existing
# CUDA 12.8 NGC baseline pinned by tag and digest.
ARG CONTROL_BASE_IMAGE=python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b
ARG GPU_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3@sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb

FROM ${CONTROL_BASE_IMAGE} AS control

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/kernelblaster
COPY docker/requirements.control.txt /tmp/requirements.control.txt
COPY . /opt/kernelblaster
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.control.txt \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin kernelblaster \
    && chown -R kernelblaster:kernelblaster /opt/kernelblaster

USER kernelblaster
EXPOSE 8000
CMD ["python", "-m", "src.kernelblaster.servers.control", "--host", "0.0.0.0", "--port", "8000"]

FROM ${GPU_BASE_IMAGE} AS cuda-base

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH=/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=8.6 \
    CMAKE_CUDA_ARCHITECTURES=86 \
    CUDAARCHS=86 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ccache cmake git ninja-build sudo \
    && rm -rf /var/lib/apt/lists/* \
    && nvcc --version

WORKDIR /workspace

FROM cuda-base AS gpu-supervisor

COPY docker/requirements.gpu.txt /tmp/requirements.gpu.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.gpu.txt
COPY . /opt/kernelblaster
RUN python -m pip install --no-cache-dir --no-deps /opt/kernelblaster \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin kernelblaster \
    && mkdir -p /tmp/kernelblaster \
    && chown -R kernelblaster:kernelblaster /tmp/kernelblaster /opt/kernelblaster

WORKDIR /opt/kernelblaster
ENV HOME=/home/kernelblaster \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_TF32_OVERRIDE=0
USER kernelblaster
EXPOSE 2002
CMD ["python", "scripts/start_gpu_worker.py", "--host", "0.0.0.0", "--port", "2002"]

FROM cuda-base AS dev

COPY . /opt/kernelblaster
RUN python -m pip install --no-cache-dir \
       "pip==24.3.1" "setuptools==80.9.0" "wheel==0.45.1" \
    && python -m pip install --no-cache-dir -e "/opt/kernelblaster[server,llm,local-llm,gpu,dev]" \
    && python -m pip install --no-cache-dir \
       "packaging==24.2" "setuptools==80.9.0" "six==1.16.0" \
    && mkdir -p /tmp/home /workspace \
    && chmod 1777 /tmp/home /workspace

ENV HOME=/tmp/home \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TRITON_CACHE_DIR=/cache/triton \
    PYTHONPATH=/workspace
USER 1000:1000
CMD ["bash"]
