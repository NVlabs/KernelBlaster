#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 <command> [args...]" >&2
  exit 64
fi

run_root=${RUN_ROOT:-/runs}
run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
run_dir="$run_root/$run_id"
mkdir "$run_dir"

{
  echo "run_id=$run_id"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -C /workspace rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "command=$(printf '%q ' "$@")"
  echo "python=$(python --version 2>&1)"
  echo "cuda=$(nvcc --version 2>/dev/null | tail -1 || echo unavailable)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
} > "$run_dir/metadata.txt"

set +e
"$@" 2>&1 | tee "$run_dir/run.log"
status=${PIPESTATUS[0]}
set -e
printf 'exit_code=%s\nfinished_at=%s\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_dir/metadata.txt"
exit "$status"
