#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Runner Script for KernelBlaster
# This script runs a single problem by default 

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to root directory
cd "$ROOT_DIR"

echo "=========================================="
echo "KernelBlaster Runtime Timing Analysis"  
echo "=========================================="

# Set default values
DATASET="${DATASET:-kernelbench-cuda}"
PRECISION="${PRECISION:-fp16}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-timing_analysis}"
MODEL="${MODEL:-gpt-5-mini-2025-08-07}"
GPU_TYPE="${GPU_TYPE:-L40S}"
PROBLEM_NUMBERS_DEFAULT="1"
SUBSET_DEFAULT="level1"

# CLI overrides (keeps defaults if flags not provided)
PROBLEM_NUMBERS="$PROBLEM_NUMBERS_DEFAULT"
SUBSET="$SUBSET_DEFAULT"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--problem-numbers N] [--subset NAME]

Options:
  --problem-numbers N   Problem number(s) to run (default: ${PROBLEM_NUMBERS_DEFAULT})
  --subset NAME         Dataset subset to run (default: ${SUBSET_DEFAULT})
  -h, --help            Show this help message

Examples:
  $(basename "$0") --problem-numbers 1 --subset level1
  $(basename "$0") --problem-numbers=5 --subset=level2
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem-numbers)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "Error: --problem-numbers requires a value."
                usage
                exit 1
            fi
            PROBLEM_NUMBERS="$2"
            shift 2
            ;;
        --problem-numbers=*)
            PROBLEM_NUMBERS="${1#*=}"
            shift 1
            ;;
        --subset)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "Error: --subset requires a value."
                usage
                exit 1
            fi
            SUBSET="$2"
            shift 2
            ;;
        --subset=*)
            SUBSET="${1#*=}"
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Precision: $PRECISION"
echo "  Model: $MODEL"
echo "  GPU: $GPU_TYPE"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Problem numbers: $PROBLEM_NUMBERS"
echo "  Subset: $SUBSET"
echo ""

# Check if we're in the right directory
if [ ! -d "src" ]; then
    echo "Error: src directory not found. Please ensure this script is in the KernelBlaster repository."
    exit 1
fi

# Check if GPU is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Warning: nvidia-smi not available. GPU operations may fail."
fi

# Test the timing system first (if it exists)
if [ -f "test_timing.py" ]; then
    echo "Testing timing system..."
    python test_timing.py
    if [ $? -ne 0 ]; then
        echo "Error: Timing system test failed. Please check the setup."
        exit 1
    fi
    echo ""
else
    echo "Note: test_timing.py not found, skipping timing system test."
    echo ""
fi

# Run the actual timing analysis
echo "Starting timing analysis..."
echo ""

# Create output directory
mkdir -p "out/${DATASET}/${PRECISION}/${EXPERIMENT_NAME}"

# Export environment variables
export MODEL="$MODEL"
export DATASET="$DATASET"

# Start a shared GPU server (if it exists)
GPU_STARTER_PID=""
GPU_SERVER_PID=""
GPU_SERVER_URL="http://localhost:2002"

if [ -f "scripts/start_gpu_server.py" ]; then
    echo "Starting shared GPU server..."
    python scripts/start_gpu_server.py --port 2002 --log-file "out/gpu_server.log" --info-file gpu_server_info.txt &
    GPU_STARTER_PID=$!
    
    # Wait a moment for GPU server to start and write info
    sleep 5

    # Read GPU server info
    if [ ! -f "gpu_server_info.txt" ]; then
        echo "Error: Failed to start GPU server"
        kill $GPU_STARTER_PID 2>/dev/null
        exit 1
    fi

    GPU_SERVER_PID=$(tail -n 1 gpu_server_info.txt)
    echo "Shared GPU server started at: $GPU_SERVER_URL (PID: $GPU_SERVER_PID)"
else
    echo "Note: start_gpu_server.py not found, skipping GPU server startup."
    echo "Using existing GPU server or will start one via run_RL.py"
fi

# Signal handler for cleanup
cleanup_processes() {
    echo "Cleaning up processes..."
    
    # Kill compiler servers by port (backup cleanup)
    if command -v lsof >/dev/null 2>&1; then
        for port in 2011; do
            COMPILER_PID=$(lsof -ti:$port 2>/dev/null)
            if [ -n "$COMPILER_PID" ]; then
                echo "Terminating compiler server on port $port (PID $COMPILER_PID)..."
                kill -TERM $COMPILER_PID 2>/dev/null
                sleep 1
                if kill -0 $COMPILER_PID 2>/dev/null; then
                    kill -KILL $COMPILER_PID 2>/dev/null
                fi
            fi
        done
    else
        # Kill any python processes with compile server keywords
        pkill -f "src.kernelblaster.servers.compile" 2>/dev/null || true
    fi
    
    # Kill GPU server
    if [ -n "$GPU_SERVER_PID" ] && kill -0 $GPU_SERVER_PID 2>/dev/null; then
        echo "Terminating GPU server (PID $GPU_SERVER_PID)..."
        kill -TERM $GPU_SERVER_PID 2>/dev/null
        sleep 1
        if kill -0 $GPU_SERVER_PID 2>/dev/null; then
            kill -KILL $GPU_SERVER_PID 2>/dev/null
        fi
    fi
    
    # Kill GPU starter process (if we started one)
    if [ -n "$GPU_STARTER_PID" ] && kill -0 $GPU_STARTER_PID 2>/dev/null; then
        echo "Terminating GPU starter (PID $GPU_STARTER_PID)..."
        kill -TERM $GPU_STARTER_PID 2>/dev/null
        sleep 1  
        if kill -0 $GPU_STARTER_PID 2>/dev/null; then
            kill -KILL $GPU_STARTER_PID 2>/dev/null
        fi
    fi
    
    # Clean up temporary files
    rm -f gpu_server_info.txt
}

cleanup_all() {
    echo ""
    echo "=========================================="
    echo "Interrupt received! Cleaning up..."
    echo "=========================================="
    
    cleanup_processes
    
    echo "Cleanup completed. Exiting..."
    exit 130  # Standard exit code for Ctrl+C
}

# Set up signal trap for SIGINT (Ctrl+C) and SIGTERM
trap cleanup_all SIGINT SIGTERM

# Run RL optimization
RL_EXPERIMENT_NAME="${RL_EXPERIMENT_NAME:-kernelblaster}"

python scripts/run_RL.py \
  --experiment-name "$RL_EXPERIMENT_NAME" \
  --dataset ${DATASET} \
  --precision fp16 \
  --cuda \
  --cuda-perf \
  --use-rl \
  --rl-iterations 10 \
  --rl-rollout-steps 10 \
  --rl-buffer-size 100 \
  --rl-update-frequency 3 \
  --concurrency 1 \
  --problem-numbers "${PROBLEM_NUMBERS}" \
  --subset "${SUBSET}" \
  --timeout 480 \
  --compiler-port 2011 \
  --gpu-server-url "$GPU_SERVER_URL"


EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "run_RL.py exited with status $EXIT_CODE but continuing execution (non-fatal)."
fi

echo "run_RL.py finished; proceeding to cleanup."

# Cleanup: Stop the GPU server
echo ""
echo "Cleaning up GPU server..."
cleanup_processes

# Exit with appropriate code
if [ $EXIT_CODE -eq 0 ]; then
    exit 0
else
    exit 1
fi 
