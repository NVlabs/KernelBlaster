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
set -e

# Default values if not provided
USER_NAME=${USER_NAME:-root}
USER_ID=${USER_ID:-0}
GROUP_ID=${GROUP_ID:-0}
HOME_DIR="/home/$USER_NAME"
RUN_AS_ROOT=${RUN_AS_ROOT:-false}  # New flag to control execution user

# Create group if it doesn't exist
# Use group name same as username, or create a group with the specified GID
GROUP_NAME="${GROUP_NAME:-$USER_NAME}"
if ! getent group "$GROUP_ID" > /dev/null 2>&1 && ! getent group "$GROUP_NAME" > /dev/null 2>&1; then
    groupadd -g "$GROUP_ID" "$GROUP_NAME" 2>/dev/null || \
    groupadd -g "$GROUP_ID" "g${GROUP_ID}" 2>/dev/null || true
fi

# Get the actual group name for the GID (in case it already existed)
ACTUAL_GROUP=$(getent group "$GROUP_ID" | cut -d: -f1)
if [ -z "$ACTUAL_GROUP" ]; then
    ACTUAL_GROUP="$GROUP_NAME"
fi

# Create user if it doesn't exist
if ! id -u "$USER_NAME" > /dev/null 2>&1; then
    useradd -u "$USER_ID" -g "$ACTUAL_GROUP" -m "$USER_NAME" -s /bin/bash
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

echo "User: ${USER_NAME} with uid ${USER_ID} and gid ${GROUP_ID}."
echo "Run as root: ${RUN_AS_ROOT}"
echo " - To switch to a different user, call docker run with the following environment variables:"
echo "   - USER_NAME=\$(whoami)"
echo "   - USER_ID=\$(id -u)"
echo "   - GROUP_ID=\$(id -g)"
echo "   - RUN_AS_ROOT=true (to run as root instead of user)"

# Ensure home dir ownership (if re-used volumes are mounted)
chown "$USER_ID":"$GROUP_ID" "$HOME_DIR"

# Ensure /kernelblaster directory has proper permissions (if mounted volume)
# This is critical for volume mounts - fix permissions aggressively
if [ -d "/kernelblaster" ]; then
    # First, ensure root can access it
    chmod 755 /kernelblaster 2>/dev/null || true
    # Then fix ownership recursively
    chown -R "$USER_ID":"$GROUP_ID" /kernelblaster 2>/dev/null || true
    # Set permissions for user
    chmod -R u+rwX,go+rX /kernelblaster 2>/dev/null || true
    # Ensure the directory itself is accessible
    chmod 755 /kernelblaster 2>/dev/null || true
fi

# Switch to user with proper environment
export HOME="$HOME_DIR"
cd "$HOME_DIR"

# Function to execute command with appropriate user
execute_command() {
    if [[ "$RUN_AS_ROOT" == "true" ]]; then
        echo "Executing as root..."
        exec sudo -E sh -c "cd /kernelblaster && exec \"\$@\"" -- "$@"
    else
        echo "Executing as user: $USER_NAME"
        exec sudo -E -u "$USER_NAME" sh -c "cd /kernelblaster && exec \"\$@\"" -- "$@"
    fi
}


# Only chown .ssh if it exists
if [ -d "$HOME_DIR/.ssh" ]; then
    sudo chown -R "$USER_NAME":"$GROUP_ID" "$HOME_DIR/.ssh" && \
    sudo chmod 700 "$HOME_DIR/.ssh" && \
    sudo chmod -R go-rwx "$HOME_DIR/.ssh"
fi

# Create sshfs mount point (optional, disabled by default)
# Set ENABLE_SSHFS=true to enable
if [[ "${ENABLE_SSHFS:-false}" == "true" ]]; then
    sudo -u "$USER_NAME" mkdir -p "$HOME_DIR/sshfs_mnt" 2>/dev/null || true
    # ensure ownership if directory pre-existed
    sudo chown "$USER_NAME":"$GROUP_ID" "$HOME_DIR/sshfs_mnt" 2>/dev/null || true
    # Try to mount sshfs (non-fatal if it fails - this is optional)
    sudo -u "$USER_NAME" sshfs -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER_NAME"@avs:/data/"$USER_NAME" "$HOME_DIR/sshfs_mnt" 2>/dev/null || \
        echo "Note: sshfs mount failed (optional feature)"
fi


# Ensure we can access /kernelblaster before trying to cd
if [ ! -d "/kernelblaster" ]; then
    mkdir -p /kernelblaster
    chown "$USER_ID":"$GROUP_ID" /kernelblaster
    chmod 755 /kernelblaster
fi

# Verify we can access the directory
if [ ! -r "/kernelblaster" ] || [ ! -x "/kernelblaster" ]; then
    echo "Warning: /kernelblaster is not accessible, attempting to fix permissions..."
    chmod 755 /kernelblaster 2>/dev/null || true
    chown "$USER_ID":"$GROUP_ID" /kernelblaster 2>/dev/null || true
fi

# Handle different commands
case "${1:-api}" in
    "dev")
        echo "Starting development environment..."
        execute_command bash
        ;;
    "gpu")
        echo "Starting GPU server..."
        python /kernelblaster/scripts/check_runtime_versions.py --require-gpu
        execute_command python -m src.kernelblaster.servers.gpu --port 2002
        ;;
    "api")
        echo "Starting API server..."
        execute_command python -m src.kernelblaster.servers.serve_api --port 8000 --output-dir out/server/ "${@:2}"
        ;;
    *)
        echo "Running custom command: $@"
        execute_command "$@"
        ;;
esac
