#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Compose control/worker authentication boundary."""

from __future__ import annotations

import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def expect_status(url: str, token: str, expected: int) -> None:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=5) as response:
            actual = response.status
    except HTTPError as error:
        actual = error.code
    if actual != expected:
        raise SystemExit(f"{url} returned {actual}; expected {expected}")


def main() -> int:
    control_token = os.environ["KERNELBLASTER_CONTROL_TOKEN"]
    worker_token = os.environ["KERNELBLASTER_WORKER_TOKEN"]
    supervisor_token = os.environ["KERNELBLASTER_SUPERVISOR_TOKEN"]
    expect_status("http://control:8000/health", control_token, 200)
    expect_status("http://control:8000/health", worker_token, 401)
    expect_status("http://gpu-supervisor:2002/ready", supervisor_token, 200)
    expect_status("http://gpu-supervisor:2002/ready", worker_token, 401)
    expect_status("http://gpu-supervisor:2002/ready", control_token, 401)
    print("Compose control/worker/supervisor token audiences are isolated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
