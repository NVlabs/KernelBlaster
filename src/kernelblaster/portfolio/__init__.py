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

"""导出可复现 Benchmark Portfolio 的能力清单与 Suite 加载接口。"""

from .capabilities import (
    CAPABILITY_MARKER,
    CapabilityResult,
    canonical_shape,
    describe_capabilities,
    load_capability_manifest,
    parse_shape,
    task_map,
    validate_candidate_request,
)
from .suite import PortfolioSuite, PortfolioTask, load_suite, resolve_suite_path

__all__ = [
    "CAPABILITY_MARKER",
    "CapabilityResult",
    "PortfolioSuite",
    "PortfolioTask",
    "canonical_shape",
    "describe_capabilities",
    "load_capability_manifest",
    "load_suite",
    "parse_shape",
    "resolve_suite_path",
    "task_map",
    "validate_candidate_request",
]
