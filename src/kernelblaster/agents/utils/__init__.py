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

"""汇总优化 Agent 共用的命令执行、解析、计时和 LLM 辅助工具。"""

from .error import FeedbackError
from .file_operations import write_code_to_file, write_jsonl, read_jsonl
from .query import *
from .timer import *
from .commands import *
from .parsing import *
from .annotate_ncu import *

# 默认为已检测的 LLM 包装器，因此令牌使用情况 + 计时会记录在 run.log 中。
# 这是正常实现的轻量级包装；它仍然兼容
# 与现有的调用站点一起，并且成为聚合的无操作，除非计时收集器
# 已安装（请参阅 timing_patches.py）。
try:
    from .query_instrumented import generate_code_retry_instrumented as generate_code_retry
except Exception:
    # 如果无法导入仪器，则退回到标准实施。
    pass
