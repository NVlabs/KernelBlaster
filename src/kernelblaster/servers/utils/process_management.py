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

"""以超时和信号升级策略安全终止异步子进程。"""

import os
import signal


async def safe_kill_process(proc, logger=None):
    """
    处理 `safe_kill_process` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        proc: 调用方提供的 `proc` 参数。
        logger: 记录诊断信息和任务进度的日志器。
    """
    if proc.returncode is None:
        forbidden_groups = [
            os.getpgid(0),  # 当前 shell 的组
            os.getpgid(1),  # 初始化/系统组
            0,
            1,  # 初始化/系统组
        ]
        current_pgid = os.getpgid(proc.pid)
        if logger:
            logger.info(
                f"Current PGID: {current_pgid} ; Process PID: {proc.pid} ; Forbidden groups: {forbidden_groups}"
            )

        # 关键安全检查
        if current_pgid not in forbidden_groups:
            if logger:
                logger.info(f"Safe to kill - KILLING PGID: {current_pgid}")
            os.killpg(current_pgid, signal.SIGKILL)
        else:
            if logger:
                logger.warning(f"Refusing to kill protected group {current_pgid}")
            proc.kill()

    await proc.wait()
