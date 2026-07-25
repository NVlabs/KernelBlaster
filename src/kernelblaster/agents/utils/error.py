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

"""定义优化反馈流程使用的领域异常。"""

class FeedbackError(Exception):
    """表示该领域内可被调用方识别和处理的失败。"""
    def __init__(self, feedback: str, logging_message: str = None):
        """
        初始化 FeedbackError 实例，并保存后续流程所需的配置与依赖。

        参数:
            feedback: 调用方提供的 `feedback` 参数。
            logging_message: 调用方提供的 `logging_message` 参数。
        """
        self.feedback = feedback
        if logging_message is None:
            logging_message = self.feedback
        self.logging_message = logging_message
