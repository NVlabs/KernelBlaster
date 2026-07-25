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

"""生成 Uvicorn 与应用日志共用的结构化日志配置。"""

import uvicorn.config


def get_log_config(log_filepath: str = None):
    """
    获取 `get_log_config` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        log_filepath: 调用方提供的 `log_filepath` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    base_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_format = "%(asctime)s | %(levelprefix)s | %(message)s"
    base_config["formatters"]["default"]["fmt"] = log_format
    if log_filepath is not None:
        base_config["handlers"]["file"] = {
            "formatter": "file",  # 使用没有颜色的自定义格式化程序
            "class": "logging.FileHandler",
            "filename": log_filepath,
            "mode": "a",
        }
        # 添加一个不带颜色的格式化程序用于文件记录
        base_config["formatters"]["file"] = {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": log_format,
            "use_colors": False,  # 明确禁用颜色
        }
        base_config["loggers"]["uvicorn"]["handlers"].append("file")
    return base_config
