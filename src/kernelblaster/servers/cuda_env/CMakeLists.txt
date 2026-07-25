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
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda)

enable_language(CUDA)

# 在 Docker 外运行时，链接器有时会移除看似未使用但运行期需要的库。
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt IMPORTED INTERFACE)
endif()

# 查找 Torch 前暂时降低 CMake 日志级别，避免输出无关警告。
set(CMAKE_MESSAGE_LOG_LEVEL "ERROR")

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 清除 PyTorch 自动注入的 CUDA 架构标志。
string(REGEX REPLACE "-gencode[= ]arch=compute_[0-9]+,code=sm_[0-9]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REGEX REPLACE "-gencode[= ]arch=compute_[0-9]+,code=compute_[0-9]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# 仅使用调用方通过命令行指定的目标架构。
set(GPU_ARCH_VERSION "" CACHE STRING "GPU architecture version to compile for")
if(NOT GPU_ARCH_VERSION)
    message(FATAL_ERROR "Please specify GPU_ARCH_VERSION: -DGPU_ARCH_VERSION=89")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${GPU_ARCH_VERSION},code=sm_${GPU_ARCH_VERSION}")

# 找到 Torch 后恢复调用方要求的日志级别。
set(CMAKE_MESSAGE_LOG_LEVEL "STATUS") 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

# 定义统一的可执行文件创建函数，集中设置编译和链接参数。
function(add_cuda_executable TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
    # -lineinfo 供 NCU 关联源码行；它可能增加运行开销，不应用于正式基准测量。
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_include_directories(${TARGET_NAME} PRIVATE
        "${Python3_INCLUDE_DIRS}"
    )
    # 链接 Driver 和候选 Kernel 运行所需的基础库。
    target_link_libraries(
        ${TARGET_NAME}
        "${TORCH_LIBRARIES}"
        ${Python3_LIBRARIES}
        CUDA::cublas
    )
endfunction()

# 调用上述函数创建本次请求的可执行文件。
add_cuda_executable(main main.cpp cuda_model.cu)
