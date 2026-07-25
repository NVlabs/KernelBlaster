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

"""提供异步工作流 API、任务队列、状态持久化、恢复与取消能力。"""

import argparse
import asyncio
import os
import sys
from loguru import logger
from pathlib import Path
import time
import uuid
import traceback
from typing import Optional, List
import json

import uvicorn
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..graph.state import load_state_from_json, compare_states
from ..agents.utils.file_operations import get_agent_status
from ..config import config, WorkflowConfig, GPUType
from ..resources import *
from ..workflow import *
from .auth import require_control_token, validate_token_boundaries


app = FastAPI(title="KernelBlaster API")

cors_origins = [
    value.strip()
    for value in os.getenv("KERNELBLASTER_CORS_ORIGINS", "").split(",")
    if value.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


# 任务追踪
class TaskStatus:
    """封装 `TaskStatus` 对应的领域状态与操作。"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Task:
    """封装 `Task` 对应的领域状态与操作。"""
    def __init__(self, task_id, params, user_id):
        """
        初始化 Task 实例，并保存后续流程所需的配置与依赖。

        参数:
            task_id: 调用方分配的任务唯一标识。
            params: 调用方提供的 `params` 参数。
            user_id: 调用方提供的 `user_id` 参数。

        异常:
            ValueError: 输入、外部调用或状态不满足执行要求时抛出。
        """
        self.task_id = task_id
        self.params = params
        output_root = OUTPUT_DIR.resolve()
        if params.folder:
            requested = Path(params.folder)
            if requested.is_absolute():
                raise ValueError("Task folder must be relative to the output directory")
            folder = output_root / requested
        elif user_id:
            folder = output_root / user_id / task_id
        else:
            folder = output_root / task_id
        self.folder = folder.resolve()
        if not self.folder.is_relative_to(output_root):
            raise ValueError("Task folder escapes the output directory")
        self.status = TaskStatus.PENDING
        self.creation_time = time.time()
        self.start_time = None
        self.end_time = None
        self.result: dict[str, str] = None
        self.task = None
        self.queue_position = None
        self.state = None


# 全局状态
TASKS: dict[str, Task] = {}
TASK_QUEUE: List[str] = []  # 按提交顺序排列的待处理 task_ids 队列
SEMAPHORE = None
OUTPUT_DIR: Path = None


class WorkflowRequest(BaseModel):
    """描述服务执行一次操作所需的输入字段。"""
    user_message: str
    reference_code: str
    folder: Optional[str] = None
    user_id: Optional[str] = None
    model: str
    timeout: Optional[float] = None
    run_cuda: bool = False
    run_cuda_perf: bool = False
    benchmark: bool = False
    retry_failed: bool = False
    gpu: Optional[GPUType] = None


class TaskResponse(BaseModel):
    """表示服务返回给调用方的结构化响应。"""
    task_id: str
    status: str
    queue_position: Optional[int] = None


class TaskStatusResponse(BaseModel):
    """表示服务返回给调用方的结构化响应。"""
    task_id: str
    status: str
    creation_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    folder: str
    queue_position: Optional[int] = None
    state: Optional[dict] = None


class StatusResponse(BaseModel):
    """表示服务返回给调用方的结构化响应。"""
    tasks: dict[str, dict]
    running_count: int
    pending_count: int
    task_queue: List[str]


def get_queue_position(task_id: str) -> Optional[int]:
    """
    获取任务在队列中的位置（0 表示正在运行，1+ 表示待处理）

    参数:
        task_id: 调用方分配的任务唯一标识。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    assert task_id in TASKS, f"Task {task_id} not found"

    task = TASKS[task_id]
    if task.status == TaskStatus.RUNNING:
        return 0
    elif task.status == TaskStatus.PENDING:
        try:
            return TASK_QUEUE.index(task_id) + 1
        except ValueError:
            return None
    else:
        return None


def get_task_status_data(task_id: str) -> dict:
    """
    以适合 API 响应的格式获取任务状态数据。

    参数:
        task_id: 调用方分配的任务唯一标识。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    if task_id not in TASKS:
        # 尝试从文件加载已完成的任务
        task = load_task_status(task_id)
        if not task:
            return {"error": f"Task {task_id} not found"}
    else:
        task = TASKS[task_id]

    queue_position = get_queue_position(task_id) if task_id in TASKS else None

    duration = None
    if task.start_time:
        end = task.end_time if task.end_time else time.time()
        duration = end - task.start_time

    state_path = task.folder / "state.json"
    state = (
        load_state_from_json(state_path, read_fp=True) if state_path.exists() else None
    )
    agents_status = {
        "tests": get_agent_status(task.folder, "tests"),
        "cuda": get_agent_status(task.folder, "kgen"),
        "ncu": get_agent_status(task.folder, "ncu"),
        "ncu_annot": get_agent_status(task.folder, "ncu_annot"),
    }

    return {
        "task_id": task_id,
        "status": task.status,
        "queue_position": queue_position,
        "creation_time": task.creation_time,
        "start_time": task.start_time,
        "end_time": task.end_time,
        "duration": duration,
        "state": state,
        "agents": agents_status,
    }


def save_task_status(task: Task):
    """
    将任务状态保存到任务文件夹中的文件中。

    参数:
        task: 调用方提供的 `task` 参数。
    """
    task_folder = task.folder
    task_folder.mkdir(parents=True, exist_ok=True)

    # 创建任务的可序列化版本
    task_data = {
        "task_id": task.task_id,
        "status": task.status,
        "creation_time": task.creation_time,
        "start_time": task.start_time,
        "end_time": task.end_time,
        "result": task.result,
    }

    # 保存到任务文件夹中的 status.json
    with open(task_folder / "status.json", "w") as f:
        json.dump(task_data, f)


def load_task_status_from_folder(task_folder: Path) -> Optional[Task]:
    """
    从特定文件夹路径加载任务状态。

    参数:
        task_folder: 调用方提供的 `task_folder` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    status_file = task_folder / "status.json"

    if not status_file.exists():
        return None

    try:
        with open(status_file, "r") as f:
            task_data = json.load(f)

        task_id = task_data["task_id"]

        # 尝试从 task_folder 中提取 user_id
        user_id = task_folder.parent.name if task_folder.parent.name else None

        # 从数据创建任务对象
        task = Task(task_id, type("MockParams", (), {"folder": None}), user_id)
        task.folder = task_folder  # 既然我们知道就直接设置文件夹
        task.status = task_data["status"]
        task.creation_time = task_data["creation_time"]
        task.start_time = task_data["start_time"]
        task.end_time = task_data["end_time"]
        task.result = task_data.get("result", None)
        return task
    except Exception as e:
        logger.error(f"Failed to load task status from folder {task_folder}: {e}")
        return None


def restore_previous_tasks(output_dir: Path):
    """
    从输出目录恢复以前的任务。

    参数:
        output_dir: 调用方提供的 `output_dir` 参数。
    """
    if not output_dir.exists():
        logger.info("Output directory does not exist, no previous tasks to restore")
        return

    restored_count = 0

    try:
        for item in output_dir.iterdir():
            if item.is_dir():
                tasks = []
                task = load_task_status_from_folder(item)
                if not task:
                    # 这可能是一个用户名文件夹，我们需要在其中检查任务的子文件夹
                    for child in item.iterdir():
                        if child.is_dir():
                            task = load_task_status_from_folder(child)
                            tasks.append(task)
                else:
                    tasks.append(task)

                for task in tasks:
                    # 对于待处理/正在运行的任务，更新为已取消
                    if task.status not in [
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                        TaskStatus.TIMEOUT,
                    ]:
                        task.status = TaskStatus.CANCELLED
                        task.result = "Task cancelled due to server restart"
                        save_task_status(task)
                        logger.debug(
                            f"Task {task.task_id} cancelled due to server restart"
                        )

                    TASKS[task.task_id] = task
                    restored_count += 1
                    logger.debug(
                        f"Restored task {task.task_id} from {task.folder} with status {task.status}"
                    )
    except Exception as e:
        logger.error(f"Error restoring previous tasks: {e}")

    if restored_count > 0:
        logger.info(f"Restored {restored_count} previous tasks from {output_dir}")
    else:
        logger.info("No previous tasks found to restore")


def load_task_status(task_id: str) -> Optional[Task]:
    """
    从任务文件夹加载任务状态。

    参数:
        task_id: 调用方分配的任务唯一标识。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    if task_id not in TASKS:
        return None

    task_folder = TASKS[task_id].folder
    return load_task_status_from_folder(task_folder)


async def process_task(task_id: str, params: WorkflowRequest):
    """
    执行一个排队工作流任务，并持久化其状态和最终产物。

    参数:
        task_id: 调用方分配的任务唯一标识。
        params: 调用方提供的 `params` 参数。
    """
    task = TASKS[task_id]
    job_logger_id = None
    job_logger = None
    try:
        async with SEMAPHORE:
            # 任务开始运行时从待处理队列中删除
            assert task_id in TASK_QUEUE, f"Task {task_id} not in queue"
            assert task.status == TaskStatus.PENDING, f"Task {task_id} is not pending"
            TASK_QUEUE.remove(task_id)

            task_folder = task.folder
            task_folder.mkdir(parents=True, exist_ok=True)

            logger.info(f"Task {task_id} started running: {task_folder}")

            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()

            # 为作业创建记录器
            job_logger = logger.bind(job_id=task_id)
            job_logger_id = job_logger.add(
                task_folder / "run.log",
                level="DEBUG",
                backtrace=True,
                diagnose=True,
                format=config.CUSTOM_LOGGER_FORMAT,
                # task_id指的是kernelblaster中的并行生成线程
                # 所以这里使用job_id来过滤日志
                filter=lambda record: record["extra"].get("job_id") == task_id,
            )

            workflow_config = WorkflowConfig(
                model=params.model,
                run_cuda=params.run_cuda,
                run_cuda_perf=params.run_cuda_perf,
                run_cuda_bench=params.run_cuda and params.benchmark,
                run_cuda_perf_bench=params.run_cuda_perf and params.benchmark,
                retry_failed=params.retry_failed,
                gpu=params.gpu if params.gpu else GPUType.current(),
            )

            result = await run_workflow(
                task_id,
                params.user_message,
                params.reference_code,
                task_folder,
                workflow_config,
                job_logger,
                params.timeout,
            )

            if result.timeout:
                task.status = TaskStatus.TIMEOUT
                task.result = result.error
                logger.warning(f"Task {task_id} timed out: {task.result}")
            elif result.success:
                task.status = TaskStatus.COMPLETED
                task.result = result.generated_codes
                logger.info(f"Task {task_id} completed successfully: {task.result}")
            else:
                task.status = TaskStatus.FAILED
                task.result = result.outcome.to_dict()
                logger.warning(
                    f"Task {task_id} ended with {result.outcome.status.value}: "
                    f"{result.error}"
                )

    except asyncio.CancelledError:
        logger.info(f"Task {task_id} was cancelled")
        task.status = TaskStatus.CANCELLED
    except Exception as e:
        logger.error(f"Unhandled error in task {task_id}: {e}")
        logger.error(f"Stacktrace for task {task_id}:\n{traceback.format_exc()}")
        task.status = TaskStatus.FAILED
        task.result = str(e)

    finally:
        task.end_time = time.time()
        if task_id in TASK_QUEUE:
            # 如果任务仍保留在队列中，则从队列中删除。
            # 如果在等待信号量时取消 process_task()，则会发生这种情况
            TASK_QUEUE.remove(task_id)
        if job_logger is not None:
            job_logger.remove(job_logger_id)

        # 如果任务处于最终状态，则将状态保存到文件
        if task.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        ]:
            # 将任务状态保存到其文件夹中的文件中
            save_task_status(task)
            logger.info(
                f"Task {task_id} with status {task.status} completed and saved to disk"
            )


async def create_task(task_id: str, params: WorkflowRequest):
    """
    创建 `create_task` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        task_id: 调用方分配的任务唯一标识。
        params: 调用方提供的 `params` 参数。
    """
    TASKS[task_id].task = asyncio.create_task(process_task(task_id, params))


@app.post("/submit", response_model=TaskResponse)
async def submit_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    _authorized: None = Depends(require_control_token),
):
    """
    提交 `submit_workflow` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        request: 经过类型约束的服务请求对象。
        background_tasks: 调用方提供的 `background_tasks` 参数。
        _authorized: 调用方提供的 `_authorized` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        HTTPException: 输入、外部调用或状态不满足执行要求时抛出。
    """
    task_id = str(uuid.uuid4())
    try:
        task = Task(task_id, request, request.user_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    logger.info(f"Task {task_id} created: {task.folder}")
    TASKS[task_id] = task

    # 添加到待处理任务队列
    TASK_QUEUE.append(task_id)

    background_tasks.add_task(create_task, task_id, request)

    task.queue_position = get_queue_position(task_id)
    logger.info(
        f"Task {task_id} status: {task.status}, queue position: {task.queue_position}"
    )

    return TaskResponse(
        task_id=task_id, status=task.status, queue_position=task.queue_position
    )


@app.get("/status/{task_id}")
async def get_task_status(
    task_id: str,
    _authorized: None = Depends(require_control_token),
):
    """
    用于获取任务状态的 HTTP 端点。

    参数:
        task_id: 调用方分配的任务唯一标识。
        _authorized: 调用方提供的 `_authorized` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        HTTPException: 输入、外部调用或状态不满足执行要求时抛出。
    """
    status_data = get_task_status_data(task_id)
    if "error" in status_data:
        raise HTTPException(status_code=404, detail=status_data["error"])
    return status_data


@app.post("/cancel/{task_id}", response_model=TaskResponse)
async def cancel_task(
    task_id: str,
    _authorized: None = Depends(require_control_token),
):
    """
    取消 `cancel_task` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        task_id: 调用方分配的任务唯一标识。
        _authorized: 调用方提供的 `_authorized` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        HTTPException: 输入、外部调用或状态不满足执行要求时抛出。
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = TASKS[task_id]

    if task.status in [TaskStatus.RUNNING, TaskStatus.PENDING]:
        # 任务的取消方法应该处理将其自身从队列中删除
        assert task.task is not None, "Task should be running"
        logger.info(f"Cancelling task {task_id}")
        task.task.cancel()
        await task.task

    task.status = TaskStatus.CANCELLED

    return TaskResponse(task_id=task_id, status=task.status, queue_position=None)


@app.get("/health")
async def health_check():
    """
    运行状况检查端点，用于验证 API 服务器是否正在运行并连接到依赖项。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    try:
        compile_server_status = "ok" if config.COMPILE_SERVER_URL else "not_connected"
        gpu_server_status = "ok" if config.GPU_SERVER_URL else "not_connected"

        return {
            "status": "ok",
            "dependencies": {
                "compile_server": compile_server_status,
                "gpu_server": gpu_server_status,
            },
            "tasks": {
                "running": sum(
                    1 for t in TASKS.values() if t.status == TaskStatus.RUNNING
                ),
                "pending": len(TASK_QUEUE),
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}, 500


def run_server(host, port, output_dir, gpu: Optional[GPUType]):
    # 设置日志记录
    """
    运行 `run_server` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        host: 远端服务监听或连接的主机名。
        port: 远端服务监听或连接的端口。
        output_dir: 调用方提供的 `output_dir` 参数。
        gpu: 执行或分析任务使用的 GPU 配置。
    """
    logger.configure(
        handlers=[
            dict(
                sink=sys.stderr,
                format=config.CUSTOM_LOGGER_FORMAT,
                level=config.LOG_LEVEL,
                colorize=True,
                backtrace=True,
                diagnose=True,
            ),
        ],
        extra=dict(agent_name="server", attempt_id=None, task_id=None),
    )

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 恢复之前的任务
    restore_previous_tasks(output_dir)

    # 设置资源
    try:
        compile_server = CompileServer(logger, output_dir)
        gpu_server = GPUServer(logger, output_dir, gpu=gpu)
        assert VectorDB.initialize(logger)
        compile_server.wait_for_connection()
        gpu_server.wait_for_connection()
        if compile_server.is_managed:
            config.set_compile_server_url(compile_server.url)
        if gpu_server.is_managed:
            config.set_gpu_server_url(GPUType.current(), gpu_server.url)

        config.print_config(logger)
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        sys.exit(1)

    # 启动服务器
    logger.info(
        f"Starting server on {host}:{port} with max concurrency {SEMAPHORE._value}"
    )
    uvicorn.run(app, host=host, port=port)


def main():
    """处理 `main` 对应的领域操作，并返回调用方所需的标准化结果。"""
    validate_token_boundaries()
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--concurrency", type=int, default=16, help="Maximum concurrent workflows"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/kernelblaster"),
        help="Output directory for compile server and GPU server logs",
    )
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir

    assert OUTPUT_DIR is not None, "Output directory must be specified"

    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(args.concurrency)

    # 使用 None 作为 GPU 类型，默认在主机上以其当前 GPU 类型启动 GPU 服务器
    run_server(args.host, args.port, args.output_dir, None)


if __name__ == "__main__":
    main()
