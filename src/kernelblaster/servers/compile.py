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

"""实现隔离的 CUDA 编译服务、编译队列、环境池和请求生命周期。"""

import argparse
import asyncio
from contextlib import asynccontextmanager
import os
from fastapi import Depends, FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
import logging
import shutil
import tempfile
import uuid
import uvicorn
import uvicorn.config
import re
import sysconfig
from torch.utils import cmake_prefix_path

from .server_logging import get_log_config
from .utils.process_management import safe_kill_process
from .auth import require_worker_token
from .security import allowed_source_path
from ..agents.utils import find_kernel_launch_header

logger = logging.getLogger("uvicorn")

# 修复 cmake_prefix_path 格式，以便正确对可能包含空格的路径使用引号
CMAKE_PREFIX_PATH = f'"{cmake_prefix_path};{sysconfig.get_path("include")}"'
QUEUE = asyncio.Queue()
CUDA_ENV_PATH = Path(__file__).parent / "cuda_env"
ENV_VARS = os.environ.copy()
# 模块级变量（将在服务器启动期间设置）
_ARTIFACTS_DIR = None


def _allowed_source(path: str) -> Path:
    """
    处理 `allowed_source` 所表示的内部步骤；该函数不属于稳定的公开接口。

    参数:
        path: 待读取、写入或校验的文件系统路径。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return allowed_source_path(path)


def get_cmake_prefix_path() -> str:
    """
    获取CMAKE_PREFIX_PATH进行编译

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return f'"{cmake_prefix_path};{sysconfig.get_path("include")}"'


def get_cuda_env_template_path() -> Path:
    """
    获取cuda_env模板目录的路径

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return Path(__file__).parent / "cuda_env"


def extract_arch_version(sm_version: str) -> str:
    """
    从 SM 版本字符串中提取架构版本

    参数:
        sm_version: 调用方提供的 `sm_version` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        ValueError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    match = re.fullmatch(r"sm_(\d{2,3})", sm_version)
    if match is None:
        raise ValueError(f"Invalid sm version format: {sm_version}")
    arch_version = match.group(1)
    if int(arch_version) < 50:
        raise ValueError(f"Invalid sm version: {sm_version}")
    return arch_version


def write_compilation_files(
    work_dir: Path, main_file_path: str, cuda_file_path: str | None
) -> tuple[Path, Path, Path]:
    """
    拆分编译文件并将其写入工作目录。

    返回：
    (main_fp_out、header_fp_out、cuda_fp_out) 路径的元组

    参数:
        work_dir: 调用方提供的 `work_dir` 参数。
        main_file_path: 调用方提供的 `main_file_path` 参数。
        cuda_file_path: 调用方提供的 `cuda_file_path` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    main_fp_out = work_dir / "main.cpp"
    header_fp_out = work_dir / "cuda_model.cuh"
    cuda_fp_out = work_dir / "cuda_model.cu"

    main_file_text, header_file_text, cuda_file_text = split_files_for_compilation(
        main_file_path, cuda_file_path
    )

    main_fp_out.write_text(main_file_text)
    header_fp_out.write_text(header_file_text)
    cuda_fp_out.write_text(cuda_file_text)

    return main_fp_out, header_fp_out, cuda_fp_out


def build_cmake_command(
    sm_build_dir: Path,
    arch_version: str,
    build_type: str = "Release",
) -> list[str]:
    """
    构建cmake配置命令

    参数:
        sm_build_dir: 调用方提供的 `sm_build_dir` 参数。
        arch_version: 调用方提供的 `arch_version` 参数。
        build_type: 调用方提供的 `build_type` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return [
        "cmake",
        f"-DCMAKE_PREFIX_PATH={cmake_prefix_path};{sysconfig.get_path('include')}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DGPU_ARCH_VERSION={arch_version}",
        "..",
    ]


# 在后台启动工作任务
@asynccontextmanager
async def lifespan(app):
    """
    处理 `lifespan` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        app: 调用方提供的 `app` 参数。
    """
    logger.info(
        f"Started compilation server on {args.host}:{args.port} with {args.num_workers} workers"
    )
    # 启动时启动工作任务
    _ = asyncio.create_task(start_workers(args.num_workers, args.compile_debug))
    yield
    free_cuda_envs()


APP = FastAPI(lifespan=lifespan)


def get_cuda_env_root(thread_id: int) -> Path:
    """
    获取 `get_cuda_env_root` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        thread_id: 调用方提供的 `thread_id` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    path = ENV_DIR / f"cuda_eval_{thread_id}"
    if not path.exists():
        setup_cuda_envs(path)
    assert path.exists()
    return path


def get_persistent_root(unique_name: str) -> Path:
    # 根据输出文件名创建唯一目录，避免其中的产物被后续编译覆盖。
    """
    获取 `get_persistent_root` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        unique_name: 调用方提供的 `unique_name` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    persistent_artifacts_dir = ENV_DIR / "persistent" / unique_name
    assert (
        not persistent_artifacts_dir.exists()
    ), f"Persistent artifacts directory {persistent_artifacts_dir} already exists"
    setup_cuda_envs(persistent_artifacts_dir)
    return persistent_artifacts_dir


def setup_cuda_envs(directory: Path):
    """
    处理 `setup_cuda_envs` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        directory: 调用方提供的 `directory` 参数。
    """
    shutil.copytree(CUDA_ENV_PATH, directory)
    logger.info(f"Set up CUDA environment at {directory}")


def free_cuda_envs():
    """释放 `free_cuda_envs` 对应的领域操作，并返回调用方所需的标准化结果。"""
    if ENV_DIR.exists():
        shutil.rmtree(ENV_DIR)
    logger.info("Cleaned up CUDA environment")


def get_all_includes(main_file_text: str) -> list[str]:
    # 获取尖括号包围的包含内容
    """
    获取 `get_all_includes` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        main_file_text: 调用方提供的 `main_file_text` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    system_includes = re.findall(r"(#include\s+<[^>]+>)", main_file_text)
    # 获取用引号括起来的包含内容
    user_includes = re.findall(r'(#include\s+"[^"]+")', main_file_text)
    return system_includes + user_includes


def split_files_for_compilation(
    main_file_path: str, cuda_file_path: str | None
) -> tuple[str, str, str]:
    """
    该方法解析驱动文件和cuda文件，并将其分成两个可编译单元。

    参数:
        main_file_path: 调用方提供的 `main_file_path` 参数。
        cuda_file_path: 调用方提供的 `cuda_file_path` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        CompilationError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    main_file_text = Path(main_file_path).read_text()
    header_file_text = ""
    cuda_file_text = ""

    if cuda_file_path:
        cuda_file_text = Path(cuda_file_path).read_text()
        # 如果我们正在编译cuda内核，我们必须构建一个单独的可编译单元
        # 从测试文件中解析标头并移动到标头文件
        try:
            header_decl = find_kernel_launch_header(main_file_text)
        except Exception as e:
            logger.error(
                f"Failed to find kernel launch header in {main_file_path}: {e}"
            )
            raise CompilationError(
                f"Failed to find kernel launch header in {main_file_path}: {e}"
            )
        main_file_text = main_file_text.replace(header_decl, "")

        # 添加 cstdint 以防使用固定宽度整数类型（如 int64_t）
        # 如果参数的类型为 torch::Tensor 或 c10::ScalarType，则添加 torch/torch.h
        header_file_text = (
            "#include <cstdint>\n#include <torch/torch.h>\n" + header_decl + "\n"
        )

        # 将头文件添加到主文件和 CUDA 文件中
        main_file_text = f'#include "cuda_model.cuh"\n{main_file_text}'
        cuda_file_text = f'#include "cuda_model.cuh"\n{cuda_file_text}'

        # 删除“inline”和“extern“C””，因为链接器将失败
        cuda_file_text = cuda_file_text.replace(
            "inline void launch_gpu_implementation", "void launch_gpu_implementation"
        ).replace('extern "C"', "")

    return main_file_text, header_file_text, cuda_file_text


class CompilationRequest(BaseModel):
    """描述服务执行一次操作所需的输入字段。"""
    job_name: str
    main_file: str
    cuda_file: str
    sm_version: str

    # 该标志允许编译服务将 CUDA 源码产物保存在唯一目录中，
    # 仅在关闭时修改。
    # 当后续命令需要引用原始 CUDA 源代码 e.g 时，这非常有用。 NCU 注释。
    # 此外，REST API 不支持布尔标志。
    persistent_artifacts: int = 0


class CompilationResult(BaseModel):
    """保存一次操作的标准化结果及其诊断信息。"""
    job_name: str
    main_file: str
    cuda_file: str
    success: bool = False
    message: str = None
    output_path: str = None
    persistent_artifacts_dir: str = None


class CompilationError(Exception):
    """表示该领域内可被调用方识别和处理的失败。"""
    def __init__(self, message: str):
        """
        初始化 CompilationError 实例，并保存后续流程所需的配置与依赖。

        参数:
            message: 调用方提供的 `message` 参数。
        """
        self.message = message
        super().__init__(self.message)


def complete_compilation_future(
    completion_future: asyncio.Future,
    *,
    result: bool | None = None,
    error: Exception | None = None,
) -> None:
    """
    完成 `complete_compilation_future` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        completion_future: 调用方提供的 `completion_future` 参数。
        result: 上一步产生并等待进一步处理的结果。
        error: 调用方提供的 `error` 参数。
    """
    if completion_future.done():
        return
    if error is not None:
        completion_future.set_exception(error)
    else:
        completion_future.set_result(result)


async def exec_compilation(
    job_name: str,
    main_file: str,
    cuda_file: str,
    sm_version: str,
    worker_id: int,
    output_path: Path,
    persistent_artifacts: bool,
    debug=False,
    timeout=360,
):
    """
    该函数用于编译CUDA程序。
    它将 GPU 代码声明从测试文件中分离出来并移动到它自己的头文件中。
    它将从 cuda 文件中删除标头，并向标头文件添加一条包含语句。

    结构如下：
    - main.cpp（从 <main_file> 复制，但删除了内核启动标头）
    - cuda_model.cuh（仅提取的内核启动标头）
    - cuda_model.cu（从 <cuda_file> 复制，并在顶部附加#include“cuda_model.cuh”）

    参数:
        job_name: 调用方提供的 `job_name` 参数。
        main_file: 调用方提供的 `main_file` 参数。
        cuda_file: 调用方提供的 `cuda_file` 参数。
        sm_version: 调用方提供的 `sm_version` 参数。
        worker_id: 调用方提供的 `worker_id` 参数。
        output_path: 调用方提供的 `output_path` 参数。
        persistent_artifacts: 调用方提供的 `persistent_artifacts` 参数。
        debug: 调用方提供的 `debug` 参数。
        timeout: 允许操作等待的最长秒数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        CompilationError: 输入、外部调用或状态不满足执行要求时抛出。
    """
    assert not debug, "Debug compilation is not supported"

    if persistent_artifacts:
        work_dir = get_persistent_root(output_path.name)
    else:
        # 使用标准工人环境
        work_dir = get_cuda_env_root(worker_id)

    main_fp_out, header_fp_out, cuda_fp_out = write_compilation_files(
        work_dir, main_file, cuda_file
    )

    arch_version = extract_arch_version(sm_version)

    # 这个调用是昂贵的，所以只有当 sm 版本不同时才重新生成
    sm_build_dir = work_dir / f"build_{sm_version}"
    if not sm_build_dir.exists():
        build_type = "Release"
        sm_build_dir.mkdir(parents=True, exist_ok=False)
        cmd = build_cmake_command(
            sm_build_dir,
            arch_version,
            build_type,
        )
        logger.info(f"Running cmake command: {cmd[0]}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=sm_build_dir,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            raise CompilationError(
                f"Failed to run cmake config for {work_dir}: stderr:\n{stderr.decode()}"
            )

    proc = await asyncio.create_subprocess_exec(
        "make",
        "-j8",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=sm_build_dir,
        start_new_session=True,
        env=ENV_VARS,
    )
    try:
        # 等待进程超时
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            raise CompilationError(
                f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
            )
    except asyncio.TimeoutError:
        # 如果超时则终止该进程
        await safe_kill_process(proc, logger)
        raise CompilationError(
            f"Timeout: Compilation process timed out after {timeout} seconds"
        )

    return sm_build_dir / "main"


async def compilation_worker(worker_id: int, debug: bool = False):
    """
    处理编译队列中的文件

    参数:
        worker_id: 调用方提供的 `worker_id` 参数。
        debug: 调用方提供的 `debug` 参数。
    """
    while True:
        (
            job_name,
            main_file,
            cuda_file,
            sm_version,
            persistent_artifacts,
            output_path,
            completion_future,
        ) = await QUEUE.get()
        try:
            logger.info(f"[Worker {worker_id}]: Compiling {job_name}")
            if persistent_artifacts:
                logger.info(
                    f"[Worker {worker_id}]: Using persistent_artifacts mode for {job_name}"
                )

            tmp_path = await exec_compilation(
                job_name,
                main_file,
                cuda_file,
                sm_version,
                worker_id,
                output_path,
                persistent_artifacts,
                debug=debug,
            )
            output_path.write_bytes(tmp_path.read_bytes())
            output_path.chmod(0o755)  # 使该文件可执行
            logger.info(
                f"[Worker {worker_id}]: Successfully compiled {job_name} and saved to {output_path}"
            )
            complete_compilation_future(completion_future, result=True)
        except CompilationError as e:
            logger.info(f"[Worker {worker_id}]: Error compiling {job_name}")
            complete_compilation_future(completion_future, error=e)
        except FileNotFoundError as e:
            logger.error(f"[Worker {worker_id}]: File not found: {e}")
            complete_compilation_future(completion_future, error=e)
        except Exception as e:
            error_msg = f"[Worker {worker_id}]: Unhandled exception compiling {job_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # 包装 CompilationError 以便正确处理
            complete_compilation_future(
                completion_future,
                error=CompilationError(error_msg),
            )
            # 不要重新加注——让工人继续处理其他工作
        finally:
            QUEUE.task_done()


@APP.get("/health")
async def health_check():
    """
    健康检查端点

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return {"status": "healthy", "service": "compile-server"}


@APP.get("/compile", response_model=CompilationResult)
async def process_compilation_request(
    job_name: str,
    main_file: str,
    cuda_file: str,
    sm_version: str,
    persistent_artifacts: int = 0,
    _authorized: None = Depends(require_worker_token),
):
    """
    处理 `process_compilation_request` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        job_name: 调用方提供的 `job_name` 参数。
        main_file: 调用方提供的 `main_file` 参数。
        cuda_file: 调用方提供的 `cuda_file` 参数。
        sm_version: 调用方提供的 `sm_version` 参数。
        persistent_artifacts: 调用方提供的 `persistent_artifacts` 参数。
        _authorized: 调用方提供的 `_authorized` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        HTTPException: 输入、外部调用或状态不满足执行要求时抛出。
    """
    logger.info(f"/compile request received: job_name={job_name}, main_file={main_file}, cuda_file={cuda_file}, sm_version={sm_version}, backlog: {QUEUE.qsize()}")
    
    try:
        main_path = _allowed_source(main_file)
        cuda_path = _allowed_source(cuda_file) if cuda_file else None

        if not main_path.exists():
            error_msg = f"File {main_file} not found"
            logger.error(f"/compile error: {error_msg}")
            return CompilationResult(
                job_name=job_name,
                main_file=main_file,
                cuda_file=cuda_file,
                success=False,
                message=error_msg,
            )

        if cuda_path is not None and not cuda_path.exists():
            error_msg = f"File {cuda_file} not found"
            logger.error(f"/compile error: {error_msg}")
            return CompilationResult(
                job_name=job_name,
                main_file=main_file,
                cuda_file=cuda_file,
                success=False,
                message=error_msg,
            )

        # 创建 Future，使请求协程可以等待编译 Worker 完成。
        completion_future = asyncio.Future()

        # 计算将使用的输出路径
        with tempfile.NamedTemporaryFile(delete=False, dir=OUT_DIR) as f:
            output_path = Path(f.name)

        logger.info(f"Queueing compilation: {job_name} -> {output_path}")

        # 创建一个带有 future 的特殊队列项
        await QUEUE.put(
            (
                job_name,
                str(main_path),
                str(cuda_path) if cuda_path is not None else "",
                sm_version,
                bool(persistent_artifacts),
                output_path,
                completion_future,
            )
        )

        # 等待编译完成
        try:
            await completion_future

            result = CompilationResult(
                job_name=job_name,
                main_file=main_file,
                cuda_file=cuda_file,
                success=True,
                message="Compilation successful",
                output_path=str(output_path),
            )

            if persistent_artifacts:
                persistent_artifacts_dir = (
                    args.artifacts_dir / "persistent_artifacts" / output_path.name
                )
                result.persistent_artifacts_dir = str(persistent_artifacts_dir)

            logger.info(f"/compile success: {job_name} -> {output_path}")
            return result
        except CompilationError as e:
            error_msg = str(e)
            logger.error(f"/compile CompilationError for {job_name}: {error_msg}")
            result = CompilationResult(
                job_name=job_name,
                main_file=main_file,
                cuda_file=cuda_file,
                success=False,
                message=error_msg,
            )

            if persistent_artifacts:
                persistent_artifacts_dir = (
                    args.artifacts_dir / "persistent_artifacts" / output_path.name
                )
                result.persistent_artifacts_dir = str(persistent_artifacts_dir)

            return result
        except asyncio.CancelledError:
            logger.warning(f"Compilation {job_name} was cancelled")
            raise HTTPException(status_code=500, detail="Compilation was cancelled")
        except Exception as e:
            error_msg = f"Unexpected error during compilation: {str(e)}"
            logger.error(f"/compile unexpected error for {job_name}: {error_msg}", exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing compilation request: {str(e)}"
        logger.error(f"/compile request processing error for {job_name}: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


async def start_workers(num_workers: int, debug: bool = False):
    """
    启动编译工作任务

    参数:
        num_workers: 调用方提供的 `num_workers` 参数。
        debug: 调用方提供的 `debug` 参数。
    """
    workers = [
        asyncio.create_task(compilation_worker(worker_id, debug))
        for worker_id in range(num_workers)
    ]
    try:
        await asyncio.gather(*workers)
    except Exception as e:
        logger.error(f"Worker exception: {e}")
        # 重新引发异常以使服务器崩溃
        raise


def run_compilation_server(host: str, port: int):
    """
    使用 REST API 运行编译服务器

    参数：
    host：要绑定服务器的主机
    port：服务器绑定的端口
    num_workers：并行编译工作者的数量
    debug：是否以调试模式编译

    参数:
        host: 远端服务监听或连接的主机名。
        port: 远端服务监听或连接的端口。
    """

    # 运行 FastAPI 服务器
    log_config = get_log_config()
    uvicorn.run(
        APP, host=host, port=port, log_config=log_config, timeout_graceful_shutdown=0.1
    )


def main():
    # 运行 REST API 编译服务器
    """处理 `main` 对应的领域操作，并返回调用方所需的标准化结果。"""
    run_compilation_server(
        args.host,
        args.port,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2001)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--compile-debug", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Path to directory to store artifacts",
        default=Path("/tmp/kernelblaster"),
    )
    args = parser.parse_args()

    ENV_DIR = args.artifacts_dir / str(uuid.uuid4())
    OUT_DIR = ENV_DIR / "out"
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # 将 artifacts_dir 存储在模块级变量中，以便在端点处理程序中使用
    import src.kernelblaster.servers.compile as compile_module
    compile_module._ARTIFACTS_DIR = args.artifacts_dir

    main()
