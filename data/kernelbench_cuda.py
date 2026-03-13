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
from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import Dataset


class KernelBenchCUDADataset(Dataset):
    """
    Dataset over curated CUDA artifacts produced by KernelBlaster runs.

    Expected directory layout:
      data/kernelbench-cuda/level1/<problem_name>/{driver.cpp,init.cu}
    """

    def __init__(
        self,
        level_str: str | None = None,
        problem_numbers: list[int] | None = None,
        start: int | None = None,
        end: int | None = None,
        root_dir: str | Path | None = None,
    ):
        root = Path(root_dir) if root_dir is not None else Path(__file__).parent / "kernelbench-cuda"
        super().__init__(root)
        assert level_str is None or level_str in ["level1", "level2", "level3"], "Invalid level"
        self.level_str = level_str
        self._load_dataset(problem_numbers=problem_numbers, start=start, end=end)

    def _load_dataset(
        self,
        problem_numbers: list[int] | None,
        start: int | None,
        end: int | None,
    ) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.data_dir} not found")

        levels = [self.level_str] if self.level_str else ["level1", "level2", "level3"]
        for level in levels:
            level_dir = self.data_dir / level
            if not level_dir.exists():
                continue

            for problem_dir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
                # problem_dir name expected like "001_Square_matrix_multiplication"
                try:
                    num = int(problem_dir.name.split("_", 1)[0])
                except Exception:
                    continue

                if problem_numbers is not None and num not in problem_numbers:
                    continue
                if start is not None and num < start:
                    continue
                if end is not None and num > end:
                    continue

                driver_cpp = problem_dir / "driver.cpp"
                init_cu = problem_dir / "init.cu"
                if not driver_cpp.exists() or not init_cu.exists():
                    # skip incomplete entries
                    continue

                entry: dict[str, Any] = {
                    "id": f"{level}/{problem_dir.name}",
                    "problem_name": problem_dir.name,
                    "problem_num": num,
                    "level": level,
                    "driver_cpp_fp": str(driver_cpp),
                    "init_cuda_fp": str(init_cu),
                    # Backwards compatibility for older callers that expect this key name.
                    "final_cuda_fp": str(init_cu),
                }
                self.data.append(entry)

        self.data.sort(key=lambda x: x["id"])

