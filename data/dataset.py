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
from typing import Iterator, Any
from pathlib import Path
import random
from dataclasses import dataclass

from glob import glob
from typing import Iterator, Any
from pathlib import Path

SEED = 4589


class Dataset:
    """Sample dataset class."""

    def __init__(self, data_dir: str):
        """Initialize the dataset.

        Args:
            data_dir (str): Path to the directory containing the kernelbench dataset
        """
        self.data_dir = Path(data_dir)
        self.data: list[dict[str]] = []

        # train, test split
        self.splits = {"train": 0.5, "test": 0.5}

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str]:
        """Get a sample from the dataset by index."""
        return self.data[idx]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Return an iterator over the dataset."""
        return iter(self.data)

    def format_id(self, id: int, dataset_len: int) -> str:
        assert isinstance(id, int)
        assert isinstance(dataset_len, int)
        return str(id).zfill(len(str(dataset_len)))

    def make_sample(self, id: str, reference_code: str, **kwargs) -> dict[str]:
        assert isinstance(id, str)
        assert isinstance(reference_code, str)
        return {"id": id, "reference_code": reference_code, **kwargs}

    def get_iter(self, split: str | None = None):
        if split is None:
            permutation = list(range(len(self.data)))
        else:
            permutation = self.__get_split_indices(split)
        for i in permutation:
            yield self.data[i]

    def __get_split_indices(self, split: str):
        assert split in self.splits, f"Invalid split: {split}"
        random.seed(SEED)
        permutation = list(range(len(self.data)))
        random.shuffle(permutation)
        split_sizes = {name: int(len(self.data) * frac) for name, frac in self.splits.items()}
        start = 0
        for name in self.splits:
            size = split_sizes[name]
            if name == split:
                return permutation[start : start + size]
            start += size
        return []
