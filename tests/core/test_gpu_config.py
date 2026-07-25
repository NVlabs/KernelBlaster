from __future__ import annotations

from unittest.mock import patch

from src.kernelblaster.config import gpu_config
from src.kernelblaster.config.gpu_config import GPUType


def test_current_gpu_uses_fixed_argv_without_a_shell(monkeypatch):
    monkeypatch.setattr(gpu_config, "_current_gpu", None)
    with patch.object(
        gpu_config.subprocess,
        "check_output",
        return_value=b"NVIDIA GeForce RTX 3080\n",
    ) as check_output:
        assert GPUType.current() is GPUType.RTX3080

    check_output.assert_called_once_with(
        [
            "nvidia-smi",
            "--query-gpu=gpu_name",
            "--format=csv,noheader",
        ]
    )
