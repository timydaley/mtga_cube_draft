"""Small torch helpers shared across training scripts."""

from __future__ import annotations

import torch


def select_device(requested: str) -> torch.device:
    """Resolve the training device.

    `cpu`  : force CPU — use for local dev on a machine without a GPU.
    `cuda` / `auto` : require CUDA. A silent CPU fall-back on a GPU pod means
        the run is orders of magnitude slower AND the GPU plumbing went
        unvalidated, so we error out instead. Pass `cpu` to opt into CPU.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is not available to torch (torch {torch.__version__}, "
            f"built for CUDA {torch.version.cuda}). Common cause: the installed "
            "torch wheel targets a newer CUDA than the GPU driver supports — "
            "reinstall a matching build, e.g.\n"
            "    uv pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "To run on CPU intentionally, pass --device cpu."
        )
    return torch.device("cuda")
