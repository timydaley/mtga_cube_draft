"""Checkpoint save/load with optional Cloudflare R2 mirroring.

Per docs/CLOUD_TRAINING.md, spot pods can die at any time and their disk is
wiped — only R2 survives. So checkpoints are written locally and, when an R2
prefix is configured, pushed to R2 so a fresh pod can resume.

A `latest.pt` pointer is maintained alongside the step-numbered files so
`--resume latest` always finds the newest checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from cube_draft.utils import r2

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: dict[str, Any],
    local_dir: Path,
    step: int,
    r2_prefix: str | None = None,
) -> Path:
    """Write `state` to `local_dir/step_{step}.pt` (+ latest.pt), mirror to R2.

    Returns the local path of the step checkpoint.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    step_path = local_dir / f"step_{step:08d}.pt"
    latest_path = local_dir / "latest.pt"
    torch.save(state, step_path)
    torch.save(state, latest_path)

    if r2_prefix:
        prefix = r2_prefix.rstrip("/")
        client = r2.make_client()
        r2.push(step_path, f"{prefix}/{step_path.name}", client=client)
        r2.push(latest_path, f"{prefix}/latest.pt", client=client)
        logger.info("mirrored checkpoint to r2://%s/%s/", r2.bucket_name(), prefix)

    return step_path


def load_checkpoint(source: str, local_dir: Path, device: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint from a local path or an R2 key.

    `source` resolution order:
      1. an existing local file path
      2. `r2://<key>` or a bare R2 key (pulled into `local_dir`)
      3. the literal string "latest" -> `<local_dir>/latest.pt`
    """
    if source == "latest":
        path = local_dir / "latest.pt"
    elif Path(source).exists():
        path = Path(source)
    else:
        key = source.removeprefix("r2://")
        path = local_dir / Path(key).name
        r2.pull(key, path)

    logger.info("loading checkpoint %s", path)
    return torch.load(path, map_location=device, weights_only=False)
