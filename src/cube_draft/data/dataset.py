"""On-disk format for prebuilt distillation datasets.

A dataset is a directory:

    manifest.json                 {teacher, num_cubes, config}
    cube_000_oracle.json          [oracle_id, ...]  (cube card order == index)
    cube_000_train.npz            pack/pool/seen/pack_idx/pick_idx/picked
    cube_000_eval.npz             same keys, held-out
    cube_001_...

Generated offline by scripts/gen_distill_dataset.py (e.g. with CubeCobraBot) and
consumed by scripts/train_stage2.py via --dataset. Mirrored to R2 so a fresh pod
can train without regenerating. The decision arrays are identical in shape to
what the on-the-fly teacher produces, so the trainer is source-agnostic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cube_draft.utils import r2

logger = logging.getLogger(__name__)

_ARRAY_KEYS = ("pack", "pool", "seen", "pack_idx", "pick_idx", "picked")
_LOCAL_CACHE = Path("data/datasets")


@dataclass
class CubeData:
    oracle_ids: list[str]
    train: dict[str, np.ndarray]
    eval: dict[str, np.ndarray]


@dataclass
class Dataset:
    teacher: str
    cubes: list[CubeData]

    def __len__(self) -> int:
        return len(self.cubes)


def _split_path(d: Path, i: int, split: str) -> Path:
    return d / f"cube_{i:03d}_{split}.npz"


def _oracle_path(d: Path, i: int) -> Path:
    return d / f"cube_{i:03d}_oracle.json"


def save_dataset(
    out_dir: Path,
    teacher: str,
    cubes: list[CubeData],
    config: dict | None = None,
    r2_prefix: str | None = None,
) -> Path:
    """Write a dataset directory and optionally mirror it to R2."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps({"teacher": teacher, "num_cubes": len(cubes), "config": config or {}}, indent=2)
    )
    for i, cube in enumerate(cubes):
        _oracle_path(out_dir, i).write_text(json.dumps(cube.oracle_ids))
        np.savez(_split_path(out_dir, i, "train"), **{k: cube.train[k] for k in _ARRAY_KEYS})
        np.savez(_split_path(out_dir, i, "eval"), **{k: cube.eval[k] for k in _ARRAY_KEYS})
    logger.info("wrote dataset (%d cubes) to %s", len(cubes), out_dir)

    if r2_prefix:
        n = r2.push_dir(out_dir, r2_prefix)
        logger.info("mirrored %d dataset files to r2://%s/%s/", n, r2.bucket_name(), r2_prefix.rstrip("/"))
    return out_dir


def load_dataset(source: str) -> Dataset:
    """Load a dataset from a local directory or an R2 prefix.

    If `source` is an existing local directory it's read directly; otherwise it's
    treated as an R2 prefix and pulled into data/datasets/<name>/ first.
    """
    path = Path(source)
    if not path.is_dir():
        prefix = source.removeprefix("r2://").rstrip("/")
        path = _LOCAL_CACHE / Path(prefix).name
        logger.info("pulling dataset from r2://%s/%s/ -> %s", r2.bucket_name(), prefix, path)
        r2.pull_prefix(prefix, path)

    manifest = json.loads((path / "manifest.json").read_text())
    cubes: list[CubeData] = []
    for i in range(manifest["num_cubes"]):
        oracle_ids = json.loads(_oracle_path(path, i).read_text())
        train = {k: v for k, v in np.load(_split_path(path, i, "train")).items()}
        eval_ = {k: v for k, v in np.load(_split_path(path, i, "eval")).items()}
        cubes.append(CubeData(oracle_ids=oracle_ids, train=train, eval=eval_))
    return Dataset(teacher=manifest["teacher"], cubes=cubes)
