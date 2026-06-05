"""CubeCobra data ingestion — S3 bulk export + REST API.

S3 export layout (per plan §4.1):
    s3://cubecobra-public/export/
        indexToOracleMap.json     # int idx -> Scryfall oracle_id
        simpleCardDict.json       # oracle_id -> metadata
        cubes.json                # all public cube lists
        decks/{n}.json            # finished draft decks
        picks/{n}.json            # individual pick decisions
        cubeInstances/{n}.json    # cube card pool for each draft

Anonymous read; no auth needed. Updated quarterly.

The bucket is `public-read`; we use boto3's anonymous-credentials path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config

logger = logging.getLogger(__name__)

CUBECOBRA_BUCKET = "cubecobra-public"
CUBECOBRA_PREFIX = "export/"


def _anon_s3():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_export_objects(max_keys: int = 1000) -> list[dict[str, Any]]:
    """List objects under the export prefix. Used by the Week 0 audit."""
    s3 = _anon_s3()
    resp = s3.list_objects_v2(
        Bucket=CUBECOBRA_BUCKET, Prefix=CUBECOBRA_PREFIX, MaxKeys=max_keys
    )
    return resp.get("Contents", [])


def download_object(key: str, dest: Path) -> Path:
    """Download a single object from the export bucket to `dest`."""
    s3 = _anon_s3()
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading s3://%s/%s -> %s", CUBECOBRA_BUCKET, key, dest)
    s3.download_file(CUBECOBRA_BUCKET, key, str(dest))
    return dest


def fetch_picks_sample(out_dir: Path, n: int = 5) -> list[Path]:
    """Download a small sample of picks/{n}.json files for the audit.

    We don't know the numbering scheme in advance, so we list first and
    pull the first `n` keys under `export/picks/`.
    """
    s3 = _anon_s3()
    resp = s3.list_objects_v2(
        Bucket=CUBECOBRA_BUCKET, Prefix=f"{CUBECOBRA_PREFIX}picks/", MaxKeys=n
    )
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key in keys:
        local = out_dir / Path(key).name
        download_object(key, local)
        paths.append(local)
    return paths


def fetch_cube_instances_sample(out_dir: Path, n: int = 5) -> list[Path]:
    s3 = _anon_s3()
    resp = s3.list_objects_v2(
        Bucket=CUBECOBRA_BUCKET, Prefix=f"{CUBECOBRA_PREFIX}cubeInstances/", MaxKeys=n
    )
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key in keys:
        local = out_dir / Path(key).name
        download_object(key, local)
        paths.append(local)
    return paths


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
