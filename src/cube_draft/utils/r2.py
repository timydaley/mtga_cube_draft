"""Cloudflare R2 client wrapper.

R2 is S3-compatible; we use boto3 with a custom endpoint. The bucket and
credentials come from env vars (.env locally, RunPod secrets in the cloud).

Required env vars:
    R2_ACCOUNT_ID           — account UUID from the Cloudflare dashboard
    R2_ACCESS_KEY_ID        — API token access key
    R2_SECRET_ACCESS_KEY    — API token secret
    R2_BUCKET               — bucket name (default: cube-draft)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "cube-draft"


def _env(name: str, *, required: bool = True, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"missing env var {name}. Source .env or set RunPod pod secrets."
        )
    return value or ""


def make_client() -> Any:
    """Return an S3 client configured for Cloudflare R2."""
    account_id = _env("R2_ACCOUNT_ID")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def bucket_name() -> str:
    return _env("R2_BUCKET", required=False, default=DEFAULT_BUCKET)


def push(local: Path, remote_key: str, client: Any | None = None) -> None:
    """Upload a local file to `r2://<bucket>/<remote_key>`."""
    client = client or make_client()
    bucket = bucket_name()
    logger.info("push %s -> r2://%s/%s", local, bucket, remote_key)
    client.upload_file(str(local), bucket, remote_key)


def pull(remote_key: str, local: Path, client: Any | None = None) -> Path:
    """Download `r2://<bucket>/<remote_key>` to `local`."""
    client = client or make_client()
    bucket = bucket_name()
    local.parent.mkdir(parents=True, exist_ok=True)
    logger.info("pull r2://%s/%s -> %s", bucket, remote_key, local)
    client.download_file(bucket, remote_key, str(local))
    return local


def list_prefix(prefix: str = "", client: Any | None = None) -> list[dict[str, Any]]:
    """List objects under a prefix, paginated."""
    client = client or make_client()
    bucket = bucket_name()
    paginator = client.get_paginator("list_objects_v2")
    out: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        out.extend(page.get("Contents", []))
    return out


def push_dir(local_dir: Path, remote_prefix: str, client: Any | None = None) -> int:
    """Recursively upload every file under `local_dir` to `remote_prefix/`."""
    client = client or make_client()
    n = 0
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir)
        key = f"{remote_prefix.rstrip('/')}/{rel.as_posix()}"
        push(p, key, client=client)
        n += 1
    return n


def pull_prefix(remote_prefix: str, local_dir: Path, client: Any | None = None) -> int:
    """Pull every object under `remote_prefix/` into `local_dir`, preserving relative paths."""
    client = client or make_client()
    n = 0
    for obj in list_prefix(remote_prefix, client=client):
        key = obj["Key"]
        rel = key[len(remote_prefix) :].lstrip("/")
        if not rel:
            continue  # skip the "directory" key itself
        dest = local_dir / rel
        pull(key, dest, client=client)
        n += 1
    return n
