"""Scryfall bulk data ingest.

Fetches https://api.scryfall.com/bulk-data, downloads `oracle_cards`, and
normalizes to our `Card` schema. Persists Parquet locally; upload to R2
via scripts/upload_to_r2.py.

Refreshes weekly per the plan; bulk data updated daily by Scryfall.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import requests

from cube_draft.cards.card import Card

logger = logging.getLogger(__name__)

BULK_INDEX_URL = "https://api.scryfall.com/bulk-data"
USER_AGENT = "cube-draft/0.0.1 (https://github.com/personal/mtga_cube_draft)"


def fetch_bulk_index(timeout: int = 30) -> list[dict[str, Any]]:
    """List available bulk data files from Scryfall."""
    resp = requests.get(
        BULK_INDEX_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def download_oracle_cards(dest: Path, timeout: int = 600) -> Path:
    """Download the latest `oracle_cards` bulk JSON.

    Returns the path to the downloaded file. Streams to disk to avoid
    holding ~110MB in memory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = fetch_bulk_index()
    target = next(f for f in files if f["type"] == "oracle_cards")
    url = target["download_uri"]
    logger.info("downloading oracle_cards from %s (%d bytes)", url, target.get("size", 0))
    with requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=timeout) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return dest


def _normalize_card(raw: dict[str, Any]) -> Card | None:
    """Convert a raw Scryfall card dict into our Card schema.

    Returns None for cards we exclude (tokens, art series, memorabilia).
    """
    layout = raw.get("layout", "")
    if layout in {"token", "double_faced_token", "emblem", "art_series", "vanguard"}:
        return None
    if raw.get("set_type") == "memorabilia":
        return None

    type_line = raw.get("type_line", "")
    if not type_line or "Token" in type_line:
        return None

    return Card(
        oracle_id=raw["oracle_id"],
        name=raw["name"],
        mana_cost=raw.get("mana_cost", "") or "",
        cmc=float(raw.get("cmc", 0.0) or 0.0),
        type_line=type_line,
        oracle_text=raw.get("oracle_text", "") or "",
        colors=tuple(raw.get("colors", []) or []),  # type: ignore[arg-type]
        color_identity=tuple(raw.get("color_identity", []) or []),  # type: ignore[arg-type]
        power=raw.get("power"),
        toughness=raw.get("toughness"),
        keywords=tuple(raw.get("keywords", []) or []),
        rarity=raw.get("rarity", "common"),
        arena_id=raw.get("arena_id"),
    )


def parse_oracle_cards(path: Path) -> list[Card]:
    """Parse oracle_cards JSON (gzipped or plain) into Card objects."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        raw = json.load(f)
    cards: list[Card] = []
    for entry in raw:
        card = _normalize_card(entry)
        if card is not None:
            cards.append(card)
    logger.info("parsed %d cards (filtered from %d)", len(cards), len(raw))
    return cards


def cards_to_parquet(cards: list[Card], dest: Path) -> Path:
    """Persist a Card list to Parquet via Polars."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame([c.model_dump() for c in cards])
    df.write_parquet(dest, compression="zstd")
    return dest


def cards_from_parquet(path: Path) -> list[Card]:
    """Load Cards from Parquet."""
    df = pl.read_parquet(path)
    return [Card(**row) for row in df.to_dicts()]
