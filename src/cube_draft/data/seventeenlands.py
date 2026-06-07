"""17lands card-level ratings as a draft teacher signal (plan §4.2 17lands source).

Per-draft cube data isn't public, but 17lands' card_ratings API exposes per-card
metrics from real human Arena Cube drafts. We use them as the Stage 2 teacher:
a card's "value" drives the teacher's pick, which the CC-CPR model distills (and
then generalizes to cards 17lands hasn't rated, via card features).

Metrics of interest:
  drawn_improvement_win_rate  GIH WR minus not-drawn WR — marginal win contribution
                              (least deck-quality-confounded; "draft what wins")
  ever_drawn_win_rate         GIH WR — raw games-in-hand win rate
  avg_pick                    ATA — average pick position humans took it (lower =
                              earlier = more wanted; negated so higher value = sooner)

CAVEAT: these are aggregate/context-free, so the distilled policy is effectively a
real-data card ranking. Context-aware drafting (synergy, color commitment, signal
reading) needs per-draft data not public for cube — deferred (plan §4.7).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from cube_draft.cards.vocab import CardVocab, CubeVocab
from cube_draft.data import cube_list

logger = logging.getLogger(__name__)

_API = "https://www.17lands.com/card_ratings/data"
_UA = {"User-Agent": "Mozilla/5.0 (cube-draft research)"}
DEFAULT_EXPANSION = "Cube - Powered"
DEFAULT_FORMAT = "PremierDraft"
METRICS = ("drawn_improvement_win_rate", "ever_drawn_win_rate", "avg_pick")
_KEEP_FIELDS = ("name", "game_count", *METRICS)
_LOCAL_CACHE = Path("data/17lands")


def fetch_card_ratings(
    expansion: str = DEFAULT_EXPANSION,
    fmt: str = DEFAULT_FORMAT,
    start: str = "2025-10-01",
    end: str = "2026-06-06",
    timeout: int = 60,
) -> list[dict]:
    """Fetch per-card ratings for an expansion/format/date window."""
    import requests

    resp = requests.get(
        _API,
        params={"expansion": expansion, "format": fmt, "start_date": start, "end_date": end},
        headers=_UA,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def ratings_by_oracle(rows: list[dict], vocab: CardVocab) -> dict[str, dict]:
    """Join 17lands rows (keyed by card name) to oracle_ids via the name resolver."""
    index = cube_list.build_name_index(vocab)
    out: dict[str, dict] = {}
    for c in rows:
        if c.get("game_count", 0) <= 0:
            continue
        oid = index.get(cube_list.normalize_name(c["name"])) or index.get(
            cube_list._front_face(c["name"])  # noqa: SLF001
        )
        if oid is not None:
            out.setdefault(oid, {k: c.get(k) for k in _KEEP_FIELDS})
    return out


def value_array(cube: CubeVocab, by_oracle: dict[str, dict], metric: str) -> np.ndarray:
    """Per-cube-local-index teacher value (z-scored; higher = pick sooner).

    `avg_pick` is negated (earlier pick = higher value). Cube cards with no
    17lands data fall to the minimum value (least preferred).
    """
    raw = np.full(cube.size, np.nan, dtype=np.float64)
    for i in range(cube.size):
        m = by_oracle.get(cube.oracle_id(i))
        if m is not None and m.get(metric) is not None:
            v = float(m[metric])
            raw[i] = -v if metric == "avg_pick" else v
    if np.isnan(raw).all():
        return np.zeros(cube.size, dtype=np.float32)
    raw = np.where(np.isnan(raw), np.nanmin(raw), raw)
    std = float(raw.std()) or 1.0
    return ((raw - raw.mean()) / std).astype(np.float32)


def save_ratings(by_oracle: dict[str, dict], path: Path, meta: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "cards": by_oracle}, indent=0))
    return path


def load_ratings(source: str) -> dict[str, dict]:
    """Load an oracle-keyed ratings file from a local path or R2 key."""
    path = Path(source)
    if not path.is_file():
        from cube_draft.utils import r2

        key = source.removeprefix("r2://")
        path = _LOCAL_CACHE / Path(key).name
        r2.pull(key, path)
    return json.loads(path.read_text())["cards"]
