"""Resolve a published cube card list (names) to oracle_ids (plan §2.5).

The Arena Cube list is published as card names (e.g. on magic.wizards.com). To
use it we must map those names onto our Scryfall vocab's oracle_ids. Name strings
vary from Scryfall's canonical form — commas/apostrophes dropped, and DFC /
adventure cards often given by front face only — so matching is done on a
normalized key, with front-face fallback. Unresolved names are reported rather
than silently dropped, so the list can be corrected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from cube_draft.cards.vocab import CardVocab

_LOCAL_CACHE = Path("data/cubes")
_USER_AGENT = "cube-draft/0.0.1 (https://github.com/timydaley/mtga_cube_draft)"


def fetch_cubecobra_list(cube_id: str, timeout: int = 30) -> list[str]:
    """Fetch a cube's card names from CubeCobra's plain-text list API.

    Cleaner and more complete than scraping an article. `cube_id` is the cube's
    short id or UUID (e.g. "mtgapc" for the official Arena Powered Cube).
    """
    import requests

    url = f"https://cubecobra.com/cube/api/cubelist/{cube_id}"
    resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return [line.strip() for line in resp.text.splitlines() if line.strip()]


def normalize_name(name: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Apostrophes are removed (so "Yawgmoth's" -> "yawgmoths"); other punctuation
    (commas, hyphens, the "//" DFC separator) becomes a space.
    """
    name = name.lower().replace("//", " ")
    name = name.replace("'", "").replace("’", "")  # straight + curly apostrophes
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _front_face(name: str) -> str:
    return normalize_name(name.split("//")[0])


def build_name_index(vocab: CardVocab) -> dict[str, str]:
    """normalized-name -> oracle_id, indexing both full and front-face keys.

    Full-name keys take precedence over front-face keys (added first); a
    front-face key is only used if no card claims it as a full name.
    """
    index: dict[str, str] = {}
    cards = [vocab.card(i) for i in range(len(vocab))]
    for c in cards:  # full names first
        index.setdefault(normalize_name(c.name), c.oracle_id)
    for c in cards:  # then front-face fallbacks
        index.setdefault(_front_face(c.name), c.oracle_id)
    return index


@dataclass
class Resolution:
    oracle_ids: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)  # name resolved to an already-seen oracle


def resolve_names(vocab: CardVocab, names: list[str]) -> Resolution:
    """Resolve card names to a de-duplicated list of oracle_ids."""
    index = build_name_index(vocab)
    res = Resolution()
    seen: set[str] = set()
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        oid = index.get(normalize_name(name)) or index.get(_front_face(name))
        if oid is None:
            res.unmatched.append(name)
        elif oid in seen:
            res.duplicates.append(name)
        else:
            seen.add(oid)
            res.oracle_ids.append(oid)
            res.matched.append(name)
    return res


def parse_name_file(text: str) -> list[str]:
    """One card name per line; blank lines and `#` comments ignored."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def load_oracle_ids(source: str) -> list[str]:
    """Load an oracle_id cube file (JSON list) from a local path or R2 key."""
    path = Path(source)
    if not path.is_file():
        from cube_draft.utils import r2

        key = source.removeprefix("r2://")
        path = _LOCAL_CACHE / Path(key).name
        r2.pull(key, path)
    return json.loads(path.read_text())
