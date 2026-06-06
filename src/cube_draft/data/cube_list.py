"""Resolve a published cube card list (names) to oracle_ids (plan §2.5).

The Arena Cube list is published as card names (e.g. on magic.wizards.com). To
use it we must map those names onto our Scryfall vocab's oracle_ids. Name strings
vary from Scryfall's canonical form — commas/apostrophes dropped, and DFC /
adventure cards often given by front face only — so matching is done on a
normalized key, with front-face fallback. Unresolved names are reported rather
than silently dropped, so the list can be corrected.
"""

from __future__ import annotations

import difflib
import json
import re
import time
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
class FuzzyMatch:
    """A name resolved only via fuzzy fallback — flagged for human review."""

    input: str
    matched: str  # canonical card name it was mapped to
    oracle_id: str
    score: float  # 1.0 for the Scryfall backend (authoritative)
    backend: str


@dataclass
class Resolution:
    oracle_ids: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)  # name resolved to an already-seen oracle
    fuzzy: list[FuzzyMatch] = field(default_factory=list)  # exact-miss, fuzzy-recovered


def _name_of(vocab: CardVocab, oracle_id: str) -> str:
    return vocab.card(vocab.index(oracle_id)).name


def _fuzzy_local(name: str, keys: list[str], index: dict[str, str], cutoff: float) -> tuple[str, float] | None:
    """Closest normalized vocab key by edit-distance ratio (offline)."""
    key = normalize_name(name)
    close = difflib.get_close_matches(key, keys, n=1, cutoff=cutoff)
    if not close:
        return None
    best = close[0]
    return index[best], difflib.SequenceMatcher(None, key, best).ratio()


def _fuzzy_scryfall(name: str, timeout: int = 20) -> str | None:
    """Scryfall's MTG-tuned fuzzy name endpoint -> oracle_id (network)."""
    import requests

    resp = requests.get(
        "https://api.scryfall.com/cards/named",
        params={"fuzzy": name},
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    time.sleep(0.1)  # Scryfall asks callers to throttle
    if resp.status_code != 200:
        return None
    return resp.json().get("oracle_id")


def resolve_names(
    vocab: CardVocab,
    names: list[str],
    *,
    fuzzy: bool = False,
    fuzzy_cutoff: float = 0.85,
    fuzzy_backend: str = "local",
) -> Resolution:
    """Resolve card names to a de-duplicated list of oracle_ids.

    Exact normalized matching first; if `fuzzy`, names that miss are retried via
    `fuzzy_backend` ("local" difflib or "scryfall"). Fuzzy hits are added to the
    cube AND recorded in `res.fuzzy` for review — never silently trusted.
    """
    index = build_name_index(vocab)
    keys = list(index.keys()) if fuzzy and fuzzy_backend == "local" else []
    res = Resolution()
    seen: set[str] = set()

    def add(oid: str, name: str, fm: FuzzyMatch | None = None) -> None:
        if oid in seen:
            res.duplicates.append(name)
            return
        seen.add(oid)
        res.oracle_ids.append(oid)
        res.matched.append(name)
        if fm is not None:
            res.fuzzy.append(fm)

    for raw in names:
        name = raw.strip()
        if not name:
            continue
        oid = index.get(normalize_name(name)) or index.get(_front_face(name))
        if oid is not None:
            add(oid, name)
            continue
        if fuzzy:
            fm = _try_fuzzy(vocab, name, keys, index, fuzzy_backend, fuzzy_cutoff)
            if fm is not None:
                add(fm.oracle_id, name, fm)
                continue
        res.unmatched.append(name)
    return res


def _try_fuzzy(
    vocab: CardVocab, name: str, keys: list[str], index: dict[str, str], backend: str, cutoff: float
) -> FuzzyMatch | None:
    if backend == "local":
        hit = _fuzzy_local(name, keys, index, cutoff)
        if hit is None:
            return None
        oid, score = hit
        return FuzzyMatch(name, _name_of(vocab, oid), oid, score, "local")
    if backend == "scryfall":
        oid = _fuzzy_scryfall(name)
        if oid is None or not vocab.has_oracle(oid):
            return None  # Scryfall match not in our vocab is unusable
        return FuzzyMatch(name, _name_of(vocab, oid), oid, 1.0, "scryfall")
    raise ValueError(f"unknown fuzzy backend: {backend}")


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
