"""Structured card features f(c) for the card representation (plan §4.3).

`card_repr(c) = E[c] + alpha * g(f(c))`. This module produces `f(c)`: a
fixed-width vector of Scryfall-derived structured features (mana value, color
bits, type bits, P/T, rarity, evergreen keywords). The learned embedding `E`
and the feature MLP `g` live in the model (cc_cpr).

The MiniLM oracle-text embedding (§4.3, a 384-d vector) is intentionally NOT
included here yet — it needs a one-time per-card precompute and is gated behind
the §4.5 ablation. When added, it concatenates onto this vector and FEATURE_DIM
grows accordingly.
"""

from __future__ import annotations

import numpy as np

from cube_draft.cards.card import Card

# Feature layout (order matters; FEATURE_DIM is derived from these).
_CMC_BUCKETS = 8  # mana values 0..6, with 7 capturing 7+
_TYPE_TOKENS = (
    "Creature",
    "Land",
    "Instant",
    "Sorcery",
    "Artifact",
    "Enchantment",
    "Planeswalker",
    "Battle",
)
_RARITIES = ("common", "uncommon", "rare", "mythic", "special", "bonus")
# Evergreen / near-evergreen keywords — a small curated set, matched
# case-insensitively against Scryfall's `keywords`.
_KEYWORDS = (
    "Flying",
    "Deathtouch",
    "Lifelink",
    "Trample",
    "Vigilance",
    "Haste",
    "First strike",
    "Double strike",
    "Menace",
    "Reach",
    "Hexproof",
    "Ward",
    "Flash",
    "Defender",
)
_COLOR_ORDER = ("W", "U", "B", "R", "G")

FEATURE_DIM = (
    1  # cmc normalized
    + _CMC_BUCKETS  # cmc one-hot
    + 5  # colors WUBRG
    + 5  # color identity WUBRG
    + 1  # num colors normalized
    + len(_TYPE_TOKENS)  # type bits
    + 4  # power norm, power-is-variable, toughness norm, toughness-is-variable
    + len(_RARITIES)  # rarity one-hot
    + len(_KEYWORDS)  # keyword multi-hot
    + 1  # keyword count normalized
)


def _parse_pt(value: str | None) -> tuple[float, float]:
    """Return (normalized_value, is_variable). '*'/'1+*' etc. -> variable."""
    if value is None or value == "":
        return 0.0, 0.0
    try:
        return float(value) / 10.0, 0.0
    except ValueError:
        return 0.0, 1.0  # '*', '1+*', 'X' — power/toughness is non-numeric


def card_feature_vector(card: Card) -> np.ndarray:
    """Build f(c) for a single card. Length == FEATURE_DIM."""
    parts: list[np.ndarray] = []

    cmc = max(card.cmc, 0.0)
    parts.append(np.array([min(cmc / 16.0, 1.0)], dtype=np.float32))

    cmc_onehot = np.zeros(_CMC_BUCKETS, dtype=np.float32)
    cmc_onehot[min(int(cmc), _CMC_BUCKETS - 1)] = 1.0
    parts.append(cmc_onehot)

    parts.append(np.array(card.color_vector(), dtype=np.float32))
    parts.append(
        np.array([int(c in card.color_identity) for c in _COLOR_ORDER], dtype=np.float32)
    )
    parts.append(np.array([len(card.colors) / 5.0], dtype=np.float32))

    type_bits = np.array(
        [int(tok in card.type_line) for tok in _TYPE_TOKENS], dtype=np.float32
    )
    parts.append(type_bits)

    p_norm, p_var = _parse_pt(card.power)
    t_norm, t_var = _parse_pt(card.toughness)
    parts.append(np.array([p_norm, p_var, t_norm, t_var], dtype=np.float32))

    rarity_onehot = np.array(
        [int(card.rarity == r) for r in _RARITIES], dtype=np.float32
    )
    parts.append(rarity_onehot)

    kw_lower = {k.lower() for k in card.keywords}
    kw_multihot = np.array(
        [int(k.lower() in kw_lower) for k in _KEYWORDS], dtype=np.float32
    )
    parts.append(kw_multihot)
    parts.append(np.array([min(len(card.keywords) / 5.0, 1.0)], dtype=np.float32))

    return np.concatenate(parts)


def feature_matrix(cards: list[Card]) -> np.ndarray:
    """Stack f(c) for many cards into a (len(cards), FEATURE_DIM) matrix."""
    if not cards:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([card_feature_vector(c) for c in cards])


def color_matrix(cards: list[Card]) -> np.ndarray:
    """(len(cards), 5) WUBRG one-hot, for the explicit color-commit vector (§4.4)."""
    if not cards:
        return np.zeros((0, 5), dtype=np.float32)
    return np.stack([np.array(c.color_vector(), dtype=np.float32) for c in cards])
