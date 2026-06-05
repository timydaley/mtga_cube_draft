"""Shared fixtures: a synthetic cube vocab for tests that don't need Scryfall data."""

from __future__ import annotations

import pytest

from cube_draft.cards.card import Card
from cube_draft.cards.vocab import CardVocab, CubeVocab


def _synthetic_card(i: int) -> Card:
    """Generate a deterministic synthetic card for testing.

    Distributes colors and rarities so RaredraftBot and color-aware bots
    have something to discriminate on.
    """
    colors_pool = [("W",), ("U",), ("B",), ("R",), ("G",), ()]
    rarities = ["common", "uncommon", "rare", "mythic"]
    color = colors_pool[i % len(colors_pool)]
    rarity = rarities[i % len(rarities)]
    return Card(
        oracle_id=f"oracle-{i:05d}",
        name=f"Test Card {i}",
        mana_cost=f"{{{i % 7}}}",
        cmc=float(i % 7),
        type_line="Creature" if i % 2 == 0 else "Sorcery",
        oracle_text="",
        colors=color,  # type: ignore[arg-type]
        color_identity=color,  # type: ignore[arg-type]
        rarity=rarity,  # type: ignore[arg-type]
        arena_id=70000 + i,
    )


@pytest.fixture
def synthetic_cards() -> list[Card]:
    return [_synthetic_card(i) for i in range(540)]


@pytest.fixture
def global_vocab(synthetic_cards) -> CardVocab:
    return CardVocab(synthetic_cards)


@pytest.fixture
def cube(global_vocab) -> CubeVocab:
    oracle_ids = [global_vocab.card(i).oracle_id for i in range(len(global_vocab))]
    return CubeVocab(global_vocab, oracle_ids)
