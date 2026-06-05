"""Card vocabulary: maps cards to dense indices used by env and model.

Two index spaces:
- Global vocab: every card we know about (loaded from Scryfall Parquet).
- Cube vocab: a per-cube restriction. The env operates on a cube vocab
  because pack/pool tensors are over the ~360-540 cards in the cube, not
  the ~30k global vocab.

arena_id -> oracle_id resolution is for the MTGA log follower (§2.2 of
the plan). Many arena_ids map to the same oracle_id (art variants and
promos); canonicalization is one-way.
"""

from __future__ import annotations

from pathlib import Path

from cube_draft.cards.card import Card
from cube_draft.cards.scryfall import cards_from_parquet


class CardVocab:
    """Bidirectional oracle_id <-> index mapping plus arena_id resolution.

    Indices are stable for the lifetime of the instance. Reload to rebuild.
    """

    def __init__(self, cards: list[Card]) -> None:
        self._cards: list[Card] = list(cards)
        self._by_oracle: dict[str, int] = {c.oracle_id: i for i, c in enumerate(self._cards)}
        self._by_name: dict[str, int] = {c.name: i for i, c in enumerate(self._cards)}
        self._arena_to_oracle: dict[int, str] = {
            c.arena_id: c.oracle_id for c in self._cards if c.arena_id is not None
        }

    @classmethod
    def from_parquet(cls, path: Path) -> CardVocab:
        return cls(cards_from_parquet(path))

    def __len__(self) -> int:
        return len(self._cards)

    def card(self, idx: int) -> Card:
        return self._cards[idx]

    def index(self, oracle_id: str) -> int:
        return self._by_oracle[oracle_id]

    def index_by_name(self, name: str) -> int:
        return self._by_name[name]

    def has_oracle(self, oracle_id: str) -> bool:
        return oracle_id in self._by_oracle

    def arena_to_oracle(self, arena_id: int) -> str | None:
        """Resolve an MTGA log `CardId` to an oracle_id, or None if unknown."""
        return self._arena_to_oracle.get(arena_id)

    def arena_to_index(self, arena_id: int) -> int | None:
        oracle = self.arena_to_oracle(arena_id)
        return self._by_oracle.get(oracle) if oracle is not None else None


class CubeVocab:
    """Restricted vocab over the cards in a specific cube.

    Maps between global vocab indices and local cube indices. The env
    uses cube indices for compactness; the model bridges back to global
    indices for card embeddings.
    """

    def __init__(self, global_vocab: CardVocab, cube_oracle_ids: list[str]) -> None:
        # Drop unknown oracles (cards not in the global vocab) — warn caller
        # via the discarded list, but proceed.
        self._global = global_vocab
        self.discarded: list[str] = [
            oid for oid in cube_oracle_ids if not global_vocab.has_oracle(oid)
        ]
        kept = [oid for oid in cube_oracle_ids if global_vocab.has_oracle(oid)]
        self._oracle_ids: list[str] = kept
        self._global_idx: list[int] = [global_vocab.index(oid) for oid in kept]
        self._local_by_global: dict[int, int] = {g: i for i, g in enumerate(self._global_idx)}

    def __len__(self) -> int:
        return len(self._oracle_ids)

    @property
    def size(self) -> int:
        return len(self._oracle_ids)

    def to_global(self, local_idx: int) -> int:
        return self._global_idx[local_idx]

    def to_local(self, global_idx: int) -> int | None:
        return self._local_by_global.get(global_idx)

    def card(self, local_idx: int) -> Card:
        return self._global.card(self._global_idx[local_idx])

    def oracle_id(self, local_idx: int) -> str:
        return self._oracle_ids[local_idx]

    def global_indices(self) -> list[int]:
        return list(self._global_idx)
