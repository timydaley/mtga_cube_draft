"""Pack: a multiset of card indices that gets passed between seats.

In a singleton cube, each pack contains 15 distinct cards sampled
without replacement from the cube pool. The pack's contents shrink by
one each pick as it rotates.
"""

from __future__ import annotations

from collections.abc import Iterable


class Pack:
    __slots__ = ("_cards",)

    def __init__(self, cards: Iterable[int]) -> None:
        # Order matters for reproducible serialization; preserve insertion.
        self._cards: list[int] = list(cards)

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self):
        return iter(self._cards)

    def __contains__(self, card_idx: int) -> bool:
        return card_idx in self._cards

    def cards(self) -> list[int]:
        return list(self._cards)

    def pick(self, card_idx: int) -> int:
        """Remove and return `card_idx`. Raises ValueError if absent."""
        try:
            self._cards.remove(card_idx)
        except ValueError as e:
            raise ValueError(f"card {card_idx} not in pack {self._cards}") from e
        return card_idx

    def is_empty(self) -> bool:
        return len(self._cards) == 0

    def __repr__(self) -> str:
        return f"Pack({self._cards!r})"
