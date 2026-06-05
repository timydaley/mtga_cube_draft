from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Color = Literal["W", "U", "B", "R", "G"]
Rarity = Literal["common", "uncommon", "rare", "mythic", "special", "bonus"]


class Card(BaseModel):
    """Minimal card representation derived from Scryfall oracle_cards.

    Stored in Parquet, keyed by oracle_id. arena_id is the join column
    against MTGA log events; it is absent for paper-only printings.
    """

    model_config = ConfigDict(frozen=True)

    oracle_id: str
    name: str
    mana_cost: str = ""
    cmc: float = 0.0
    type_line: str = ""
    oracle_text: str = ""
    colors: tuple[Color, ...] = ()
    color_identity: tuple[Color, ...] = ()
    power: str | None = None
    toughness: str | None = None
    keywords: tuple[str, ...] = ()
    rarity: Rarity = "common"
    arena_id: int | None = None

    @property
    def is_land(self) -> bool:
        return "Land" in self.type_line

    @property
    def is_creature(self) -> bool:
        return "Creature" in self.type_line

    def color_vector(self) -> tuple[int, int, int, int, int]:
        """Returns (W, U, B, R, G) one-hot of card colors."""
        order: tuple[Color, ...] = ("W", "U", "B", "R", "G")
        return tuple(int(c in self.colors) for c in order)  # type: ignore[return-value]
