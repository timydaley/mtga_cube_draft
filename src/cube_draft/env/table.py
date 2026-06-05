"""Table — runs a full 8-seat draft with a homogeneous-or-mixed bot population.

Used for pick-accuracy and tournament evaluation. Unlike CubeDraftEnv,
which exposes one seat as the "agent," Table treats every seat
symmetrically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cube_draft.cards.vocab import CubeVocab
from cube_draft.env.cube_draft import DraftConfig, SeatState
from cube_draft.env.pack import Pack

if TYPE_CHECKING:
    from cube_draft.bots.base import Drafter


@dataclass
class DraftResult:
    pools: list[list[int]]  # one per seat, in seat order (cube indices)
    pick_log: list[list[int]] = field(default_factory=list)  # per-seat sequence of picks


class Table:
    def __init__(
        self,
        cube_vocab: CubeVocab,
        drafters: list[Drafter],
        config: DraftConfig | None = None,
    ) -> None:
        self.cube = cube_vocab
        self.cfg = config or DraftConfig()
        if len(drafters) != self.cfg.num_seats:
            raise ValueError(
                f"need {self.cfg.num_seats} drafters, got {len(drafters)}"
            )
        self.drafters = list(drafters)

    def run(self, seed: int | None = None) -> DraftResult:
        rng = np.random.default_rng(seed if seed is not None else self.cfg.seed)
        seats = [SeatState() for _ in range(self.cfg.num_seats)]
        pick_log: list[list[int]] = [[] for _ in range(self.cfg.num_seats)]

        for pack_idx in range(self.cfg.num_packs):
            packs = self._open_packs(rng)
            direction = 1 if pack_idx % 2 == 0 else -1

            for pick_idx in range(self.cfg.pack_size):
                # Every seat picks simultaneously from its current pack.
                for seat in range(self.cfg.num_seats):
                    pack = packs[seat]
                    obs = self._observe(seats[seat], pack, pack_idx, pick_idx, seat)
                    choice = self.drafters[seat].pick(obs)
                    if choice not in pack:
                        raise ValueError(
                            f"seat {seat} chose {choice} not in pack {pack.cards()}"
                        )
                    # seen_unpicked is updated with cards in pack *other than* the choice
                    seats[seat].seen_unpicked.extend(c for c in pack if c != choice)
                    seats[seat].pool.append(choice)
                    pick_log[seat].append(choice)
                    pack.pick(choice)

                # Rotate.
                n = self.cfg.num_seats
                new_packs = [Pack([])] * n
                for i, pack in enumerate(packs):
                    new_packs[(i + direction) % n] = pack
                packs = new_packs

        return DraftResult(
            pools=[list(s.pool) for s in seats],
            pick_log=pick_log,
        )

    def _open_packs(self, rng: np.random.Generator) -> list[Pack]:
        n_cards = self.cfg.pack_size * self.cfg.num_seats
        if n_cards > self.cube.size:
            raise ValueError(
                f"cube has {self.cube.size} cards; need {n_cards} for one pack-round"
            )
        sampled = rng.choice(self.cube.size, size=n_cards, replace=False)
        return [
            Pack(sampled[i * self.cfg.pack_size : (i + 1) * self.cfg.pack_size].tolist())
            for i in range(self.cfg.num_seats)
        ]

    def _observe(
        self,
        seat: SeatState,
        pack: Pack,
        pack_idx: int,
        pick_idx: int,
        seat_idx: int,
    ) -> dict:
        n = self.cube.size
        pack_vec = np.zeros(n, dtype=np.int8)
        for c in pack:
            pack_vec[c] = 1
        return {
            "pack": pack_vec,
            "pool": seat.pool_counts(n),
            "seen_unpicked": seat.seen_counts(n),
            "cube_mask": np.ones(n, dtype=np.int8),
            "pack_idx": pack_idx,
            "pick_idx": pick_idx,
            "seat_idx": seat_idx,
        }
