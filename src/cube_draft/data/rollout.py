"""Record teacher pick decisions from simulated drafts (for distillation).

Runs N drafts where every seat is a provided Drafter and records, for every
seat at every pick, the observation tensors and the chosen card. The output
arrays are exactly what the Stage 2 trainer consumes, so the same recorder
serves both the on-the-fly simulator teacher and an offline CubeCobraBot
dataset build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cube_draft.cards.vocab import CubeVocab
from cube_draft.env.cube_draft import DraftConfig, SeatState
from cube_draft.env.pack import Pack

if TYPE_CHECKING:
    from cube_draft.bots.base import Drafter


def observe(
    seat: SeatState, pack: Pack, pack_idx: int, pick_idx: int, seat_idx: int, n: int
) -> dict:
    """Build a single seat's observation (cube-local tensors), matching the env."""
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


def collect_decisions(
    cube: CubeVocab,
    drafters: list[Drafter],
    n_drafts: int,
    cfg: DraftConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Run `n_drafts` drafts with the given per-seat drafters; record every pick.

    Each draft yields num_seats * total_picks decisions. seen_unpicked
    accumulates per the §2.4 own-seat-only signal. Returns arrays keyed
    pack / pool / seen / pack_idx / pick_idx / picked.
    """
    if len(drafters) != cfg.num_seats:
        raise ValueError(f"need {cfg.num_seats} drafters, got {len(drafters)}")
    n = cube.size
    packs_buf: list[np.ndarray] = []
    pools_buf: list[np.ndarray] = []
    seen_buf: list[np.ndarray] = []
    pack_idx_buf: list[int] = []
    pick_idx_buf: list[int] = []
    picked_buf: list[int] = []

    for _ in range(n_drafts):
        seats = [SeatState() for _ in range(cfg.num_seats)]
        for pack_idx in range(cfg.num_packs):
            n_cards = cfg.pack_size * cfg.num_seats
            sampled = rng.choice(n, size=n_cards, replace=False)
            packs = [
                Pack(sampled[i * cfg.pack_size : (i + 1) * cfg.pack_size].tolist())
                for i in range(cfg.num_seats)
            ]
            direction = 1 if pack_idx % 2 == 0 else -1
            for pick_idx in range(cfg.pack_size):
                for seat in range(cfg.num_seats):
                    pack = packs[seat]
                    obs = observe(seats[seat], pack, pack_idx, pick_idx, seat, n)
                    choice = drafters[seat].pick(obs)
                    packs_buf.append(obs["pack"])
                    pools_buf.append(obs["pool"])
                    seen_buf.append(obs["seen_unpicked"])
                    pack_idx_buf.append(pack_idx)
                    pick_idx_buf.append(pick_idx)
                    picked_buf.append(choice)
                    seats[seat].seen_unpicked.extend(c for c in pack if c != choice)
                    seats[seat].pool.append(choice)
                    pack.pick(choice)
                new_packs = [Pack([])] * cfg.num_seats
                for i, p in enumerate(packs):
                    new_packs[(i + direction) % cfg.num_seats] = p
                packs = new_packs

    return {
        "pack": np.stack(packs_buf),
        "pool": np.stack(pools_buf),
        "seen": np.stack(seen_buf),
        "pack_idx": np.array(pack_idx_buf, dtype=np.int64),
        "pick_idx": np.array(pick_idx_buf, dtype=np.int64),
        "picked": np.array(picked_buf, dtype=np.int64),
    }
