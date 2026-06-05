"""RaredraftBot — pick the highest-rarity in-pack card. Trivial baseline.

In cube the heuristic is weaker than in set draft (cubes are mostly rares
already) but it's still a non-trivial baseline above random.
"""

from __future__ import annotations

import numpy as np

from cube_draft.bots.base import Drafter, DraftObs
from cube_draft.cards.vocab import CubeVocab

_RARITY_ORDER = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3, "special": 4, "bonus": 1}


class RaredraftBot(Drafter):
    def __init__(self, cube: CubeVocab, seed: int | None = None) -> None:
        self._cube = cube
        self._rng = np.random.default_rng(seed)
        # Precompute rarity score per cube index.
        self._rarity_score = np.array(
            [_RARITY_ORDER.get(cube.card(i).rarity, 0) for i in range(cube.size)],
            dtype=np.int32,
        )

    def pick(self, obs: DraftObs) -> int:
        pack = obs["pack"]
        in_pack = np.flatnonzero(pack > 0)
        if len(in_pack) == 0:
            raise ValueError("empty pack passed to RaredraftBot")
        scores = self._rarity_score[in_pack]
        best = scores.max()
        winners = in_pack[scores == best]
        return int(self._rng.choice(winners))
