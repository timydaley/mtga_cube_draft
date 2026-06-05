from __future__ import annotations

import numpy as np

from cube_draft.bots.base import Drafter, DraftObs


class RandomBot(Drafter):
    """Uniform random over in-pack cards. Seeded for reproducibility."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def pick(self, obs: DraftObs) -> int:
        pack = obs["pack"]
        choices = np.flatnonzero(pack > 0)
        if len(choices) == 0:
            raise ValueError("empty pack passed to RandomBot")
        return int(self._rng.choice(choices))
