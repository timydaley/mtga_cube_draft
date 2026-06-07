"""SeventeenLandsBot — teacher Drafter driven by 17lands card values.

Picks the highest-value in-pack card (or samples by softmax of value when a
temperature is given). Value is a per-cube-local-index array built from 17lands
metrics (see cube_draft.data.seventeenlands.value_array) — e.g. win-rate
contribution, so the teacher "drafts what wins."
"""

from __future__ import annotations

import numpy as np

from cube_draft.bots.base import Drafter, DraftObs


class SeventeenLandsBot(Drafter):
    def __init__(self, values: np.ndarray, seed: int = 0, temperature: float = 0.0) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)

    def pick(self, obs: DraftObs) -> int:
        in_pack = np.flatnonzero(obs["pack"] > 0)
        if len(in_pack) == 0:
            raise ValueError("empty pack passed to SeventeenLandsBot")
        v = self.values[in_pack]
        if self.temperature <= 0.0:
            winners = in_pack[v == v.max()]  # break exact ties at random
            return int(self._rng.choice(winners))
        p = np.exp((v - v.max()) / self.temperature)
        p /= p.sum()
        return int(self._rng.choice(in_pack, p=p))
