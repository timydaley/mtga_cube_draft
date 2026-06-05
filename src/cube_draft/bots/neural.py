"""NeuralBot — a Drafter backed by a trained CC-CPR model (plan §3.2 #5).

Bound to a single cube's static tensors. `pick` builds a batch-of-one
observation, scores the pack, and returns the argmax in-pack card (cube-local
index). Used both as the eval policy during training and, eventually, behind
the inference UI.
"""

from __future__ import annotations

import numpy as np
import torch

from cube_draft.bots.base import Drafter, DraftObs
from cube_draft.model.cc_cpr import CCCPR, CubeTensors, DraftBatch


class NeuralBot(Drafter):
    def __init__(
        self,
        model: CCCPR,
        cube: CubeTensors,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.cube = cube.to(device)
        self.device = torch.device(device)

    @torch.no_grad()
    def pick(self, obs: DraftObs) -> int:
        self.model.eval()
        batch = DraftBatch(
            pack=self._row(obs["pack"]),
            pool=self._row(obs["pool"]),
            seen=self._row(obs["seen_unpicked"]),
            cube_mask=self._row(obs["cube_mask"]),
            pack_idx=torch.tensor([int(obs["pack_idx"])], device=self.device),
            pick_idx=torch.tensor([int(obs["pick_idx"])], device=self.device),
        )
        scores = self.model(batch, self.cube)[0]  # (C,)
        pack_mask = torch.as_tensor(obs["pack"] > 0, device=self.device)
        if not bool(pack_mask.any()):
            raise ValueError("empty pack passed to NeuralBot")
        masked = scores.masked_fill(~pack_mask, float("-inf"))
        return int(masked.argmax().item())

    def _row(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=self.device).unsqueeze(0)
