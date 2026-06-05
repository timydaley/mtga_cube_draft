"""CC-CPR — Cube-Context Contextual Preference Ranking model (plan §4.4).

    card_repr(c)  = E[c] + alpha * g(f(c))
    pool_repr     = DeepSet({card_repr(c) for c in pool})
    seen_repr     = DeepSet({card_repr(c) for c in seen_unpicked})
    cube_repr     = DeepSet({card_repr(c) for c in cube_list})
    context       = MLP([pool_repr, seen_repr, cube_repr,
                         color_commit_vec(pool), emb(pack_idx), emb(pick_idx)])
    score(c|ctx)  = <context, card_repr(c)> + bias(c)

Operates over a *cube* vocab (≤~540 cards), not the global vocab, so the
DeepSets and the score vector are bounded. The learned embedding `E` is indexed
by GLOBAL vocab id (shared across cubes); a trailing `<UNK>` row backs OOV cards
(§4.3). Card representations for a cube are computed once per forward and reused
across the whole observation batch.

The primary loss is the §4.4 triplet, not an auxiliary softmax. `triplet_loss`
implements it over the in-pack support only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


@dataclass
class CubeTensors:
    """Per-cube static tensors consumed by the model.

    feats      : (C, F) structured card features f(c)
    global_idx : (C,)   global-vocab id of each cube card, for the E lookup
    colors     : (C, 5) WUBRG one-hot, for the color-commit vector
    """

    feats: torch.Tensor
    global_idx: torch.Tensor
    colors: torch.Tensor

    @property
    def size(self) -> int:
        return self.feats.shape[0]

    def to(self, device: torch.device | str) -> CubeTensors:
        return CubeTensors(
            feats=self.feats.to(device),
            global_idx=self.global_idx.to(device),
            colors=self.colors.to(device),
        )


class DeepSet(nn.Module):
    """Permutation-invariant set encoder: rho(weighted_mean(phi(x))).

    Accepts per-element weights (pool/seen counts, or a 0/1 cube mask) so the
    same module handles all three set inputs.
    """

    def __init__(self, repr_dim: int, hidden: int) -> None:
        super().__init__()
        self.phi = _mlp(repr_dim, hidden, hidden)
        self.rho = _mlp(hidden, hidden, repr_dim)

    def forward(self, card_repr: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # card_repr: (C, D); weights: (B, C) -> (B, D)
        phi = self.phi(card_repr)  # (C, H)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
        agg = (weights @ phi) / denom  # (B, H)
        return self.rho(agg)  # (B, D)


@dataclass
class DraftBatch:
    """A batch of pick decisions over a single cube. All tensors on-device."""

    pack: torch.Tensor  # (B, C) 0/1 in-pack mask
    pool: torch.Tensor  # (B, C) counts
    seen: torch.Tensor  # (B, C) counts
    cube_mask: torch.Tensor  # (B, C) 0/1 known-in-cube mask
    pack_idx: torch.Tensor  # (B,) long
    pick_idx: torch.Tensor  # (B,) long


class CCCPR(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        repr_dim: int = 128,
        hidden: int = 256,
        num_packs: int = 3,
        pack_size: int = 15,
        idx_emb: int = 16,
        alpha_floor: float = 0.3,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        # +1 row for the shared <UNK> embedding (global_idx == vocab_size).
        self.embed = nn.Embedding(vocab_size + 1, repr_dim)
        self.unk_index = vocab_size
        self.g = _mlp(feature_dim, hidden, repr_dim)

        self.pool_set = DeepSet(repr_dim, hidden)
        self.seen_set = DeepSet(repr_dim, hidden)
        self.cube_set = DeepSet(repr_dim, hidden)

        self.pack_emb = nn.Embedding(num_packs, idx_emb)
        self.pick_emb = nn.Embedding(pack_size, idx_emb)

        ctx_in = repr_dim * 3 + 5 + idx_emb * 2
        self.context_mlp = _mlp(ctx_in, hidden, repr_dim)
        self.bias = nn.Linear(repr_dim, 1)

        self.alpha_floor = alpha_floor
        # alpha is a buffer (not a parameter): the trainer anneals it 1.0 -> floor.
        self.register_buffer("alpha", torch.tensor(1.0))

    def set_alpha(self, value: float) -> None:
        self.alpha.fill_(max(value, self.alpha_floor))

    def card_repr(self, cube: CubeTensors) -> torch.Tensor:
        """(C, D) representation for every card in the cube."""
        return self.embed(cube.global_idx) + self.alpha * self.g(cube.feats)

    def forward(self, batch: DraftBatch, cube: CubeTensors) -> torch.Tensor:
        """Return per-card scores (B, C). Caller masks to the pack for picks."""
        cr = self.card_repr(cube)  # (C, D)

        pool_repr = self.pool_set(cr, batch.pool.float())
        seen_repr = self.seen_set(cr, batch.seen.float())
        cube_repr = self.cube_set(cr, batch.cube_mask.float())

        pool_f = batch.pool.float()
        color_commit = (pool_f @ cube.colors) / pool_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        context = self.context_mlp(
            torch.cat(
                [
                    pool_repr,
                    seen_repr,
                    cube_repr,
                    color_commit,
                    self.pack_emb(batch.pack_idx),
                    self.pick_emb(batch.pick_idx),
                ],
                dim=1,
            )
        )  # (B, D)

        scores = context @ cr.t() + self.bias(cr).squeeze(-1)  # (B, C)
        return scores


def triplet_loss(
    scores: torch.Tensor,
    pack: torch.Tensor,
    picked: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """CPR triplet loss over the in-pack support (plan §4.4).

    For each example, sum max(0, margin - score(picked) + score(unpicked)) over
    every other card in the pack, averaged over all (example, unpicked) pairs.

    scores : (B, C)  raw scores
    pack   : (B, C)  0/1 in-pack mask
    picked : (B,)    long, index of the chosen card (must be in-pack)
    """
    b = scores.shape[0]
    rows = torch.arange(b, device=scores.device)
    score_picked = scores[rows, picked].unsqueeze(1)  # (B, 1)

    # Unpicked-in-pack cards only.
    neg_mask = pack.bool().clone()
    neg_mask[rows, picked] = False  # exclude the picked card itself

    diff = (margin - score_picked + scores).clamp_min(0.0)  # (B, C)
    losses = diff * neg_mask.float()
    denom = neg_mask.float().sum().clamp_min(1.0)
    return losses.sum() / denom
