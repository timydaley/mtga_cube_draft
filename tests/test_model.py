"""Tests for the CC-CPR model and triplet loss.

Requires torch (the `ml` extra); skipped cleanly when it isn't installed, so the
base test suite still runs on a torch-less machine.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cube_draft.cards import features  # noqa: E402
from cube_draft.model.cc_cpr import CCCPR, CubeTensors, DraftBatch, triplet_loss  # noqa: E402


@pytest.fixture
def cube_tensors(cube):
    cards = [cube.card(i) for i in range(cube.size)]
    return CubeTensors(
        feats=torch.from_numpy(features.feature_matrix(cards)),
        global_idx=torch.tensor(cube.global_indices(), dtype=torch.long),
        colors=torch.from_numpy(features.color_matrix(cards)),
    )


def _random_batch(n_cards: int, batch: int, rng: np.random.Generator):
    pack = np.zeros((batch, n_cards), dtype=np.float32)
    picked = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        in_pack = rng.choice(n_cards, size=5, replace=False)
        pack[b, in_pack] = 1.0
        picked[b] = in_pack[0]
    return DraftBatch(
        pack=torch.from_numpy(pack),
        pool=torch.from_numpy(rng.integers(0, 2, (batch, n_cards)).astype(np.float32)),
        seen=torch.from_numpy(rng.integers(0, 2, (batch, n_cards)).astype(np.float32)),
        cube_mask=torch.ones((batch, n_cards)),
        pack_idx=torch.zeros(batch, dtype=torch.long),
        pick_idx=torch.zeros(batch, dtype=torch.long),
    ), torch.from_numpy(picked)


def test_forward_shape(cube, cube_tensors):
    model = CCCPR(vocab_size=len(cube.global_indices()) + 100, feature_dim=features.FEATURE_DIM)
    batch, _ = _random_batch(cube.size, 8, np.random.default_rng(0))
    scores = model(batch, cube_tensors)
    assert scores.shape == (8, cube.size)
    assert torch.isfinite(scores).all()


def test_triplet_loss_nonnegative_and_finite(cube, cube_tensors):
    model = CCCPR(vocab_size=max(cube.global_indices()) + 1, feature_dim=features.FEATURE_DIM)
    batch, picked = _random_batch(cube.size, 8, np.random.default_rng(1))
    scores = model(batch, cube_tensors)
    loss = triplet_loss(scores, batch.pack, picked, margin=0.2)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_training_step_reduces_loss(cube, cube_tensors):
    """A few gradient steps on a fixed batch should drive the triplet loss down."""
    torch.manual_seed(0)
    model = CCCPR(vocab_size=max(cube.global_indices()) + 1, feature_dim=features.FEATURE_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch, picked = _random_batch(cube.size, 16, np.random.default_rng(2))

    first = None
    for _ in range(50):
        scores = model(batch, cube_tensors)
        loss = triplet_loss(scores, batch.pack, picked, margin=0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first


def test_alpha_floor(cube):
    model = CCCPR(vocab_size=10, feature_dim=features.FEATURE_DIM, alpha_floor=0.3)
    model.set_alpha(0.0)
    assert float(model.alpha) == pytest.approx(0.3)
    model.set_alpha(0.9)
    assert float(model.alpha) == pytest.approx(0.9)
